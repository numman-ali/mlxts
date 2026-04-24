/**
 * Static batched generation helpers for full-cache decoder models.
 * @module
 */

import { argmax, array, type MxArray, mxEval, reshape, slice } from "@mlxts/core";

import type {
  BatchGenerationOptions,
  BatchTokenGenerationEvent,
  CausalLM,
  GenerationResult,
} from "../../types";
import { BatchKVCache } from "../cache";
import { throwIfGenerationAborted } from "./cancellation";
import { resolveGenerationOptions } from "./defaults";
import { takeLastLogits } from "./helpers";
import { finishIfEos, runGenerationScope } from "./runtime";

function validateBatchPrompts(promptTokenIdsBatch: readonly (readonly number[])[]): void {
  if (promptTokenIdsBatch.length === 0) {
    throw new Error("generateBatchTokens: promptTokenIdsBatch must contain at least one prompt.");
  }
  for (let index = 0; index < promptTokenIdsBatch.length; index += 1) {
    const prompt = promptTokenIdsBatch[index];
    if (prompt === undefined || prompt.length === 0) {
      throw new Error(`generateBatchTokens: prompt ${index} must contain at least one token.`);
    }
  }
}

function normalizeBatchMaxTokens(
  value: BatchGenerationOptions["maxTokens"],
  batchSize: number,
): number[] {
  if (typeof value === "number") {
    if (!Number.isInteger(value) || value < 0) {
      throw new Error(
        `generateBatchTokens: maxTokens must be a non-negative integer, got ${value}.`,
      );
    }
    return Array.from({ length: batchSize }, () => value);
  }

  if (value.length !== batchSize) {
    throw new Error(
      `generateBatchTokens: maxTokens length ${value.length} must match batch size ${batchSize}.`,
    );
  }
  for (const maxTokens of value) {
    if (!Number.isInteger(maxTokens) || maxTokens < 0) {
      throw new Error(
        `generateBatchTokens: maxTokens values must be non-negative integers, got ${maxTokens}.`,
      );
    }
  }

  return [...value];
}

function validateBatchOptions(options: BatchGenerationOptions): void {
  if (options.cache !== undefined) {
    throw new Error(
      "generateBatchTokens: external caches are not supported for batched generation.",
    );
  }
  if (options.useCache === false) {
    throw new Error("generateBatchTokens: batched generation requires cache-enabled decoding.");
  }
  if ((options.temperature ?? 1.0) > 0) {
    throw new Error("generateBatchTokens: batched generation currently requires temperature: 0.");
  }
  if ((options.repetitionPenalty ?? 1.0) !== 1.0) {
    throw new Error(
      "generateBatchTokens: batched generation does not yet support repetition penalty.",
    );
  }
}

function leftPadPrompts(
  prompts: readonly (readonly number[])[],
  padTokenId: number,
): { padded: number[][]; leftPadding: number[] } {
  const maxLength = Math.max(...prompts.map((prompt) => prompt.length));
  const padded: number[][] = [];
  const leftPadding: number[] = [];

  for (const prompt of prompts) {
    const padding = maxLength - prompt.length;
    leftPadding.push(padding);
    padded.push([...Array<number>(padding).fill(padTokenId), ...prompt]);
  }

  return { padded, leftPadding };
}

function sampleGreedyBatchTokenTensor(logits: MxArray): MxArray {
  const batchSize = logits.shape[0];
  if (batchSize === undefined || logits.shape.length !== 2) {
    throw new Error(
      `generateBatchTokens: expected rank-2 logits for batch sampling, got [${logits.shape.join(", ")}].`,
    );
  }
  using greedy = argmax(logits, -1);
  return reshape(greedy, [batchSize, 1]);
}

function tokenTensorToIds(tokens: MxArray): number[] {
  const [batchSize, width] = tokens.shape;
  if (batchSize === undefined || width !== 1 || tokens.shape.length !== 2) {
    throw new Error(
      `generateBatchTokens: expected token tensor shape [batch, 1], got [${tokens.shape.join(", ")}].`,
    );
  }

  const tokenIds: number[] = [];
  for (let index = 0; index < batchSize; index += 1) {
    using token = slice(tokens, [index, 0], [index + 1, 1]);
    tokenIds.push(token.item());
  }
  return tokenIds;
}

function nextInputTensor(tokenIds: readonly number[], keepPositions: readonly number[]): MxArray {
  return array(
    keepPositions.map((position) => [tokenIds[position] ?? 0]),
    "int32",
  );
}

function sampleNextBatchToken(model: CausalLM, inputIds: MxArray, cache: BatchKVCache): MxArray {
  using logits = model.forward(inputIds, { cache });
  using lastLogits = takeLastLogits(logits, "generateBatchTokens");
  const nextToken = sampleGreedyBatchTokenTensor(lastLogits);
  mxEval(nextToken);
  return nextToken;
}

function createInitialResults(batchSize: number): GenerationResult[] {
  return Array.from({ length: batchSize }, () => ({
    tokenIds: [],
    finishReason: "length" as const,
  }));
}

function activePromptIndices(maxTokens: readonly number[]): number[] {
  const activeIndices: number[] = [];
  for (let index = 0; index < maxTokens.length; index += 1) {
    if ((maxTokens[index] ?? 0) > 0) {
      activeIndices.push(index);
    }
  }
  return activeIndices;
}

function activePrompts(
  prompts: readonly (readonly number[])[],
  activeIndices: readonly number[],
): readonly (readonly number[])[] {
  return activeIndices.map((index) => prompts[index] ?? []);
}

function maxTokensForPrompt(maxTokens: readonly number[], originalIndex: number): number {
  const value = maxTokens[originalIndex];
  if (value === undefined) {
    throw new Error(`generateBatchTokens: missing max token limit for prompt ${originalIndex}.`);
  }
  return value;
}

function emitBatchDone(
  onEvent: ((event: BatchTokenGenerationEvent) => void) | undefined,
  batchIndex: number,
  result: GenerationResult,
): void {
  onEvent?.({
    type: "done",
    batchIndex,
    tokenIds: [...result.tokenIds],
    finishReason: result.finishReason,
  });
}

function emitZeroTokenDoneEvents(
  results: readonly GenerationResult[],
  maxTokens: readonly number[],
  onEvent: ((event: BatchTokenGenerationEvent) => void) | undefined,
): void {
  if (onEvent === undefined) {
    return;
  }
  for (let index = 0; index < maxTokens.length; index += 1) {
    if ((maxTokens[index] ?? 0) === 0) {
      const result = results[index];
      if (result !== undefined) {
        emitBatchDone(onEvent, index, result);
      }
    }
  }
}

type BatchStepResult = {
  keepPositions: number[];
  nextActiveOriginalIndices: number[];
};

function recordGeneratedBatchTokens(
  activeOriginalIndices: readonly number[],
  tokenIds: readonly number[],
  eosTokenIds: ReadonlySet<number>,
  maxTokens: readonly number[],
  results: GenerationResult[],
  onEvent: ((event: BatchTokenGenerationEvent) => void) | undefined,
): BatchStepResult {
  const keepPositions: number[] = [];
  const nextActiveOriginalIndices: number[] = [];

  for (let position = 0; position < activeOriginalIndices.length; position += 1) {
    const originalIndex = activeOriginalIndices[position];
    const tokenId = tokenIds[position];
    if (originalIndex === undefined || tokenId === undefined) {
      continue;
    }

    const result = results[originalIndex];
    if (result === undefined) {
      continue;
    }
    result.tokenIds.push(tokenId);
    onEvent?.({
      type: "token",
      batchIndex: originalIndex,
      tokenId,
      completionTokens: result.tokenIds.length,
    });

    const eosFinishReason = finishIfEos(eosTokenIds, tokenId);
    if (eosFinishReason !== null) {
      result.finishReason = eosFinishReason;
      emitBatchDone(onEvent, originalIndex, result);
      continue;
    }

    if (result.tokenIds.length >= maxTokensForPrompt(maxTokens, originalIndex)) {
      emitBatchDone(onEvent, originalIndex, result);
      continue;
    }

    keepPositions.push(position);
    nextActiveOriginalIndices.push(originalIndex);
  }

  return { keepPositions, nextActiveOriginalIndices };
}

function runGreedyBatchDecode(
  model: CausalLM,
  cache: BatchKVCache,
  initialToken: MxArray,
  initialActiveOriginalIndices: readonly number[],
  maxTokens: readonly number[],
  eosTokenIds: ReadonlySet<number>,
  results: GenerationResult[],
  onEvent: ((event: BatchTokenGenerationEvent) => void) | undefined,
  abortSignal: AbortSignal | undefined,
): void {
  let activeOriginalIndices = [...initialActiveOriginalIndices];
  let currentToken: MxArray | null = initialToken;
  const maxDecodeSteps = Math.max(...maxTokens);

  try {
    for (let step = 0; step < maxDecodeSteps; step += 1) {
      throwIfGenerationAborted(abortSignal, "generateBatchTokens");
      const tokenIds = tokenTensorToIds(currentToken);
      const { keepPositions, nextActiveOriginalIndices } = recordGeneratedBatchTokens(
        activeOriginalIndices,
        tokenIds,
        eosTokenIds,
        maxTokens,
        results,
        onEvent,
      );
      throwIfGenerationAborted(abortSignal, "generateBatchTokens");

      currentToken.free();
      currentToken = null;

      if (step + 1 >= maxDecodeSteps || keepPositions.length === 0) {
        break;
      }

      if (keepPositions.length !== activeOriginalIndices.length) {
        cache.filter(keepPositions);
      }
      activeOriginalIndices = nextActiveOriginalIndices;

      using nextInput = nextInputTensor(tokenIds, keepPositions);
      currentToken = sampleNextBatchToken(model, nextInput, cache);
    }
  } finally {
    currentToken?.free();
  }
}

/** Generate token continuations for a fixed prompt batch using greedy full-cache decoding. */
export function generateBatchTokensInternal(
  model: CausalLM,
  promptTokenIdsBatch: readonly (readonly number[])[],
  options: BatchGenerationOptions,
  onEvent?: (event: BatchTokenGenerationEvent) => void,
): GenerationResult[] {
  validateBatchPrompts(promptTokenIdsBatch);
  const maxTokens = normalizeBatchMaxTokens(options.maxTokens, promptTokenIdsBatch.length);
  const padTokenId = options.padTokenId ?? 0;
  const resolvedBaseOptions = resolveGenerationOptions(model, undefined, {
    ...options,
    maxTokens: Math.max(...maxTokens),
  });
  const resolvedOptions: BatchGenerationOptions = {
    ...resolvedBaseOptions,
    maxTokens,
    ...(options.padTokenId === undefined ? {} : { padTokenId: options.padTokenId }),
  };
  validateBatchOptions(resolvedOptions);

  const results = createInitialResults(promptTokenIdsBatch.length);
  emitZeroTokenDoneEvents(results, maxTokens, onEvent);
  const initialActiveOriginalIndices = activePromptIndices(maxTokens);
  if (initialActiveOriginalIndices.length === 0) {
    return results;
  }

  const eosTokenIds = new Set(resolvedOptions.eosTokenIds ?? []);
  const { padded, leftPadding } = leftPadPrompts(
    activePrompts(promptTokenIdsBatch, initialActiveOriginalIndices),
    padTokenId,
  );

  return runGenerationScope(() => {
    throwIfGenerationAborted(resolvedOptions.abortSignal, "generateBatchTokens");
    using cache = new BatchKVCache(model.layerCount, leftPadding);
    using promptInput = array(padded, "int32");
    const initialToken = sampleNextBatchToken(model, promptInput, cache);
    throwIfGenerationAborted(resolvedOptions.abortSignal, "generateBatchTokens");
    runGreedyBatchDecode(
      model,
      cache,
      initialToken,
      initialActiveOriginalIndices,
      maxTokens,
      eosTokenIds,
      results,
      onEvent,
      resolvedOptions.abortSignal,
    );
    return results;
  });
}
