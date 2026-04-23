/**
 * Static batched generation helpers for full-cache decoder models.
 * @module
 */

import { argmax, array, type MxArray, mxEval, reshape, slice } from "@mlxts/core";

import type { BatchGenerationOptions, CausalLM, GenerationResult } from "../../types";
import { BatchKVCache } from "../cache";
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

function validateBatchOptions(options: BatchGenerationOptions): void {
  if (options.maxTokens < 0) {
    throw new Error(`generateBatchTokens: maxTokens must be >= 0, got ${options.maxTokens}.`);
  }
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

type BatchStepResult = {
  keepPositions: number[];
  nextActiveOriginalIndices: number[];
};

function recordGeneratedBatchTokens(
  activeOriginalIndices: readonly number[],
  tokenIds: readonly number[],
  eosTokenIds: ReadonlySet<number>,
  results: GenerationResult[],
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

    const eosFinishReason = finishIfEos(eosTokenIds, tokenId);
    if (eosFinishReason !== null) {
      result.finishReason = eosFinishReason;
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
  maxTokens: number,
  eosTokenIds: ReadonlySet<number>,
  results: GenerationResult[],
): void {
  let activeOriginalIndices = [...initialActiveOriginalIndices];
  let currentToken: MxArray | null = initialToken;

  try {
    for (let step = 0; step < maxTokens; step += 1) {
      const tokenIds = tokenTensorToIds(currentToken);
      const { keepPositions, nextActiveOriginalIndices } = recordGeneratedBatchTokens(
        activeOriginalIndices,
        tokenIds,
        eosTokenIds,
        results,
      );

      currentToken.free();
      currentToken = null;

      if (step + 1 >= maxTokens || keepPositions.length === 0) {
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
): GenerationResult[] {
  validateBatchPrompts(promptTokenIdsBatch);
  const padTokenId = options.padTokenId ?? 0;
  const resolvedOptions = resolveGenerationOptions(model, undefined, options);
  validateBatchOptions(resolvedOptions);

  const results = createInitialResults(promptTokenIdsBatch.length);
  if (resolvedOptions.maxTokens === 0) {
    return results;
  }

  const eosTokenIds = new Set(resolvedOptions.eosTokenIds ?? []);
  const { padded, leftPadding } = leftPadPrompts(promptTokenIdsBatch, padTokenId);

  return runGenerationScope(() => {
    using cache = new BatchKVCache(model.layerCount, leftPadding);
    using promptInput = array(padded, "int32");
    const initialToken = sampleNextBatchToken(model, promptInput, cache);
    const activeOriginalIndices = padded.map((_, index) => index);
    runGreedyBatchDecode(
      model,
      cache,
      initialToken,
      activeOriginalIndices,
      resolvedOptions.maxTokens,
      eosTokenIds,
      results,
    );
    return results;
  });
}
