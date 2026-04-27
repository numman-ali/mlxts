import {
  argmax,
  array,
  clearMemoryCache,
  concatenate,
  type MxArray,
  mxEval,
  reshape,
  slice,
  takeAxis,
} from "@mlxts/core";

import type {
  BatchGenerationOptions,
  CausalLM,
  GenerationResult,
  PrefillProgressEvent,
  SamplerOptions,
  TransformerBatchCache,
} from "../../types";
import { SamplerState } from "../sampling";
import { createBatchCacheForModel } from "./batch-cache-factory";
import { GenerationAbortError } from "./cancellation";
import type { ContinuousBatchTokenRequest, ScheduledRequest } from "./continuous-batch-types";
import { takeLastLogits } from "./helpers";

type AbortablePrefillRequest = {
  abortSignal?: AbortSignal;
  reject(error: unknown): void;
};

type AbortablePrefillRow<Request extends AbortablePrefillRequest> = {
  request: Request;
  cache: TransformerBatchCache;
};

export function integerAtLeast(value: number, name: string, minimum: number): number {
  if (!Number.isInteger(value) || value < minimum) {
    throw new Error(`ContinuousBatchTokenScheduler: ${name} must be >= ${minimum}.`);
  }
  return value;
}

export function validateContinuousBatchOptions(options: BatchGenerationOptions): void {
  if (options.cache !== undefined) {
    throw new Error("ContinuousBatchTokenScheduler: external caches are not supported.");
  }
  if (options.useCache === false) {
    throw new Error("ContinuousBatchTokenScheduler: cache-enabled decoding is required.");
  }
}

export function samplerOptionsFrom(options: SamplerOptions): SamplerOptions {
  return {
    ...(options.temperature === undefined ? {} : { temperature: options.temperature }),
    ...(options.topK === undefined ? {} : { topK: options.topK }),
    ...(options.topP === undefined ? {} : { topP: options.topP }),
    ...(options.minP === undefined ? {} : { minP: options.minP }),
    ...(options.repetitionPenalty === undefined
      ? {}
      : { repetitionPenalty: options.repetitionPenalty }),
    ...(options.seed === undefined ? {} : { seed: options.seed }),
  };
}

export function createScheduledRequest(
  id: string,
  request: ContinuousBatchTokenRequest,
  samplerOptions: SamplerOptions,
  resolve: (result: GenerationResult) => void,
  reject: (error: unknown) => void,
): ScheduledRequest {
  return {
    id,
    promptTokenIds: request.promptTokenIds,
    maxTokens: request.maxTokens,
    samplerOptions,
    samplerState: new SamplerState(request.promptTokenIds, samplerOptions),
    enqueuedAtMs: performance.now(),
    generated: [],
    firstTokenEmitted: false,
    finishReason: "length",
    admissionDeferred: false,
    ...(request.abortSignal === undefined ? {} : { abortSignal: request.abortSignal }),
    ...(request.onPrefillProgress === undefined
      ? {}
      : { onPrefillProgress: request.onPrefillProgress }),
    ...(request.onToken === undefined ? {} : { onToken: request.onToken }),
    resolve,
    reject,
  };
}

export function remainingPrefillTokens(prefilling: {
  request: { promptTokenIds: readonly number[] };
  cursor: number;
}): number {
  return prefilling.request.promptTokenIds.length - prefilling.cursor - 1;
}

export function yieldToScheduler(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

export function rejectAbortedPrefillingRows<Request extends AbortablePrefillRequest>(
  prefilling: AbortablePrefillRow<Request>[],
  cleanup: (request: Request) => void,
): void {
  for (let index = prefilling.length - 1; index >= 0; index -= 1) {
    const row = prefilling[index];
    if (row === undefined || !row.request.abortSignal?.aborted) {
      continue;
    }
    prefilling.splice(index, 1);
    row.cache[Symbol.dispose]();
    cleanup(row.request);
    row.request.reject(
      new GenerationAbortError("ContinuousBatchTokenScheduler: generation was cancelled."),
    );
  }
}

export function disposePrefillingCaches(
  prefilling: readonly { cache: TransformerBatchCache }[],
): void {
  for (const row of prefilling) {
    row.cache[Symbol.dispose]();
  }
}

export function dropAbortedActiveRows(
  active: ScheduledRequest[],
  currentToken: MxArray | null,
  cache: TransformerBatchCache | null,
  onCancelled: (request: ScheduledRequest) => void,
): {
  active: ScheduledRequest[];
  currentToken: MxArray | null;
  cache: TransformerBatchCache | null;
} {
  if (active.length === 0 || currentToken === null || cache === null) {
    return { active, currentToken, cache };
  }

  const keepPositions: number[] = [];
  for (let index = 0; index < active.length; index += 1) {
    const request = active[index];
    if (request === undefined) {
      continue;
    }
    if (request.abortSignal?.aborted) {
      onCancelled(request);
      continue;
    }
    keepPositions.push(index);
  }

  if (keepPositions.length === active.length) {
    return { active, currentToken, cache };
  }
  if (keepPositions.length === 0) {
    currentToken.free();
    cache[Symbol.dispose]();
    return { active: [], currentToken: null, cache: null };
  }

  const nextActive = keepPositions
    .map((index) => active[index])
    .filter((entry) => entry !== undefined);
  cache.filter(keepPositions);
  const nextToken = tokenRows(currentToken, keepPositions);
  mxEval(nextToken);
  currentToken.free();
  return { active: nextActive, currentToken: nextToken, cache };
}

export function shouldChunkInitialRows(
  waiting: readonly { promptTokenIds: readonly number[] }[],
  count: number,
  prefillStepSize: number,
): boolean {
  return waiting
    .slice(0, count)
    .some((request) => request.promptTokenIds.length - 1 > prefillStepSize);
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

export function prefillReadyBatch(
  model: CausalLM,
  rows: readonly { promptTokenIds: readonly number[]; samplerState: SamplerState }[],
  padTokenId: number,
  options: SamplerOptions,
): { cache: TransformerBatchCache; nextToken: MxArray } {
  const prompts = rows.map((row) => row.promptTokenIds);
  const { padded, leftPadding } = leftPadPrompts(prompts, padTokenId);
  const cache = createBatchCacheForModel(model, leftPadding, "ContinuousBatchTokenScheduler");
  using promptInput = array(padded, "int32");
  const nextToken = sampleNextBatchToken(
    model,
    promptInput,
    cache,
    rows.map((row) => row.samplerState),
    options,
  );
  return { cache, nextToken };
}

function sampleGreedyBatchTokenTensor(logits: MxArray, context: string): MxArray {
  const batchSize = logits.shape[0];
  if (batchSize === undefined || logits.shape.length !== 2) {
    throw new Error(
      `${context}: expected rank-2 logits for batch sampling, got [${logits.shape.join(", ")}].`,
    );
  }
  using greedy = argmax(logits, -1);
  return reshape(greedy, [batchSize, 1]);
}

export function sampleNextBatchToken(
  model: CausalLM,
  inputIds: MxArray,
  cache: TransformerBatchCache,
  samplerStates: readonly SamplerState[],
  options: SamplerOptions,
): MxArray {
  using logits = model.forward(inputIds, { cache });
  using lastLogits = takeLastLogits(logits, "ContinuousBatchTokenScheduler");
  const nextToken = sampleBatchTokenTensor(
    lastLogits,
    samplerStates,
    options,
    "ContinuousBatchTokenScheduler",
  );
  mxEval(nextToken);
  return nextToken;
}

function sampleBatchTokenTensor(
  logits: MxArray,
  samplerStates: readonly SamplerState[],
  options: SamplerOptions,
  context: string,
): MxArray {
  const batchSize = logits.shape[0];
  const vocabSize = logits.shape[1];
  if (
    batchSize === undefined ||
    vocabSize === undefined ||
    logits.shape.length !== 2 ||
    batchSize !== samplerStates.length
  ) {
    throw new Error(
      `${context}: expected rank-2 logits with ${samplerStates.length} sampler rows, got [${logits.shape.join(", ")}].`,
    );
  }

  if ((options.temperature ?? 1.0) <= 0 && (options.repetitionPenalty ?? 1.0) <= 1.0) {
    return sampleGreedyBatchTokenTensor(logits, context);
  }

  const sampledRows: MxArray[] = [];
  try {
    for (let row = 0; row < batchSize; row += 1) {
      using rowLogits = slice(logits, [row, 0], [row + 1, vocabSize]);
      const samplerState = samplerStates[row];
      if (samplerState === undefined) {
        throw new Error(`${context}: missing sampler state for row ${row}.`);
      }
      const rowToken = samplerState.sampleTokenTensor(rowLogits, options);
      sampledRows.push(rowToken);
      mxEval(rowToken);
    }

    if (sampledRows.length === 1) {
      const only = sampledRows.pop();
      if (only === undefined) {
        throw new Error(`${context}: sampled no token rows.`);
      }
      return only;
    }

    const sampled = concatenate(sampledRows, 0);
    try {
      mxEval(sampled);
      return sampled;
    } catch (error) {
      sampled.free();
      throw error;
    }
  } finally {
    for (const row of sampledRows) {
      row.free();
    }
  }
}

function materializeBatchCacheState(cache: TransformerBatchCache): boolean {
  const stateArrays = cache.arrays();
  try {
    if (stateArrays.length === 0) {
      return false;
    }
    mxEval(...stateArrays);
    return true;
  } finally {
    for (const stateArray of stateArrays) {
      stateArray.free();
    }
  }
}

export function prefillPromptChunk(
  model: CausalLM,
  promptTokenIds: readonly number[],
  cache: TransformerBatchCache,
  cursor: number,
  prefillStepSize: number,
  remainingPrefillTokens: number,
): { cursor: number; event: PrefillProgressEvent } {
  const chunkSize = Math.min(prefillStepSize, remainingPrefillTokens);
  const chunk = promptTokenIds.slice(cursor, cursor + chunkSize);
  using chunkInput = array([chunk], "int32");
  using chunkLogits = model.forward(chunkInput, { cache });
  if (!materializeBatchCacheState(cache)) {
    mxEval(chunkLogits);
  }
  const nextCursor = cursor + chunkSize;
  clearMemoryCache();
  return {
    cursor: nextCursor,
    event: {
      processedTokens: nextCursor,
      totalTokens: Math.max(promptTokenIds.length - 1, 0),
      chunkTokens: chunkSize,
    },
  };
}

export function tokenTensorToIds(tokens: MxArray): number[] {
  const [batchSize, width] = tokens.shape;
  if (batchSize === undefined || width !== 1 || tokens.shape.length !== 2) {
    throw new Error(
      `ContinuousBatchTokenScheduler: expected token tensor shape [batch, 1], got [${tokens.shape.join(", ")}].`,
    );
  }

  const tokenIds: number[] = [];
  for (let index = 0; index < batchSize; index += 1) {
    using token = slice(tokens, [index, 0], [index + 1, 1]);
    tokenIds.push(token.item());
  }
  return tokenIds;
}

export function tokenRows(tokens: MxArray, indices: readonly number[]): MxArray {
  using indexTensor = array([...indices], "int32");
  return takeAxis(tokens, indexTensor, 0);
}

export function nextInputTensor(
  tokenIds: readonly number[],
  keepPositions: readonly number[],
): MxArray {
  return array(
    keepPositions.map((position) => [tokenIds[position] ?? 0]),
    "int32",
  );
}

export function combineCurrentTokens(left: MxArray | null, right: MxArray): MxArray {
  if (left === null) {
    return tokenRows(
      right,
      Array.from({ length: right.shape[0] ?? 0 }, (_, index) => index),
    );
  }
  using retainedLeft = tokenRows(
    left,
    Array.from({ length: left.shape[0] ?? 0 }, (_, index) => index),
  );
  using retainedRight = tokenRows(
    right,
    Array.from({ length: right.shape[0] ?? 0 }, (_, index) => index),
  );
  const combined = concatenate([retainedLeft, retainedRight], 0);
  mxEval(combined);
  return combined;
}
