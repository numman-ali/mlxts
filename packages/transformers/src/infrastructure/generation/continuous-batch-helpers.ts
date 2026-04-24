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

import type { BatchGenerationOptions, CausalLM, PrefillProgressEvent } from "../../types";
import { BatchKVCache } from "../cache";
import { GenerationAbortError } from "./cancellation";
import { takeLastLogits } from "./helpers";

type AbortablePrefillRequest = {
  abortSignal?: AbortSignal;
  reject(error: unknown): void;
};

type AbortablePrefillRow<Request extends AbortablePrefillRequest> = {
  request: Request;
  cache: BatchKVCache;
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
  if ((options.temperature ?? 1.0) > 0) {
    throw new Error("ContinuousBatchTokenScheduler: continuous batching requires temperature: 0.");
  }
  if ((options.repetitionPenalty ?? 1.0) !== 1.0) {
    throw new Error("ContinuousBatchTokenScheduler: repetition penalty is not supported yet.");
  }
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
  prompts: readonly (readonly number[])[],
  padTokenId: number,
): { cache: BatchKVCache; nextToken: MxArray } {
  const { padded, leftPadding } = leftPadPrompts(prompts, padTokenId);
  const cache = new BatchKVCache(model.layerCount, leftPadding);
  using promptInput = array(padded, "int32");
  const nextToken = sampleNextBatchToken(model, promptInput, cache);
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
  cache: BatchKVCache,
): MxArray {
  using logits = model.forward(inputIds, { cache });
  using lastLogits = takeLastLogits(logits, "ContinuousBatchTokenScheduler");
  const nextToken = sampleGreedyBatchTokenTensor(lastLogits, "ContinuousBatchTokenScheduler");
  mxEval(nextToken);
  return nextToken;
}

function materializeBatchCacheState(cache: BatchKVCache): boolean {
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
  cache: BatchKVCache,
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
