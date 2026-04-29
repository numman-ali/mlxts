import { array, clearMemoryCache, type MxArray, mxEval } from "@mlxts/core";

import type {
  CausalLM,
  PrefillProgressEvent,
  SamplerOptions,
  TransformerBatchCache,
} from "../../types";
import { createBatchCacheForModel } from "./batch-cache-factory";
import { sampleNextBatchToken } from "./continuous-batch-helpers";
import type { PrefillingRequest, ScheduledRequest } from "./continuous-batch-types";

export function emitPrefillingPromptCacheSnapshot(prefilling: PrefillingRequest): void {
  const onPromptCacheSnapshot = prefilling.request.onPromptCacheSnapshot;
  if (onPromptCacheSnapshot === undefined) {
    return;
  }

  using cache = prefilling.cache.extract(0);
  if (cache.offset <= 0) {
    return;
  }
  const snapshot = cache.snapshot();
  try {
    onPromptCacheSnapshot({ offset: snapshot.offset, snapshot });
  } catch (error) {
    snapshot[Symbol.dispose]();
    throw error;
  }
}

function emitReadyBatchPromptCacheSnapshots(
  requests: readonly ScheduledRequest[],
  cache: TransformerBatchCache,
): void {
  for (let batchIndex = 0; batchIndex < requests.length; batchIndex += 1) {
    const request = requests[batchIndex];
    if (request?.onPromptCacheSnapshot === undefined) {
      continue;
    }

    using rowCache = cache.extract(batchIndex);
    if (rowCache.offset <= 0) {
      continue;
    }
    const snapshot = rowCache.snapshot();
    try {
      request.onPromptCacheSnapshot({ offset: snapshot.offset, snapshot });
    } catch (error) {
      snapshot[Symbol.dispose]();
      throw error;
    }
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

function hasPromptCacheSnapshotRows(rows: readonly { onPromptCacheSnapshot?: unknown }[]): boolean {
  return rows.some((row) => row.onPromptCacheSnapshot !== undefined);
}

function finalPromptToken(row: { promptTokenIds: readonly number[] }): number {
  const token = row.promptTokenIds.at(-1);
  if (token === undefined) {
    throw new Error("ContinuousBatchTokenScheduler: prompt row unexpectedly has no final token.");
  }
  return token;
}

export function prefillReadyBatch(
  model: CausalLM,
  rows: readonly ScheduledRequest[],
  padTokenId: number,
  options: SamplerOptions,
): { cache: TransformerBatchCache; nextToken: MxArray } {
  if (hasPromptCacheSnapshotRows(rows)) {
    return prefillReadyBatchWithSnapshots(model, rows, padTokenId, options);
  }

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

function prefillReadyBatchWithSnapshots(
  model: CausalLM,
  rows: readonly ScheduledRequest[],
  padTokenId: number,
  options: SamplerOptions,
): { cache: TransformerBatchCache; nextToken: MxArray } {
  const reusablePrompts = rows.map((row) => row.promptTokenIds.slice(0, -1));
  const { padded, leftPadding } = leftPadPrompts(reusablePrompts, padTokenId);
  const cache = createBatchCacheForModel(model, leftPadding, "ContinuousBatchTokenScheduler");
  const maxReusableLength = Math.max(...reusablePrompts.map((prompt) => prompt.length));

  try {
    if (maxReusableLength > 0) {
      using prefixInput = array(padded, "int32");
      using prefixLogits = model.forward(prefixInput, { cache });
      if (!materializeBatchCacheState(cache)) {
        mxEval(prefixLogits);
      }
    }
    emitReadyBatchPromptCacheSnapshots(rows, cache);
    using finalInput = array(
      rows.map((row) => [finalPromptToken(row)]),
      "int32",
    );
    return {
      cache,
      nextToken: sampleNextBatchToken(
        model,
        finalInput,
        cache,
        rows.map((row) => row.samplerState),
        options,
      ),
    };
  } catch (error) {
    cache[Symbol.dispose]();
    throw error;
  }
}

export function takeReadyPrefillingRow(
  model: CausalLM,
  prefilling: PrefillingRequest[],
  index: number,
): { request: ScheduledRequest; cache: TransformerBatchCache; nextToken: MxArray } | null {
  const prefillingRow = prefilling[index];
  if (prefillingRow === undefined) {
    return null;
  }
  const tail = prefillingRow.request.promptTokenIds.slice(prefillingRow.cursor);
  using tailInput = array([tail], "int32");
  const nextToken = sampleNextBatchToken(
    model,
    tailInput,
    prefillingRow.cache,
    [prefillingRow.request.samplerState],
    prefillingRow.request.samplerOptions,
  );
  const [ready] = prefilling.splice(index, 1);
  if (ready === undefined) {
    nextToken.free();
    return null;
  }
  return { request: ready.request, cache: ready.cache, nextToken };
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
