/**
 * Static request batching for transformer-backed serving.
 * @module
 */

import { ServeError } from "../errors";
import { linkAbortSignals } from "../http/abort";
import { transformersRuntimeStrategy } from "../runtime/strategy";
import type { NormalizedGenerationRequest, NormalizedGenerationResult } from "../types";
import { generateTransformersBatch } from "./batch";
import type { ModelExecutionLane, ModelExecutionLaneStats } from "./execution-lane";
import type { TransformersGenerationEngineOptions } from "./index";
import { continuousBatchIneligibilityReason, staticBatchIneligibilityReason } from "./routing";

type PendingStaticGeneration = {
  request: NormalizedGenerationRequest;
  resolve(result: NormalizedGenerationResult): void;
  reject(error: unknown): void;
  onAbort?: () => void;
};

function maxBatchSize(options: TransformersGenerationEngineOptions): number {
  return transformersRuntimeStrategy(options).scheduler.maxBatchSize;
}

function batchWindowMs(options: TransformersGenerationEngineOptions): number {
  return transformersRuntimeStrategy(options).scheduler.batchWindowMs;
}

function cancellationError(): ServeError {
  return new ServeError("Request was cancelled before static batch generation started.", {
    code: "client_cancelled",
    status: 499,
  });
}

function emitModelLaneWaitForBatch(
  options: TransformersGenerationEngineOptions,
  requests: readonly NormalizedGenerationRequest[],
  queuedStats: ModelExecutionLaneStats,
  dispatchStats: ModelExecutionLaneStats,
  waitMs: number,
): void {
  for (const request of requests) {
    options.onEvent?.({
      type: "generation_model_lane_wait",
      id: request.id,
      protocol: request.protocol,
      model: request.model,
      lane: "model",
      waitMs,
      inFlightAtQueue: queuedStats.inFlight,
      queuedAhead: queuedStats.queued,
      inFlightAtDispatch: dispatchStats.inFlight,
      queuedAtDispatch: dispatchStats.queued,
      maxConcurrentJobs: queuedStats.maxConcurrentJobs,
    });
  }
}

function staticBatchOnlyEligible(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
): boolean {
  return (
    maxBatchSize(options) > 1 &&
    staticBatchIneligibilityReason(request, options) === "eligible" &&
    continuousBatchIneligibilityReason(request, options) !== "eligible"
  );
}

function cleanupPending(entry: PendingStaticGeneration): void {
  if (entry.onAbort !== undefined) {
    entry.request.abortSignal?.removeEventListener("abort", entry.onAbort);
    delete entry.onAbort;
  }
}

function activeEntries(entries: readonly PendingStaticGeneration[]): PendingStaticGeneration[] {
  const active: PendingStaticGeneration[] = [];
  for (const entry of entries) {
    cleanupPending(entry);
    if (entry.request.abortSignal?.aborted) {
      entry.reject(cancellationError());
      continue;
    }
    active.push(entry);
  }
  return active;
}

function linkedBatchSignal(
  requests: readonly NormalizedGenerationRequest[],
): ReturnType<typeof linkAbortSignals> | undefined {
  const signals: AbortSignal[] = [];
  for (const request of requests) {
    if (request.abortSignal !== undefined) {
      signals.push(request.abortSignal);
    }
  }
  return signals.length === 0 ? undefined : linkAbortSignals(...signals);
}

/** Run static batch generation while holding this model's execution lane. */
export async function runStaticBatchOnModelLane(
  lane: ModelExecutionLane,
  options: TransformersGenerationEngineOptions,
  requests: readonly NormalizedGenerationRequest[],
): Promise<NormalizedGenerationResult[]> {
  const queuedStats = lane.stats();
  const queuedAt = performance.now();
  const abortScope = linkedBatchSignal(requests);
  let release: (() => void) | undefined;
  try {
    release = await lane.acquire(abortScope?.signal);
  } catch (error) {
    abortScope?.dispose();
    throw error;
  }
  emitModelLaneWaitForBatch(
    options,
    requests,
    queuedStats,
    lane.stats(),
    performance.now() - queuedAt,
  );
  try {
    return await generateTransformersBatch(requests, options);
  } finally {
    release?.();
    abortScope?.dispose();
  }
}

/** Coalesce concurrent static-only requests into `generateBatchTokens()`. */
export function createStaticTransformersGeneration(
  options: TransformersGenerationEngineOptions,
  lane: ModelExecutionLane,
): {
  generate(request: NormalizedGenerationRequest): Promise<NormalizedGenerationResult> | null;
} {
  const pending: PendingStaticGeneration[] = [];
  let flushScheduled = false;
  let flushInFlight = false;

  async function settleBatch(batch: readonly PendingStaticGeneration[]): Promise<void> {
    const active = activeEntries(batch);
    if (active.length === 0) {
      return;
    }

    const requests = active.map((entry) => entry.request);
    try {
      const results = await runStaticBatchOnModelLane(lane, options, requests);
      for (let index = 0; index < active.length; index += 1) {
        const entry = active[index];
        const result = results[index];
        if (entry === undefined) {
          continue;
        }
        if (result === undefined) {
          entry.reject(new Error("Static batch generation returned an incomplete result."));
          continue;
        }
        entry.resolve(result);
      }
    } catch (error) {
      for (const entry of active) {
        entry.reject(error);
      }
    }
  }

  async function flushLoop(): Promise<void> {
    if (flushInFlight) {
      return;
    }
    flushInFlight = true;
    flushScheduled = false;
    try {
      while (pending.length > 0) {
        await settleBatch(pending.splice(0, maxBatchSize(options)));
      }
    } finally {
      flushInFlight = false;
      if (pending.length > 0) {
        scheduleFlush();
      }
    }
  }

  function startFlush(): void {
    flushScheduled = false;
    if (!flushInFlight) {
      void flushLoop();
    }
  }

  function scheduleFlush(): void {
    if (flushScheduled || flushInFlight || pending.length === 0) {
      return;
    }
    flushScheduled = true;
    const waitMs = batchWindowMs(options);
    if (waitMs === 0) {
      queueMicrotask(startFlush);
      return;
    }
    setTimeout(startFlush, waitMs);
  }

  function enqueue(request: NormalizedGenerationRequest): Promise<NormalizedGenerationResult> {
    if (request.abortSignal?.aborted) {
      return Promise.reject(cancellationError());
    }
    return new Promise((resolve, reject) => {
      const entry: PendingStaticGeneration = { request, resolve, reject };
      if (request.abortSignal !== undefined) {
        entry.onAbort = () => {
          const index = pending.indexOf(entry);
          if (index >= 0) {
            pending.splice(index, 1);
          }
          cleanupPending(entry);
          reject(cancellationError());
        };
        request.abortSignal.addEventListener("abort", entry.onAbort, { once: true });
      }
      pending.push(entry);
      if (pending.length >= maxBatchSize(options) && !flushInFlight) {
        startFlush();
        return;
      }
      scheduleFlush();
    });
  }

  return {
    generate(request) {
      return staticBatchOnlyEligible(request, options) ? enqueue(request) : null;
    },
  };
}
