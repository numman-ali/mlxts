/**
 * Small request coalescing wrapper for non-streaming generation engines.
 * @module
 */

import { ServeError } from "./errors";
import type {
  GenerationEngine,
  NormalizedGenerationRequest,
  NormalizedGenerationResult,
} from "./types";

export type MicroBatchingGenerationEngineOptions = {
  engine: GenerationEngine;
  maxBatchSize?: number;
  batchWindowMs?: number;
};

type PendingGeneration = {
  request: NormalizedGenerationRequest;
  resolve(result: NormalizedGenerationResult): void;
  reject(error: unknown): void;
};

function positiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value <= 0) {
    throw new ServeError(`${name} must be a positive integer.`, {
      code: "invalid_batching_options",
      param: name,
    });
  }
  return value;
}

function nonNegativeInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 0) {
    throw new ServeError(`${name} must be a non-negative integer.`, {
      code: "invalid_batching_options",
      param: name,
    });
  }
  return value;
}

async function runBatch(
  engine: GenerationEngine,
  requests: readonly NormalizedGenerationRequest[],
): Promise<NormalizedGenerationResult[]> {
  if (engine.generateBatch !== undefined) {
    const results = await engine.generateBatch(requests);
    if (results.length !== requests.length) {
      throw new ServeError("Generation engine returned the wrong number of batch results.", {
        code: "invalid_engine_result",
        status: 500,
      });
    }
    return [...results];
  }

  const results: NormalizedGenerationResult[] = [];
  for (const request of requests) {
    results.push(await engine.generate(request));
  }
  return results;
}

/**
 * Coalesce nearby non-streaming `generate()` calls into `generateBatch()` calls.
 *
 * This is admission micro-batching, not continuous token-level batching. It is
 * the right lightweight seam for testing concurrency and for future engines
 * that own a true decode scheduler underneath the same contract.
 */
export function createMicroBatchingGenerationEngine(
  options: MicroBatchingGenerationEngineOptions,
): GenerationEngine {
  const maxBatchSize = positiveInteger(options.maxBatchSize ?? 32, "maxBatchSize");
  const batchWindowMs = nonNegativeInteger(options.batchWindowMs ?? 1, "batchWindowMs");
  const pending: PendingGeneration[] = [];
  let flushScheduled = false;
  let flushInFlight = false;

  async function settleBatch(batch: readonly PendingGeneration[]): Promise<void> {
    try {
      const results = await runBatch(
        options.engine,
        batch.map((entry) => entry.request),
      );
      for (const [index, entry] of batch.entries()) {
        const result = results[index];
        if (result === undefined) {
          entry.reject(
            new ServeError("Generation engine returned an incomplete batch result.", {
              code: "invalid_engine_result",
              status: 500,
            }),
          );
        } else {
          entry.resolve(result);
        }
      }
    } catch (error) {
      for (const entry of batch) {
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
        const batch = pending.splice(0, maxBatchSize);
        if (batch.length === 0) {
          return;
        }
        await settleBatch(batch);
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
    if (flushInFlight) {
      return;
    }
    void flushLoop();
  }

  function scheduleFlush(): void {
    if (flushScheduled || flushInFlight || pending.length === 0) {
      return;
    }
    flushScheduled = true;
    if (batchWindowMs === 0) {
      queueMicrotask(startFlush);
      return;
    }
    setTimeout(startFlush, batchWindowMs);
  }

  function enqueue(request: NormalizedGenerationRequest): Promise<NormalizedGenerationResult> {
    return new Promise((resolve, reject) => {
      pending.push({ request, resolve, reject });
      if (pending.length >= maxBatchSize && !flushInFlight) {
        startFlush();
        return;
      }
      scheduleFlush();
    });
  }

  const batched: GenerationEngine = {
    generate(request) {
      return enqueue(request);
    },
    generateBatch(requests) {
      return Promise.all(requests.map((request) => enqueue(request)));
    },
  };

  const stream = options.engine.stream;
  if (stream !== undefined) {
    batched.stream = (request) => stream(request);
  }

  return batched;
}
