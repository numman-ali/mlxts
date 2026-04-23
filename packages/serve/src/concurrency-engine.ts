/**
 * Concurrency gate for generation engines.
 * @module
 */

import { ServeError } from "./errors";
import type {
  GenerationEngine,
  GenerationStreamEvent,
  NormalizedGenerationRequest,
  NormalizedGenerationResult,
} from "./types";

export type ConcurrencyLimitGenerationEngineOptions = {
  engine: GenerationEngine;
  maxConcurrentRequests?: number;
};

type Release = () => void;

function positiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value <= 0) {
    throw new ServeError(`${name} must be a positive integer.`, {
      code: "invalid_concurrency_options",
      param: name,
    });
  }
  return value;
}

function runBatch(
  engine: GenerationEngine,
  requests: readonly NormalizedGenerationRequest[],
): Promise<readonly NormalizedGenerationResult[]> | readonly NormalizedGenerationResult[] {
  if (engine.generateBatch !== undefined) {
    return engine.generateBatch(requests);
  }

  return Promise.all(requests.map((request) => engine.generate(request)));
}

/**
 * Serialize generation work through a bounded in-flight gate.
 *
 * This is admission control, not continuous batching. It guarantees that one
 * model-backed engine only runs a bounded number of concurrent generation jobs
 * across `generate`, `generateBatch`, and `stream`.
 */
export function createConcurrencyLimitGenerationEngine(
  options: ConcurrencyLimitGenerationEngineOptions,
): GenerationEngine {
  const maxConcurrentRequests = positiveInteger(
    options.maxConcurrentRequests ?? 1,
    "maxConcurrentRequests",
  );
  const waiters: Array<(release: Release) => void> = [];
  let inFlight = 0;

  function release(): void {
    inFlight -= 1;
    const next = waiters.shift();
    if (next !== undefined) {
      inFlight += 1;
      next(release);
    }
  }

  function acquire(): Promise<Release> {
    if (inFlight < maxConcurrentRequests) {
      inFlight += 1;
      return Promise.resolve(release);
    }
    return new Promise((resolve) => {
      waiters.push(resolve);
    });
  }

  async function withPermit<T>(work: () => Promise<T> | T): Promise<T> {
    const permit = await acquire();
    try {
      return await work();
    } finally {
      permit();
    }
  }

  const limited: GenerationEngine = {
    generate(request) {
      return withPermit(() => options.engine.generate(request));
    },
    generateBatch(requests) {
      return withPermit(() => runBatch(options.engine, requests));
    },
  };

  const stream = options.engine.stream;
  if (stream !== undefined) {
    limited.stream = async (request) => {
      const permit = await acquire();
      let source: AsyncIterable<GenerationStreamEvent>;
      try {
        source = await stream(request);
      } catch (error) {
        permit();
        throw error;
      }
      return (async function* () {
        try {
          yield* source;
        } finally {
          permit();
        }
      })();
    };
  }

  return limited;
}
