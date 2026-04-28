/**
 * Concurrency gate for generation engines.
 * @module
 */

import { ServeError } from "../errors";
import type {
  GenerationEngine,
  GenerationStreamEvent,
  NormalizedGenerationRequest,
  NormalizedGenerationResult,
} from "../types";

export type ConcurrencyLimitGenerationEngineOptions = {
  engine: GenerationEngine;
  maxConcurrentRequests?: number;
};

type Release = () => void;
type Waiter = {
  resolve(release: Release): void;
  reject(error: unknown): void;
  signal?: AbortSignal;
  onAbort?: () => void;
};
type AbortSignalScope = {
  signal?: AbortSignal;
  dispose(): void;
};

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

function cancellationError(): ServeError {
  return new ServeError("Request was cancelled before generation started.", {
    code: "client_cancelled",
    status: 499,
  });
}

function combinedAbortSignalScope(
  requests: readonly NormalizedGenerationRequest[],
): AbortSignalScope {
  const signals = requests
    .map((request) => request.abortSignal)
    .filter((signal): signal is AbortSignal => signal !== undefined);
  const firstSignal = signals[0];
  if (firstSignal === undefined) {
    return {
      dispose() {},
    };
  }
  if (signals.length === 1) {
    return {
      signal: firstSignal,
      dispose() {},
    };
  }

  const controller = new AbortController();
  const cleanups: Array<() => void> = [];
  for (const signal of signals) {
    if (signal.aborted) {
      controller.abort(signal.reason);
      break;
    }
    const onAbort = () => controller.abort(signal.reason);
    signal.addEventListener("abort", onAbort, { once: true });
    cleanups.push(() => signal.removeEventListener("abort", onAbort));
  }
  return {
    signal: controller.signal,
    dispose() {
      for (const cleanup of cleanups) {
        cleanup();
      }
    },
  };
}

async function withAbortSignalScope<T>(
  scope: AbortSignalScope,
  work: (signal: AbortSignal | undefined) => Promise<T>,
): Promise<T> {
  try {
    return await work(scope.signal);
  } finally {
    scope.dispose();
  }
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
  const waiters: Waiter[] = [];
  let inFlight = 0;

  function removeWaiter(waiter: Waiter): void {
    const index = waiters.indexOf(waiter);
    if (index >= 0) {
      waiters.splice(index, 1);
    }
  }

  function cleanupWaiter(waiter: Waiter): void {
    if (waiter.signal !== undefined && waiter.onAbort !== undefined) {
      waiter.signal.removeEventListener("abort", waiter.onAbort);
    }
  }

  function release(): void {
    inFlight -= 1;
    while (waiters.length > 0) {
      const next = waiters.shift();
      if (next === undefined) {
        return;
      }
      cleanupWaiter(next);
      if (next.signal?.aborted) {
        next.reject(cancellationError());
        continue;
      }
      inFlight += 1;
      next.resolve(release);
      return;
    }
  }

  function acquire(signal: AbortSignal | undefined): Promise<Release> {
    if (signal?.aborted) {
      return Promise.reject(cancellationError());
    }
    if (inFlight < maxConcurrentRequests) {
      inFlight += 1;
      return Promise.resolve(release);
    }
    return new Promise((resolve, reject) => {
      const waiter: Waiter = { resolve, reject };
      if (signal !== undefined) {
        waiter.signal = signal;
        waiter.onAbort = () => {
          removeWaiter(waiter);
          cleanupWaiter(waiter);
          reject(cancellationError());
        };
        signal.addEventListener("abort", waiter.onAbort, { once: true });
      }
      waiters.push(waiter);
    });
  }

  async function withPermit<T>(
    signal: AbortSignal | undefined,
    work: () => Promise<T> | T,
  ): Promise<T> {
    const permit = await acquire(signal);
    try {
      return await work();
    } finally {
      permit();
    }
  }

  const limited: GenerationEngine = {
    generate(request) {
      return withPermit(request.abortSignal, () => options.engine.generate(request));
    },
    generateBatch(requests) {
      return withAbortSignalScope(combinedAbortSignalScope(requests), (signal) =>
        withPermit(signal, () => runBatch(options.engine, requests)),
      );
    },
  };

  const stream = options.engine.stream;
  if (stream !== undefined) {
    limited.stream = async (request) => {
      const permit = await acquire(request.abortSignal);
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
