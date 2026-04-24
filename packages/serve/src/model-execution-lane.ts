/**
 * Single-lane execution guard for one loaded MLX model.
 * @module
 */

import { GenerationAbortError } from "@mlxts/transformers";

type Release = () => void;
type Waiter = {
  resolve(release: Release): void;
  reject(error: unknown): void;
  signal?: AbortSignal;
  onAbort?: () => void;
};

function cancellationError(): GenerationAbortError {
  return new GenerationAbortError("ModelExecutionLane: generation was cancelled before dispatch.");
}

/** Serialize model-backed work while still allowing a scheduler to batch internally. */
export class ModelExecutionLane {
  readonly #maxConcurrentJobs: number;
  readonly #waiters: Waiter[] = [];
  #inFlight = 0;

  constructor(maxConcurrentJobs = 1) {
    if (!Number.isInteger(maxConcurrentJobs) || maxConcurrentJobs <= 0) {
      throw new Error("ModelExecutionLane: maxConcurrentJobs must be a positive integer.");
    }
    this.#maxConcurrentJobs = maxConcurrentJobs;
  }

  acquire(signal?: AbortSignal): Promise<Release> {
    if (signal?.aborted) {
      return Promise.reject(cancellationError());
    }
    if (this.#inFlight < this.#maxConcurrentJobs) {
      this.#inFlight += 1;
      return Promise.resolve(() => this.#release());
    }

    return new Promise((resolve, reject) => {
      const waiter: Waiter = { resolve, reject };
      if (signal !== undefined) {
        waiter.signal = signal;
        waiter.onAbort = () => {
          this.#removeWaiter(waiter);
          this.#cleanup(waiter);
          reject(cancellationError());
        };
        signal.addEventListener("abort", waiter.onAbort, { once: true });
      }
      this.#waiters.push(waiter);
    });
  }

  async run<T>(work: () => Promise<T>, signal?: AbortSignal): Promise<T> {
    const release = await this.acquire(signal);
    try {
      return await work();
    } finally {
      release();
    }
  }

  #release(): void {
    this.#inFlight -= 1;
    while (this.#waiters.length > 0) {
      const next = this.#waiters.shift();
      if (next === undefined) {
        return;
      }
      this.#cleanup(next);
      if (next.signal?.aborted) {
        next.reject(cancellationError());
        continue;
      }
      this.#inFlight += 1;
      next.resolve(() => this.#release());
      return;
    }
  }

  #removeWaiter(waiter: Waiter): void {
    const index = this.#waiters.indexOf(waiter);
    if (index >= 0) {
      this.#waiters.splice(index, 1);
    }
  }

  #cleanup(waiter: Waiter): void {
    if (waiter.signal !== undefined && waiter.onAbort !== undefined) {
      waiter.signal.removeEventListener("abort", waiter.onAbort);
    }
  }
}
