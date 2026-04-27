/**
 * Shared token reservation budget for continuous transformer schedulers.
 * @module
 */

import type {
  ContinuousBatchAdmissionController,
  ContinuousBatchAdmissionRequest,
  ContinuousBatchAdmissionReservation,
} from "@mlxts/transformers";

type ContinuousSchedulerTokenBudgetOptions = {
  maxBatchSize: number;
  maxTotalTokens?: number;
};

class TokenBudgetReservation implements ContinuousBatchAdmissionReservation {
  readonly #release: () => void;
  #released = false;

  constructor(release: () => void) {
    this.#release = release;
  }

  [Symbol.dispose](): void {
    if (this.#released) {
      return;
    }
    this.#released = true;
    this.#release();
  }
}

class ContinuousSchedulerTokenBudget implements ContinuousBatchAdmissionController {
  readonly #maxScheduledTotalTokens: number;
  readonly #listeners = new Set<() => void>();
  #scheduledTotalTokens = 0;

  constructor(maxScheduledTotalTokens: number) {
    this.#maxScheduledTotalTokens = maxScheduledTotalTokens;
  }

  tryReserve(request: ContinuousBatchAdmissionRequest) {
    if (request.totalTokens > this.#maxScheduledTotalTokens) {
      return {
        type: "rejected" as const,
        message: `Continuous scheduler request ${request.id} requires ${request.totalTokens} total tokens, exceeding the model-level scheduled token budget of ${this.#maxScheduledTotalTokens}.`,
      };
    }
    if (this.#scheduledTotalTokens + request.totalTokens > this.#maxScheduledTotalTokens) {
      return {
        type: "deferred" as const,
        ...this.snapshot(),
      };
    }
    this.#scheduledTotalTokens += request.totalTokens;
    return {
      type: "reserved" as const,
      reservation: new TokenBudgetReservation(() => this.#release(request.totalTokens)),
    };
  }

  snapshot() {
    return {
      scheduledTotalTokens: this.#scheduledTotalTokens,
      maxScheduledTotalTokens: this.#maxScheduledTotalTokens,
    };
  }

  onRelease(listener: () => void): () => void {
    this.#listeners.add(listener);
    return () => {
      this.#listeners.delete(listener);
    };
  }

  #release(totalTokens: number): void {
    this.#scheduledTotalTokens = Math.max(0, this.#scheduledTotalTokens - totalTokens);
    for (const listener of this.#listeners) {
      listener();
    }
  }
}

export function createContinuousSchedulerTokenBudget(
  options: ContinuousSchedulerTokenBudgetOptions,
): ContinuousBatchAdmissionController | undefined {
  if (options.maxTotalTokens === undefined || options.maxBatchSize <= 1) {
    return undefined;
  }
  return new ContinuousSchedulerTokenBudget(options.maxTotalTokens * options.maxBatchSize);
}
