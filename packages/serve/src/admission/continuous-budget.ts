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
  maxPromptTokens?: number;
  maxGeneratedTokens?: number;
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
  readonly #maxScheduledPromptTokens: number;
  readonly #maxScheduledCompletionTokens: number;
  readonly #maxScheduledTotalTokens: number;
  readonly #listeners = new Set<() => void>();
  #scheduledPromptTokens = 0;
  #scheduledCompletionTokens = 0;

  constructor(
    maxScheduledPromptTokens: number,
    maxScheduledCompletionTokens: number,
    maxScheduledTotalTokens: number,
  ) {
    this.#maxScheduledPromptTokens = maxScheduledPromptTokens;
    this.#maxScheduledCompletionTokens = maxScheduledCompletionTokens;
    this.#maxScheduledTotalTokens = maxScheduledTotalTokens;
  }

  tryReserve(request: ContinuousBatchAdmissionRequest) {
    if (request.promptTokens > this.#maxScheduledPromptTokens) {
      return {
        type: "rejected" as const,
        message: `Continuous scheduler request ${request.id} requires ${request.promptTokens} prompt tokens, exceeding the model-level scheduled prompt token budget of ${this.#maxScheduledPromptTokens}.`,
      };
    }
    if (request.maxTokens > this.#maxScheduledCompletionTokens) {
      return {
        type: "rejected" as const,
        message: `Continuous scheduler request ${request.id} requires ${request.maxTokens} completion tokens, exceeding the model-level scheduled completion token budget of ${this.#maxScheduledCompletionTokens}.`,
      };
    }
    if (request.totalTokens > this.#maxScheduledTotalTokens) {
      return {
        type: "rejected" as const,
        message: `Continuous scheduler request ${request.id} requires ${request.totalTokens} total tokens, exceeding the model-level scheduled total token budget of ${this.#maxScheduledTotalTokens}.`,
      };
    }
    if (
      this.#scheduledPromptTokens + request.promptTokens > this.#maxScheduledPromptTokens ||
      this.#scheduledCompletionTokens + request.maxTokens > this.#maxScheduledCompletionTokens ||
      this.#scheduledPromptTokens + this.#scheduledCompletionTokens + request.totalTokens >
        this.#maxScheduledTotalTokens
    ) {
      return {
        type: "deferred" as const,
        reason: this.#deferredReason(request),
        ...this.snapshot(),
      };
    }
    this.#scheduledPromptTokens += request.promptTokens;
    this.#scheduledCompletionTokens += request.maxTokens;
    return {
      type: "reserved" as const,
      reservation: new TokenBudgetReservation(() =>
        this.#release(request.promptTokens, request.maxTokens),
      ),
    };
  }

  snapshot() {
    const scheduledTotalTokens = this.#scheduledPromptTokens + this.#scheduledCompletionTokens;
    return {
      scheduledPromptTokens: this.#scheduledPromptTokens,
      maxScheduledPromptTokens: this.#maxScheduledPromptTokens,
      scheduledCompletionTokens: this.#scheduledCompletionTokens,
      maxScheduledCompletionTokens: this.#maxScheduledCompletionTokens,
      scheduledTotalTokens,
      maxScheduledTotalTokens: this.#maxScheduledTotalTokens,
    };
  }

  onRelease(listener: () => void): () => void {
    this.#listeners.add(listener);
    return () => {
      this.#listeners.delete(listener);
    };
  }

  #release(promptTokens: number, completionTokens: number): void {
    this.#scheduledPromptTokens = Math.max(0, this.#scheduledPromptTokens - promptTokens);
    this.#scheduledCompletionTokens = Math.max(
      0,
      this.#scheduledCompletionTokens - completionTokens,
    );
    for (const listener of this.#listeners) {
      listener();
    }
  }

  #deferredReason(
    request: ContinuousBatchAdmissionRequest,
  ): "scheduled_prompt_budget" | "scheduled_completion_budget" | "scheduled_token_budget" {
    const promptExceeded =
      this.#scheduledPromptTokens + request.promptTokens > this.#maxScheduledPromptTokens;
    const completionExceeded =
      this.#scheduledCompletionTokens + request.maxTokens > this.#maxScheduledCompletionTokens;
    const totalExceeded =
      this.#scheduledPromptTokens + this.#scheduledCompletionTokens + request.totalTokens >
      this.#maxScheduledTotalTokens;
    if (totalExceeded || (promptExceeded && completionExceeded)) {
      return "scheduled_token_budget";
    }
    return promptExceeded ? "scheduled_prompt_budget" : "scheduled_completion_budget";
  }
}

function scaledBudget(limit: number | undefined, maxBatchSize: number): number | undefined {
  return limit === undefined ? undefined : limit * maxBatchSize;
}

export function createContinuousSchedulerTokenBudget(
  options: ContinuousSchedulerTokenBudgetOptions,
): ContinuousBatchAdmissionController | undefined {
  if (options.maxBatchSize <= 1) {
    return undefined;
  }
  const maxScheduledPromptTokens = scaledBudget(
    options.maxPromptTokens ?? options.maxTotalTokens,
    options.maxBatchSize,
  );
  const maxScheduledCompletionTokens = scaledBudget(
    options.maxGeneratedTokens ?? options.maxTotalTokens,
    options.maxBatchSize,
  );
  if (maxScheduledPromptTokens === undefined || maxScheduledCompletionTokens === undefined) {
    return undefined;
  }
  const maxScheduledTotalTokens =
    scaledBudget(options.maxTotalTokens, options.maxBatchSize) ??
    maxScheduledPromptTokens + maxScheduledCompletionTokens;
  return new ContinuousSchedulerTokenBudget(
    maxScheduledPromptTokens,
    maxScheduledCompletionTokens,
    maxScheduledTotalTokens,
  );
}
