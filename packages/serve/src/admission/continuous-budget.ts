/**
 * Shared reservation budget for continuous transformer schedulers.
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
  maxScheduledMemoryBytes?: number;
  estimateMemoryBytes?: (request: ContinuousBatchAdmissionRequest) => number | undefined;
};

class SchedulerBudgetReservation implements ContinuousBatchAdmissionReservation {
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

class ContinuousSchedulerBudget implements ContinuousBatchAdmissionController {
  readonly #maxScheduledPromptTokens: number | null;
  readonly #maxScheduledCompletionTokens: number | null;
  readonly #maxScheduledTotalTokens: number | null;
  readonly #maxScheduledMemoryBytes: number | null;
  readonly #estimateMemoryBytes: (request: ContinuousBatchAdmissionRequest) => number | undefined;
  readonly #listeners = new Set<() => void>();
  #scheduledPromptTokens = 0;
  #scheduledCompletionTokens = 0;
  #scheduledMemoryBytes = 0;

  constructor(
    maxScheduledPromptTokens: number | null,
    maxScheduledCompletionTokens: number | null,
    maxScheduledTotalTokens: number | null,
    maxScheduledMemoryBytes: number | null,
    estimateMemoryBytes: (request: ContinuousBatchAdmissionRequest) => number | undefined,
  ) {
    this.#maxScheduledPromptTokens = maxScheduledPromptTokens;
    this.#maxScheduledCompletionTokens = maxScheduledCompletionTokens;
    this.#maxScheduledTotalTokens = maxScheduledTotalTokens;
    this.#maxScheduledMemoryBytes = maxScheduledMemoryBytes;
    this.#estimateMemoryBytes = estimateMemoryBytes;
  }

  tryReserve(request: ContinuousBatchAdmissionRequest) {
    const memoryBytes = this.#requestMemoryBytes(request);
    if (exceedsLimit(request.promptTokens, this.#maxScheduledPromptTokens)) {
      return {
        type: "rejected" as const,
        message: `Continuous scheduler request ${request.id} requires ${request.promptTokens} prompt tokens, exceeding the model-level scheduled prompt token budget of ${this.#maxScheduledPromptTokens}.`,
      };
    }
    if (exceedsLimit(request.maxTokens, this.#maxScheduledCompletionTokens)) {
      return {
        type: "rejected" as const,
        message: `Continuous scheduler request ${request.id} requires ${request.maxTokens} completion tokens, exceeding the model-level scheduled completion token budget of ${this.#maxScheduledCompletionTokens}.`,
      };
    }
    if (exceedsLimit(request.totalTokens, this.#maxScheduledTotalTokens)) {
      return {
        type: "rejected" as const,
        message: `Continuous scheduler request ${request.id} requires ${request.totalTokens} total tokens, exceeding the model-level scheduled total token budget of ${this.#maxScheduledTotalTokens}.`,
      };
    }
    if (exceedsLimit(memoryBytes, this.#maxScheduledMemoryBytes)) {
      return {
        type: "rejected" as const,
        message: `Continuous scheduler request ${request.id} requires estimated memory ${formatBytes(memoryBytes)}, exceeding the model-level scheduled memory budget of ${formatBytes(this.#maxScheduledMemoryBytes ?? 0)}.`,
      };
    }
    if (
      exceedsLimit(
        this.#scheduledPromptTokens + request.promptTokens,
        this.#maxScheduledPromptTokens,
      ) ||
      exceedsLimit(
        this.#scheduledCompletionTokens + request.maxTokens,
        this.#maxScheduledCompletionTokens,
      ) ||
      exceedsLimit(
        this.#scheduledPromptTokens + this.#scheduledCompletionTokens + request.totalTokens,
        this.#maxScheduledTotalTokens,
      ) ||
      exceedsLimit(this.#scheduledMemoryBytes + memoryBytes, this.#maxScheduledMemoryBytes)
    ) {
      return {
        type: "deferred" as const,
        reason: this.#deferredReason(request, memoryBytes),
        ...this.snapshot(),
      };
    }
    this.#scheduledPromptTokens += request.promptTokens;
    this.#scheduledCompletionTokens += request.maxTokens;
    this.#scheduledMemoryBytes += memoryBytes;
    return {
      type: "reserved" as const,
      reservation: new SchedulerBudgetReservation(() =>
        this.#release(request.promptTokens, request.maxTokens, memoryBytes),
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
      scheduledMemoryBytes: this.#scheduledMemoryBytes,
      maxScheduledMemoryBytes: this.#maxScheduledMemoryBytes,
    };
  }

  onRelease(listener: () => void): () => void {
    this.#listeners.add(listener);
    return () => {
      this.#listeners.delete(listener);
    };
  }

  #release(promptTokens: number, completionTokens: number, memoryBytes: number): void {
    this.#scheduledPromptTokens = Math.max(0, this.#scheduledPromptTokens - promptTokens);
    this.#scheduledCompletionTokens = Math.max(
      0,
      this.#scheduledCompletionTokens - completionTokens,
    );
    this.#scheduledMemoryBytes = Math.max(0, this.#scheduledMemoryBytes - memoryBytes);
    for (const listener of this.#listeners) {
      listener();
    }
  }

  #requestMemoryBytes(request: ContinuousBatchAdmissionRequest): number {
    if (this.#maxScheduledMemoryBytes === null) {
      return 0;
    }
    const estimated = this.#estimateMemoryBytes(request);
    if (estimated === undefined || !Number.isFinite(estimated) || estimated <= 0) {
      return 0;
    }
    return Math.ceil(estimated);
  }

  #deferredReason(
    request: ContinuousBatchAdmissionRequest,
    memoryBytes: number,
  ):
    | "scheduled_prompt_budget"
    | "scheduled_completion_budget"
    | "scheduled_token_budget"
    | "scheduled_memory_budget" {
    const promptExceeded = exceedsLimit(
      this.#scheduledPromptTokens + request.promptTokens,
      this.#maxScheduledPromptTokens,
    );
    const completionExceeded = exceedsLimit(
      this.#scheduledCompletionTokens + request.maxTokens,
      this.#maxScheduledCompletionTokens,
    );
    const totalExceeded = exceedsLimit(
      this.#scheduledPromptTokens + this.#scheduledCompletionTokens + request.totalTokens,
      this.#maxScheduledTotalTokens,
    );
    if (totalExceeded || (promptExceeded && completionExceeded)) {
      return "scheduled_token_budget";
    }
    if (promptExceeded || completionExceeded) {
      return promptExceeded ? "scheduled_prompt_budget" : "scheduled_completion_budget";
    }
    if (exceedsLimit(this.#scheduledMemoryBytes + memoryBytes, this.#maxScheduledMemoryBytes)) {
      return "scheduled_memory_budget";
    }
    return "scheduled_token_budget";
  }
}

function exceedsLimit(value: number, limit: number | null): boolean {
  return limit !== null && value > limit;
}

function scaledBudget(limit: number | undefined, maxBatchSize: number): number | null {
  return limit === undefined ? null : limit * maxBatchSize;
}

function finiteByteBudget(bytes: number | undefined): number | null {
  if (bytes === undefined || !Number.isFinite(bytes) || bytes < 0) {
    return null;
  }
  return Math.floor(bytes);
}

function formatBytes(bytes: number): string {
  if (bytes >= 1e9) {
    return `${(bytes / 1e9).toFixed(1)} GB`;
  }
  if (bytes >= 1e6) {
    return `${(bytes / 1e6).toFixed(1)} MB`;
  }
  if (bytes >= 1e3) {
    return `${(bytes / 1e3).toFixed(1)} KB`;
  }
  return `${bytes} B`;
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
  const maxScheduledTotalTokens =
    scaledBudget(options.maxTotalTokens, options.maxBatchSize) ??
    (maxScheduledPromptTokens === null || maxScheduledCompletionTokens === null
      ? null
      : maxScheduledPromptTokens + maxScheduledCompletionTokens);
  const maxScheduledMemoryBytes =
    options.estimateMemoryBytes === undefined
      ? null
      : finiteByteBudget(options.maxScheduledMemoryBytes);
  if (
    maxScheduledPromptTokens === null &&
    maxScheduledCompletionTokens === null &&
    maxScheduledTotalTokens === null &&
    maxScheduledMemoryBytes === null
  ) {
    return undefined;
  }
  return new ContinuousSchedulerBudget(
    maxScheduledPromptTokens,
    maxScheduledCompletionTokens,
    maxScheduledTotalTokens,
    maxScheduledMemoryBytes,
    options.estimateMemoryBytes ?? (() => undefined),
  );
}
