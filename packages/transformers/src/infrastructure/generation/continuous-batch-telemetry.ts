import type { GenerationResult } from "../../types";
import type {
  ContinuousBatchEvent,
  ContinuousBatchQueueSnapshot,
  ContinuousBatchSchedulerEvent,
} from "./continuous-batch-events";

export type ContinuousBatchTelemetryRequest = {
  id: string;
  promptTokenIds: readonly number[];
  maxTokens: number;
  enqueuedAtMs: number;
  generated: readonly number[];
  finishReason: GenerationResult["finishReason"];
};

type OnBatch = ((event: ContinuousBatchEvent) => void) | undefined;
type OnSchedulerEvent = ((event: ContinuousBatchSchedulerEvent) => void) | undefined;

export class ContinuousBatchTelemetry {
  readonly #onBatch: OnBatch;
  readonly #onSchedulerEvent: OnSchedulerEvent;

  constructor(onBatch: OnBatch, onSchedulerEvent: OnSchedulerEvent) {
    this.#onBatch = onBatch;
    this.#onSchedulerEvent = onSchedulerEvent;
  }

  #elapsedMs(request: ContinuousBatchTelemetryRequest): number {
    return performance.now() - request.enqueuedAtMs;
  }

  queued(
    request: ContinuousBatchTelemetryRequest,
    queuedAhead: number,
    snapshot: ContinuousBatchQueueSnapshot,
  ): void {
    this.#onSchedulerEvent?.({
      type: "queued",
      id: request.id,
      queuedAhead,
      promptTokens: request.promptTokenIds.length,
      maxTokens: request.maxTokens,
      schedulerMs: 0,
      ...snapshot,
    });
  }

  prefillStart(
    request: ContinuousBatchTelemetryRequest,
    snapshot: ContinuousBatchQueueSnapshot,
  ): void {
    this.#onSchedulerEvent?.({
      type: "prefill_start",
      id: request.id,
      promptTokens: request.promptTokenIds.length,
      maxTokens: request.maxTokens,
      queuedMs: this.#elapsedMs(request),
      schedulerMs: this.#elapsedMs(request),
      ...snapshot,
    });
  }

  admitted(
    active: readonly ContinuousBatchTelemetryRequest[],
    snapshot: ContinuousBatchQueueSnapshot,
  ): void {
    if (active.length === 0) {
      return;
    }
    const maxTokensByRequest = active.map((request) => request.maxTokens);
    const event = {
      ids: active.map((request) => request.id),
      batchSize: active.length,
      maxTokens: Math.max(...maxTokensByRequest),
      maxTokensByRequest,
      queuedMsByRequest: active.map((request) => this.#elapsedMs(request)),
      schedulerMs: Math.max(...active.map((request) => this.#elapsedMs(request))),
      ...snapshot,
    };
    this.#onBatch?.(event);
    this.#onSchedulerEvent?.({ type: "admitted", ...event });
  }

  firstToken(
    request: ContinuousBatchTelemetryRequest,
    snapshot: ContinuousBatchQueueSnapshot,
  ): void {
    this.#onSchedulerEvent?.({
      type: "first_token",
      id: request.id,
      completionTokens: request.generated.length,
      queuedMs: this.#elapsedMs(request),
      schedulerMs: this.#elapsedMs(request),
      ...snapshot,
    });
  }

  finished(request: ContinuousBatchTelemetryRequest, snapshot: ContinuousBatchQueueSnapshot): void {
    this.#onSchedulerEvent?.({
      type: "finished",
      id: request.id,
      completionTokens: request.generated.length,
      finishReason: request.finishReason,
      queuedMs: this.#elapsedMs(request),
      schedulerMs: this.#elapsedMs(request),
      ...snapshot,
    });
  }

  cancelled(
    request: ContinuousBatchTelemetryRequest,
    snapshot: ContinuousBatchQueueSnapshot,
  ): void {
    this.#onSchedulerEvent?.({
      type: "cancelled",
      id: request.id,
      completionTokens: request.generated.length,
      queuedMs: this.#elapsedMs(request),
      schedulerMs: this.#elapsedMs(request),
      ...snapshot,
    });
  }
}
