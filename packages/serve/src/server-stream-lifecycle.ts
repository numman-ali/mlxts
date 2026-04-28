/**
 * Shared terminal handling for HTTP generation streams.
 * @module
 */

import {
  emitGenerationComplete,
  emitGenerationError,
  emitRequestComplete,
  emitRequestError,
  serveErrorDetails,
} from "./server-events";
import type { GenerationStreamObserver } from "./server-stream-observability";
import type { StreamSummary } from "./server-stream-runtime";
import type { NormalizedGenerationRequest, NormalizedGenerationResult, ServeEvent } from "./types";

type EventSink = {
  onEvent?: (event: ServeEvent) => void;
};

type DisposableAbortScope = {
  abort?(): void;
  dispose(): void;
};

type StreamLifecycleContext = {
  options: EventSink;
  request: Request;
  generationRequest: NormalizedGenerationRequest;
  abortScope: DisposableAbortScope;
  observer: GenerationStreamObserver;
  generationStartedAt: number;
  requestStartedAt: number;
  controller: ReadableStreamDefaultController<Uint8Array>;
  isCancelled(): boolean;
};

/** Finish a successful or client-cancelled HTTP generation stream. */
export function completeGenerationStream(
  context: StreamLifecycleContext,
  summary: StreamSummary,
  clientCancelledMessage: string,
): void {
  context.abortScope.dispose();
  const durationMs = performance.now() - context.generationStartedAt;
  context.observer.end(
    summary.finishReason === "cancelled" ? "cancelled" : "completed",
    summary.finishReason,
    durationMs,
  );
  const result: NormalizedGenerationResult = {
    text: "",
    finishReason: summary.finishReason,
    ...(summary.usage === undefined ? {} : { usage: summary.usage }),
  };
  emitGenerationComplete(context.options, context.generationRequest, result, durationMs);
  if (summary.finishReason === "cancelled") {
    emitRequestError(
      context.options,
      context.request,
      { message: clientCancelledMessage, code: "client_cancelled", status: 499 },
      context.requestStartedAt,
    );
  } else {
    emitRequestComplete(context.options, context.request, 200, context.requestStartedAt);
  }
  if (!context.isCancelled()) {
    context.controller.close();
  }
}

/** Finish a failed HTTP generation stream. */
export function failGenerationStream(context: StreamLifecycleContext, error: unknown): void {
  context.abortScope.abort?.();
  context.abortScope.dispose();
  const durationMs = performance.now() - context.generationStartedAt;
  context.observer.end("error", "error", durationMs);
  emitGenerationError(context.options, context.generationRequest, error, durationMs);
  emitRequestError(
    context.options,
    context.request,
    serveErrorDetails(error),
    context.requestStartedAt,
  );
  if (!context.isCancelled()) {
    context.controller.error(error);
  }
}
