/**
 * Structured serving event helpers.
 * @module
 */

import { ServeError } from "./errors";
import { readGenerationMemoryUsage } from "./memory-telemetry";
import type { NormalizedGenerationRequest, NormalizedGenerationResult, ServeEvent } from "./types";

type EventSink = {
  onEvent?: (event: ServeEvent) => void;
};

export function emitServeEvent(options: EventSink, event: ServeEvent): void {
  options.onEvent?.(event);
}

export function emitRequestComplete(
  options: EventSink,
  request: Request,
  status: number,
  startedAt: number,
): void {
  const url = new URL(request.url);
  emitServeEvent(options, {
    type: "request_complete",
    method: request.method,
    path: url.pathname,
    status,
    durationMs: performance.now() - startedAt,
  });
}

export function emitRequestError(
  options: EventSink,
  request: Request,
  details: { message: string; code: string; status: number },
  startedAt: number,
): void {
  const url = new URL(request.url);
  emitServeEvent(options, {
    type: "request_error",
    method: request.method,
    path: url.pathname,
    message: details.message,
    code: details.code,
    status: details.status,
    durationMs: performance.now() - startedAt,
  });
}

export function serveErrorDetails(error: unknown): {
  message: string;
  code: string;
  status: number;
} {
  if (error instanceof ServeError) {
    return { message: error.message, code: error.code, status: error.status };
  }
  return {
    message: error instanceof Error ? error.message : String(error),
    code: "internal_error",
    status: 500,
  };
}

export function emitGenerationStart(
  options: EventSink,
  request: NormalizedGenerationRequest,
): void {
  emitServeEvent(options, {
    type: "generation_start",
    id: request.id,
    protocol: request.protocol,
    model: request.model,
    inputKind: request.input.kind,
    maxTokens: request.sampling.maxTokens,
  });
}

export function emitGenerationComplete(
  options: EventSink,
  request: NormalizedGenerationRequest,
  result: NormalizedGenerationResult,
  durationMs: number,
): void {
  const completionTokens = result.usage?.completionTokens ?? result.tokenIds?.length;
  const memory = readGenerationMemoryUsage();
  emitServeEvent(options, {
    type: "generation_complete",
    id: request.id,
    protocol: request.protocol,
    model: request.model,
    finishReason: result.finishReason,
    ...(result.usage?.promptTokens === undefined
      ? {}
      : { promptTokens: result.usage.promptTokens }),
    ...(completionTokens === undefined ? {} : { completionTokens }),
    ...(result.usage?.totalTokens === undefined ? {} : { totalTokens: result.usage.totalTokens }),
    durationMs,
    ...(memory === undefined ? {} : { memory }),
  });
}

export function emitGenerationError(
  options: EventSink,
  request: NormalizedGenerationRequest,
  error: unknown,
  durationMs: number,
): void {
  const details = serveErrorDetails(error);
  emitServeEvent(options, {
    type: "generation_error",
    id: request.id,
    protocol: request.protocol,
    model: request.model,
    code: details.code,
    message: details.message,
    durationMs,
  });
}
