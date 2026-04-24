/**
 * Shared serving error helpers.
 * @module
 */

import { GenerationAbortError } from "@mlxts/transformers";

export class ServeError extends Error {
  readonly status: number;
  readonly code: string;
  readonly param: string | null;

  constructor(message: string, options: { status?: number; code?: string; param?: string } = {}) {
    super(message);
    this.name = "ServeError";
    this.status = options.status ?? 400;
    this.code = options.code ?? "invalid_request";
    this.param = options.param ?? null;
  }
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
    },
  });
}

/** Convert thrown values into the serving error envelope used by HTTP and events. */
export function toServeError(error: unknown): ServeError {
  if (error instanceof ServeError) {
    return error;
  }
  if (
    error instanceof GenerationAbortError ||
    (error instanceof Error && error.name === "AbortError")
  ) {
    return new ServeError("Client disconnected before generation completed.", {
      status: 499,
      code: "client_cancelled",
    });
  }
  return new ServeError(error instanceof Error ? error.message : String(error), {
    status: 500,
    code: "internal_error",
  });
}

export function openAIErrorResponse(error: unknown): Response {
  const serveError = toServeError(error);

  return jsonResponse(
    {
      error: {
        message: serveError.message,
        type: serveError.status >= 500 ? "server_error" : "invalid_request_error",
        param: serveError.param,
        code: serveError.code,
      },
    },
    serveError.status,
  );
}
