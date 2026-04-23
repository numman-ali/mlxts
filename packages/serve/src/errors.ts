/**
 * Shared serving error helpers.
 * @module
 */

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

export function openAIErrorResponse(error: unknown): Response {
  const serveError =
    error instanceof ServeError
      ? error
      : new ServeError(error instanceof Error ? error.message : String(error), {
          status: 500,
          code: "internal_error",
        });

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
