/**
 * Bounded JSON body parsing for serving routes.
 * @module
 */

import { ServeError } from "../errors";

/** Default maximum JSON request body size accepted by local serving routes. */
export const DEFAULT_SERVE_JSON_BODY_MAX_BYTES = 64 * 1024 * 1024;

export type ReadJsonOptions = {
  maxBytes?: number;
};

function bodyTooLarge(maxBytes: number): ServeError {
  return new ServeError(`Request body exceeds the ${maxBytes} byte limit.`, {
    code: "request_too_large",
    status: 413,
  });
}

function contentLength(request: Request): number | null {
  const header = request.headers.get("content-length");
  if (header === null) {
    return null;
  }
  const parsed = Number.parseInt(header, 10);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : null;
}

async function readBoundedBody(request: Request, maxBytes: number): Promise<Uint8Array> {
  const declaredLength = contentLength(request);
  if (declaredLength !== null && declaredLength > maxBytes) {
    throw bodyTooLarge(maxBytes);
  }

  if (request.body === null) {
    throw new ServeError("Request body must be valid JSON.", { code: "invalid_json" });
  }

  const reader = request.body.getReader();
  const chunks: Uint8Array[] = [];
  let totalBytes = 0;
  try {
    while (true) {
      const next = await reader.read();
      if (next.done) {
        break;
      }
      const chunk = next.value;
      totalBytes += chunk.byteLength;
      if (totalBytes > maxBytes) {
        throw bodyTooLarge(maxBytes);
      }
      chunks.push(chunk);
    }
  } finally {
    reader.releaseLock();
  }

  const body = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    body.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return body;
}

/** Read and parse one bounded JSON request body. */
export async function readJson(request: Request, options: ReadJsonOptions = {}): Promise<unknown> {
  const maxBytes = options.maxBytes ?? DEFAULT_SERVE_JSON_BODY_MAX_BYTES;
  try {
    const bytes = await readBoundedBody(request, maxBytes);
    return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  } catch (error) {
    if (error instanceof ServeError) {
      throw error;
    }
    throw new ServeError("Request body must be valid JSON.", { code: "invalid_json" });
  }
}
