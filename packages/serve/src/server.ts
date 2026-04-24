/**
 * Bun-native serving shell.
 * @module
 */

import { jsonResponse, openAIErrorResponse, ServeError } from "./errors";
import {
  formatOpenAIChatCompletionResponse,
  normalizeOpenAIChatCompletionRequest,
} from "./protocols/openai-chat-completions";
import {
  formatOpenAICompletionResponse,
  normalizeOpenAICompletionRequest,
} from "./protocols/openai-completions";
import type { ServedModelInfo } from "./protocols/openai-models";
import { linkAbortSignals, withAbortSignal } from "./server-abort";
import {
  emitGenerationComplete,
  emitGenerationError,
  emitGenerationStart,
  emitRequestComplete,
  emitRequestError,
  emitServeEvent,
  serveErrorDetails,
} from "./server-events";
import {
  openAIModelRouteResponse,
  type ServeRuntimeLimits,
  serveInfoResponse,
} from "./server-info";
import { openAIResponsesRouteResponse } from "./server-responses";
import {
  closeStreamEvents,
  sseHeaders,
  writeChatStreamEvents,
  writeStreamEvents,
} from "./server-streaming";
import type {
  GenerationEngine,
  GenerationStreamEvent,
  NormalizedGenerationResult,
  ServeEvent,
} from "./types";

export type ServeAppOptions = {
  engine: GenerationEngine;
  models?: readonly ServedModelInfo[];
  limits?: ServeRuntimeLimits;
  apiKey?: string;
  abortSignal?: AbortSignal;
  idGenerator?: () => string;
  now?: () => Date;
  onEvent?: (event: ServeEvent) => void;
};

export type ServeServerOptions = ServeAppOptions & {
  port?: number;
  hostname?: string;
};

type ServerRequestControls = {
  timeout(request: Request, seconds: number): void;
};

type CompletionRequests = ReturnType<typeof normalizeOpenAICompletionRequest>["requests"];

function defaultId(): string {
  return `cmpl-${crypto.randomUUID()}`;
}

function defaultResponseId(): string {
  return `resp-${crypto.randomUUID()}`;
}

function unixSeconds(date: Date): number {
  return Math.floor(date.getTime() / 1000);
}

const GENERATION_ROUTE_PATTERN = /^\/v1\/(?:completions|chat\/completions|responses)$/;

function routeMayRunGeneration(method: string, pathname: string): boolean {
  return method === "POST" && GENERATION_ROUTE_PATTERN.test(pathname);
}

async function readJson(request: Request): Promise<unknown> {
  try {
    return await request.json();
  } catch {
    throw new ServeError("Request body must be valid JSON.", {
      code: "invalid_json",
    });
  }
}

function authorize(request: Request, apiKey: string | undefined): void {
  if (apiKey === undefined) {
    return;
  }

  const authorization = request.headers.get("authorization");
  if (authorization !== `Bearer ${apiKey}`) {
    throw new ServeError("Invalid or missing API key.", {
      code: "invalid_api_key",
      status: 401,
    });
  }
}

function assertBatchResultCount(
  results: readonly NormalizedGenerationResult[],
  requests: CompletionRequests,
): void {
  if (results.length !== requests.length) {
    throw new ServeError("Generation engine returned the wrong number of batch results.", {
      code: "invalid_engine_result",
      status: 500,
    });
  }
}

function emitBatchGenerationErrors(
  options: ServeAppOptions,
  requests: CompletionRequests,
  error: unknown,
  startedAt: number,
): void {
  const durationMs = performance.now() - startedAt;
  for (const normalized of requests) {
    emitGenerationError(options, normalized, error, durationMs);
  }
}

function emitBatchGenerationComplete(
  options: ServeAppOptions,
  requests: CompletionRequests,
  results: readonly NormalizedGenerationResult[],
  durationMs: number,
): void {
  for (let index = 0; index < requests.length; index += 1) {
    const normalized = requests[index];
    const result = results[index];
    if (normalized !== undefined && result !== undefined) {
      emitGenerationComplete(options, normalized, result, durationMs);
    }
  }
}

async function generateWithBatchEngine(
  options: ServeAppOptions,
  requests: CompletionRequests,
): Promise<NormalizedGenerationResult[]> {
  const generateBatch = options.engine.generateBatch;
  if (generateBatch === undefined) {
    throw new Error("generateWithBatchEngine requires an engine batch function.");
  }

  const startedAt = performance.now();
  for (const normalized of requests) {
    emitGenerationStart(options, normalized);
  }
  try {
    const results = await generateBatch(requests);
    assertBatchResultCount(results, requests);
    emitBatchGenerationComplete(options, requests, results, performance.now() - startedAt);
    return [...results];
  } catch (error) {
    emitBatchGenerationErrors(options, requests, error, startedAt);
    throw error;
  }
}

async function generateSequentially(
  options: ServeAppOptions,
  requests: CompletionRequests,
): Promise<NormalizedGenerationResult[]> {
  const results: NormalizedGenerationResult[] = [];
  for (const normalized of requests) {
    const startedAt = performance.now();
    emitGenerationStart(options, normalized);
    try {
      const result = await options.engine.generate(normalized);
      emitGenerationComplete(options, normalized, result, performance.now() - startedAt);
      results.push(result);
    } catch (error) {
      emitGenerationError(options, normalized, error, performance.now() - startedAt);
      throw error;
    }
  }
  return results;
}

async function generateBatch(
  options: ServeAppOptions,
  requests: CompletionRequests,
): Promise<NormalizedGenerationResult[]> {
  if (requests.length > 1 && options.engine.generateBatch !== undefined) {
    return await generateWithBatchEngine(options, requests);
  }
  return await generateSequentially(options, requests);
}

async function completionResponse(request: Request, options: ServeAppOptions): Promise<Response> {
  const id = options.idGenerator?.() ?? defaultId();
  const created = unixSeconds(options.now?.() ?? new Date());
  const batch = normalizeOpenAICompletionRequest(await readJson(request), { id });

  if (batch.stream) {
    if (batch.requests.length !== 1) {
      throw new ServeError("OpenAI completions: streaming is only supported for one prompt.", {
        param: "prompt",
      });
    }
    if (options.engine.stream === undefined) {
      throw new ServeError("This generation engine does not support streaming yet.", {
        code: "stream_not_supported",
        param: "stream",
      });
    }

    const normalizedBase = batch.requests[0];
    if (normalizedBase === undefined) {
      throw new ServeError("OpenAI completions: streaming requires one prompt.", {
        param: "prompt",
      });
    }
    const streamAbort = linkAbortSignals(request.signal, options.abortSignal);
    const normalized = withAbortSignal(normalizedBase, streamAbort.signal);
    const startedAt = performance.now();
    emitGenerationStart(options, normalized);
    let stream: AsyncIterable<GenerationStreamEvent>;
    try {
      stream = await options.engine.stream(normalized);
    } catch (error) {
      streamAbort.dispose();
      emitGenerationError(options, normalized, error, performance.now() - startedAt);
      throw error;
    }
    const iterator = stream[Symbol.asyncIterator]();
    let cancelled = false;
    return new Response(
      new ReadableStream<Uint8Array>({
        start(controller) {
          void writeStreamEvents(controller, iterator, batch, normalized, {
            id,
            created,
            signal: streamAbort.signal,
          }).then(
            (summary) => {
              streamAbort.dispose();
              const result: NormalizedGenerationResult = {
                text: "",
                finishReason: summary.finishReason,
                ...(summary.usage === undefined ? {} : { usage: summary.usage }),
              };
              emitGenerationComplete(options, normalized, result, performance.now() - startedAt);
              if (summary.finishReason === "cancelled") {
                emitRequestError(
                  options,
                  request,
                  {
                    message: "Client disconnected during streaming completion output.",
                    code: "client_cancelled",
                    status: 499,
                  },
                  startedAt,
                );
              } else {
                emitRequestComplete(options, request, 200, startedAt);
              }
              if (!cancelled) {
                controller.close();
              }
            },
            (error: unknown) => {
              streamAbort.dispose();
              emitGenerationError(options, normalized, error, performance.now() - startedAt);
              emitRequestError(options, request, serveErrorDetails(error), startedAt);
              if (!cancelled) {
                controller.error(error);
              }
            },
          );
        },
        async cancel() {
          cancelled = true;
          streamAbort.abort();
          void closeStreamEvents(iterator);
        },
      }),
      { headers: sseHeaders() },
    );
  }

  const abortScope = linkAbortSignals(request.signal, options.abortSignal);
  try {
    const results = await generateBatch(
      options,
      batch.requests.map((normalized) => withAbortSignal(normalized, abortScope.signal)),
    );
    return jsonResponse(formatOpenAICompletionResponse(batch, results, { id, created }));
  } finally {
    abortScope.dispose();
  }
}

async function chatCompletionResponse(
  request: Request,
  options: ServeAppOptions,
): Promise<Response> {
  const id = options.idGenerator?.() ?? defaultId();
  const created = unixSeconds(options.now?.() ?? new Date());
  const chat = normalizeOpenAIChatCompletionRequest(await readJson(request), { id });
  if (chat.stream) {
    if (options.engine.stream === undefined) {
      throw new ServeError("This generation engine does not support streaming yet.", {
        code: "stream_not_supported",
        param: "stream",
      });
    }

    const streamAbort = linkAbortSignals(request.signal, options.abortSignal);
    const chatRequest = withAbortSignal(chat.request, streamAbort.signal);
    const startedAt = performance.now();
    emitGenerationStart(options, chatRequest);
    let stream: AsyncIterable<GenerationStreamEvent>;
    try {
      stream = await options.engine.stream(chatRequest);
    } catch (error) {
      streamAbort.dispose();
      emitGenerationError(options, chatRequest, error, performance.now() - startedAt);
      throw error;
    }
    const iterator = stream[Symbol.asyncIterator]();
    let cancelled = false;
    return new Response(
      new ReadableStream<Uint8Array>({
        start(controller) {
          void writeChatStreamEvents(controller, iterator, chat, {
            id,
            created,
            signal: streamAbort.signal,
          }).then(
            (summary) => {
              streamAbort.dispose();
              const result: NormalizedGenerationResult = {
                text: "",
                finishReason: summary.finishReason,
                ...(summary.usage === undefined ? {} : { usage: summary.usage }),
              };
              emitGenerationComplete(options, chatRequest, result, performance.now() - startedAt);
              if (summary.finishReason === "cancelled") {
                emitRequestError(
                  options,
                  request,
                  {
                    message: "Client disconnected during streaming chat output.",
                    code: "client_cancelled",
                    status: 499,
                  },
                  startedAt,
                );
              } else {
                emitRequestComplete(options, request, 200, startedAt);
              }
              if (!cancelled) {
                controller.close();
              }
            },
            (error: unknown) => {
              streamAbort.dispose();
              emitGenerationError(options, chatRequest, error, performance.now() - startedAt);
              emitRequestError(options, request, serveErrorDetails(error), startedAt);
              if (!cancelled) {
                controller.error(error);
              }
            },
          );
        },
        async cancel() {
          cancelled = true;
          streamAbort.abort();
          void closeStreamEvents(iterator);
        },
      }),
      { headers: sseHeaders() },
    );
  }

  const abortScope = linkAbortSignals(request.signal, options.abortSignal);
  const chatRequest = withAbortSignal(chat.request, abortScope.signal);
  const startedAt = performance.now();
  emitGenerationStart(options, chatRequest);
  try {
    const result = await options.engine.generate(chatRequest);
    emitGenerationComplete(options, chatRequest, result, performance.now() - startedAt);
    return jsonResponse(formatOpenAIChatCompletionResponse(chat, result, { id, created }));
  } catch (error) {
    emitGenerationError(options, chatRequest, error, performance.now() - startedAt);
    throw error;
  } finally {
    abortScope.dispose();
  }
}

async function openAIRouteResponse(
  request: Request,
  options: ServeAppOptions,
  pathname: string,
  startedAt: number,
): Promise<Response | null> {
  authorize(request, options.apiKey);

  if (request.method === "GET") {
    const response = openAIModelRouteResponse(pathname, options);
    if (response !== null) {
      return response;
    }
  }

  if (request.method === "POST" && pathname === "/v1/completions") {
    return await completionResponse(request, options);
  }

  if (request.method === "POST" && pathname === "/v1/chat/completions") {
    return await chatCompletionResponse(request, options);
  }

  if (request.method === "POST" && pathname === "/v1/responses") {
    return await openAIResponsesRouteResponse(request, options, {
      id: options.idGenerator?.() ?? defaultResponseId(),
      created: unixSeconds(options.now?.() ?? new Date()),
      startedAt,
    });
  }

  return null;
}

/** Create a Bun-compatible fetch handler for the serving API. */
export function createFetchHandler(
  options: ServeAppOptions,
): (request: Request, server?: ServerRequestControls) => Promise<Response> {
  return async (request, server) => {
    const url = new URL(request.url);
    const startedAt = performance.now();
    emitServeEvent(options, { type: "request_start", method: request.method, path: url.pathname });
    if (request.method === "GET" && url.pathname === "/health") {
      const response = jsonResponse({ status: "ok" });
      emitRequestComplete(options, request, response.status, startedAt);
      return response;
    }

    try {
      if (request.method === "GET" && url.pathname === "/info") {
        authorize(request, options.apiKey);
        const response = serveInfoResponse(options);
        emitRequestComplete(options, request, response.status, startedAt);
        return response;
      }

      if (routeMayRunGeneration(request.method, url.pathname)) {
        server?.timeout(request, 0);
      }
      const response = await openAIRouteResponse(request, options, url.pathname, startedAt);
      if (response !== null) {
        const isStreaming = response.headers.get("content-type")?.startsWith("text/event-stream");
        if (isStreaming !== true) {
          emitRequestComplete(options, request, response.status, startedAt);
        }
        return response;
      }
    } catch (error) {
      emitRequestError(options, request, serveErrorDetails(error), startedAt);
      return openAIErrorResponse(error);
    }

    const response = jsonResponse({ error: { message: "Not found" } }, 404);
    emitRequestComplete(options, request, response.status, startedAt);
    return response;
  };
}

/** Start a Bun server for the serving API. */
export function startServeServer(options: ServeServerOptions): ReturnType<typeof Bun.serve> {
  const fetch = createFetchHandler(options);
  const serverOptions: Parameters<typeof Bun.serve>[0] = {
    port: options.port ?? 3000,
    fetch(request, server) {
      return fetch(request, server);
    },
  };
  if (options.hostname !== undefined) {
    serverOptions.hostname = options.hostname;
  }
  return Bun.serve(serverOptions);
}
