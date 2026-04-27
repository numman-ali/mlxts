/**
 * Bun-native serving shell.
 * @module
 */

import { anthropicErrorResponse, jsonResponse, openAIErrorResponse, ServeError } from "./errors";
import {
  formatOpenAIChatCompletionResponse,
  normalizeOpenAIChatCompletionRequest,
} from "./protocols/openai-chat-completions";
import {
  formatOpenAICompletionResponse,
  normalizeOpenAICompletionRequest,
} from "./protocols/openai-completions";
import type { ServedModelInfo } from "./protocols/openai-models";
import {
  createServeMetrics,
  createServeMetricsSink,
  type ServeMetrics,
  serveMetricsResponse,
} from "./serve-metrics";
import { linkAbortSignals, withAbortSignal } from "./server-abort";
import { anthropicMessagesRouteResponse } from "./server-anthropic-messages";
import {
  emitGenerationComplete,
  emitGenerationError,
  emitGenerationStart,
  emitRequestComplete,
  emitRequestError,
  emitServeEvent,
  serveErrorDetails,
} from "./server-events";
import { generateCompletionBatch } from "./server-generation";
import {
  openAIModelRouteResponse,
  type ServeRuntimeLimits,
  serveInfoResponse,
} from "./server-info";
import { openAIResponsesRouteResponse } from "./server-responses";
import { completeGenerationStream, failGenerationStream } from "./server-stream-lifecycle";
import { createGenerationStreamObserver } from "./server-stream-observability";
import {
  closeStreamEvents,
  sseHeaders,
  writeChatStreamEvents,
  writeStreamEvents,
} from "./server-streaming";
import type { GenerationEngine, GenerationStreamEvent, ServeEvent } from "./types";

export type ServeAppOptions = {
  engine: GenerationEngine;
  models?: readonly ServedModelInfo[];
  limits?: ServeRuntimeLimits;
  apiKey?: string;
  abortSignal?: AbortSignal;
  idGenerator?: () => string;
  now?: () => Date;
  onEvent?: (event: ServeEvent) => void;
  /** @internal Reuse one collector when engines and the HTTP router share metrics. */
  metrics?: ServeMetrics;
};

export type ServeServerOptions = ServeAppOptions & {
  port?: number;
  hostname?: string;
};

type ServerRequestControls = {
  timeout(request: Request, seconds: number): void;
};

function defaultId(): string {
  return `cmpl-${crypto.randomUUID()}`;
}

function defaultResponseId(): string {
  return `resp-${crypto.randomUUID()}`;
}

function defaultAnthropicMessageId(): string {
  return `msg_${crypto.randomUUID()}`;
}

function unixSeconds(date: Date): number {
  return Math.floor(date.getTime() / 1000);
}

const GENERATION_ROUTE_PATTERN = /^\/v1\/(?:completions|chat\/completions|responses|messages)$/;

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

async function completionResponse(
  request: Request,
  options: ServeAppOptions,
  requestStartedAt: number,
): Promise<Response> {
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
    const streamObserver = createGenerationStreamObserver(options, normalized, startedAt);
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
            observer: streamObserver,
          }).then(
            (summary) => {
              completeGenerationStream(
                {
                  options,
                  request,
                  generationRequest: normalized,
                  abortScope: streamAbort,
                  observer: streamObserver,
                  generationStartedAt: startedAt,
                  requestStartedAt,
                  controller,
                  isCancelled: () => cancelled,
                },
                summary,
                "Client disconnected during streaming completion output.",
              );
            },
            (error: unknown) => {
              failGenerationStream(
                {
                  options,
                  request,
                  generationRequest: normalized,
                  abortScope: streamAbort,
                  observer: streamObserver,
                  generationStartedAt: startedAt,
                  requestStartedAt,
                  controller,
                  isCancelled: () => cancelled,
                },
                error,
              );
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
    const results = await generateCompletionBatch(
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
  requestStartedAt: number,
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
    const streamObserver = createGenerationStreamObserver(options, chatRequest, startedAt);
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
            observer: streamObserver,
          }).then(
            (summary) => {
              completeGenerationStream(
                {
                  options,
                  request,
                  generationRequest: chatRequest,
                  abortScope: streamAbort,
                  observer: streamObserver,
                  generationStartedAt: startedAt,
                  requestStartedAt,
                  controller,
                  isCancelled: () => cancelled,
                },
                summary,
                "Client disconnected during streaming chat output.",
              );
            },
            (error: unknown) => {
              failGenerationStream(
                {
                  options,
                  request,
                  generationRequest: chatRequest,
                  abortScope: streamAbort,
                  observer: streamObserver,
                  generationStartedAt: startedAt,
                  requestStartedAt,
                  controller,
                  isCancelled: () => cancelled,
                },
                error,
              );
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
    return await completionResponse(request, options, startedAt);
  }

  if (request.method === "POST" && pathname === "/v1/chat/completions") {
    return await chatCompletionResponse(request, options, startedAt);
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

async function generationRouteResponse(
  request: Request,
  options: ServeAppOptions,
  pathname: string,
  startedAt: number,
): Promise<Response | null> {
  const openAIResponse = await openAIRouteResponse(request, options, pathname, startedAt);
  if (openAIResponse !== null) {
    return openAIResponse;
  }

  authorize(request, options.apiKey);
  if (request.method === "POST" && pathname === "/v1/messages") {
    return await anthropicMessagesRouteResponse(request, options, {
      id: options.idGenerator?.() ?? defaultAnthropicMessageId(),
      startedAt,
    });
  }

  return null;
}

function lightweightGetResponse(
  request: Request,
  options: ServeAppOptions,
  metrics: ServeMetrics,
  pathname: string,
  startedAt: number,
): Response | null {
  if (request.method !== "GET") {
    return null;
  }
  if (pathname === "/health") {
    const response = jsonResponse({ status: "ok" });
    emitRequestComplete(options, request, response.status, startedAt);
    return response;
  }
  if (pathname === "/metrics") {
    authorize(request, options.apiKey);
    const response = serveMetricsResponse(metrics);
    emitRequestComplete(options, request, response.status, startedAt);
    return response;
  }
  if (pathname === "/info") {
    authorize(request, options.apiKey);
    const response = serveInfoResponse(options);
    emitRequestComplete(options, request, response.status, startedAt);
    return response;
  }
  return null;
}

/** Create a Bun-compatible fetch handler for the serving API. */
export function createFetchHandler(
  options: ServeAppOptions,
): (request: Request, server?: ServerRequestControls) => Promise<Response> {
  const metrics =
    options.metrics ??
    createServeMetrics({
      ...(options.models === undefined
        ? {}
        : { modelIds: options.models.map((model) => model.id) }),
    });
  const observedOptions: ServeAppOptions = {
    ...options,
    metrics,
    onEvent: createServeMetricsSink(metrics, options.onEvent),
  };
  return async (request, server) => {
    const url = new URL(request.url);
    const startedAt = performance.now();
    emitServeEvent(observedOptions, {
      type: "request_start",
      method: request.method,
      path: url.pathname,
    });

    try {
      const lightweightResponse = lightweightGetResponse(
        request,
        observedOptions,
        metrics,
        url.pathname,
        startedAt,
      );
      if (lightweightResponse !== null) {
        return lightweightResponse;
      }

      if (routeMayRunGeneration(request.method, url.pathname)) {
        server?.timeout(request, 0);
      }
      const response = await generationRouteResponse(
        request,
        observedOptions,
        url.pathname,
        startedAt,
      );
      if (response !== null) {
        const isStreaming = response.headers.get("content-type")?.startsWith("text/event-stream");
        if (isStreaming !== true) {
          emitRequestComplete(observedOptions, request, response.status, startedAt);
        }
        return response;
      }
    } catch (error) {
      emitRequestError(observedOptions, request, serveErrorDetails(error), startedAt);
      return url.pathname === "/v1/messages"
        ? anthropicErrorResponse(error)
        : openAIErrorResponse(error);
    }

    const response = jsonResponse({ error: { message: "Not found" } }, 404);
    emitRequestComplete(observedOptions, request, response.status, startedAt);
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
