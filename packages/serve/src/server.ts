/**
 * Bun-native serving shell.
 * @module
 */

import { jsonResponse, openAIErrorResponse, ServeError } from "./errors";
import { readGenerationMemoryUsage } from "./memory-telemetry";
import {
  formatOpenAIChatCompletionResponse,
  normalizeOpenAIChatCompletionRequest,
} from "./protocols/openai-chat-completions";
import {
  formatOpenAICompletionResponse,
  formatOpenAICompletionStreamChunk,
  formatOpenAICompletionUsageStreamChunk,
  normalizeOpenAICompletionRequest,
} from "./protocols/openai-completions";
import { formatOpenAIModelsResponse, type ServedModelInfo } from "./protocols/openai-models";
import type {
  GenerationEngine,
  GenerationStreamEvent,
  GenerationUsage,
  NormalizedFinishReason,
  NormalizedGenerationRequest,
  NormalizedGenerationResult,
  ServeEvent,
} from "./types";

export type ServeAppOptions = {
  engine: GenerationEngine;
  models?: readonly ServedModelInfo[];
  apiKey?: string;
  idGenerator?: () => string;
  now?: () => Date;
  onEvent?: (event: ServeEvent) => void;
};

export type ServeServerOptions = ServeAppOptions & {
  port?: number;
  hostname?: string;
};

function defaultId(): string {
  return `cmpl-${crypto.randomUUID()}`;
}

function unixSeconds(date: Date): number {
  return Math.floor(date.getTime() / 1000);
}

function emit(options: ServeAppOptions, event: ServeEvent): void {
  options.onEvent?.(event);
}

function serveErrorDetails(error: unknown): { message: string; code: string; status: number } {
  if (error instanceof ServeError) {
    return { message: error.message, code: error.code, status: error.status };
  }
  return {
    message: error instanceof Error ? error.message : String(error),
    code: "internal_error",
    status: 500,
  };
}

function emitGenerationStart(options: ServeAppOptions, request: NormalizedGenerationRequest): void {
  emit(options, {
    type: "generation_start",
    id: request.id,
    protocol: request.protocol,
    model: request.model,
    inputKind: request.input.kind,
    maxTokens: request.sampling.maxTokens,
  });
}

function emitGenerationComplete(
  options: ServeAppOptions,
  request: NormalizedGenerationRequest,
  result: NormalizedGenerationResult,
  durationMs: number,
): void {
  const completionTokens = result.usage?.completionTokens ?? result.tokenIds?.length;
  const memory = readGenerationMemoryUsage();
  emit(options, {
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

async function generateBatch(
  options: ServeAppOptions,
  requests: ReturnType<typeof normalizeOpenAICompletionRequest>["requests"],
): Promise<NormalizedGenerationResult[]> {
  if (requests.length > 1 && options.engine.generateBatch !== undefined) {
    const startedAt = performance.now();
    for (const normalized of requests) {
      emitGenerationStart(options, normalized);
    }
    const results = await options.engine.generateBatch(requests);
    if (results.length !== requests.length) {
      throw new ServeError("Generation engine returned the wrong number of batch results.", {
        code: "invalid_engine_result",
        status: 500,
      });
    }
    const durationMs = performance.now() - startedAt;
    for (let index = 0; index < requests.length; index += 1) {
      const normalized = requests[index];
      const result = results[index];
      if (normalized !== undefined && result !== undefined) {
        emitGenerationComplete(options, normalized, result, durationMs);
      }
    }
    return [...results];
  }

  const results: NormalizedGenerationResult[] = [];
  for (const normalized of requests) {
    const startedAt = performance.now();
    emitGenerationStart(options, normalized);
    const result = await options.engine.generate(normalized);
    emitGenerationComplete(options, normalized, result, performance.now() - startedAt);
    results.push(result);
  }
  return results;
}

function encodeSse(payload: string): Uint8Array {
  return new TextEncoder().encode(payload);
}

async function writeStreamEvents(
  controller: ReadableStreamDefaultController<Uint8Array>,
  stream: AsyncIterable<GenerationStreamEvent>,
  batch: ReturnType<typeof normalizeOpenAICompletionRequest>,
  request: ReturnType<typeof normalizeOpenAICompletionRequest>["requests"][number],
  options: { id: string; created: number },
): Promise<{ finishReason: NormalizedFinishReason; usage?: GenerationUsage }> {
  let finalUsage: GenerationUsage | undefined;
  let finalFinishReason: NormalizedFinishReason = "stop";
  for await (const event of stream) {
    if (event.type === "text") {
      const chunk = formatOpenAICompletionStreamChunk(request, event.text, {
        ...options,
        includeUsage: batch.streamOptions.includeUsage,
      });
      controller.enqueue(encodeSse(`data: ${JSON.stringify(chunk)}\n\n`));
    } else {
      finalUsage = event.usage;
      finalFinishReason = event.finishReason;
      const chunk = formatOpenAICompletionStreamChunk(request, "", {
        ...options,
        finishReason: event.finishReason,
        includeUsage: batch.streamOptions.includeUsage,
      });
      controller.enqueue(encodeSse(`data: ${JSON.stringify(chunk)}\n\n`));
    }
  }
  if (batch.streamOptions.includeUsage) {
    const chunk = formatOpenAICompletionUsageStreamChunk(batch, finalUsage, options);
    controller.enqueue(encodeSse(`data: ${JSON.stringify(chunk)}\n\n`));
  }
  controller.enqueue(encodeSse("data: [DONE]\n\n"));
  return finalUsage === undefined
    ? { finishReason: finalFinishReason }
    : { finishReason: finalFinishReason, usage: finalUsage };
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

    const normalized = batch.requests[0];
    if (normalized === undefined) {
      throw new ServeError("OpenAI completions: streaming requires one prompt.", {
        param: "prompt",
      });
    }
    const startedAt = performance.now();
    emitGenerationStart(options, normalized);
    const stream = await options.engine.stream(normalized);
    return new Response(
      new ReadableStream<Uint8Array>({
        start(controller) {
          return writeStreamEvents(controller, stream, batch, normalized, { id, created }).then(
            (summary) => {
              const result: NormalizedGenerationResult = {
                text: "",
                finishReason: summary.finishReason,
                ...(summary.usage === undefined ? {} : { usage: summary.usage }),
              };
              emitGenerationComplete(options, normalized, result, performance.now() - startedAt);
              controller.close();
            },
            (error: unknown) => {
              controller.error(error);
            },
          );
        },
      }),
      {
        headers: {
          "content-type": "text/event-stream; charset=utf-8",
          "cache-control": "no-cache",
          connection: "keep-alive",
        },
      },
    );
  }

  const results = await generateBatch(options, batch.requests);
  return jsonResponse(formatOpenAICompletionResponse(batch, results, { id, created }));
}

async function chatCompletionResponse(
  request: Request,
  options: ServeAppOptions,
): Promise<Response> {
  const id = options.idGenerator?.() ?? defaultId();
  const created = unixSeconds(options.now?.() ?? new Date());
  const chat = normalizeOpenAIChatCompletionRequest(await readJson(request), { id });
  const startedAt = performance.now();
  emitGenerationStart(options, chat.request);
  const result = await options.engine.generate(chat.request);
  emitGenerationComplete(options, chat.request, result, performance.now() - startedAt);
  return jsonResponse(formatOpenAIChatCompletionResponse(chat, result, { id, created }));
}

async function openAIRouteResponse(
  request: Request,
  options: ServeAppOptions,
  pathname: string,
): Promise<Response | null> {
  authorize(request, options.apiKey);

  if (request.method === "GET" && pathname === "/v1/models") {
    const created = unixSeconds(options.now?.() ?? new Date());
    return jsonResponse(formatOpenAIModelsResponse(options.models ?? [], { created }));
  }

  if (request.method === "POST" && pathname === "/v1/completions") {
    return await completionResponse(request, options);
  }

  if (request.method === "POST" && pathname === "/v1/chat/completions") {
    return await chatCompletionResponse(request, options);
  }

  return null;
}

/** Create a Bun-compatible fetch handler for the serving API. */
export function createFetchHandler(
  options: ServeAppOptions,
): (request: Request) => Promise<Response> {
  return async (request) => {
    const url = new URL(request.url);
    const startedAt = performance.now();
    emit(options, { type: "request_start", method: request.method, path: url.pathname });
    if (request.method === "GET" && url.pathname === "/health") {
      const response = jsonResponse({ status: "ok" });
      emit(options, {
        type: "request_complete",
        method: request.method,
        path: url.pathname,
        status: response.status,
        durationMs: performance.now() - startedAt,
      });
      return response;
    }

    try {
      const response = await openAIRouteResponse(request, options, url.pathname);
      if (response !== null) {
        emit(options, {
          type: "request_complete",
          method: request.method,
          path: url.pathname,
          status: response.status,
          durationMs: performance.now() - startedAt,
        });
        return response;
      }
    } catch (error) {
      const details = serveErrorDetails(error);
      emit(options, {
        type: "request_error",
        method: request.method,
        path: url.pathname,
        message: details.message,
        code: details.code,
        status: details.status,
        durationMs: performance.now() - startedAt,
      });
      return openAIErrorResponse(error);
    }

    const response = jsonResponse({ error: { message: "Not found" } }, 404);
    emit(options, {
      type: "request_complete",
      method: request.method,
      path: url.pathname,
      status: response.status,
      durationMs: performance.now() - startedAt,
    });
    return response;
  };
}

/** Start a Bun server for the serving API. */
export function startServeServer(options: ServeServerOptions): ReturnType<typeof Bun.serve> {
  const serverOptions: Parameters<typeof Bun.serve>[0] = {
    port: options.port ?? 3000,
    fetch: createFetchHandler(options),
  };
  if (options.hostname !== undefined) {
    serverOptions.hostname = options.hostname;
  }
  return Bun.serve(serverOptions);
}
