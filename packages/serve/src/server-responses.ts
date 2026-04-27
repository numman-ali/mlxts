/**
 * OpenAI Responses route handling.
 * @module
 */

import { jsonResponse, ServeError } from "./errors";
import { formatOpenAIResponse, normalizeOpenAIResponseRequest } from "./protocols/openai-responses";
import { linkAbortSignals, withAbortSignal } from "./server-abort";
import {
  emitGenerationComplete,
  emitGenerationError,
  emitGenerationStart,
  emitRequestComplete,
  emitRequestError,
  serveErrorDetails,
} from "./server-events";
import { writeOpenAIResponseStreamEvents } from "./server-responses-streaming";
import { createGenerationStreamObserver } from "./server-stream-observability";
import { closeStreamEvents, sseHeaders } from "./server-streaming";
import type {
  GenerationEngine,
  GenerationStreamEvent,
  NormalizedGenerationRequest,
  NormalizedGenerationResult,
  ServeEvent,
} from "./types";

export type OpenAIResponsesRouteOptions = {
  engine: GenerationEngine;
  abortSignal?: AbortSignal;
  onEvent?: (event: ServeEvent) => void;
};

type ResponseStreamControl = {
  id: string;
  created: number;
  generationStartedAt: number;
  requestStartedAt: number;
  signal: AbortSignal;
  abort(): void;
  dispose(): void;
};

async function readJson(request: Request): Promise<unknown> {
  try {
    return await request.json();
  } catch {
    throw new ServeError("Request body must be valid JSON.", {
      code: "invalid_json",
    });
  }
}

function streamNotSupported(): ServeError {
  return new ServeError("This generation engine does not support streaming yet.", {
    code: "stream_not_supported",
    param: "stream",
  });
}

function responseStreamBody(
  request: Request,
  options: OpenAIResponsesRouteOptions,
  response: ReturnType<typeof normalizeOpenAIResponseRequest>,
  stream: AsyncIterable<GenerationStreamEvent>,
  control: ResponseStreamControl,
  generationRequest: NormalizedGenerationRequest,
): ReadableStream<Uint8Array> {
  const iterator = stream[Symbol.asyncIterator]();
  const streamObserver = createGenerationStreamObserver(
    options,
    generationRequest,
    control.generationStartedAt,
  );
  let cancelled = false;
  return new ReadableStream<Uint8Array>({
    start(controller) {
      void writeOpenAIResponseStreamEvents(controller, iterator, response, {
        id: control.id,
        created: control.created,
        signal: control.signal,
        observer: streamObserver,
      }).then(
        (summary) => {
          control.dispose();
          const durationMs = performance.now() - control.generationStartedAt;
          streamObserver.end(
            summary.finishReason === "cancelled" ? "cancelled" : "completed",
            summary.finishReason,
            durationMs,
          );
          const result: NormalizedGenerationResult = {
            text: "",
            finishReason: summary.finishReason,
            ...(summary.usage === undefined ? {} : { usage: summary.usage }),
          };
          emitGenerationComplete(options, generationRequest, result, durationMs);
          if (summary.finishReason === "cancelled") {
            emitRequestError(
              options,
              request,
              {
                message: "Client disconnected during streaming response output.",
                code: "client_cancelled",
                status: 499,
              },
              control.requestStartedAt,
            );
          } else {
            emitRequestComplete(options, request, 200, control.requestStartedAt);
          }
          if (!cancelled) {
            controller.close();
          }
        },
        (error: unknown) => {
          control.dispose();
          const durationMs = performance.now() - control.generationStartedAt;
          streamObserver.end("error", "error", durationMs);
          emitGenerationError(options, generationRequest, error, durationMs);
          emitRequestError(options, request, serveErrorDetails(error), control.requestStartedAt);
          if (!cancelled) {
            controller.error(error);
          }
        },
      );
    },
    async cancel() {
      cancelled = true;
      control.abort();
      void closeStreamEvents(iterator);
    },
  });
}

/** Handle one OpenAI Responses HTTP route. */
export async function openAIResponsesRouteResponse(
  request: Request,
  options: OpenAIResponsesRouteOptions,
  responseOptions: { id: string; created: number; startedAt: number },
): Promise<Response> {
  const response = normalizeOpenAIResponseRequest(await readJson(request), {
    id: responseOptions.id,
  });
  const abortScope = linkAbortSignals(request.signal, options.abortSignal);
  const generationRequest = withAbortSignal(response.request, abortScope.signal);
  if (!response.stream) {
    const startedAt = performance.now();
    emitGenerationStart(options, generationRequest);
    try {
      const result = await options.engine.generate(generationRequest);
      emitGenerationComplete(options, generationRequest, result, performance.now() - startedAt);
      return jsonResponse(formatOpenAIResponse(response, result, responseOptions));
    } catch (error) {
      emitGenerationError(options, generationRequest, error, performance.now() - startedAt);
      throw error;
    } finally {
      abortScope.dispose();
    }
  }

  const streamGenerator = options.engine.stream;
  if (streamGenerator === undefined) {
    abortScope.dispose();
    throw streamNotSupported();
  }

  const startedAt = performance.now();
  emitGenerationStart(options, generationRequest);
  let stream: AsyncIterable<GenerationStreamEvent>;
  try {
    stream = await streamGenerator(generationRequest);
  } catch (error) {
    abortScope.dispose();
    emitGenerationError(options, generationRequest, error, performance.now() - startedAt);
    throw error;
  }

  return new Response(
    responseStreamBody(
      request,
      options,
      response,
      stream,
      {
        id: responseOptions.id,
        created: responseOptions.created,
        generationStartedAt: startedAt,
        requestStartedAt: responseOptions.startedAt,
        signal: abortScope.signal,
        abort: abortScope.abort,
        dispose: abortScope.dispose,
      },
      generationRequest,
    ),
    { headers: sseHeaders() },
  );
}
