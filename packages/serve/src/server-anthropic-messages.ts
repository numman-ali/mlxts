/**
 * Anthropic Messages route handling.
 * @module
 */

import { jsonResponse, ServeError } from "./errors";
import {
  formatAnthropicMessageResponse,
  normalizeAnthropicMessageRequest,
} from "./protocols/anthropic-messages";
import { linkAbortSignals, withAbortSignal } from "./server-abort";
import { writeAnthropicMessageStreamEvents } from "./server-anthropic-messages-streaming";
import { emitGenerationComplete, emitGenerationError, emitGenerationStart } from "./server-events";
import { readJson } from "./server-json";
import { completeGenerationStream, failGenerationStream } from "./server-stream-lifecycle";
import { createGenerationStreamObserver } from "./server-stream-observability";
import { closeStreamEvents, sseHeaders } from "./server-streaming";
import type { GenerationEngine, GenerationStreamEvent, ServeEvent } from "./types";

export type AnthropicMessagesRouteOptions = {
  engine: GenerationEngine;
  abortSignal?: AbortSignal;
  onEvent?: (event: ServeEvent) => void;
};

function streamNotSupported(): ServeError {
  return new ServeError("This generation engine does not support streaming yet.", {
    code: "stream_not_supported",
    param: "stream",
  });
}

/** Handle one Anthropic Messages HTTP route. */
export async function anthropicMessagesRouteResponse(
  request: Request,
  options: AnthropicMessagesRouteOptions,
  responseOptions: { id: string; startedAt: number },
): Promise<Response> {
  const message = normalizeAnthropicMessageRequest(await readJson(request), {
    id: responseOptions.id,
  });
  const abortScope = linkAbortSignals(request.signal, options.abortSignal);
  const generationRequest = withAbortSignal(message.request, abortScope.signal);

  if (!message.stream) {
    const startedAt = performance.now();
    emitGenerationStart(options, generationRequest);
    try {
      const result = await options.engine.generate(generationRequest);
      emitGenerationComplete(options, generationRequest, result, performance.now() - startedAt);
      return jsonResponse(formatAnthropicMessageResponse(message, result, responseOptions));
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

  const iterator = stream[Symbol.asyncIterator]();
  const observer = createGenerationStreamObserver(options, generationRequest, startedAt);
  let cancelled = false;
  return new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        void writeAnthropicMessageStreamEvents(controller, iterator, message, {
          id: responseOptions.id,
          created: 0,
          signal: abortScope.signal,
          abort: () => abortScope.abort(),
          observer,
        }).then(
          (summary) => {
            completeGenerationStream(
              {
                options,
                request,
                generationRequest,
                abortScope,
                observer,
                generationStartedAt: startedAt,
                requestStartedAt: responseOptions.startedAt,
                controller,
                isCancelled: () => cancelled,
              },
              summary,
              "Client disconnected during streaming Anthropic message output.",
            );
          },
          (error: unknown) => {
            failGenerationStream(
              {
                options,
                request,
                generationRequest,
                abortScope,
                observer,
                generationStartedAt: startedAt,
                requestStartedAt: responseOptions.startedAt,
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
        abortScope.abort();
        void closeStreamEvents(iterator);
      },
    }),
    { headers: sseHeaders() },
  );
}
