/**
 * OpenAI Responses route handling.
 * @module
 */

import { jsonResponse, ServeError } from "./errors";
import { formatOpenAIResponse, normalizeOpenAIResponseRequest } from "./protocols/openai-responses";
import { linkAbortSignals, withAbortSignal } from "./server-abort";
import { emitGenerationComplete, emitGenerationError, emitGenerationStart } from "./server-events";
import { readJson } from "./server-json";
import { writeOpenAIResponseStreamEvents } from "./server-responses-streaming";
import { completeGenerationStream, failGenerationStream } from "./server-stream-lifecycle";
import { createGenerationStreamObserver } from "./server-stream-observability";
import { closeStreamEvents, sseHeaders } from "./server-streaming";
import type {
  GenerationEngine,
  GenerationStreamEvent,
  NormalizedGenerationRequest,
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
        abort: () => control.abort(),
        observer: streamObserver,
      }).then(
        (summary) => {
          completeGenerationStream(
            {
              options,
              request,
              generationRequest,
              abortScope: control,
              observer: streamObserver,
              generationStartedAt: control.generationStartedAt,
              requestStartedAt: control.requestStartedAt,
              controller,
              isCancelled: () => cancelled,
            },
            summary,
            "Client disconnected during streaming response output.",
          );
        },
        (error: unknown) => {
          failGenerationStream(
            {
              options,
              request,
              generationRequest,
              abortScope: control,
              observer: streamObserver,
              generationStartedAt: control.generationStartedAt,
              requestStartedAt: control.requestStartedAt,
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
