/**
 * OpenAI Responses route handling.
 * @module
 */

import { jsonResponse } from "./errors";
import { formatOpenAIResponse, normalizeOpenAIResponseRequest } from "./protocols/openai-responses";
import { emitGenerationComplete, emitGenerationError, emitGenerationStart } from "./server-events";
import type { GenerationEngine, NormalizedGenerationRequest, ServeEvent } from "./types";

export type OpenAIResponsesRouteOptions = {
  engine: GenerationEngine;
  onEvent?: (event: ServeEvent) => void;
};

function withAbortSignal(
  request: NormalizedGenerationRequest,
  signal: AbortSignal | undefined,
): NormalizedGenerationRequest {
  return signal === undefined ? request : { ...request, abortSignal: signal };
}

/** Handle one non-streaming OpenAI Responses request body. */
export async function openAIResponseResponse(
  body: unknown,
  options: OpenAIResponsesRouteOptions,
  responseOptions: { id: string; created: number; signal?: AbortSignal },
): Promise<Response> {
  const response = normalizeOpenAIResponseRequest(body, { id: responseOptions.id });
  const generationRequest = withAbortSignal(response.request, responseOptions.signal);
  const startedAt = performance.now();
  emitGenerationStart(options, generationRequest);
  try {
    const result = await options.engine.generate(generationRequest);
    emitGenerationComplete(options, generationRequest, result, performance.now() - startedAt);
    return jsonResponse(formatOpenAIResponse(response, result, responseOptions));
  } catch (error) {
    emitGenerationError(options, generationRequest, error, performance.now() - startedAt);
    throw error;
  }
}
