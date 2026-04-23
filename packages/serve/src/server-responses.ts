/**
 * OpenAI Responses route handling.
 * @module
 */

import { jsonResponse } from "./errors";
import { formatOpenAIResponse, normalizeOpenAIResponseRequest } from "./protocols/openai-responses";
import { emitGenerationComplete, emitGenerationError, emitGenerationStart } from "./server-events";
import type { GenerationEngine, ServeEvent } from "./types";

export type OpenAIResponsesRouteOptions = {
  engine: GenerationEngine;
  onEvent?: (event: ServeEvent) => void;
};

/** Handle one non-streaming OpenAI Responses request body. */
export async function openAIResponseResponse(
  body: unknown,
  options: OpenAIResponsesRouteOptions,
  responseOptions: { id: string; created: number },
): Promise<Response> {
  const response = normalizeOpenAIResponseRequest(body, { id: responseOptions.id });
  const startedAt = performance.now();
  emitGenerationStart(options, response.request);
  try {
    const result = await options.engine.generate(response.request);
    emitGenerationComplete(options, response.request, result, performance.now() - startedAt);
    return jsonResponse(formatOpenAIResponse(response, result, responseOptions));
  } catch (error) {
    emitGenerationError(options, response.request, error, performance.now() - startedAt);
    throw error;
  }
}
