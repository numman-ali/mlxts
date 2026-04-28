/**
 * OpenAI Responses output formatting.
 * @module
 */

import type { GenerationUsage, NormalizedFinishReason, NormalizedGenerationResult } from "../types";
import type {
  NormalizedOpenAIResponse,
  OpenAIResponseObject,
  OpenAIResponseOutputItem,
  OpenAIResponseReasoningItem,
  OpenAIResponseUsage,
} from "./openai-responses";

function responseUsage(usage: GenerationUsage | undefined): OpenAIResponseUsage | null {
  if (usage === undefined) {
    return null;
  }
  const inputTokens = usage.promptTokens ?? 0;
  const outputTokens = usage.completionTokens ?? 0;
  const cachedTokens = usage.cacheReadTokens ?? 0;
  return {
    input_tokens: inputTokens,
    input_tokens_details: { cached_tokens: cachedTokens },
    output_tokens: outputTokens,
    output_tokens_details: { reasoning_tokens: 0 },
    total_tokens: usage.totalTokens ?? inputTokens + outputTokens,
  };
}

function responseStatus(
  finishReason: NormalizedFinishReason,
): Pick<OpenAIResponseObject, "completed_at" | "incomplete_details" | "status"> {
  if (finishReason === "length") {
    return {
      status: "incomplete",
      completed_at: null,
      incomplete_details: { reason: "max_output_tokens" },
    };
  }
  return {
    status: "completed",
    completed_at: 0,
    incomplete_details: null,
  };
}

function pendingStatus(): Pick<
  OpenAIResponseObject,
  "completed_at" | "incomplete_details" | "status"
> {
  return {
    status: "in_progress",
    completed_at: null,
    incomplete_details: null,
  };
}

function outputItems(
  result: NormalizedGenerationResult,
  options: { id: string },
): OpenAIResponseOutputItem[] {
  const reasoning: OpenAIResponseReasoningItem[] = [];
  if (result.reasoningContent !== undefined && result.reasoningContent.trim() !== "") {
    reasoning.push({
      id: `${options.id}-rsn`,
      type: "reasoning",
      status: "completed",
      summary: [],
      content: [{ type: "reasoning_text", text: result.reasoningContent }],
    });
  }
  return [
    ...reasoning,
    {
      id: `${options.id}-msg`,
      type: "message",
      status: "completed",
      role: "assistant",
      content: [{ type: "output_text", text: result.text, annotations: [] }],
    },
  ];
}

/** Format a generation result as an OpenAI Responses object. */
export function formatOpenAIResponse(
  response: NormalizedOpenAIResponse,
  result: NormalizedGenerationResult,
  options: { id: string; created: number },
): OpenAIResponseObject {
  const status = responseStatus(result.finishReason);
  return {
    id: options.id,
    object: "response",
    created_at: options.created,
    status: status.status,
    completed_at: status.completed_at === null ? null : options.created,
    error: null,
    incomplete_details: status.incomplete_details,
    instructions: response.instructions,
    max_output_tokens: response.maxOutputTokens,
    model: response.model,
    output: outputItems(result, { id: options.id }),
    output_text: result.text,
    parallel_tool_calls: response.parallelToolCalls,
    previous_response_id: null,
    reasoning: { effort: null, summary: null },
    store: false,
    temperature: response.temperature,
    text: { format: { type: "text" } },
    tool_choice: response.toolChoice,
    tools: [],
    top_p: response.topP,
    truncation: "disabled",
    usage: responseUsage(result.usage),
    user: response.user,
    metadata: response.metadata,
  };
}

/** Format the initial in-progress object for an OpenAI Responses SSE stream. */
export function formatOpenAIResponsePending(
  response: NormalizedOpenAIResponse,
  options: { id: string; created: number },
): OpenAIResponseObject {
  const status = pendingStatus();
  return {
    id: options.id,
    object: "response",
    created_at: options.created,
    status: status.status,
    completed_at: status.completed_at,
    error: null,
    incomplete_details: status.incomplete_details,
    instructions: response.instructions,
    max_output_tokens: response.maxOutputTokens,
    model: response.model,
    output: [],
    output_text: "",
    parallel_tool_calls: response.parallelToolCalls,
    previous_response_id: null,
    reasoning: { effort: null, summary: null },
    store: false,
    temperature: response.temperature,
    text: { format: { type: "text" } },
    tool_choice: response.toolChoice,
    tools: [],
    top_p: response.topP,
    truncation: "disabled",
    usage: null,
    user: response.user,
    metadata: response.metadata,
  };
}
