/**
 * OpenAI Responses output formatting.
 * @module
 */

import type { ChatToolCall } from "@mlxts/transformers";
import type { GenerationUsage, NormalizedFinishReason, NormalizedGenerationResult } from "../types";
import { stripGeneratedChatControlTokens } from "./openai-chat-completions";
import { extractOpenAIChatToolCalls } from "./openai-chat-tool-calls";
import type {
  NormalizedOpenAIResponse,
  OpenAIResponseFunctionCallItem,
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

type ParsedResponseOutput = {
  items: OpenAIResponseOutputItem[];
  outputText: string;
};

function hasToolOutputEnabled(response: NormalizedOpenAIResponse): boolean {
  return (
    (response.request.input.kind === "messages" || response.request.input.kind === "content") &&
    response.request.input.tools !== undefined &&
    response.request.input.tools.length > 0
  );
}

function functionCallItem(
  toolCall: ChatToolCall,
  index: number,
  options: { id: string },
): OpenAIResponseFunctionCallItem {
  const callId = toolCall.id ?? `call_${index + 1}`;
  return {
    id: `${options.id}-fc-${index + 1}`,
    type: "function_call",
    status: "completed",
    call_id: callId,
    name: toolCall.function.name,
    arguments: toolCall.function.arguments,
  };
}

function outputItems(
  response: NormalizedOpenAIResponse,
  result: NormalizedGenerationResult,
  options: { id: string },
): ParsedResponseOutput {
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
  const extractedToolCalls = hasToolOutputEnabled(response)
    ? extractOpenAIChatToolCalls(result.text)
    : null;
  const outputText = stripGeneratedChatControlTokens(extractedToolCalls?.content ?? result.text);
  const messageItems: OpenAIResponseOutputItem[] =
    outputText !== "" || (reasoning.length === 0 && extractedToolCalls === null)
      ? [
          {
            id: `${options.id}-msg`,
            type: "message",
            status: "completed",
            role: "assistant",
            content: [{ type: "output_text", text: outputText, annotations: [] }],
          },
        ]
      : [];
  const toolItems =
    extractedToolCalls?.toolCalls.map((toolCall, index) =>
      functionCallItem(toolCall, index, options),
    ) ?? [];
  return {
    outputText,
    items: [...reasoning, ...messageItems, ...toolItems],
  };
}

/** Format a generation result as an OpenAI Responses object. */
export function formatOpenAIResponse(
  response: NormalizedOpenAIResponse,
  result: NormalizedGenerationResult,
  options: { id: string; created: number },
): OpenAIResponseObject {
  const status = responseStatus(result.finishReason);
  const output = outputItems(response, result, { id: options.id });
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
    output: output.items,
    output_text: output.outputText,
    parallel_tool_calls: response.parallelToolCalls,
    previous_response_id: null,
    reasoning: { effort: null, summary: null },
    store: false,
    temperature: response.temperature,
    text: { format: { type: "text" } },
    tool_choice: response.toolChoice,
    tools: [...response.tools],
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
    tools: [...response.tools],
    top_p: response.topP,
    truncation: "disabled",
    usage: null,
    user: response.user,
    metadata: response.metadata,
  };
}
