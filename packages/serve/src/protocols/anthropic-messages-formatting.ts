/**
 * Anthropic Messages output formatting.
 * @module
 */

import type { ChatToolCall } from "@mlxts/transformers";
import { isRecord } from "../errors";
import type { GenerationUsage, NormalizedFinishReason, NormalizedGenerationResult } from "../types";
import type {
  AnthropicContentBlock,
  AnthropicMessageResponse,
  AnthropicStopReason,
  NormalizedAnthropicMessage,
} from "./anthropic-messages";
import { stripGeneratedChatControlTokens } from "./openai-chat-completions";
import { extractOpenAIChatToolCalls } from "./openai-chat-tool-calls";
import { splitReasoningTags } from "./reasoning-tags";

function splitReasoningText(text: string): { content: string; reasoningContent?: string } {
  return splitReasoningTags(text);
}

type FormattedAnthropicContent = {
  blocks: AnthropicContentBlock[];
  stopReason?: AnthropicStopReason;
};

function hasToolOutputEnabled(message: NormalizedAnthropicMessage): boolean {
  return (
    (message.request.input.kind === "messages" || message.request.input.kind === "content") &&
    message.request.input.tools !== undefined &&
    message.request.input.tools.length > 0
  );
}

function toolInput(toolCall: ChatToolCall): Record<string, unknown> {
  try {
    const parsed: unknown = JSON.parse(toolCall.function.arguments);
    return isRecord(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function toolUseBlocks(toolCalls: readonly ChatToolCall[]): AnthropicContentBlock[] {
  return toolCalls.map((toolCall, index) => ({
    type: "tool_use",
    id: toolCall.id ?? `toolu_${index + 1}`,
    name: toolCall.function.name,
    input: toolInput(toolCall),
  }));
}

function contentBlocks(
  message: NormalizedAnthropicMessage,
  result: NormalizedGenerationResult,
): FormattedAnthropicContent {
  const split = splitReasoningText(result.text);
  const reasoningContent = result.reasoningContent ?? split.reasoningContent;
  const toolCandidate = result.reasoningContent === undefined ? split.content : result.text.trim();
  const extractedToolCalls = hasToolOutputEnabled(message)
    ? extractOpenAIChatToolCalls(toolCandidate)
    : null;
  const text = stripGeneratedChatControlTokens(extractedToolCalls?.content ?? toolCandidate);
  const blocks: AnthropicContentBlock[] = [];
  if (reasoningContent !== undefined && reasoningContent.trim() !== "") {
    blocks.push({ type: "thinking", thinking: reasoningContent, signature: "" });
  }
  if (text !== "" || (blocks.length === 0 && extractedToolCalls === null)) {
    blocks.push({ type: "text", text });
  }
  if (extractedToolCalls !== null) {
    blocks.push(...toolUseBlocks(extractedToolCalls.toolCalls));
  }
  return {
    blocks,
    ...(extractedToolCalls === null ? {} : { stopReason: "tool_use" }),
  };
}

function formatUsage(usage: GenerationUsage | undefined): {
  input_tokens: number;
  output_tokens: number;
} {
  return {
    input_tokens: usage?.promptTokens ?? 0,
    output_tokens: usage?.completionTokens ?? 0,
  };
}

export function anthropicStopReason(
  reason: NormalizedFinishReason,
  stoppedByStopSequence = false,
): AnthropicStopReason {
  if (stoppedByStopSequence) {
    return "stop_sequence";
  }
  if (reason === "length") {
    return "max_tokens";
  }
  if (reason === "cancelled") {
    return null;
  }
  if (reason === "error") {
    return "refusal";
  }
  return "end_turn";
}

/** Format a generation result as an Anthropic Messages response. */
export function formatAnthropicMessageResponse(
  message: NormalizedAnthropicMessage,
  result: NormalizedGenerationResult,
  options: { id: string },
): AnthropicMessageResponse {
  const content = contentBlocks(message, result);
  return {
    id: options.id,
    type: "message",
    role: "assistant",
    content: content.blocks,
    model: message.model,
    stop_reason: content.stopReason ?? anthropicStopReason(result.finishReason),
    stop_sequence: null,
    usage: formatUsage(result.usage),
  };
}
