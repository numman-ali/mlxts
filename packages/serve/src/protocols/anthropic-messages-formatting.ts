/**
 * Anthropic Messages output formatting.
 * @module
 */

import type { GenerationUsage, NormalizedFinishReason, NormalizedGenerationResult } from "../types";
import type {
  AnthropicContentBlock,
  AnthropicMessageResponse,
  AnthropicStopReason,
  NormalizedAnthropicMessage,
} from "./anthropic-messages";
import { splitReasoningTags } from "./reasoning-tags";

function splitReasoningText(text: string): { content: string; reasoningContent?: string } {
  return splitReasoningTags(text);
}

function contentBlocks(result: NormalizedGenerationResult): AnthropicContentBlock[] {
  const split = splitReasoningText(result.text);
  const reasoningContent = result.reasoningContent ?? split.reasoningContent;
  const text = result.reasoningContent === undefined ? split.content : result.text.trim();
  const blocks: AnthropicContentBlock[] = [];
  if (reasoningContent !== undefined && reasoningContent.trim() !== "") {
    blocks.push({ type: "thinking", thinking: reasoningContent, signature: "" });
  }
  if (text !== "" || blocks.length === 0) {
    blocks.push({ type: "text", text });
  }
  return blocks;
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
  return {
    id: options.id,
    type: "message",
    role: "assistant",
    content: contentBlocks(result),
    model: message.model,
    stop_reason: anthropicStopReason(result.finishReason),
    stop_sequence: null,
    usage: formatUsage(result.usage),
  };
}
