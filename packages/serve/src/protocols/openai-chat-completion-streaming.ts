/**
 * OpenAI chat-completion streaming helpers.
 * @module
 */

import { isRecord, ServeError } from "../errors";
import type { GenerationUsage, NormalizedFinishReason } from "../types";
import { formatOpenAICompletionLikeUsage, type OpenAICompletionLikeUsage } from "./openai-usage";
import { createReasoningTagStream } from "./reasoning-tags";

export type OpenAIChatCompletionStreamToolCall = {
  index: number;
  id?: string;
  type?: "function";
  function?: {
    name?: string;
    arguments?: string;
  };
};

export type OpenAIChatCompletionUsage = OpenAICompletionLikeUsage;

export type OpenAIChatCompletionChunkChoice = {
  index: number;
  delta: {
    role?: "assistant";
    content?: string;
    reasoning_content?: string;
    tool_calls?: OpenAIChatCompletionStreamToolCall[];
  };
  finish_reason: "stop" | "length" | "tool_calls" | "content_filter" | null;
};

export type OpenAIChatCompletionChunk = {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  choices: OpenAIChatCompletionChunkChoice[];
  usage?: OpenAIChatCompletionUsage | null;
};

export type OpenAIChatCompletionStreamDelta = {
  content?: string;
  reasoningContent?: string;
  toolCalls?: OpenAIChatCompletionStreamToolCall[];
};

function optionalBoolean(record: Record<string, unknown>, key: string): boolean | undefined {
  const value = record[key];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "boolean") {
    throw new ServeError(`OpenAI chat completions: "${key}" must be a boolean.`, { param: key });
  }
  return value;
}

function finishReason(
  reason: NormalizedFinishReason | "tool_calls",
): OpenAIChatCompletionChunkChoice["finish_reason"] {
  if (reason === "tool_calls") {
    return "tool_calls";
  }
  if (reason === "length") {
    return "length";
  }
  if (reason === "stop" || reason === "eos") {
    return "stop";
  }
  if (reason === "cancelled") {
    return null;
  }
  return "content_filter";
}

function formatUsage(usage: GenerationUsage): OpenAIChatCompletionUsage {
  return formatOpenAICompletionLikeUsage(usage);
}

export function parseOpenAIChatCompletionStreamOptions(
  record: Record<string, unknown>,
  stream: boolean,
): { includeUsage: boolean } {
  const value = record.stream_options;
  if (value === undefined || value === null) {
    return { includeUsage: false };
  }
  if (!stream) {
    throw new ServeError('OpenAI chat completions: "stream_options" requires "stream": true.', {
      param: "stream_options",
    });
  }
  if (!isRecord(value)) {
    throw new ServeError('OpenAI chat completions: "stream_options" must be an object or null.', {
      param: "stream_options",
    });
  }

  const includeUsage = optionalBoolean(value, "include_usage") ?? false;
  const includeObfuscation = optionalBoolean(value, "include_obfuscation") ?? false;
  if (includeObfuscation) {
    throw new ServeError(
      'OpenAI chat completions: "stream_options.include_obfuscation" is not supported yet.',
      { param: "stream_options" },
    );
  }
  return { includeUsage };
}

export function createOpenAIChatCompletionReasoningStream(): {
  push(text: string): OpenAIChatCompletionStreamDelta[];
  finish(): OpenAIChatCompletionStreamDelta[];
} {
  return createReasoningTagStream();
}

export function formatOpenAIChatCompletionStreamChunk(
  chat: { model: string; streamOptions?: { includeUsage: boolean } },
  delta: OpenAIChatCompletionStreamDelta,
  options: {
    id: string;
    created: number;
    includeRole?: boolean;
    includeUsage?: boolean;
    finishReason?: NormalizedFinishReason | "tool_calls" | null;
  },
): OpenAIChatCompletionChunk {
  const reason = options.finishReason;
  const includeUsage = options.includeUsage ?? chat.streamOptions?.includeUsage ?? false;
  return {
    id: options.id,
    object: "chat.completion.chunk",
    created: options.created,
    model: chat.model,
    choices: [
      {
        index: 0,
        delta: {
          ...(options.includeRole ? { role: "assistant" } : {}),
          ...(delta.content === undefined ? {} : { content: delta.content }),
          ...(delta.reasoningContent === undefined
            ? {}
            : { reasoning_content: delta.reasoningContent }),
          ...(delta.toolCalls === undefined ? {} : { tool_calls: [...delta.toolCalls] }),
        },
        finish_reason: reason === undefined || reason === null ? null : finishReason(reason),
      },
    ],
    ...(includeUsage ? { usage: null } : {}),
  };
}

export function formatOpenAIChatCompletionUsageStreamChunk(
  chat: { model: string },
  usage: GenerationUsage | undefined,
  options: { id: string; created: number },
): OpenAIChatCompletionChunk {
  return {
    id: options.id,
    object: "chat.completion.chunk",
    created: options.created,
    model: chat.model,
    choices: [],
    usage: usage === undefined ? null : formatUsage(usage),
  };
}
