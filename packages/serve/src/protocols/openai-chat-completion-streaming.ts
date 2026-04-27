/**
 * OpenAI chat-completion streaming helpers.
 * @module
 */

import { isRecord, ServeError } from "../errors";
import type { GenerationUsage, NormalizedFinishReason } from "../types";
import { formatOpenAICompletionLikeUsage, type OpenAICompletionLikeUsage } from "./openai-usage";

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

const THINK_OPEN = "<think>";
const THINK_CLOSE = "</think>";

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

function pushDelta(
  deltas: OpenAIChatCompletionStreamDelta[],
  key: keyof OpenAIChatCompletionStreamDelta,
  text: string,
): void {
  if (text === "") {
    return;
  }
  deltas.push(key === "content" ? { content: text } : { reasoningContent: text });
}

function flushTailLength(buffer: string): number {
  return Math.max(0, buffer.length - (Math.max(THINK_OPEN.length, THINK_CLOSE.length) - 1));
}

type ReasoningStreamState = {
  buffer: string;
  mode: "content" | "reasoning";
  trimLeadingContent: boolean;
};

function pushContentDelta(
  state: ReasoningStreamState,
  deltas: OpenAIChatCompletionStreamDelta[],
  text: string,
): void {
  if (!state.trimLeadingContent) {
    pushDelta(deltas, "content", text);
    return;
  }

  const trimmed = text.trimStart();
  if (trimmed === "") {
    return;
  }
  state.trimLeadingContent = false;
  pushDelta(deltas, "content", trimmed);
}

function consumeContentBuffer(
  state: ReasoningStreamState,
  deltas: OpenAIChatCompletionStreamDelta[],
): boolean {
  const openIndex = state.buffer.indexOf(THINK_OPEN);
  const closeIndex = state.buffer.indexOf(THINK_CLOSE);
  if (openIndex < 0 && closeIndex < 0) {
    const safeLength = flushTailLength(state.buffer);
    if (safeLength === 0) {
      return true;
    }
    pushContentDelta(state, deltas, state.buffer.slice(0, safeLength));
    state.buffer = state.buffer.slice(safeLength);
    return true;
  }

  if (closeIndex >= 0 && (openIndex < 0 || closeIndex < openIndex)) {
    pushDelta(deltas, "reasoningContent", state.buffer.slice(0, closeIndex));
    state.buffer = state.buffer.slice(closeIndex + THINK_CLOSE.length);
    return false;
  }

  pushContentDelta(state, deltas, state.buffer.slice(0, openIndex));
  state.buffer = state.buffer.slice(openIndex + THINK_OPEN.length);
  state.mode = "reasoning";
  return false;
}

function consumeReasoningBuffer(
  state: ReasoningStreamState,
  deltas: OpenAIChatCompletionStreamDelta[],
): boolean {
  const closeIndex = state.buffer.indexOf(THINK_CLOSE);
  if (closeIndex < 0) {
    const safeLength = flushTailLength(state.buffer);
    if (safeLength === 0) {
      return true;
    }
    pushDelta(deltas, "reasoningContent", state.buffer.slice(0, safeLength));
    state.buffer = state.buffer.slice(safeLength);
    return true;
  }

  pushDelta(deltas, "reasoningContent", state.buffer.slice(0, closeIndex));
  state.buffer = state.buffer.slice(closeIndex + THINK_CLOSE.length);
  state.mode = "content";
  state.trimLeadingContent = true;
  return false;
}

function appendReasoningStreamChunk(
  state: ReasoningStreamState,
  text: string,
): OpenAIChatCompletionStreamDelta[] {
  state.buffer += text;
  const deltas: OpenAIChatCompletionStreamDelta[] = [];

  while (state.buffer !== "") {
    const stalled =
      state.mode === "content"
        ? consumeContentBuffer(state, deltas)
        : consumeReasoningBuffer(state, deltas);
    if (stalled) {
      break;
    }
  }

  return deltas;
}

function flushReasoningStream(state: ReasoningStreamState): OpenAIChatCompletionStreamDelta[] {
  if (state.buffer === "") {
    return [];
  }

  const deltas: OpenAIChatCompletionStreamDelta[] = [];
  if (state.mode === "content") {
    pushContentDelta(state, deltas, state.buffer);
  } else {
    pushDelta(deltas, "reasoningContent", state.buffer);
  }
  state.buffer = "";
  return deltas;
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
  const state: ReasoningStreamState = { buffer: "", mode: "content", trimLeadingContent: false };
  return {
    push(text) {
      return appendReasoningStreamChunk(state, text);
    },
    finish() {
      return flushReasoningStream(state);
    },
  };
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
