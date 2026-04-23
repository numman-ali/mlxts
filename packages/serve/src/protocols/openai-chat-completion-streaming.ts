/**
 * OpenAI chat-completion streaming helpers.
 * @module
 */

import { isRecord, ServeError } from "../errors";
import type { GenerationUsage, NormalizedFinishReason } from "../types";

export type OpenAIChatCompletionUsage = {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
};

export type OpenAIChatCompletionChunkChoice = {
  index: number;
  delta: {
    role?: "assistant";
    content?: string;
    reasoning_content?: string;
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
};

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
    pushDelta(deltas, "content", state.buffer.slice(0, safeLength));
    state.buffer = state.buffer.slice(safeLength);
    return true;
  }

  if (closeIndex >= 0 && (openIndex < 0 || closeIndex < openIndex)) {
    pushDelta(deltas, "reasoningContent", state.buffer.slice(0, closeIndex));
    state.buffer = state.buffer.slice(closeIndex + THINK_CLOSE.length);
    return false;
  }

  pushDelta(deltas, "content", state.buffer.slice(0, openIndex));
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
  pushDelta(deltas, state.mode === "content" ? "content" : "reasoningContent", state.buffer);
  state.buffer = "";
  return deltas;
}

function finishReason(
  reason: NormalizedFinishReason,
): OpenAIChatCompletionChunkChoice["finish_reason"] {
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
  return {
    ...(usage.promptTokens === undefined ? {} : { prompt_tokens: usage.promptTokens }),
    ...(usage.completionTokens === undefined ? {} : { completion_tokens: usage.completionTokens }),
    ...(usage.totalTokens === undefined ? {} : { total_tokens: usage.totalTokens }),
  };
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
  const state: ReasoningStreamState = { buffer: "", mode: "content" };
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
  chat: { model: string },
  delta: OpenAIChatCompletionStreamDelta,
  options: {
    id: string;
    created: number;
    includeRole?: boolean;
    finishReason?: NormalizedFinishReason | null;
  },
): OpenAIChatCompletionChunk {
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
        },
        finish_reason:
          options.finishReason === undefined || options.finishReason === null
            ? null
            : finishReason(options.finishReason),
      },
    ],
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
