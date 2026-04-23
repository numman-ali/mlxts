/**
 * Streaming parser for generated chat tool-call envelopes.
 * @module
 */

import type { OpenAIChatCompletionStreamDelta } from "./openai-chat-completion-streaming";
import { parseOpenAIChatToolCallPayload } from "./openai-chat-tool-calls";

const TOOL_CALL_OPEN = "<tool_call>";
const TOOL_CALL_CLOSE = "</tool_call>";

type ToolCallStreamState = {
  buffer: string;
  mode: "content" | "tool_call";
  nextToolCallIndex: number;
};

function pushContent(deltas: OpenAIChatCompletionStreamDelta[], text: string): void {
  if (text !== "") {
    deltas.push({ content: text });
  }
}

function pushToolCall(
  state: ToolCallStreamState,
  deltas: OpenAIChatCompletionStreamDelta[],
  payload: string,
): void {
  try {
    const toolCall = parseOpenAIChatToolCallPayload(payload.trim(), state.nextToolCallIndex);
    deltas.push({
      toolCalls: [
        {
          index: state.nextToolCallIndex,
          ...(toolCall.id === undefined ? {} : { id: toolCall.id }),
          type: "function",
          function: {
            name: toolCall.function.name,
            arguments: toolCall.function.arguments,
          },
        },
      ],
    });
    state.nextToolCallIndex += 1;
  } catch {
    pushContent(deltas, `${TOOL_CALL_OPEN}${payload}${TOOL_CALL_CLOSE}`);
  }
}

function flushSafeContentTail(state: ToolCallStreamState): string {
  const safeLength = Math.max(0, state.buffer.length - (TOOL_CALL_OPEN.length - 1));
  const emitted = state.buffer.slice(0, safeLength);
  state.buffer = state.buffer.slice(safeLength);
  return emitted;
}

function consumeContentBuffer(
  state: ToolCallStreamState,
  deltas: OpenAIChatCompletionStreamDelta[],
): boolean {
  const openIndex = state.buffer.indexOf(TOOL_CALL_OPEN);
  if (openIndex < 0) {
    const emitted = flushSafeContentTail(state);
    pushContent(deltas, emitted);
    return emitted === "";
  }

  pushContent(deltas, state.buffer.slice(0, openIndex));
  state.buffer = state.buffer.slice(openIndex + TOOL_CALL_OPEN.length);
  state.mode = "tool_call";
  return false;
}

function consumeToolCallBuffer(
  state: ToolCallStreamState,
  deltas: OpenAIChatCompletionStreamDelta[],
): boolean {
  const closeIndex = state.buffer.indexOf(TOOL_CALL_CLOSE);
  if (closeIndex < 0) {
    return true;
  }

  pushToolCall(state, deltas, state.buffer.slice(0, closeIndex));
  state.buffer = state.buffer.slice(closeIndex + TOOL_CALL_CLOSE.length);
  state.mode = "content";
  return false;
}

function appendToolCallStreamChunk(
  state: ToolCallStreamState,
  text: string,
): OpenAIChatCompletionStreamDelta[] {
  state.buffer += text;
  const deltas: OpenAIChatCompletionStreamDelta[] = [];

  while (state.buffer !== "") {
    const stalled =
      state.mode === "content"
        ? consumeContentBuffer(state, deltas)
        : consumeToolCallBuffer(state, deltas);
    if (stalled) {
      break;
    }
  }

  return deltas;
}

function flushToolCallStream(state: ToolCallStreamState): OpenAIChatCompletionStreamDelta[] {
  if (state.buffer === "") {
    return [];
  }

  const deltas: OpenAIChatCompletionStreamDelta[] = [];
  pushContent(deltas, state.mode === "content" ? state.buffer : `${TOOL_CALL_OPEN}${state.buffer}`);
  state.buffer = "";
  state.mode = "content";
  return deltas;
}

/** Create a streaming parser that emits structured tool-call deltas only when tools are enabled. */
export function createOpenAIChatCompletionToolCallStream(enabled: boolean): {
  push(text: string): OpenAIChatCompletionStreamDelta[];
  finish(): OpenAIChatCompletionStreamDelta[];
} {
  if (!enabled) {
    return {
      push(text) {
        return text === "" ? [] : [{ content: text }];
      },
      finish() {
        return [];
      },
    };
  }

  const state: ToolCallStreamState = { buffer: "", mode: "content", nextToolCallIndex: 0 };
  return {
    push(text) {
      return appendToolCallStreamChunk(state, text);
    },
    finish() {
      return flushToolCallStream(state);
    },
  };
}
