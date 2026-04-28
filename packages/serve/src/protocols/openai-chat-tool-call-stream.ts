/**
 * Streaming parser for generated chat tool-call envelopes.
 * @module
 */

import type { OpenAIChatCompletionStreamDelta } from "./openai-chat-completion-streaming";
import { parseOpenAIChatToolCallPayload } from "./openai-chat-tool-calls";

const TOOL_CALL_OPEN = "<tool_call>";
const TOOL_CALL_CLOSE = "</tool_call>";
const GEMMA_TOOL_CALL_OPEN = "<|tool_call>";
const GEMMA_TOOL_CALL_CLOSE = "<tool_call|>";
const GEMMA_TOOL_RESPONSE_OPEN = "<|tool_response>";
const GEMMA_TOOL_RESPONSE_CLOSE = "<tool_response|>";
const TOOL_CALL_OPEN_MARKERS = [TOOL_CALL_OPEN, GEMMA_TOOL_CALL_OPEN] as const;
const GENERATED_CONTROL_MARKERS = [GEMMA_TOOL_RESPONSE_OPEN, GEMMA_TOOL_RESPONSE_CLOSE] as const;
const MAX_TOOL_CALL_OPEN_LENGTH = Math.max(
  ...TOOL_CALL_OPEN_MARKERS.map((marker) => marker.length),
  ...GENERATED_CONTROL_MARKERS.map((marker) => marker.length),
);

type ToolCallStreamState = {
  buffer: string;
  mode: "content" | "tool_call";
  nextToolCallIndex: number;
  openMarker: string;
  closeMarker: string;
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
    pushContent(deltas, `${state.openMarker}${payload}${state.closeMarker}`);
  }
}

function flushSafeContentTail(state: ToolCallStreamState): string {
  const safeLength = Math.max(0, state.buffer.length - (MAX_TOOL_CALL_OPEN_LENGTH - 1));
  const emitted = state.buffer.slice(0, safeLength);
  state.buffer = state.buffer.slice(safeLength);
  return emitted;
}

function consumeContentBuffer(
  state: ToolCallStreamState,
  deltas: OpenAIChatCompletionStreamDelta[],
): boolean {
  const toolCallMatches = TOOL_CALL_OPEN_MARKERS.map((marker) => ({
    marker,
    index: state.buffer.indexOf(marker),
  })).filter((match) => match.index >= 0);
  const controlMatches = GENERATED_CONTROL_MARKERS.map((marker) => ({
    marker,
    index: state.buffer.indexOf(marker),
  })).filter((match) => match.index >= 0);
  const toolCallMatch = toolCallMatches.sort((left, right) => left.index - right.index)[0];
  const controlMatch = controlMatches.sort((left, right) => left.index - right.index)[0];
  if (toolCallMatch === undefined && controlMatch === undefined) {
    const emitted = flushSafeContentTail(state);
    pushContent(deltas, emitted);
    return emitted === "";
  }

  if (
    controlMatch !== undefined &&
    (toolCallMatch === undefined || controlMatch.index < toolCallMatch.index)
  ) {
    pushContent(deltas, state.buffer.slice(0, controlMatch.index));
    state.buffer = state.buffer.slice(controlMatch.index + controlMatch.marker.length);
    return false;
  }

  if (toolCallMatch === undefined) {
    return true;
  }
  state.openMarker = toolCallMatch.marker;
  state.closeMarker =
    toolCallMatch.marker === GEMMA_TOOL_CALL_OPEN ? GEMMA_TOOL_CALL_CLOSE : TOOL_CALL_CLOSE;
  pushContent(deltas, state.buffer.slice(0, toolCallMatch.index));
  state.buffer = state.buffer.slice(toolCallMatch.index + toolCallMatch.marker.length);
  state.mode = "tool_call";
  return false;
}

function consumeToolCallBuffer(
  state: ToolCallStreamState,
  deltas: OpenAIChatCompletionStreamDelta[],
): boolean {
  const closeIndex = state.buffer.indexOf(state.closeMarker);
  if (closeIndex < 0) {
    return true;
  }

  pushToolCall(state, deltas, state.buffer.slice(0, closeIndex));
  state.buffer = state.buffer.slice(closeIndex + state.closeMarker.length);
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
  pushContent(
    deltas,
    state.mode === "content" ? state.buffer : `${state.openMarker}${state.buffer}`,
  );
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

  const state: ToolCallStreamState = {
    buffer: "",
    mode: "content",
    nextToolCallIndex: 0,
    openMarker: TOOL_CALL_OPEN,
    closeMarker: TOOL_CALL_CLOSE,
  };
  return {
    push(text) {
      return appendToolCallStreamChunk(state, text);
    },
    finish() {
      return flushToolCallStream(state);
    },
  };
}
