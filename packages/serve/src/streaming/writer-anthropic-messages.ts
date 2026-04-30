/**
 * Anthropic Messages SSE streaming writer.
 * @module
 */

import {
  anthropicStopReason,
  type NormalizedAnthropicMessage,
} from "../protocols/anthropic-messages";
import type { OpenAIChatCompletionStreamToolCall } from "../protocols/openai-chat-completion-streaming";
import {
  createOpenAIChatCompletionReasoningStream,
  stripGeneratedChatControlTokens,
} from "../protocols/openai-chat-completions";
import { createOpenAIChatCompletionToolCallStream } from "../protocols/openai-chat-tool-call-stream";
import type { GenerationStreamEvent, GenerationUsage, NormalizedFinishReason } from "../types";
import {
  enqueueObservedSse,
  type StreamControlOptions,
  type StreamObserver,
  type StreamObserverChunkKind,
  type StreamSummary,
  streamSummary,
  streamWasCancelled,
} from "./runtime";
import { createStopSequenceFilter } from "./stop-filter";
import { runSseGenerationStream } from "./writer-base";

type BlockKind = "thinking" | "text" | "tool_use";

type AnthropicStreamState = {
  reasoning: ReturnType<typeof createOpenAIChatCompletionReasoningStream>;
  toolCallParser: ReturnType<typeof createOpenAIChatCompletionToolCallStream>;
  stopFilter: ReturnType<typeof createStopSequenceFilter>;
  finalUsage: GenerationUsage | undefined;
  finalFinishReason: NormalizedFinishReason;
  stoppedByStopSequence: boolean;
  activeBlock: BlockKind | null;
  nextBlockIndex: number;
  toolUseCount: number;
  observer?: StreamObserver;
};

function observerOptions(state: AnthropicStreamState): { observer: StreamObserver } | undefined {
  return state.observer === undefined ? undefined : { observer: state.observer };
}

function enqueueEvent(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: AnthropicStreamState,
  event: string,
  data: unknown,
  kind: StreamObserverChunkKind,
): void {
  enqueueObservedSse(
    controller,
    `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`,
    observerOptions(state),
    kind,
  );
}

function usage(usageInfo: GenerationUsage | undefined): {
  input_tokens: number;
  output_tokens: number;
} {
  return {
    input_tokens: usageInfo?.promptTokens ?? 0,
    output_tokens: usageInfo?.completionTokens ?? 0,
  };
}

function hasStreamingToolOutput(message: NormalizedAnthropicMessage): boolean {
  return (
    (message.request.input.kind === "messages" || message.request.input.kind === "content") &&
    message.request.input.tools !== undefined &&
    message.request.input.tools.length > 0
  );
}

function blockIndex(state: AnthropicStreamState, kind: BlockKind): number {
  return kind === "thinking" && state.activeBlock === "thinking"
    ? state.nextBlockIndex - 1
    : kind === "text" && state.activeBlock === "text"
      ? state.nextBlockIndex - 1
      : kind === "tool_use" && state.activeBlock === "tool_use"
        ? state.nextBlockIndex - 1
        : state.nextBlockIndex;
}

function closeActiveBlock(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: AnthropicStreamState,
): void {
  if (state.activeBlock === null) {
    return;
  }
  enqueueEvent(
    controller,
    state,
    "content_block_stop",
    { type: "content_block_stop", index: state.nextBlockIndex - 1 },
    "protocol",
  );
  state.activeBlock = null;
}

function ensureBlock(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: AnthropicStreamState,
  kind: BlockKind,
  toolCall?: OpenAIChatCompletionStreamToolCall,
): number {
  if (state.activeBlock === kind) {
    return blockIndex(state, kind);
  }
  closeActiveBlock(controller, state);
  const index = state.nextBlockIndex;
  state.nextBlockIndex += 1;
  state.activeBlock = kind;
  const contentBlock =
    kind === "thinking"
      ? { type: "thinking", thinking: "", signature: "" }
      : kind === "tool_use"
        ? {
            type: "tool_use",
            id: toolCall?.id ?? `toolu_${state.toolUseCount + 1}`,
            name: toolCall?.function?.name ?? "",
            input: {},
          }
        : { type: "text", text: "" };
  enqueueEvent(
    controller,
    state,
    "content_block_start",
    { type: "content_block_start", index, content_block: contentBlock },
    "protocol",
  );
  return index;
}

function emitThinkingDelta(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: AnthropicStreamState,
  text: string,
): void {
  if (text === "") {
    return;
  }
  const index = ensureBlock(controller, state, "thinking");
  enqueueEvent(
    controller,
    state,
    "content_block_delta",
    { type: "content_block_delta", index, delta: { type: "thinking_delta", thinking: text } },
    "output",
  );
}

function emitTextDelta(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: AnthropicStreamState,
  text: string,
): void {
  if (text === "") {
    return;
  }
  const index = ensureBlock(controller, state, "text");
  enqueueEvent(
    controller,
    state,
    "content_block_delta",
    { type: "content_block_delta", index, delta: { type: "text_delta", text } },
    "output",
  );
}

function emitToolUse(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: AnthropicStreamState,
  toolCall: OpenAIChatCompletionStreamToolCall,
): void {
  const index = ensureBlock(controller, state, "tool_use", toolCall);
  const partialJson = toolCall.function?.arguments ?? "";
  if (partialJson !== "") {
    enqueueEvent(
      controller,
      state,
      "content_block_delta",
      {
        type: "content_block_delta",
        index,
        delta: { type: "input_json_delta", partial_json: partialJson },
      },
      "output",
    );
  }
  closeActiveBlock(controller, state);
  state.toolUseCount += 1;
}

function processParsedContentDelta(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: AnthropicStreamState,
  text: string,
): void {
  const filtered = state.stopFilter.push(stripGeneratedChatControlTokens(text));
  emitTextDelta(controller, state, filtered.text);
  if (filtered.stopped) {
    state.finalFinishReason = "stop";
    state.stoppedByStopSequence = true;
  }
}

function processParsedToolDelta(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: AnthropicStreamState,
  delta: { content?: string; toolCalls?: OpenAIChatCompletionStreamToolCall[] },
): void {
  if (delta.toolCalls !== undefined) {
    for (const toolCall of delta.toolCalls) {
      emitToolUse(controller, state, toolCall);
    }
    return;
  }

  processParsedContentDelta(controller, state, delta.content ?? "");
}

function processDelta(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: AnthropicStreamState,
  delta: { content?: string; reasoningContent?: string },
): void {
  if (delta.reasoningContent !== undefined) {
    emitThinkingDelta(controller, state, delta.reasoningContent);
    return;
  }

  for (const parsed of state.toolCallParser.push(delta.content ?? "")) {
    processParsedToolDelta(controller, state, parsed);
    if (state.stoppedByStopSequence) {
      return;
    }
  }
}

function flushTail(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: AnthropicStreamState,
): void {
  for (const delta of state.reasoning.finish()) {
    processDelta(controller, state, delta);
    if (state.stoppedByStopSequence) {
      return;
    }
  }
  for (const delta of state.toolCallParser.finish()) {
    processParsedToolDelta(controller, state, delta);
    if (state.stoppedByStopSequence) {
      return;
    }
  }
  const tail = state.stopFilter.finish();
  emitTextDelta(controller, state, tail.text);
  if (tail.stopped) {
    state.finalFinishReason = "stop";
    state.stoppedByStopSequence = true;
  }
}

function handleTextEvent(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: AnthropicStreamState,
  text: string,
): boolean {
  for (const delta of state.reasoning.push(text)) {
    processDelta(controller, state, delta);
    if (state.stoppedByStopSequence) {
      return true;
    }
  }
  return false;
}

function handleStreamEvent(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: AnthropicStreamState,
  event: GenerationStreamEvent,
): boolean {
  if (event.type === "text") {
    return handleTextEvent(controller, state, event.text);
  }
  state.finalUsage = event.usage;
  state.finalFinishReason = event.finishReason;
  return state.stoppedByStopSequence;
}

function finalizeStream(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: AnthropicStreamState,
  options: StreamControlOptions,
): StreamSummary {
  state.finalFinishReason = streamWasCancelled(options.signal)
    ? "cancelled"
    : state.finalFinishReason;
  if (streamWasCancelled(options.signal)) {
    return streamSummary(state.finalFinishReason, state.finalUsage);
  }
  if (!state.stoppedByStopSequence) {
    flushTail(controller, state);
  }
  if (state.activeBlock === null && state.nextBlockIndex === 0) {
    ensureBlock(controller, state, "text");
  }
  closeActiveBlock(controller, state);
  enqueueEvent(
    controller,
    state,
    "message_delta",
    {
      type: "message_delta",
      delta: {
        stop_reason:
          state.toolUseCount > 0
            ? "tool_use"
            : anthropicStopReason(state.finalFinishReason, state.stoppedByStopSequence),
        stop_sequence: null,
      },
      usage: { output_tokens: state.finalUsage?.completionTokens ?? 0 },
    },
    "protocol",
  );
  enqueueEvent(controller, state, "message_stop", { type: "message_stop" }, "protocol");
  return streamSummary(state.finalFinishReason, state.finalUsage);
}

/** Write Anthropic Messages SSE events for a generation stream. */
export async function writeAnthropicMessageStreamEvents(
  controller: ReadableStreamDefaultController<Uint8Array>,
  stream: AsyncIterable<GenerationStreamEvent> | AsyncIterator<GenerationStreamEvent>,
  message: NormalizedAnthropicMessage,
  options: StreamControlOptions,
): Promise<StreamSummary> {
  const state: AnthropicStreamState = {
    reasoning: createOpenAIChatCompletionReasoningStream(),
    toolCallParser: createOpenAIChatCompletionToolCallStream(hasStreamingToolOutput(message)),
    stopFilter: createStopSequenceFilter(message.request.sampling.stop),
    finalUsage: undefined,
    finalFinishReason: "stop",
    stoppedByStopSequence: false,
    activeBlock: null,
    nextBlockIndex: 0,
    toolUseCount: 0,
  };
  if (options.observer !== undefined) {
    state.observer = options.observer;
  }
  enqueueEvent(
    controller,
    state,
    "message_start",
    {
      type: "message_start",
      message: {
        id: options.id,
        type: "message",
        role: "assistant",
        content: [],
        model: message.model,
        stop_reason: null,
        stop_sequence: null,
        usage: usage(undefined),
      },
    },
    "protocol",
  );

  await runSseGenerationStream(controller, stream, options, (event) =>
    handleStreamEvent(controller, state, event),
  );
  return finalizeStream(controller, state, options);
}
