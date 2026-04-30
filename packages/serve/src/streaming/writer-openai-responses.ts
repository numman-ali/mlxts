/**
 * OpenAI Responses SSE streaming writer.
 * @module
 */

import type { OpenAIChatCompletionStreamToolCall } from "../protocols/openai-chat-completion-streaming";
import {
  createOpenAIChatCompletionReasoningStream,
  createOpenAIChatCompletionToolCallStream,
  stripGeneratedChatControlTokens,
} from "../protocols/openai-chat-completions";
import {
  formatOpenAIResponse,
  formatOpenAIResponsePending,
  type NormalizedOpenAIResponse,
} from "../protocols/openai-responses";
import type {
  GenerationStreamEvent,
  GenerationUsage,
  NormalizedFinishReason,
  NormalizedGenerationResult,
} from "../types";
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
import {
  emitOpenAIResponseFunctionCall,
  finalOpenAIResponseOutputItems,
  hasOpenAIResponseStreamingToolOutput,
  type OpenAIResponseFunctionCallState,
  type OpenAIResponseOutputState,
} from "./writer-openai-responses-tools";

type ResponseStreamState = {
  reasoningParser: ReturnType<typeof createOpenAIChatCompletionReasoningStream>;
  toolCallParser: ReturnType<typeof createOpenAIChatCompletionToolCallStream>;
  stopFilter: ReturnType<typeof createStopSequenceFilter>;
  sequenceNumber: number;
  outputCount: number;
  visibleText: string;
  reasoningText: string;
  functionCalls: OpenAIResponseFunctionCallState[];
  finalUsage: GenerationUsage | undefined;
  finalFinishReason: NormalizedFinishReason;
  stoppedByStopSequence: boolean;
  messageItem?: OpenAIResponseOutputState;
  reasoningItem?: OpenAIResponseOutputState;
  observer?: StreamObserver;
};

function observerOptions(state: ResponseStreamState): { observer: StreamObserver } | undefined {
  return state.observer === undefined ? undefined : { observer: state.observer };
}

function enqueueSseEvent(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: ResponseStreamState,
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

function emitResponseEvent(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: ResponseStreamState,
  event: string,
  payload: Record<string, unknown>,
  kind: StreamObserverChunkKind = "protocol",
): void {
  const data = { type: event, ...payload, sequence_number: state.sequenceNumber };
  state.sequenceNumber += 1;
  enqueueSseEvent(controller, state, event, data, kind);
}

function ensureReasoningItem(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: ResponseStreamState,
  options: { id: string },
): OpenAIResponseOutputState {
  if (state.reasoningItem !== undefined) {
    return state.reasoningItem;
  }
  const item = { id: `${options.id}-rsn`, outputIndex: state.outputCount };
  state.outputCount += 1;
  state.reasoningItem = item;
  emitResponseEvent(controller, state, "response.output_item.added", {
    output_index: item.outputIndex,
    item: {
      id: item.id,
      type: "reasoning",
      status: "in_progress",
      summary: [],
      content: [],
    },
  });
  return item;
}

function ensureMessageItem(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: ResponseStreamState,
  options: { id: string },
): OpenAIResponseOutputState {
  if (state.messageItem !== undefined) {
    return state.messageItem;
  }
  const item = { id: `${options.id}-msg`, outputIndex: state.outputCount };
  state.outputCount += 1;
  state.messageItem = item;
  emitResponseEvent(controller, state, "response.output_item.added", {
    output_index: item.outputIndex,
    item: {
      id: item.id,
      type: "message",
      status: "in_progress",
      role: "assistant",
      content: [],
    },
  });
  emitResponseEvent(controller, state, "response.content_part.added", {
    item_id: item.id,
    output_index: item.outputIndex,
    content_index: 0,
    part: { type: "output_text", text: "", annotations: [] },
  });
  return item;
}

function emitReasoningDelta(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: ResponseStreamState,
  options: { id: string },
  text: string,
): void {
  if (text === "") {
    return;
  }
  const item = ensureReasoningItem(controller, state, options);
  state.reasoningText += text;
  emitResponseEvent(
    controller,
    state,
    "response.reasoning_text.delta",
    {
      item_id: item.id,
      output_index: item.outputIndex,
      content_index: 0,
      delta: text,
    },
    "output",
  );
}

function emitTextDelta(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: ResponseStreamState,
  options: { id: string },
  text: string,
): void {
  if (text === "") {
    return;
  }
  const item = ensureMessageItem(controller, state, options);
  state.visibleText += text;
  emitResponseEvent(
    controller,
    state,
    "response.output_text.delta",
    {
      item_id: item.id,
      output_index: item.outputIndex,
      content_index: 0,
      delta: text,
    },
    "output",
  );
}

function processParsedContentDelta(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: ResponseStreamState,
  options: { id: string },
  text: string,
): void {
  const filtered = state.stopFilter.push(stripGeneratedChatControlTokens(text));
  emitTextDelta(controller, state, options, filtered.text);
  if (filtered.stopped) {
    state.finalFinishReason = "stop";
    state.stoppedByStopSequence = true;
  }
}

function processParsedToolDelta(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: ResponseStreamState,
  options: { id: string },
  delta: { content?: string; toolCalls?: OpenAIChatCompletionStreamToolCall[] },
): void {
  if (delta.toolCalls !== undefined) {
    for (const toolCall of delta.toolCalls) {
      emitOpenAIResponseFunctionCall(
        state,
        (event, payload, kind) =>
          emitResponseEvent(controller, state, event, payload, kind ?? "protocol"),
        options,
        toolCall,
      );
    }
    return;
  }
  processParsedContentDelta(controller, state, options, delta.content ?? "");
}

function processDelta(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: ResponseStreamState,
  options: { id: string },
  delta: { content?: string; reasoningContent?: string },
): void {
  if (delta.reasoningContent !== undefined) {
    emitReasoningDelta(controller, state, options, delta.reasoningContent);
    return;
  }

  for (const parsed of state.toolCallParser.push(delta.content ?? "")) {
    processParsedToolDelta(controller, state, options, parsed);
    if (state.stoppedByStopSequence) {
      return;
    }
  }
}

function handleTextEvent(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: ResponseStreamState,
  options: { id: string },
  text: string,
): boolean {
  for (const delta of state.reasoningParser.push(text)) {
    processDelta(controller, state, options, delta);
    if (state.stoppedByStopSequence) {
      return true;
    }
  }
  return false;
}

function flushTail(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: ResponseStreamState,
  options: { id: string },
): void {
  for (const delta of state.reasoningParser.finish()) {
    processDelta(controller, state, options, delta);
    if (state.stoppedByStopSequence) {
      return;
    }
  }

  for (const delta of state.toolCallParser.finish()) {
    processParsedToolDelta(controller, state, options, delta);
    if (state.stoppedByStopSequence) {
      return;
    }
  }

  const tail = state.stopFilter.finish();
  emitTextDelta(controller, state, options, tail.text);
  if (tail.stopped) {
    state.finalFinishReason = "stop";
    state.stoppedByStopSequence = true;
  }
}

function emitReasoningDone(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: ResponseStreamState,
): void {
  const item = state.reasoningItem;
  if (item === undefined) {
    return;
  }
  emitResponseEvent(controller, state, "response.reasoning_text.done", {
    item_id: item.id,
    output_index: item.outputIndex,
    content_index: 0,
    text: state.reasoningText,
  });
  emitResponseEvent(controller, state, "response.output_item.done", {
    output_index: item.outputIndex,
    item: {
      id: item.id,
      type: "reasoning",
      status: "completed",
      summary: [],
      content: [{ type: "reasoning_text", text: state.reasoningText }],
    },
  });
}

function emitMessageDone(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: ResponseStreamState,
  options: { id: string },
): void {
  const item = ensureMessageItem(controller, state, options);
  const part = { type: "output_text", text: state.visibleText, annotations: [] };
  emitResponseEvent(controller, state, "response.output_text.done", {
    item_id: item.id,
    output_index: item.outputIndex,
    content_index: 0,
    text: state.visibleText,
  });
  emitResponseEvent(controller, state, "response.content_part.done", {
    item_id: item.id,
    output_index: item.outputIndex,
    content_index: 0,
    part,
  });
  emitResponseEvent(controller, state, "response.output_item.done", {
    output_index: item.outputIndex,
    item: {
      id: item.id,
      type: "message",
      status: "completed",
      role: "assistant",
      content: [part],
    },
  });
}

function shouldEmitMessageDone(state: ResponseStreamState): boolean {
  return state.visibleText !== "" || state.functionCalls.length === 0;
}

function markCancelled(
  finishReason: NormalizedFinishReason,
  signal: AbortSignal | undefined,
): NormalizedFinishReason {
  return streamWasCancelled(signal) ? "cancelled" : finishReason;
}

function finalResult(state: ResponseStreamState): NormalizedGenerationResult {
  return {
    text: state.visibleText,
    finishReason: state.finalFinishReason,
    ...(state.reasoningText.trim() === "" ? {} : { reasoningContent: state.reasoningText }),
    ...(state.finalUsage === undefined ? {} : { usage: state.finalUsage }),
  };
}

function terminalEventType(
  finishReason: NormalizedFinishReason,
): "response.completed" | "response.incomplete" {
  return finishReason === "length" ? "response.incomplete" : "response.completed";
}

function finalResponseObject(
  state: ResponseStreamState,
  response: NormalizedOpenAIResponse,
  options: { id: string; created: number },
) {
  const formatted = formatOpenAIResponse(response, finalResult(state), options);
  return {
    ...formatted,
    output: finalOpenAIResponseOutputItems(state),
    output_text: state.visibleText,
  };
}

function finalizeResponseStream(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: ResponseStreamState,
  response: NormalizedOpenAIResponse,
  options: StreamControlOptions,
): StreamSummary {
  state.finalFinishReason = markCancelled(state.finalFinishReason, options.signal);
  if (streamWasCancelled(options.signal)) {
    return streamSummary(state.finalFinishReason, state.finalUsage);
  }
  if (!state.stoppedByStopSequence) {
    flushTail(controller, state, options);
  }
  emitReasoningDone(controller, state);
  if (shouldEmitMessageDone(state)) {
    emitMessageDone(controller, state, options);
  }
  const terminalType = terminalEventType(state.finalFinishReason);
  emitResponseEvent(controller, state, terminalType, {
    response: finalResponseObject(state, response, options),
  });
  enqueueObservedSse(controller, "data: [DONE]\n\n", observerOptions(state), "protocol");
  return streamSummary(state.finalFinishReason, state.finalUsage);
}

function handleStreamEvent(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: ResponseStreamState,
  options: { id: string },
  event: GenerationStreamEvent,
): boolean {
  if (event.type === "text") {
    return handleTextEvent(controller, state, options, event.text);
  }

  state.finalUsage = event.usage;
  state.finalFinishReason = event.finishReason;
  return state.stoppedByStopSequence;
}

/** Write semantic OpenAI Responses SSE events for a generation stream. */
export async function writeOpenAIResponseStreamEvents(
  controller: ReadableStreamDefaultController<Uint8Array>,
  stream: AsyncIterable<GenerationStreamEvent> | AsyncIterator<GenerationStreamEvent>,
  response: NormalizedOpenAIResponse,
  options: StreamControlOptions,
): Promise<StreamSummary> {
  const state: ResponseStreamState = {
    reasoningParser: createOpenAIChatCompletionReasoningStream(),
    toolCallParser: createOpenAIChatCompletionToolCallStream(
      hasOpenAIResponseStreamingToolOutput(response),
    ),
    stopFilter: createStopSequenceFilter(response.request.sampling.stop),
    sequenceNumber: 0,
    outputCount: 0,
    visibleText: "",
    reasoningText: "",
    functionCalls: [],
    finalUsage: undefined,
    finalFinishReason: "stop",
    stoppedByStopSequence: false,
  };
  if (options.observer !== undefined) {
    state.observer = options.observer;
  }
  const pending = formatOpenAIResponsePending(response, options);
  emitResponseEvent(controller, state, "response.created", { response: pending });
  emitResponseEvent(controller, state, "response.in_progress", { response: pending });

  await runSseGenerationStream(controller, stream, options, (event) =>
    handleStreamEvent(controller, state, options, event),
  );

  return finalizeResponseStream(controller, state, response, options);
}
