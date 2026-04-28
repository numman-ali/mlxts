/**
 * OpenAI Responses SSE streaming writer.
 * @module
 */

import { createOpenAIChatCompletionReasoningStream } from "../protocols/openai-chat-completions";
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

type OutputState = {
  id: string;
  outputIndex: number;
};

type ResponseStreamState = {
  reasoningParser: ReturnType<typeof createOpenAIChatCompletionReasoningStream>;
  stopFilter: ReturnType<typeof createStopSequenceFilter>;
  sequenceNumber: number;
  outputCount: number;
  visibleText: string;
  reasoningText: string;
  finalUsage: GenerationUsage | undefined;
  finalFinishReason: NormalizedFinishReason;
  stoppedByStopSequence: boolean;
  messageItem?: OutputState;
  reasoningItem?: OutputState;
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
): OutputState {
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
): OutputState {
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

  const filtered = state.stopFilter.push(delta.content ?? "");
  emitTextDelta(controller, state, options, filtered.text);
  if (filtered.stopped) {
    state.finalFinishReason = "stop";
    state.stoppedByStopSequence = true;
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
  emitMessageDone(controller, state, options);
  const terminalType = terminalEventType(state.finalFinishReason);
  emitResponseEvent(controller, state, terminalType, {
    response: formatOpenAIResponse(response, finalResult(state), options),
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
    stopFilter: createStopSequenceFilter(response.request.sampling.stop),
    sequenceNumber: 0,
    outputCount: 0,
    visibleText: "",
    reasoningText: "",
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
