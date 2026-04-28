import {
  createOpenAIChatCompletionReasoningStream,
  createOpenAIChatCompletionToolCallStream,
  formatOpenAIChatCompletionStreamChunk,
  formatOpenAIChatCompletionUsageStreamChunk,
  type normalizeOpenAIChatCompletionRequest,
  stripGeneratedChatControlTokens,
} from "../protocols/openai-chat-completions";
import {
  formatOpenAICompletionStreamChunk,
  formatOpenAICompletionUsageStreamChunk,
  type normalizeOpenAICompletionRequest,
} from "../protocols/openai-completions";
import type { GenerationStreamEvent, GenerationUsage, NormalizedFinishReason } from "../types";
import { withSseHeartbeat } from "./heartbeat";
import {
  enqueueObservedSse,
  enqueueSseJson,
  readStreamEvent,
  type StreamControlOptions,
  type StreamObserverChunkKind,
  type StreamSummary,
  streamSummary,
  streamWasCancelled,
  toAsyncIterator,
  yieldToHttpWriter,
} from "./runtime";
import { createStopSequenceFilter } from "./stop-filter";

export { closeStreamEvents } from "./runtime";

type ChatChunkOptions = Parameters<typeof formatOpenAIChatCompletionStreamChunk>[2] & {
  observer?: StreamControlOptions["observer"];
};

export function sseHeaders(): HeadersInit {
  return {
    "content-type": "text/event-stream; charset=utf-8",
    "cache-control": "no-cache",
    connection: "keep-alive",
  };
}

type CompletionStreamState = {
  stopFilter: ReturnType<typeof createStopSequenceFilter>;
  finalUsage: GenerationUsage | undefined;
  finalFinishReason: NormalizedFinishReason;
  sentTerminalChunk: boolean;
};

function enqueueCompletionStreamChunk(
  controller: ReadableStreamDefaultController<Uint8Array>,
  batch: ReturnType<typeof normalizeOpenAICompletionRequest>,
  request: ReturnType<typeof normalizeOpenAICompletionRequest>["requests"][number],
  options: StreamControlOptions & { finishReason?: NormalizedFinishReason },
  text: string,
  kind: StreamObserverChunkKind,
): void {
  enqueueSseJson(
    controller,
    formatOpenAICompletionStreamChunk(request, text, {
      ...options,
      includeUsage: batch.streamOptions.includeUsage,
    }),
    options,
    kind,
  );
}

function handleCompletionTextStreamEvent(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: CompletionStreamState,
  batch: ReturnType<typeof normalizeOpenAICompletionRequest>,
  request: ReturnType<typeof normalizeOpenAICompletionRequest>["requests"][number],
  options: StreamControlOptions,
  text: string,
): boolean {
  const filtered = state.stopFilter.push(text);
  if (filtered.text !== "") {
    enqueueCompletionStreamChunk(controller, batch, request, options, filtered.text, "output");
  }
  if (filtered.stopped) {
    state.finalFinishReason = "stop";
    return true;
  }
  return false;
}

function flushCompletionTail(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: CompletionStreamState,
  batch: ReturnType<typeof normalizeOpenAICompletionRequest>,
  request: ReturnType<typeof normalizeOpenAICompletionRequest>["requests"][number],
  options: StreamControlOptions,
): void {
  const tail = state.stopFilter.finish();
  if (tail.text !== "") {
    enqueueCompletionStreamChunk(controller, batch, request, options, tail.text, "output");
  }
  if (tail.stopped) {
    state.finalFinishReason = "stop";
  }
}

function emitCompletionTerminalChunk(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: CompletionStreamState,
  batch: ReturnType<typeof normalizeOpenAICompletionRequest>,
  request: ReturnType<typeof normalizeOpenAICompletionRequest>["requests"][number],
  options: StreamControlOptions,
): void {
  if (state.sentTerminalChunk) {
    return;
  }
  enqueueCompletionStreamChunk(
    controller,
    batch,
    request,
    { ...options, finishReason: state.finalFinishReason },
    "",
    "protocol",
  );
}

function handleCompletionDoneStreamEvent(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: CompletionStreamState,
  batch: ReturnType<typeof normalizeOpenAICompletionRequest>,
  request: ReturnType<typeof normalizeOpenAICompletionRequest>["requests"][number],
  options: StreamControlOptions,
  event: Extract<GenerationStreamEvent, { type: "done" }>,
): void {
  flushCompletionTail(controller, state, batch, request, options);
  state.finalUsage = event.usage;
  state.finalFinishReason = state.finalFinishReason === "stop" ? "stop" : event.finishReason;
  emitCompletionTerminalChunk(controller, state, batch, request, options);
  state.sentTerminalChunk = true;
}

type ChatStreamState = {
  reasoning: ReturnType<typeof createOpenAIChatCompletionReasoningStream>;
  toolCalls: ReturnType<typeof createOpenAIChatCompletionToolCallStream>;
  stopFilter: ReturnType<typeof createStopSequenceFilter>;
  finalUsage: GenerationUsage | undefined;
  finalFinishReason: NormalizedFinishReason;
  sentTerminalChunk: boolean;
  stoppedByStopSequence: boolean;
  emittedToolCall: boolean;
};

function emitChatStreamChunk(
  controller: ReadableStreamDefaultController<Uint8Array>,
  chat: ReturnType<typeof normalizeOpenAIChatCompletionRequest>,
  delta: Parameters<typeof formatOpenAIChatCompletionStreamChunk>[1],
  options: ChatChunkOptions,
  kind: StreamObserverChunkKind = "output",
): void {
  enqueueSseJson(
    controller,
    formatOpenAIChatCompletionStreamChunk(chat, delta, options),
    options,
    kind,
  );
}

function processParsedChatDelta(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: ChatStreamState,
  chat: ReturnType<typeof normalizeOpenAIChatCompletionRequest>,
  options: StreamControlOptions,
  delta: Parameters<typeof formatOpenAIChatCompletionStreamChunk>[1],
): void {
  if (delta.reasoningContent !== undefined || delta.toolCalls !== undefined) {
    if (delta.toolCalls !== undefined) {
      state.emittedToolCall = true;
    }
    emitChatStreamChunk(controller, chat, delta, options);
    return;
  }

  const filtered = state.stopFilter.push(stripGeneratedChatControlTokens(delta.content ?? ""));
  if (filtered.text !== "") {
    emitChatStreamChunk(controller, chat, { content: filtered.text }, options);
  }
  if (filtered.stopped) {
    state.finalFinishReason = "stop";
    state.stoppedByStopSequence = true;
  }
}

function processChatDelta(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: ChatStreamState,
  chat: ReturnType<typeof normalizeOpenAIChatCompletionRequest>,
  options: StreamControlOptions,
  delta: { content?: string; reasoningContent?: string },
): void {
  if (delta.reasoningContent !== undefined) {
    emitChatStreamChunk(controller, chat, delta, options);
    return;
  }

  for (const parsed of state.toolCalls.push(delta.content ?? "")) {
    processParsedChatDelta(controller, state, chat, options, parsed);
    if (state.stoppedByStopSequence) {
      return;
    }
  }
}

function handleChatTextStreamEvent(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: ChatStreamState,
  chat: ReturnType<typeof normalizeOpenAIChatCompletionRequest>,
  options: StreamControlOptions,
  text: string,
): boolean {
  for (const delta of state.reasoning.push(text)) {
    processChatDelta(controller, state, chat, options, delta);
    if (state.stoppedByStopSequence) {
      return true;
    }
  }
  return false;
}

function flushChatTail(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: ChatStreamState,
  chat: ReturnType<typeof normalizeOpenAIChatCompletionRequest>,
  options: StreamControlOptions,
): void {
  for (const delta of state.reasoning.finish()) {
    processChatDelta(controller, state, chat, options, delta);
    if (state.stoppedByStopSequence) {
      return;
    }
  }

  for (const delta of state.toolCalls.finish()) {
    processParsedChatDelta(controller, state, chat, options, delta);
    if (state.stoppedByStopSequence) {
      return;
    }
  }

  const tail = state.stopFilter.finish();
  if (tail.text !== "") {
    emitChatStreamChunk(controller, chat, { content: tail.text }, options);
  }
  if (tail.stopped) {
    state.finalFinishReason = "stop";
    state.stoppedByStopSequence = true;
  }
}

function emitChatTerminalChunk(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: ChatStreamState,
  chat: ReturnType<typeof normalizeOpenAIChatCompletionRequest>,
  options: StreamControlOptions,
): void {
  if (state.sentTerminalChunk) {
    return;
  }
  emitChatStreamChunk(
    controller,
    chat,
    {},
    {
      ...options,
      finishReason: state.emittedToolCall ? "tool_calls" : state.finalFinishReason,
    },
    "protocol",
  );
}

function handleChatDoneStreamEvent(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: ChatStreamState,
  chat: ReturnType<typeof normalizeOpenAIChatCompletionRequest>,
  options: StreamControlOptions,
  event: Extract<GenerationStreamEvent, { type: "done" }>,
): void {
  flushChatTail(controller, state, chat, options);
  state.finalUsage = event.usage;
  state.finalFinishReason = state.finalFinishReason === "stop" ? "stop" : event.finishReason;
  emitChatTerminalChunk(controller, state, chat, options);
  state.sentTerminalChunk = true;
}

function hasStreamingToolOutput(chat: ReturnType<typeof normalizeOpenAIChatCompletionRequest>) {
  return (
    chat.request.input.kind === "messages" &&
    chat.request.input.tools !== undefined &&
    chat.request.input.tools.length > 0
  );
}

function handleCompletionStreamEvent(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: CompletionStreamState,
  batch: ReturnType<typeof normalizeOpenAICompletionRequest>,
  request: ReturnType<typeof normalizeOpenAICompletionRequest>["requests"][number],
  options: StreamControlOptions,
  event: GenerationStreamEvent,
): boolean {
  if (event.type === "text") {
    return handleCompletionTextStreamEvent(controller, state, batch, request, options, event.text);
  }
  handleCompletionDoneStreamEvent(controller, state, batch, request, options, event);
  return false;
}

function finalizeCompletionStream(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: CompletionStreamState,
  batch: ReturnType<typeof normalizeOpenAICompletionRequest>,
  request: ReturnType<typeof normalizeOpenAICompletionRequest>["requests"][number],
  options: StreamControlOptions,
): StreamSummary {
  state.finalFinishReason = streamWasCancelled(options.signal)
    ? "cancelled"
    : state.finalFinishReason;
  if (streamWasCancelled(options.signal)) {
    return streamSummary(state.finalFinishReason, state.finalUsage);
  }
  if (!state.sentTerminalChunk) {
    flushCompletionTail(controller, state, batch, request, options);
  }
  emitCompletionTerminalChunk(controller, state, batch, request, options);
  if (batch.streamOptions.includeUsage) {
    enqueueSseJson(
      controller,
      formatOpenAICompletionUsageStreamChunk(batch, state.finalUsage, options),
      options,
      "protocol",
    );
  }
  enqueueObservedSse(controller, "data: [DONE]\n\n", options, "protocol");
  return streamSummary(state.finalFinishReason, state.finalUsage);
}

function handleChatStreamEvent(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: ChatStreamState,
  chat: ReturnType<typeof normalizeOpenAIChatCompletionRequest>,
  options: StreamControlOptions,
  event: GenerationStreamEvent,
): boolean {
  if (event.type === "text") {
    return handleChatTextStreamEvent(controller, state, chat, options, event.text);
  }
  handleChatDoneStreamEvent(controller, state, chat, options, event);
  return state.stoppedByStopSequence;
}

function finalizeChatStream(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: ChatStreamState,
  chat: ReturnType<typeof normalizeOpenAIChatCompletionRequest>,
  options: StreamControlOptions,
): StreamSummary {
  state.finalFinishReason = streamWasCancelled(options.signal)
    ? "cancelled"
    : state.finalFinishReason;
  if (streamWasCancelled(options.signal)) {
    return streamSummary(state.finalFinishReason, state.finalUsage);
  }
  if (!state.sentTerminalChunk && !state.stoppedByStopSequence) {
    flushChatTail(controller, state, chat, options);
  }
  emitChatTerminalChunk(controller, state, chat, options);
  if (chat.streamOptions.includeUsage) {
    enqueueSseJson(
      controller,
      formatOpenAIChatCompletionUsageStreamChunk(chat, state.finalUsage, options),
      options,
      "protocol",
    );
  }
  enqueueObservedSse(controller, "data: [DONE]\n\n", options, "protocol");
  return streamSummary(state.finalFinishReason, state.finalUsage);
}

export async function writeStreamEvents(
  controller: ReadableStreamDefaultController<Uint8Array>,
  stream: AsyncIterable<GenerationStreamEvent> | AsyncIterator<GenerationStreamEvent>,
  batch: ReturnType<typeof normalizeOpenAICompletionRequest>,
  request: ReturnType<typeof normalizeOpenAICompletionRequest>["requests"][number],
  options: StreamControlOptions,
): Promise<StreamSummary> {
  const state: CompletionStreamState = {
    stopFilter: createStopSequenceFilter(request.sampling.stop),
    finalUsage: undefined,
    finalFinishReason: "stop",
    sentTerminalChunk: false,
  };
  const iterator = toAsyncIterator(stream);
  enqueueObservedSse(controller, ": mlxts-serve stream started\n\n", options, "protocol");
  await yieldToHttpWriter();
  while (true) {
    const next = await withSseHeartbeat(
      controller,
      () => readStreamEvent(iterator, options.signal),
      options.abort,
    );
    if (next.type === "finished" || next.type === "cancelled") {
      break;
    }
    const shouldStop = handleCompletionStreamEvent(
      controller,
      state,
      batch,
      request,
      options,
      next.event,
    );
    await yieldToHttpWriter();
    if (shouldStop) {
      await iterator.return?.();
      break;
    }
  }
  return finalizeCompletionStream(controller, state, batch, request, options);
}

export async function writeChatStreamEvents(
  controller: ReadableStreamDefaultController<Uint8Array>,
  stream: AsyncIterable<GenerationStreamEvent> | AsyncIterator<GenerationStreamEvent>,
  chat: ReturnType<typeof normalizeOpenAIChatCompletionRequest>,
  options: StreamControlOptions,
): Promise<StreamSummary> {
  const state: ChatStreamState = {
    reasoning: createOpenAIChatCompletionReasoningStream(),
    toolCalls: createOpenAIChatCompletionToolCallStream(hasStreamingToolOutput(chat)),
    stopFilter: createStopSequenceFilter(chat.request.sampling.stop),
    finalUsage: undefined,
    finalFinishReason: "stop",
    sentTerminalChunk: false,
    stoppedByStopSequence: false,
    emittedToolCall: false,
  };
  emitChatStreamChunk(controller, chat, {}, { ...options, includeRole: true }, "protocol");

  const iterator = toAsyncIterator(stream);
  while (true) {
    const next = await withSseHeartbeat(
      controller,
      () => readStreamEvent(iterator, options.signal),
      options.abort,
    );
    if (next.type === "finished" || next.type === "cancelled") {
      break;
    }
    const shouldStop = handleChatStreamEvent(controller, state, chat, options, next.event);
    await yieldToHttpWriter();
    if (shouldStop) {
      await iterator.return?.();
      break;
    }
  }
  return finalizeChatStream(controller, state, chat, options);
}
