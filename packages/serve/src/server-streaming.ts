import {
  createOpenAIChatCompletionReasoningStream,
  createOpenAIChatCompletionToolCallStream,
  formatOpenAIChatCompletionStreamChunk,
  formatOpenAIChatCompletionUsageStreamChunk,
  type normalizeOpenAIChatCompletionRequest,
} from "./protocols/openai-chat-completions";
import {
  formatOpenAICompletionStreamChunk,
  formatOpenAICompletionUsageStreamChunk,
  type normalizeOpenAICompletionRequest,
} from "./protocols/openai-completions";
import { enqueueSseComment, withSseHeartbeat } from "./server-sse-heartbeat";
import { createStopSequenceFilter } from "./server-stop-filter";
import type { GenerationStreamEvent, GenerationUsage, NormalizedFinishReason } from "./types";

type StreamControlOptions = {
  id: string;
  created: number;
  signal?: AbortSignal;
};

function encodeSse(payload: string): Uint8Array {
  return new TextEncoder().encode(payload);
}

export function sseHeaders(): HeadersInit {
  return {
    "content-type": "text/event-stream; charset=utf-8",
    "cache-control": "no-cache",
    connection: "keep-alive",
  };
}

function enqueueSseJson(
  controller: ReadableStreamDefaultController<Uint8Array>,
  payload: unknown,
): void {
  controller.enqueue(encodeSse(`data: ${JSON.stringify(payload)}\n\n`));
}

const yieldToHttpWriter = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

type CompletionStreamState = {
  stopFilter: ReturnType<typeof createStopSequenceFilter>;
  finalUsage: GenerationUsage | undefined;
  finalFinishReason: NormalizedFinishReason;
  sentTerminalChunk: boolean;
};

function handleCompletionTextStreamEvent(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: CompletionStreamState,
  batch: ReturnType<typeof normalizeOpenAICompletionRequest>,
  request: ReturnType<typeof normalizeOpenAICompletionRequest>["requests"][number],
  options: { id: string; created: number },
  text: string,
): boolean {
  const filtered = state.stopFilter.push(text);
  if (filtered.text !== "") {
    enqueueSseJson(
      controller,
      formatOpenAICompletionStreamChunk(request, filtered.text, {
        ...options,
        includeUsage: batch.streamOptions.includeUsage,
      }),
    );
  }
  if (filtered.stopped) {
    state.finalFinishReason = "stop";
    return true;
  }
  return false;
}

function handleCompletionDoneStreamEvent(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: CompletionStreamState,
  batch: ReturnType<typeof normalizeOpenAICompletionRequest>,
  request: ReturnType<typeof normalizeOpenAICompletionRequest>["requests"][number],
  options: { id: string; created: number },
  event: Extract<GenerationStreamEvent, { type: "done" }>,
): void {
  flushCompletionTail(controller, state, batch, request, options);
  state.finalUsage = event.usage;
  state.finalFinishReason = event.finishReason;
  enqueueSseJson(
    controller,
    formatOpenAICompletionStreamChunk(request, "", {
      ...options,
      finishReason: event.finishReason,
      includeUsage: batch.streamOptions.includeUsage,
    }),
  );
  state.sentTerminalChunk = true;
}

function flushCompletionTail(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: CompletionStreamState,
  batch: ReturnType<typeof normalizeOpenAICompletionRequest>,
  request: ReturnType<typeof normalizeOpenAICompletionRequest>["requests"][number],
  options: { id: string; created: number },
): void {
  const tail = state.stopFilter.finish();
  if (tail !== "") {
    enqueueSseJson(
      controller,
      formatOpenAICompletionStreamChunk(request, tail, {
        ...options,
        includeUsage: batch.streamOptions.includeUsage,
      }),
    );
  }
}

function emitCompletionTerminalChunk(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: CompletionStreamState,
  batch: ReturnType<typeof normalizeOpenAICompletionRequest>,
  request: ReturnType<typeof normalizeOpenAICompletionRequest>["requests"][number],
  options: { id: string; created: number },
): void {
  if (state.sentTerminalChunk) {
    return;
  }
  enqueueSseJson(
    controller,
    formatOpenAICompletionStreamChunk(request, "", {
      ...options,
      finishReason: state.finalFinishReason,
      includeUsage: batch.streamOptions.includeUsage,
    }),
  );
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
  options: Parameters<typeof formatOpenAIChatCompletionStreamChunk>[2],
): void {
  enqueueSseJson(controller, formatOpenAIChatCompletionStreamChunk(chat, delta, options));
}

function processChatDelta(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: ChatStreamState,
  chat: ReturnType<typeof normalizeOpenAIChatCompletionRequest>,
  options: { id: string; created: number },
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

function processParsedChatDelta(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: ChatStreamState,
  chat: ReturnType<typeof normalizeOpenAIChatCompletionRequest>,
  options: { id: string; created: number },
  delta: Parameters<typeof formatOpenAIChatCompletionStreamChunk>[1],
): void {
  if (delta.reasoningContent !== undefined || delta.toolCalls !== undefined) {
    if (delta.toolCalls !== undefined) {
      state.emittedToolCall = true;
    }
    emitChatStreamChunk(controller, chat, delta, options);
    return;
  }

  const filtered = state.stopFilter.push(delta.content ?? "");
  if (filtered.text !== "") {
    emitChatStreamChunk(controller, chat, { content: filtered.text }, options);
  }
  if (filtered.stopped) {
    state.finalFinishReason = "stop";
    state.stoppedByStopSequence = true;
  }
}

function handleChatTextStreamEvent(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: ChatStreamState,
  chat: ReturnType<typeof normalizeOpenAIChatCompletionRequest>,
  options: { id: string; created: number },
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

function handleChatDoneStreamEvent(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: ChatStreamState,
  chat: ReturnType<typeof normalizeOpenAIChatCompletionRequest>,
  options: { id: string; created: number },
  event: Extract<GenerationStreamEvent, { type: "done" }>,
): void {
  flushChatTail(controller, state, chat, options);
  state.finalUsage = event.usage;
  state.finalFinishReason = event.finishReason;
  emitChatTerminalChunk(controller, state, chat, options);
  state.sentTerminalChunk = true;
}

function flushChatTail(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: ChatStreamState,
  chat: ReturnType<typeof normalizeOpenAIChatCompletionRequest>,
  options: { id: string; created: number },
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
  if (tail !== "") {
    emitChatStreamChunk(controller, chat, { content: tail }, options);
  }
}

function emitChatTerminalChunk(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: ChatStreamState,
  chat: ReturnType<typeof normalizeOpenAIChatCompletionRequest>,
  options: { id: string; created: number },
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
  );
}

function hasStreamingToolOutput(chat: ReturnType<typeof normalizeOpenAIChatCompletionRequest>) {
  return (
    chat.request.input.kind === "messages" &&
    chat.request.input.tools !== undefined &&
    chat.request.input.tools.length > 0
  );
}

function toAsyncIterator(
  stream: AsyncIterable<GenerationStreamEvent> | AsyncIterator<GenerationStreamEvent>,
): AsyncIterator<GenerationStreamEvent> {
  return Symbol.asyncIterator in stream ? stream[Symbol.asyncIterator]() : stream;
}

function streamWasCancelled(signal: AbortSignal | undefined): boolean {
  return signal?.aborted ?? false;
}

function markCancelled(
  finishReason: NormalizedFinishReason,
  signal: AbortSignal | undefined,
): NormalizedFinishReason {
  return streamWasCancelled(signal) ? "cancelled" : finishReason;
}

export async function closeStreamEvents(
  stream: AsyncIterable<GenerationStreamEvent> | AsyncIterator<GenerationStreamEvent>,
): Promise<void> {
  const iterator = toAsyncIterator(stream);
  await iterator.return?.();
}

type StreamReadResult =
  | { type: "finished" }
  | { type: "cancelled" }
  | { type: "event"; event: GenerationStreamEvent };

async function readStreamEvent(
  iterator: AsyncIterator<GenerationStreamEvent>,
  signal: AbortSignal | undefined,
): Promise<StreamReadResult> {
  if (streamWasCancelled(signal)) {
    await iterator.return?.();
    return { type: "cancelled" };
  }
  const next = await iterator.next();
  if (next.done) {
    return { type: "finished" };
  }
  if (streamWasCancelled(signal)) {
    await iterator.return?.();
    return { type: "cancelled" };
  }
  return { type: "event", event: next.value };
}

function streamSummary(
  finishReason: NormalizedFinishReason,
  usage: GenerationUsage | undefined,
): { finishReason: NormalizedFinishReason; usage?: GenerationUsage } {
  return usage === undefined ? { finishReason } : { finishReason, usage };
}

function handleCompletionStreamEvent(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: CompletionStreamState,
  batch: ReturnType<typeof normalizeOpenAICompletionRequest>,
  request: ReturnType<typeof normalizeOpenAICompletionRequest>["requests"][number],
  options: { id: string; created: number },
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
): { finishReason: NormalizedFinishReason; usage?: GenerationUsage } {
  state.finalFinishReason = markCancelled(state.finalFinishReason, options.signal);
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
    );
  }
  controller.enqueue(encodeSse("data: [DONE]\n\n"));
  return streamSummary(state.finalFinishReason, state.finalUsage);
}

function handleChatStreamEvent(
  controller: ReadableStreamDefaultController<Uint8Array>,
  state: ChatStreamState,
  chat: ReturnType<typeof normalizeOpenAIChatCompletionRequest>,
  options: { id: string; created: number },
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
): { finishReason: NormalizedFinishReason; usage?: GenerationUsage } {
  state.finalFinishReason = markCancelled(state.finalFinishReason, options.signal);
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
    );
  }
  controller.enqueue(encodeSse("data: [DONE]\n\n"));
  return streamSummary(state.finalFinishReason, state.finalUsage);
}

export async function writeStreamEvents(
  controller: ReadableStreamDefaultController<Uint8Array>,
  stream: AsyncIterable<GenerationStreamEvent> | AsyncIterator<GenerationStreamEvent>,
  batch: ReturnType<typeof normalizeOpenAICompletionRequest>,
  request: ReturnType<typeof normalizeOpenAICompletionRequest>["requests"][number],
  options: StreamControlOptions,
): Promise<{ finishReason: NormalizedFinishReason; usage?: GenerationUsage }> {
  const state: CompletionStreamState = {
    stopFilter: createStopSequenceFilter(request.sampling.stop),
    finalUsage: undefined,
    finalFinishReason: "stop",
    sentTerminalChunk: false,
  };
  const iterator = toAsyncIterator(stream);
  enqueueSseComment(controller, "mlxts-serve stream started");
  await yieldToHttpWriter();
  while (true) {
    const next = await withSseHeartbeat(controller, () =>
      readStreamEvent(iterator, options.signal),
    );
    if (next.type === "finished") {
      break;
    }
    if (next.type === "cancelled") {
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
): Promise<{ finishReason: NormalizedFinishReason; usage?: GenerationUsage }> {
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
  emitChatStreamChunk(controller, chat, {}, { ...options, includeRole: true });

  const iterator = toAsyncIterator(stream);
  while (true) {
    const next = await withSseHeartbeat(controller, () =>
      readStreamEvent(iterator, options.signal),
    );
    if (next.type === "finished") {
      break;
    }
    if (next.type === "cancelled") {
      break;
    }
    const shouldStop = handleChatStreamEvent(controller, state, chat, options, next.event);
    await yieldToHttpWriter();
    if (shouldStop) {
      break;
    }
  }
  return finalizeChatStream(controller, state, chat, options);
}
