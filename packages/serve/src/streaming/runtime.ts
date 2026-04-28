/**
 * Shared runtime helpers for server-sent event streams.
 * @module
 */

import type { GenerationStreamEvent, GenerationUsage, NormalizedFinishReason } from "../types";

const encoder = new TextEncoder();

export type StreamObserverChunkKind = "protocol" | "output";

export type StreamObserver = {
  observeChunk(bytes: number, kind: StreamObserverChunkKind): void;
};

type StreamObserverOptions = {
  observer?: StreamObserver | undefined;
  abort?: (() => void) | undefined;
};

export type StreamControlOptions = {
  id: string;
  created: number;
  signal?: AbortSignal;
  abort?: () => void;
  observer?: StreamObserver;
};

export type StreamSummary = {
  finishReason: NormalizedFinishReason;
  usage?: GenerationUsage;
};

export type StreamReadResult =
  | { type: "finished" }
  | { type: "cancelled" }
  | { type: "event"; event: GenerationStreamEvent };

export function encodeSse(payload: string): Uint8Array {
  return encoder.encode(payload);
}

export function enqueueObservedSse(
  controller: ReadableStreamDefaultController<Uint8Array>,
  payload: string,
  options: StreamObserverOptions | undefined,
  kind: StreamObserverChunkKind,
): void {
  const bytes = encodeSse(payload);
  try {
    controller.enqueue(bytes);
  } catch (error) {
    options?.abort?.();
    throw error;
  }
  options?.observer?.observeChunk(bytes.byteLength, kind);
}

export function enqueueSseJson(
  controller: ReadableStreamDefaultController<Uint8Array>,
  payload: unknown,
  options: StreamObserverOptions | undefined,
  kind: StreamObserverChunkKind,
): void {
  enqueueObservedSse(controller, `data: ${JSON.stringify(payload)}\n\n`, options, kind);
}

export async function yieldToHttpWriter(): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

export function toAsyncIterator(
  stream: AsyncIterable<GenerationStreamEvent> | AsyncIterator<GenerationStreamEvent>,
): AsyncIterator<GenerationStreamEvent> {
  return Symbol.asyncIterator in stream ? stream[Symbol.asyncIterator]() : stream;
}

export function streamWasCancelled(signal: AbortSignal | undefined): boolean {
  return signal?.aborted ?? false;
}

export async function closeStreamEvents(
  stream: AsyncIterable<GenerationStreamEvent> | AsyncIterator<GenerationStreamEvent>,
): Promise<void> {
  const iterator = toAsyncIterator(stream);
  await iterator.return?.();
}

export async function readStreamEvent(
  iterator: AsyncIterator<GenerationStreamEvent>,
  signal: AbortSignal | undefined,
): Promise<StreamReadResult> {
  if (streamWasCancelled(signal)) {
    await iterator.return?.();
    return { type: "cancelled" };
  }
  if (signal === undefined) {
    const next = await iterator.next();
    if (next.done) {
      return { type: "finished" };
    }
    return { type: "event", event: next.value };
  }

  let onAbort: (() => void) | undefined;
  const cancelled = new Promise<"cancelled">((resolve) => {
    onAbort = () => resolve("cancelled");
    signal.addEventListener("abort", onAbort, { once: true });
    if (signal.aborted) {
      onAbort();
    }
  });
  const nextPromise = iterator.next();
  const raced = await Promise.race([nextPromise, cancelled]);
  if (onAbort !== undefined) {
    signal.removeEventListener("abort", onAbort);
  }
  if (raced === "cancelled") {
    void nextPromise.catch(() => undefined);
    await iterator.return?.();
    return { type: "cancelled" };
  }
  const next = raced;
  if (next.done) {
    return { type: "finished" };
  }
  if (streamWasCancelled(signal)) {
    await iterator.return?.();
    return { type: "cancelled" };
  }
  return { type: "event", event: next.value };
}

export function streamSummary(
  finishReason: NormalizedFinishReason,
  usage: GenerationUsage | undefined,
): StreamSummary {
  return usage === undefined ? { finishReason } : { finishReason, usage };
}
