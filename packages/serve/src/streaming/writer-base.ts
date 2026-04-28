/**
 * Shared SSE stream scaffolding.
 * @module
 */

import type { GenerationStreamEvent } from "../types";
import { withSseHeartbeat } from "./heartbeat";
import {
  readStreamEvent,
  type StreamControlOptions,
  toAsyncIterator,
  yieldToHttpWriter,
} from "./runtime";

export function sseHeaders(): HeadersInit {
  return {
    "content-type": "text/event-stream; charset=utf-8",
    "cache-control": "no-cache",
    connection: "keep-alive",
  };
}

export async function runSseGenerationStream(
  controller: ReadableStreamDefaultController<Uint8Array>,
  stream: AsyncIterable<GenerationStreamEvent> | AsyncIterator<GenerationStreamEvent>,
  options: StreamControlOptions,
  handleEvent: (event: GenerationStreamEvent) => boolean,
): Promise<void> {
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
    const shouldStop = handleEvent(next.event);
    await yieldToHttpWriter();
    if (shouldStop) {
      await iterator.return?.();
      break;
    }
  }
}
