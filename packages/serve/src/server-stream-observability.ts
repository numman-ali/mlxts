/**
 * Event-backed observability for server-sent generation streams.
 * @module
 */

import { emitServeEvent } from "./server-events";
import type { StreamObserver, StreamObserverChunkKind } from "./server-stream-runtime";
import type { NormalizedFinishReason, NormalizedGenerationRequest, ServeEvent } from "./types";

type EventSink = {
  onEvent?: (event: ServeEvent) => void;
};

export type StreamTerminalResult = "completed" | "cancelled" | "error";

export type GenerationStreamObserver = StreamObserver & {
  end(result: StreamTerminalResult, finishReason: NormalizedFinishReason, durationMs: number): void;
};

/** Track server-side SSE frames and emit bounded stream telemetry events. */
export function createGenerationStreamObserver(
  options: EventSink,
  request: NormalizedGenerationRequest,
  startedAt: number,
): GenerationStreamObserver {
  let chunks = 0;
  let bytes = 0;
  let outputChunks = 0;
  let outputBytes = 0;
  let ttftMs: number | undefined;
  let ended = false;

  return {
    observeChunk(chunkBytes: number, kind: StreamObserverChunkKind) {
      chunks += 1;
      bytes += chunkBytes;
      if (kind !== "output") {
        return;
      }
      outputChunks += 1;
      outputBytes += chunkBytes;
      const elapsedMs = performance.now() - startedAt;
      ttftMs ??= elapsedMs;
      emitServeEvent(options, {
        type: "generation_stream_chunk",
        id: request.id,
        protocol: request.protocol,
        model: request.model,
        chunkIndex: outputChunks,
        elapsedMs,
        bytes: chunkBytes,
      });
    },
    end(result, finishReason, durationMs) {
      if (ended) {
        return;
      }
      ended = true;
      emitServeEvent(options, {
        type: "generation_stream_end",
        id: request.id,
        protocol: request.protocol,
        model: request.model,
        result,
        finishReason,
        chunks,
        bytes,
        outputChunks,
        outputBytes,
        ...(ttftMs === undefined ? {} : { ttftMs }),
        durationMs,
      });
    },
  };
}
