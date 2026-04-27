/**
 * Prometheus metrics for generation streaming lifecycle events.
 * @module
 */

import { HistogramMetric, NumberMetric } from "./serve-metrics-registry";
import type { ServeEvent } from "./types";

const STREAM_DURATION_BUCKETS = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60];
const STREAM_CHUNK_BUCKETS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

function seconds(milliseconds: number): number {
  return milliseconds / 1000;
}

export class ServeStreamMetrics {
  readonly #streams = new NumberMetric({
    name: "mlxts_serve_generation_streams_total",
    help: "Generation streams that reached a terminal writer result.",
    type: "counter",
    labelNames: ["model", "protocol", "result", "finish_reason"],
  });
  readonly #streamDurations = new HistogramMetric(
    {
      name: "mlxts_serve_generation_stream_duration_seconds",
      help: "Server-side generation stream writer duration.",
      type: "histogram",
      labelNames: ["model", "protocol", "result"],
    },
    STREAM_DURATION_BUCKETS,
  );
  readonly #streamTtft = new HistogramMetric(
    {
      name: "mlxts_serve_generation_stream_ttft_seconds",
      help: "Server-side time from generation start to first emitted output SSE frame.",
      type: "histogram",
      labelNames: ["model", "protocol"],
    },
    STREAM_DURATION_BUCKETS,
  );
  readonly #streamChunks = new NumberMetric({
    name: "mlxts_serve_generation_stream_chunks_total",
    help: "SSE frames emitted by generation streams.",
    type: "counter",
    labelNames: ["model", "protocol", "kind"],
  });
  readonly #streamBytes = new NumberMetric({
    name: "mlxts_serve_generation_stream_bytes_total",
    help: "SSE bytes emitted by generation streams.",
    type: "counter",
    labelNames: ["model", "protocol", "kind"],
  });
  readonly #streamOutputChunks = new HistogramMetric(
    {
      name: "mlxts_serve_generation_stream_output_chunks",
      help: "Output-bearing SSE frame counts per completed stream.",
      type: "histogram",
      labelNames: ["model", "protocol", "result"],
    },
    STREAM_CHUNK_BUCKETS,
  );

  recordEnd(model: string, event: Extract<ServeEvent, { type: "generation_stream_end" }>): void {
    this.#streams.add([model, event.protocol, event.result, event.finishReason], 1);
    this.#streamDurations.observe([model, event.protocol, event.result], seconds(event.durationMs));
    if (event.ttftMs !== undefined) {
      this.#streamTtft.observe([model, event.protocol], seconds(event.ttftMs));
    }
    this.#streamChunks.add([model, event.protocol, "sse"], event.chunks);
    this.#streamChunks.add([model, event.protocol, "output"], event.outputChunks);
    this.#streamBytes.add([model, event.protocol, "sse"], event.bytes);
    this.#streamBytes.add([model, event.protocol, "output"], event.outputBytes);
    this.#streamOutputChunks.observe([model, event.protocol, event.result], event.outputChunks);
  }

  format(): string[] {
    return [
      ...this.#streams.format(),
      ...this.#streamDurations.format(),
      ...this.#streamTtft.format(),
      ...this.#streamChunks.format(),
      ...this.#streamBytes.format(),
      ...this.#streamOutputChunks.format(),
    ];
  }
}
