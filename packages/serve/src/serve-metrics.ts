/**
 * Prometheus-style metrics for the serving event stream.
 * @module
 */

import { HistogramMetric, metricKey, NumberMetric } from "./serve-metrics-registry";
import type { GenerationMemoryUsage, GenerationProtocol, ServeEvent } from "./types";

export const SERVE_METRICS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8";

const HTTP_DURATION_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30];
const GENERATION_DURATION_BUCKETS = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60];
const TOKEN_COUNT_BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192];
const WAIT_DURATION_BUCKETS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10];
const BATCH_SIZE_BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128];

function boolLabel(value: boolean): string {
  return value ? "true" : "false";
}

function seconds(milliseconds: number): number {
  return milliseconds / 1000;
}

export function normalizeServeMetricPath(path: string): string {
  if (
    path === "/health" ||
    path === "/info" ||
    path === "/metrics" ||
    path === "/v1/models" ||
    path === "/v1/completions" ||
    path === "/v1/chat/completions" ||
    path === "/v1/responses"
  ) {
    return path;
  }
  if (path.startsWith("/v1/models/")) {
    return "/v1/models/:model";
  }
  return "__unmatched__";
}

export type ServeMetricsOptions = {
  modelIds?: readonly string[];
};

export class ServeMetrics {
  readonly #knownModelIds: ReadonlySet<string> | undefined;
  readonly #generationActiveCounts = new Map<string, number>();
  readonly #httpInFlightCounts = new Map<string, number>();
  readonly #httpRequests = new NumberMetric({
    name: "mlxts_serve_http_requests_total",
    help: "HTTP requests handled by the serving router.",
    type: "counter",
    labelNames: ["method", "path", "status"],
  });
  readonly #httpErrors = new NumberMetric({
    name: "mlxts_serve_http_request_errors_total",
    help: "HTTP requests that completed through the error path.",
    type: "counter",
    labelNames: ["method", "path", "status", "code"],
  });
  readonly #httpInFlight = new NumberMetric({
    name: "mlxts_serve_http_requests_in_flight",
    help: "HTTP requests currently in flight.",
    type: "gauge",
    labelNames: ["method", "path"],
  });
  readonly #httpDurations = new HistogramMetric(
    {
      name: "mlxts_serve_http_request_duration_seconds",
      help: "End-to-end HTTP request latency.",
      type: "histogram",
      labelNames: ["method", "path", "status"],
    },
    HTTP_DURATION_BUCKETS,
  );
  readonly #generationRequests = new NumberMetric({
    name: "mlxts_serve_generation_requests_total",
    help: "Generation requests accepted by protocol.",
    type: "counter",
    labelNames: ["model", "protocol", "input_kind", "stream"],
  });
  readonly #generationActive = new NumberMetric({
    name: "mlxts_serve_generation_active",
    help: "Generation requests currently active.",
    type: "gauge",
    labelNames: ["model", "protocol"],
  });
  readonly #generationCompletions = new NumberMetric({
    name: "mlxts_serve_generation_completions_total",
    help: "Generation requests that completed normally.",
    type: "counter",
    labelNames: ["model", "protocol", "finish_reason"],
  });
  readonly #generationErrors = new NumberMetric({
    name: "mlxts_serve_generation_errors_total",
    help: "Generation requests that failed.",
    type: "counter",
    labelNames: ["model", "protocol", "code"],
  });
  readonly #generationDurations = new HistogramMetric(
    {
      name: "mlxts_serve_generation_duration_seconds",
      help: "Generation latency by terminal result.",
      type: "histogram",
      labelNames: ["model", "protocol", "result"],
    },
    GENERATION_DURATION_BUCKETS,
  );
  readonly #generationTokens = new NumberMetric({
    name: "mlxts_serve_generation_tokens_total",
    help: "Prompt, completion, and total tokens reported by completed generations.",
    type: "counter",
    labelNames: ["model", "protocol", "kind"],
  });
  readonly #generationTokenCounts = new HistogramMetric(
    {
      name: "mlxts_serve_generation_tokens",
      help: "Prompt and completion token counts reported by completed generations.",
      type: "histogram",
      labelNames: ["model", "protocol", "kind"],
    },
    TOKEN_COUNT_BUCKETS,
  );
  readonly #generationMaxTokens = new HistogramMetric(
    {
      name: "mlxts_serve_generation_max_tokens",
      help: "Requested maximum generated tokens.",
      type: "histogram",
      labelNames: ["model", "protocol", "input_kind"],
    },
    TOKEN_COUNT_BUCKETS,
  );
  readonly #prefillTokens = new NumberMetric({
    name: "mlxts_serve_generation_prefill_tokens_total",
    help: "Prompt-prefill tokens reported chunk by chunk.",
    type: "counter",
    labelNames: ["model", "protocol"],
  });
  readonly #routeDecisions = new NumberMetric({
    name: "mlxts_serve_generation_route_decisions_total",
    help: "Serving route decisions made before generation execution.",
    type: "counter",
    labelNames: [
      "model",
      "protocol",
      "route",
      "eligible",
      "reason",
      "model_type",
      "scheduler",
      "cache",
      "attention",
      "decoding",
      "stream",
    ],
  });
  readonly #memoryBytes = new NumberMetric({
    name: "mlxts_serve_memory_bytes",
    help: "Latest observed MLX allocator memory values.",
    type: "gauge",
    labelNames: ["model", "kind"],
  });
  readonly #schedulerPhases = new NumberMetric({
    name: "mlxts_serve_scheduler_phases_total",
    help: "Continuous scheduler phase transitions.",
    type: "counter",
    labelNames: ["model", "mode", "phase"],
  });
  readonly #schedulerRequests = new NumberMetric({
    name: "mlxts_serve_scheduler_requests",
    help: "Latest continuous scheduler queue and active request counts.",
    type: "gauge",
    labelNames: ["model", "mode", "state"],
  });
  readonly #schedulerQueuedDurations = new HistogramMetric(
    {
      name: "mlxts_serve_scheduler_queue_duration_seconds",
      help: "Request time spent waiting inside the continuous scheduler.",
      type: "histogram",
      labelNames: ["model", "mode"],
    },
    WAIT_DURATION_BUCKETS,
  );
  readonly #batches = new NumberMetric({
    name: "mlxts_serve_generation_batches_total",
    help: "Generation batches started by scheduler mode.",
    type: "counter",
    labelNames: ["model", "mode"],
  });
  readonly #batchSizes = new HistogramMetric(
    {
      name: "mlxts_serve_generation_batch_size",
      help: "Generation batch sizes.",
      type: "histogram",
      labelNames: ["model", "mode"],
    },
    BATCH_SIZE_BUCKETS,
  );
  readonly #admissionBatches = new NumberMetric({
    name: "mlxts_serve_generation_admission_batches_total",
    help: "Client micro-batches admitted before model execution.",
    type: "counter",
    labelNames: ["model", "mode", "engine_mode"],
  });
  readonly #modelLaneRequests = new NumberMetric({
    name: "mlxts_serve_model_lane_requests",
    help: "Latest per-model concurrency lane request counts.",
    type: "gauge",
    labelNames: ["model", "lane", "state"],
  });
  readonly #modelLaneWaitDurations = new HistogramMetric(
    {
      name: "mlxts_serve_model_lane_wait_duration_seconds",
      help: "Time spent waiting for the per-model concurrency lane.",
      type: "histogram",
      labelNames: ["model", "lane"],
    },
    WAIT_DURATION_BUCKETS,
  );

  constructor(options: ServeMetricsOptions = {}) {
    this.#knownModelIds =
      options.modelIds === undefined ? undefined : new Set(options.modelIds.filter(Boolean));
  }

  record(event: ServeEvent): void {
    switch (event.type) {
      case "request_start":
        this.#recordHttpStart(event.method, event.path);
        break;
      case "request_complete":
        this.#recordHttpComplete(event.method, event.path, event.status, event.durationMs);
        break;
      case "request_error":
        this.#recordHttpError(event.method, event.path, event.status, event.code, event.durationMs);
        break;
      case "generation_start":
        this.#recordGenerationStart(event);
        break;
      case "generation_route_decision":
        this.#routeDecisions.add(
          [
            this.#modelLabel(event.model),
            event.protocol,
            event.route,
            boolLabel(event.eligible),
            event.reason,
            event.modelType,
            event.schedulerMode,
            event.cacheBackend,
            event.attentionBackend,
            event.decodingBackend,
            boolLabel(event.stream),
          ],
          1,
        );
        break;
      case "generation_model_lane_wait":
        this.#recordModelLaneWait(event);
        break;
      case "generation_progress":
        this.#recordMemory(this.#modelLabel(event.model), event.memory);
        break;
      case "generation_prefill_progress":
        this.#prefillTokens.add([this.#modelLabel(event.model), event.protocol], event.chunkTokens);
        this.#recordMemory(this.#modelLabel(event.model), event.memory);
        break;
      case "generation_batch_start":
        this.#recordBatch(event.model, event.mode, event.batchSize);
        break;
      case "generation_scheduler_phase":
        this.#recordSchedulerPhase(event);
        break;
      case "generation_admission_batch":
        this.#admissionBatches.add(
          [this.#modelLabel(event.model), event.mode, event.engineMode],
          1,
        );
        break;
      case "generation_complete":
        this.#recordGenerationComplete(event);
        break;
      case "generation_error":
        this.#decrementGenerationActive(event.model, event.protocol);
        this.#generationErrors.add([this.#modelLabel(event.model), event.protocol, event.code], 1);
        this.#generationDurations.observe(
          [this.#modelLabel(event.model), event.protocol, "error"],
          seconds(event.durationMs),
        );
        break;
    }
  }

  format(): string {
    const metrics = [
      this.#httpRequests,
      this.#httpErrors,
      this.#httpInFlight,
      this.#httpDurations,
      this.#generationRequests,
      this.#generationActive,
      this.#generationCompletions,
      this.#generationErrors,
      this.#generationDurations,
      this.#generationTokens,
      this.#generationTokenCounts,
      this.#generationMaxTokens,
      this.#prefillTokens,
      this.#routeDecisions,
      this.#memoryBytes,
      this.#schedulerPhases,
      this.#schedulerRequests,
      this.#schedulerQueuedDurations,
      this.#batches,
      this.#batchSizes,
      this.#admissionBatches,
      this.#modelLaneRequests,
      this.#modelLaneWaitDurations,
    ];
    return `${metrics.flatMap((metric) => metric.format()).join("\n")}\n`;
  }

  #modelLabel(model: string): string {
    if (this.#knownModelIds === undefined || this.#knownModelIds.size === 0) {
      return model;
    }
    return this.#knownModelIds.has(model) ? model : "__unknown__";
  }

  #recordHttpStart(method: string, path: string): void {
    const normalizedPath = normalizeServeMetricPath(path);
    if (normalizedPath === "/metrics") {
      return;
    }
    this.#adjustHttpInFlight(method, normalizedPath, 1);
  }

  #recordHttpComplete(method: string, path: string, status: number, durationMs: number): void {
    const normalizedPath = normalizeServeMetricPath(path);
    if (normalizedPath === "/metrics") {
      return;
    }
    this.#adjustHttpInFlight(method, normalizedPath, -1);
    const statusLabel = status.toString();
    this.#httpRequests.add([method, normalizedPath, statusLabel], 1);
    this.#httpDurations.observe([method, normalizedPath, statusLabel], seconds(durationMs));
  }

  #recordHttpError(
    method: string,
    path: string,
    status: number,
    code: string,
    durationMs: number,
  ): void {
    const normalizedPath = normalizeServeMetricPath(path);
    if (normalizedPath === "/metrics") {
      return;
    }
    this.#adjustHttpInFlight(method, normalizedPath, -1);
    const statusLabel = status.toString();
    this.#httpRequests.add([method, normalizedPath, statusLabel], 1);
    this.#httpErrors.add([method, normalizedPath, statusLabel, code], 1);
    this.#httpDurations.observe([method, normalizedPath, statusLabel], seconds(durationMs));
  }

  #adjustHttpInFlight(method: string, path: string, delta: number): void {
    const labels = [method, path];
    const key = metricKey(labels);
    const next = Math.max(0, (this.#httpInFlightCounts.get(key) ?? 0) + delta);
    this.#httpInFlightCounts.set(key, next);
    this.#httpInFlight.set(labels, next);
  }

  #recordGenerationStart(event: Extract<ServeEvent, { type: "generation_start" }>): void {
    const model = this.#modelLabel(event.model);
    this.#generationRequests.add(
      [model, event.protocol, event.inputKind, boolLabel(event.stream)],
      1,
    );
    this.#generationMaxTokens.observe([model, event.protocol, event.inputKind], event.maxTokens);
    this.#adjustGenerationActive(model, event.protocol, 1);
  }

  #recordGenerationComplete(event: Extract<ServeEvent, { type: "generation_complete" }>): void {
    const model = this.#modelLabel(event.model);
    this.#decrementGenerationActive(event.model, event.protocol);
    this.#generationCompletions.add([model, event.protocol, event.finishReason], 1);
    this.#generationDurations.observe(
      [model, event.protocol, event.finishReason],
      seconds(event.durationMs),
    );
    this.#recordTokenCount(model, event.protocol, "prompt", event.promptTokens);
    this.#recordTokenCount(model, event.protocol, "completion", event.completionTokens);
    this.#recordTokenCount(model, event.protocol, "total", event.totalTokens);
    this.#recordMemory(model, event.memory);
  }

  #recordTokenCount(
    model: string,
    protocol: GenerationProtocol,
    kind: "prompt" | "completion" | "total",
    value: number | undefined,
  ): void {
    if (value !== undefined) {
      this.#generationTokens.add([model, protocol, kind], value);
      this.#generationTokenCounts.observe([model, protocol, kind], value);
    }
  }

  #recordMemory(model: string, memory: GenerationMemoryUsage | undefined): void {
    if (memory === undefined) {
      return;
    }
    this.#memoryBytes.set([model, "active"], memory.activeBytes);
    this.#memoryBytes.set([model, "cache"], memory.cacheBytes);
    this.#memoryBytes.set([model, "peak"], memory.peakBytes);
    this.#memoryBytes.set([model, "limit"], memory.limitBytes);
  }

  #recordBatch(model: string, mode: "static" | "continuous", batchSize: number): void {
    const label = this.#modelLabel(model);
    this.#batches.add([label, mode], 1);
    this.#batchSizes.observe([label, mode], batchSize);
  }

  #recordSchedulerPhase(event: Extract<ServeEvent, { type: "generation_scheduler_phase" }>): void {
    const model = this.#modelLabel(event.model);
    this.#schedulerPhases.add([model, event.mode, event.phase], 1);
    this.#schedulerRequests.set([model, event.mode, "waiting"], event.waiting);
    this.#schedulerRequests.set([model, event.mode, "prefilling"], event.prefilling);
    this.#schedulerRequests.set([model, event.mode, "active"], event.active);
    this.#schedulerRequests.set([model, event.mode, "max_batch_size"], event.maxBatchSize);
    if (event.phase === "admitted") {
      for (const queuedMs of event.queuedMsByRequest) {
        this.#schedulerQueuedDurations.observe([model, event.mode], seconds(queuedMs));
      }
      return;
    }
    if (event.phase !== "queued") {
      this.#schedulerQueuedDurations.observe([model, event.mode], seconds(event.queuedMs));
    }
  }

  #recordModelLaneWait(event: Extract<ServeEvent, { type: "generation_model_lane_wait" }>): void {
    const model = this.#modelLabel(event.model);
    this.#modelLaneWaitDurations.observe([model, event.lane], seconds(event.waitMs));
    this.#modelLaneRequests.set([model, event.lane, "in_flight"], event.inFlightAtDispatch);
    this.#modelLaneRequests.set([model, event.lane, "queued"], event.queuedAtDispatch);
    this.#modelLaneRequests.set(
      [model, event.lane, "max_concurrent_jobs"],
      event.maxConcurrentJobs,
    );
  }

  #adjustGenerationActive(model: string, protocol: GenerationProtocol, delta: number): void {
    const labels = [model, protocol];
    const key = metricKey(labels);
    const next = Math.max(0, (this.#generationActiveCounts.get(key) ?? 0) + delta);
    this.#generationActiveCounts.set(key, next);
    this.#generationActive.set(labels, next);
  }

  #decrementGenerationActive(model: string, protocol: GenerationProtocol): void {
    this.#adjustGenerationActive(this.#modelLabel(model), protocol, -1);
  }
}

export function createServeMetrics(options?: ServeMetricsOptions): ServeMetrics {
  return new ServeMetrics(options);
}

export function createServeMetricsSink(
  metrics: ServeMetrics,
  sink: ((event: ServeEvent) => void) | undefined,
): (event: ServeEvent) => void {
  return (event) => {
    metrics.record(event);
    sink?.(event);
  };
}

export function serveMetricsResponse(metrics: ServeMetrics): Response {
  return new Response(metrics.format(), {
    headers: { "content-type": SERVE_METRICS_CONTENT_TYPE },
  });
}
