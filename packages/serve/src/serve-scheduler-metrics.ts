/**
 * Prometheus-style metrics for continuous scheduler events.
 * @module
 */

import { HistogramMetric, NumberMetric } from "./serve-metrics-registry";
import type { ServeEvent } from "./types";

const WAIT_DURATION_BUCKETS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10];

function seconds(milliseconds: number): number {
  return milliseconds / 1000;
}

/** Metrics derived from `generation_scheduler_phase` events. */
export class ServeSchedulerMetrics {
  readonly #phases = new NumberMetric({
    name: "mlxts_serve_scheduler_phases_total",
    help: "Continuous scheduler phase transitions.",
    type: "counter",
    labelNames: ["model", "mode", "phase"],
  });
  readonly #requests = new NumberMetric({
    name: "mlxts_serve_scheduler_requests",
    help: "Latest continuous scheduler queue and active request counts.",
    type: "gauge",
    labelNames: ["model", "mode", "state"],
  });
  readonly #tokens = new NumberMetric({
    name: "mlxts_serve_scheduler_tokens",
    help: "Latest continuous scheduler token pressure.",
    type: "gauge",
    labelNames: ["model", "mode", "state"],
  });
  readonly #deferrals = new NumberMetric({
    name: "mlxts_serve_scheduler_deferrals_total",
    help: "Continuous scheduler admission deferrals by bounded reason.",
    type: "counter",
    labelNames: ["model", "mode", "reason"],
  });
  readonly #queuedDurations = new HistogramMetric(
    {
      name: "mlxts_serve_scheduler_queue_duration_seconds",
      help: "Request time spent waiting inside the continuous scheduler.",
      type: "histogram",
      labelNames: ["model", "mode"],
    },
    WAIT_DURATION_BUCKETS,
  );

  record(model: string, event: Extract<ServeEvent, { type: "generation_scheduler_phase" }>): void {
    this.#phases.add([model, event.mode, event.phase], 1);
    this.#requests.set([model, event.mode, "waiting"], event.waiting);
    this.#requests.set([model, event.mode, "prefilling"], event.prefilling);
    this.#requests.set([model, event.mode, "active"], event.active);
    this.#requests.set([model, event.mode, "max_batch_size"], event.maxBatchSize);
    this.#recordTokens(model, event);
    if (event.phase === "deferred") {
      this.#deferrals.add([model, event.mode, event.reason], 1);
    }
    if (event.phase === "admitted") {
      for (const queuedMs of event.queuedMsByRequest) {
        this.#queuedDurations.observe([model, event.mode], seconds(queuedMs));
      }
      return;
    }
    if (event.phase !== "queued") {
      this.#queuedDurations.observe([model, event.mode], seconds(event.queuedMs));
    }
  }

  format(): string[] {
    return [
      ...this.#phases.format(),
      ...this.#requests.format(),
      ...this.#tokens.format(),
      ...this.#deferrals.format(),
      ...this.#queuedDurations.format(),
    ];
  }

  #recordTokens(
    model: string,
    event: Extract<ServeEvent, { type: "generation_scheduler_phase" }>,
  ): void {
    this.#tokens.set([model, event.mode, "waiting_total"], event.waitingTotalTokens);
    this.#tokens.set([model, event.mode, "prefilling_total"], event.prefillingTotalTokens);
    this.#tokens.set([model, event.mode, "active_total"], event.activeTotalTokens);
    this.#tokens.set([model, event.mode, "scheduled_prompt"], event.scheduledPromptTokens);
    this.#tokens.set([model, event.mode, "scheduled_completion"], event.scheduledCompletionTokens);
    this.#tokens.set([model, event.mode, "scheduled_total"], event.scheduledTotalTokens);
    if (event.maxScheduledPromptTokens !== null) {
      this.#tokens.set([model, event.mode, "max_scheduled_prompt"], event.maxScheduledPromptTokens);
    }
    if (event.maxScheduledCompletionTokens !== null) {
      this.#tokens.set(
        [model, event.mode, "max_scheduled_completion"],
        event.maxScheduledCompletionTokens,
      );
    }
    if (event.maxScheduledTotalTokens !== null) {
      this.#tokens.set([model, event.mode, "max_scheduled_total"], event.maxScheduledTotalTokens);
    }
  }
}
