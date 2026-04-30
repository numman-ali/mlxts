/**
 * Metrics for lazy model-pool memory-pressure relief.
 * @module
 */

import type { ServeEvent } from "../types";
import { NumberMetric } from "./metrics-registry";

export class ServeModelPoolPressureMetrics {
  readonly #events = new NumberMetric({
    name: "mlxts_serve_model_pool_pressure_events_total",
    help: "Lazy model pool memory-pressure relief actions.",
    type: "counter",
    labelNames: ["model", "action", "reason"],
  });
  readonly #affected = new NumberMetric({
    name: "mlxts_serve_model_pool_pressure_affected_total",
    help: "Lazy model pool entries or requests affected by memory-pressure relief.",
    type: "counter",
    labelNames: ["model", "action", "kind"],
  });

  record(model: string, event: Extract<ServeEvent, { type: "model_pool_pressure" }>): void {
    this.#events.add([model, event.action, event.reason], 1);
    this.#affected.add([model, event.action, "evicted_models"], event.evictedModels.length);
    this.#affected.add([model, event.action, "aborted_requests"], event.abortedRequestIds.length);
  }

  format(): string[] {
    return [...this.#events.format(), ...this.#affected.format()];
  }
}
