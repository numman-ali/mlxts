/**
 * Lazy model-pool formatting for the finite status CLI.
 * @module
 */

import type { ServeInfoResponse } from "../http/route-info";

function toon(value: string | number | boolean | null): string {
  return typeof value === "string" ? JSON.stringify(value) : String(value);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isNumberOrNull(value: unknown): value is number | null {
  return typeof value === "number" || value === null;
}

function modelPoolCounts(info: NonNullable<ServeInfoResponse["model_pool"]>) {
  return info.models.reduce(
    (counts, model) => ({
      loaded: counts.loaded + (model.state === "loaded" ? 1 : 0),
      loading: counts.loading + (model.state === "loading" ? 1 : 0),
      active: counts.active + model.active_requests,
      pressureAborted: counts.pressureAborted + model.pressure_aborted_requests,
    }),
    { loaded: 0, loading: 0, active: 0, pressureAborted: 0 },
  );
}

/** Format the compact lazy model-pool status line. */
export function formatDefaultModelPool(info: ServeInfoResponse): string[] {
  const pool = info.model_pool;
  return [
    `model_pool: ${pool === null ? "null" : toon(`${pool.load_policy}/${pool.pressure_policy}`)}`,
  ];
}

/** Format full lazy model-pool details for the status CLI. */
export function formatModelPool(info: ServeInfoResponse): string[] {
  const pool = info.model_pool;
  if (pool === null) {
    return ["model_pool: null"];
  }
  const counts = modelPoolCounts(pool);
  return [
    "model_pool:",
    `  load_policy: ${toon(pool.load_policy)}`,
    `  pressure_policy: ${toon(pool.pressure_policy)}`,
    `  pressure_release_timeout_ms: ${toon(pool.pressure_release_timeout_ms)}`,
    `  idle_ttl_ms: ${toon(pool.idle_ttl_ms)}`,
    `  loaded: ${toon(counts.loaded)}`,
    `  loading: ${toon(counts.loading)}`,
    `  active_requests: ${toon(counts.active)}`,
    `  pressure_aborted_requests: ${toon(counts.pressureAborted)}`,
    `model_pool_models[${pool.models.length}]{id,pinned,state,active_requests,pressure_aborted_requests}:`,
    ...pool.models.map(
      (model) =>
        `  ${[
          toon(model.id),
          toon(model.pinned),
          toon(model.state),
          toon(model.active_requests),
          toon(model.pressure_aborted_requests),
        ].join(",")}`,
    ),
  ];
}

function isModelPoolState(value: unknown): value is "not_loaded" | "loading" | "loaded" {
  return value === "not_loaded" || value === "loading" || value === "loaded";
}

function isModelPoolModel(value: unknown): boolean {
  if (!isRecord(value)) {
    return false;
  }
  return (
    typeof value.id === "string" &&
    typeof value.pinned === "boolean" &&
    isModelPoolState(value.state) &&
    typeof value.active_requests === "number" &&
    typeof value.pressure_aborted_requests === "number"
  );
}

/** Validate the optional lazy model-pool section from `/info`. */
export function hasModelPoolInfo(value: unknown): value is ServeInfoResponse["model_pool"] {
  if (value === null) {
    return true;
  }
  if (!isRecord(value)) {
    return false;
  }
  return (
    value.load_policy === "lazy" &&
    (value.pressure_policy === "reject" || value.pressure_policy === "shed_non_pinned") &&
    isNumberOrNull(value.pressure_release_timeout_ms) &&
    isNumberOrNull(value.idle_ttl_ms) &&
    Array.isArray(value.models) &&
    value.models.every(isModelPoolModel)
  );
}
