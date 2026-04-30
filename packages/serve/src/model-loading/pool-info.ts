/**
 * Lazy model-pool operator snapshots.
 * @module
 */

import type { GenerationModelPoolInfo } from "../types";
import { activeCount } from "./pool-pressure";
import type {
  LoadedModelState,
  ModelPoolState,
  SourceModelPoolEntry,
  SourceModelPoolGenerationEngineOptions,
} from "./pool-types";

function pressureAbortedCount(state: LoadedModelState): number {
  let count = 0;
  for (const lease of state.activeLeases) {
    if (lease.pressureAborted) {
      count += 1;
    }
  }
  return count;
}

/** Format the current lazy source-backed model-pool state for `/info`. */
export function modelPoolInfo(options: {
  entries: ReadonlyMap<string, SourceModelPoolEntry>;
  states: ReadonlyMap<string, ModelPoolState>;
  pressurePolicy: SourceModelPoolGenerationEngineOptions["pressurePolicy"];
  pressureReleaseTimeoutMs: SourceModelPoolGenerationEngineOptions["pressureReleaseTimeoutMs"];
  idleTtlMs: SourceModelPoolGenerationEngineOptions["idleTtlMs"];
}): GenerationModelPoolInfo {
  return {
    load_policy: "lazy",
    pressure_policy: options.pressurePolicy ?? "reject",
    pressure_release_timeout_ms: options.pressureReleaseTimeoutMs ?? null,
    idle_ttl_ms: options.idleTtlMs ?? null,
    models: [...options.entries.values()].map((entry) => {
      const state = options.states.get(entry.modelId);
      if (state?.kind === "loaded") {
        return {
          id: entry.modelId,
          pinned: entry.pinned === true,
          state: "loaded",
          active_requests: activeCount(state),
          pressure_aborted_requests: pressureAbortedCount(state),
        };
      }
      return {
        id: entry.modelId,
        pinned: entry.pinned === true,
        state: state?.kind === "loading" ? "loading" : "not_loaded",
        active_requests: 0,
        pressure_aborted_requests: 0,
      };
    }),
  };
}
