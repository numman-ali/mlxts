/**
 * Shared state for the lazy source-backed model pool.
 * @module
 */

import type { GenerationEngine, NormalizedGenerationRequest, ServeEvent } from "../types";

export type SourceModelPoolEntry = {
  modelId: string;
  pinned?: boolean;
};

export type LoadedSourceModelPoolEntry = {
  engine: GenerationEngine;
  dispose(): void;
};

export type SourceModelPoolGenerationEngineOptions = {
  entries: readonly SourceModelPoolEntry[];
  load(entry: SourceModelPoolEntry): Promise<LoadedSourceModelPoolEntry>;
  idleTtlMs?: number;
  pressurePolicy?: ModelPoolPressurePolicy;
  pressureReleaseTimeoutMs?: number;
  onEvent?: (event: ServeEvent) => void;
};

export type IdleTimer = ReturnType<typeof setTimeout>;

export type ActiveRequestLease = {
  id: string;
  protocol: NormalizedGenerationRequest["protocol"];
  stream: boolean;
  sequence: number;
  controller: AbortController;
  cleanup(): void;
  pressureAborted: boolean;
};

export type LoadedModelState = {
  kind: "loaded";
  entry: SourceModelPoolEntry;
  engine: GenerationEngine;
  dispose(): void;
  activeLeases: Set<ActiveRequestLease>;
  idleTimer?: IdleTimer;
};

export type LoadingModelState = {
  kind: "loading";
  promise: Promise<LoadedModelState>;
};

export type ModelPoolState = LoadedModelState | LoadingModelState;

export type ModelLease = {
  state: LoadedModelState;
  leases: readonly ActiveRequestLease[];
  requests: readonly NormalizedGenerationRequest[];
};

export type PressureReliefReason = "model_load_memory_exceeded" | "memory_budget_exceeded";

export type ModelPoolPressurePolicy = "reject" | "shed_non_pinned";

export const MODEL_POOL_MEMORY_PRESSURE = "model_pool_memory_pressure";
