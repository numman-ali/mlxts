/**
 * Memory-pressure relief helpers for the lazy source-backed model pool.
 * @module
 */

import { GenerationAbortError } from "@mlxts/transformers";
import { ServeError } from "../errors";
import type {
  GenerationStreamEvent,
  NormalizedGenerationRequest,
  NormalizedGenerationResult,
  ServeEvent,
} from "../types";
import type {
  ActiveRequestLease,
  LoadedModelState,
  ModelLease,
  ModelPoolPressurePolicy,
  ModelPoolState,
  PressureReliefReason,
} from "./pool-types";
import { MODEL_POOL_MEMORY_PRESSURE } from "./pool-types";

type ModelPoolPressureControllerOptions = {
  states: Map<string, ModelPoolState>;
  policy: ModelPoolPressurePolicy;
  onEvent: ((event: ServeEvent) => void) | undefined;
  disposeLoadedState(state: LoadedModelState): void;
  requireOpen(): void;
};

export function withAbortSignal(
  request: NormalizedGenerationRequest,
  signal: AbortSignal,
): NormalizedGenerationRequest {
  return { ...request, abortSignal: signal };
}

export function createActiveLease(request: NormalizedGenerationRequest): ActiveRequestLease {
  const controller = new AbortController();
  const cleanup =
    request.abortSignal === undefined
      ? () => {}
      : (() => {
          const source = request.abortSignal;
          if (source.aborted) {
            controller.abort(source.reason);
            return () => {};
          }
          const onAbort = () => controller.abort(source.reason);
          source.addEventListener("abort", onAbort, { once: true });
          return () => source.removeEventListener("abort", onAbort);
        })();
  return {
    id: request.id,
    protocol: request.protocol,
    stream: request.stream,
    controller,
    cleanup,
    pressureAborted: false,
  };
}

export function isServeErrorCode(error: unknown, code: string): boolean {
  return error instanceof ServeError && error.code === code;
}

export function modelPoolPressureError(): ServeError {
  return new ServeError("Generation was cancelled to relieve model-pool memory pressure.", {
    code: MODEL_POOL_MEMORY_PRESSURE,
    status: 503,
  });
}

export function mapPressureAbortError(
  error: unknown,
  leases: readonly ActiveRequestLease[],
): unknown {
  return hasPressureAbortedLease(leases) && isAbortError(error) ? modelPoolPressureError() : error;
}

export function cancelledByPressure(
  result: NormalizedGenerationResult,
  leases: readonly ActiveRequestLease[],
): boolean {
  return result.finishReason === "cancelled" && hasPressureAbortedLease(leases);
}

export function activeCount(state: LoadedModelState): number {
  return state.activeLeases.size;
}

export async function* readPressureAwareStream(
  stream: AsyncIterable<GenerationStreamEvent>,
  leases: readonly ActiveRequestLease[],
): AsyncIterable<GenerationStreamEvent> {
  for await (const event of stream) {
    if (
      event.type === "done" &&
      event.finishReason === "cancelled" &&
      hasPressureAbortedLease(leases)
    ) {
      throw modelPoolPressureError();
    }
    yield event;
  }
}

function isAbortError(error: unknown): boolean {
  return (
    error instanceof GenerationAbortError || (error instanceof Error && error.name === "AbortError")
  );
}

function hasPressureAbortedLease(leases: readonly ActiveRequestLease[]): boolean {
  return leases.some((lease) => lease.pressureAborted);
}

export class ModelPoolPressureController {
  readonly #states: Map<string, ModelPoolState>;
  readonly #pressureWaiters = new Set<() => void>();
  readonly #policy: ModelPoolPressurePolicy;
  readonly #onEvent: ((event: ServeEvent) => void) | undefined;
  readonly #disposeLoadedState: (state: LoadedModelState) => void;
  readonly #requireOpen: () => void;

  constructor(options: ModelPoolPressureControllerOptions) {
    this.#states = options.states;
    this.#policy = options.policy;
    this.#onEvent = options.onEvent;
    this.#disposeLoadedState = options.disposeLoadedState;
    this.#requireOpen = options.requireOpen;
  }

  notifyReleasedLease(): void {
    if (this.#pressureWaiters.size === 0) {
      return;
    }
    const waiters = [...this.#pressureWaiters];
    this.#pressureWaiters.clear();
    for (const waiter of waiters) {
      waiter();
    }
  }

  async relieveMemoryPressure(options: {
    targetModel: string;
    reason: PressureReliefReason;
    excludeLeases?: ReadonlySet<ActiveRequestLease>;
  }): Promise<boolean> {
    if (this.#policy === "reject") {
      return false;
    }
    const evicted = this.#evictIdlePressureCandidates(options.targetModel, options.reason);
    const abortedRequestIds = this.#abortPressureCandidates(options.excludeLeases);
    if (abortedRequestIds.length > 0) {
      this.#emit({
        targetModel: options.targetModel,
        action: "abort_active",
        reason: options.reason,
        evictedModels: [],
        abortedRequestIds,
        activeRequests: this.#activeRequestCount(),
      });
      await this.#waitForPressureAbortedLeases();
    }
    return evicted || abortedRequestIds.length > 0;
  }

  async retryMemoryBudgetAfterRelief(
    error: unknown,
    modelLease: ModelLease,
    eligible: boolean,
  ): Promise<boolean> {
    if (!eligible || !isServeErrorCode(error, "memory_budget_exceeded")) {
      return false;
    }
    const relieved = await this.relieveMemoryPressure({
      targetModel: modelLease.state.entry.modelId,
      reason: "memory_budget_exceeded",
      excludeLeases: new Set(modelLease.leases),
    });
    if (!relieved) {
      return false;
    }
    this.#requireOpen();
    return true;
  }

  #emit(event: Omit<Extract<ServeEvent, { type: "model_pool_pressure" }>, "type">): void {
    this.#onEvent?.({ type: "model_pool_pressure", ...event });
  }

  #activeRequestCount(): number {
    let count = 0;
    for (const state of this.#states.values()) {
      if (state.kind === "loaded") {
        count += activeCount(state);
      }
    }
    return count;
  }

  #pressureAbortedLeaseCount(): number {
    let count = 0;
    for (const state of this.#states.values()) {
      if (state.kind !== "loaded") {
        continue;
      }
      for (const lease of state.activeLeases) {
        if (lease.pressureAborted) {
          count += 1;
        }
      }
    }
    return count;
  }

  async #waitForPressureAbortedLeases(): Promise<void> {
    while (this.#pressureAbortedLeaseCount() > 0) {
      await new Promise<void>((resolve) => {
        this.#pressureWaiters.add(resolve);
      });
    }
  }

  #evictIdlePressureCandidates(targetModel: string, reason: PressureReliefReason): boolean {
    const evictedModels: string[] = [];
    for (const state of this.#states.values()) {
      if (
        state.kind !== "loaded" ||
        state.entry.modelId === targetModel ||
        state.entry.pinned === true ||
        activeCount(state) !== 0
      ) {
        continue;
      }
      this.#states.delete(state.entry.modelId);
      this.#disposeLoadedState(state);
      evictedModels.push(state.entry.modelId);
    }
    if (evictedModels.length === 0) {
      return false;
    }
    this.#emit({
      targetModel,
      action: "evict_idle",
      reason,
      evictedModels,
      abortedRequestIds: [],
      activeRequests: this.#activeRequestCount(),
    });
    return true;
  }

  #abortPressureCandidates(excludeLeases: ReadonlySet<ActiveRequestLease> | undefined): string[] {
    const abortedRequestIds: string[] = [];
    for (const state of this.#states.values()) {
      if (state.kind !== "loaded" || state.entry.pinned === true) {
        continue;
      }
      for (const lease of state.activeLeases) {
        if (
          excludeLeases?.has(lease) === true ||
          lease.controller.signal.aborted ||
          lease.pressureAborted
        ) {
          continue;
        }
        lease.pressureAborted = true;
        lease.controller.abort(MODEL_POOL_MEMORY_PRESSURE);
        abortedRequestIds.push(lease.id);
      }
    }
    return abortedRequestIds;
  }
}
