import { aggregateEnginePromptPrefixCacheInfo } from "../engine/prefix-cache-info";
import { ServeError } from "../errors";
import type {
  GenerationEngine,
  GenerationStreamEvent,
  NormalizedGenerationRequest,
  NormalizedGenerationResult,
} from "../types";
import { modelPoolInfo } from "./pool-info";
import {
  activeCount,
  cancelledByPressure,
  createActiveLease,
  isServeErrorCode,
  ModelPoolPressureController,
  mapPressureAbortError,
  modelPoolPressureError,
  readPressureAwareStream,
  withAbortSignal,
} from "./pool-pressure";
import { generateForEngine, groupByModel, requireEntries } from "./pool-routing";
import type {
  LoadedModelState,
  LoadedSourceModelPoolEntry,
  ModelLease,
  ModelPoolState,
  SourceModelPoolEntry,
  SourceModelPoolGenerationEngineOptions,
} from "./pool-types";

export type {
  LoadedSourceModelPoolEntry,
  SourceModelPoolEntry,
  SourceModelPoolGenerationEngineOptions,
} from "./pool-types";

type Stream = AsyncIterable<GenerationStreamEvent>;
type StreamFactory = () => Stream | Promise<Stream>;

/** Create a generation engine that loads source-backed models on first use. */
export function createSourceModelPoolGenerationEngine(
  options: SourceModelPoolGenerationEngineOptions,
): GenerationEngine {
  const entries = requireEntries(options.entries);
  const states = new Map<string, ModelPoolState>();
  let loadLane: Promise<void> = Promise.resolve();
  let disposed = false;

  function requireOpen(): void {
    if (disposed) {
      throw new ServeError("Source model pool is stopped.", {
        code: "server_stopped",
        status: 503,
      });
    }
  }

  function entryFor(modelId: string): SourceModelPoolEntry {
    const entry = entries.get(modelId);
    if (entry === undefined) {
      throw new ServeError(`Model "${modelId}" is not served by this endpoint.`, {
        code: "model_not_found",
        param: "model",
        status: 404,
      });
    }
    return entry;
  }

  function clearIdleTimer(state: LoadedModelState): void {
    if (state.idleTimer !== undefined) {
      clearTimeout(state.idleTimer);
      delete state.idleTimer;
    }
  }

  function disposeLoadedState(state: LoadedModelState): void {
    clearIdleTimer(state);
    state.dispose();
  }

  function evictLoadedState(state: LoadedModelState): void {
    const current = states.get(state.entry.modelId);
    if (current !== state || activeCount(state) !== 0 || state.entry.pinned === true) {
      return;
    }
    states.delete(state.entry.modelId);
    disposeLoadedState(state);
  }

  function scheduleIdleEviction(state: LoadedModelState): void {
    clearIdleTimer(state);
    if (
      options.idleTtlMs === undefined ||
      state.entry.pinned === true ||
      activeCount(state) !== 0
    ) {
      return;
    }
    state.idleTimer = setTimeout(() => {
      delete state.idleTimer;
      evictLoadedState(state);
    }, options.idleTtlMs);
  }

  const pressure = new ModelPoolPressureController({
    states,
    policy: options.pressurePolicy ?? "reject",
    onEvent: options.onEvent,
    disposeLoadedState,
    requireOpen,
    releaseTimeoutMs: options.pressureReleaseTimeoutMs,
  });

  async function loadSequentially(
    entry: SourceModelPoolEntry,
  ): Promise<LoadedSourceModelPoolEntry> {
    const previous = loadLane;
    let release: () => void = () => {};
    loadLane = previous.then(
      () =>
        new Promise<void>((resolve) => {
          release = resolve;
        }),
      () =>
        new Promise<void>((resolve) => {
          release = resolve;
        }),
    );
    await previous.catch(() => {});
    try {
      requireOpen();
      return await options.load(entry);
    } finally {
      release();
    }
  }

  async function loadWithMemoryPressureRelief(
    entry: SourceModelPoolEntry,
  ): Promise<LoadedSourceModelPoolEntry> {
    while (true) {
      try {
        return await loadSequentially(entry);
      } catch (error) {
        if (!isServeErrorCode(error, "model_load_memory_exceeded")) {
          throw error;
        }
        const relieved = await pressure.relieveMemoryPressure({
          targetModel: entry.modelId,
          reason: "model_load_memory_exceeded",
        });
        if (!relieved) {
          throw error;
        }
        requireOpen();
      }
    }
  }

  async function retryMemoryBudgetAfterRelief(
    error: unknown,
    modelLease: ModelLease,
    eligible: boolean,
  ): Promise<void> {
    const mapped = mapPressureAbortError(error, modelLease.leases);
    if (mapped !== error) {
      throw mapped;
    }
    const retry = await pressure.retryMemoryBudgetAfterRelief(error, modelLease, eligible);
    if (!retry) {
      throw error;
    }
  }

  async function startPressureAwareStream(
    modelLease: ModelLease,
    createStream: StreamFactory,
  ): Promise<Stream> {
    while (true) {
      try {
        const stream = await createStream();
        return streamWithGenerationPressureRetry(modelLease, stream, createStream);
      } catch (error) {
        try {
          await retryMemoryBudgetAfterRelief(error, modelLease, true);
        } catch (retryError) {
          release(modelLease);
          throw retryError;
        }
      }
    }
  }

  async function retryStreamAfterRelief(
    error: unknown,
    modelLease: ModelLease,
    createRetryStream: StreamFactory,
  ): Promise<Stream> {
    let pendingError = error;
    while (true) {
      await retryMemoryBudgetAfterRelief(pendingError, modelLease, true);
      try {
        return await createRetryStream();
      } catch (retryError) {
        pendingError = mapPressureAbortError(retryError, modelLease.leases);
      }
    }
  }

  async function ensureLoaded(entry: SourceModelPoolEntry): Promise<LoadedModelState> {
    requireOpen();
    const existing = states.get(entry.modelId);
    if (existing?.kind === "loaded") {
      return existing;
    }
    if (existing?.kind === "loading") {
      return existing.promise;
    }

    let promise: Promise<LoadedModelState>;
    promise = loadWithMemoryPressureRelief(entry).then(
      (loaded) => {
        if (disposed) {
          loaded.dispose();
          throw new ServeError("Source model pool stopped while loading a model.", {
            code: "server_stopped",
            status: 503,
          });
        }
        const state: LoadedModelState = {
          kind: "loaded",
          entry,
          engine: loaded.engine,
          dispose: loaded.dispose,
          activeLeases: new Set(),
        };
        states.set(entry.modelId, state);
        return state;
      },
      (error: unknown) => {
        const current = states.get(entry.modelId);
        if (current?.kind === "loading" && current.promise === promise) {
          states.delete(entry.modelId);
        }
        throw error;
      },
    );
    states.set(entry.modelId, { kind: "loading", promise });
    return promise;
  }

  async function acquire(
    modelId: string,
    requests: readonly NormalizedGenerationRequest[],
  ): Promise<ModelLease> {
    const state = await ensureLoaded(entryFor(modelId));
    requireOpen();
    clearIdleTimer(state);
    const leases = requests.map((request) => createActiveLease(request));
    for (const lease of leases) {
      state.activeLeases.add(lease);
    }
    const linkedRequests: NormalizedGenerationRequest[] = [];
    for (let index = 0; index < requests.length; index += 1) {
      const request = requests[index];
      const lease = leases[index];
      if (request === undefined || lease === undefined) {
        throw new ServeError("Source model pool lost request lease alignment.", {
          code: "invalid_engine_result",
          status: 500,
        });
      }
      linkedRequests.push(withAbortSignal(request, lease.controller.signal));
    }
    return {
      state,
      leases,
      requests: linkedRequests,
    };
  }

  function release(modelLease: ModelLease): void {
    for (const lease of modelLease.leases) {
      lease.cleanup();
      modelLease.state.activeLeases.delete(lease);
    }
    pressure.notifyReleasedLease();
    scheduleIdleEviction(modelLease.state);
  }

  async function runWithGenerationPressureRetry<T>(
    modelLease: ModelLease,
    work: () => Promise<T>,
  ): Promise<T> {
    while (true) {
      try {
        return await work();
      } catch (error) {
        await retryMemoryBudgetAfterRelief(error, modelLease, true);
      }
    }
  }

  async function* streamWithGenerationPressureRetry(
    modelLease: ModelLease,
    firstStream: AsyncIterable<GenerationStreamEvent>,
    createRetryStream: StreamFactory,
  ): AsyncIterable<GenerationStreamEvent> {
    let emitted = false;
    let stream = firstStream;
    try {
      while (true) {
        try {
          yield* readMappedPressureStream(modelLease, stream, () => {
            emitted = true;
          });
          return;
        } catch (error) {
          if (emitted) {
            throw error;
          }
          stream = await retryStreamAfterRelief(error, modelLease, createRetryStream);
        }
      }
    } finally {
      release(modelLease);
    }
  }

  async function* readMappedPressureStream(
    modelLease: ModelLease,
    stream: AsyncIterable<GenerationStreamEvent>,
    markEmitted?: () => void,
  ): AsyncIterable<GenerationStreamEvent> {
    try {
      for await (const event of readPressureAwareStream(stream, modelLease.leases)) {
        markEmitted?.();
        yield event;
      }
    } catch (error) {
      throw mapPressureAbortError(error, modelLease.leases);
    }
  }

  return {
    modelPoolInfo: () =>
      modelPoolInfo({
        entries,
        states,
        pressurePolicy: options.pressurePolicy,
        pressureReleaseTimeoutMs: options.pressureReleaseTimeoutMs,
        idleTtlMs: options.idleTtlMs,
      }),
    promptPrefixCacheInfo: () =>
      aggregateEnginePromptPrefixCacheInfo(
        [...states.values()].map((state) =>
          state.kind === "loaded" ? state.engine.promptPrefixCacheInfo?.() : undefined,
        ),
      ),
    async generate(request) {
      const modelLease = await acquire(request.model, [request]);
      const linkedRequest = modelLease.requests[0];
      if (linkedRequest === undefined) {
        release(modelLease);
        throw new ServeError("Source model pool failed to acquire a request lease.", {
          code: "invalid_engine_result",
          status: 500,
        });
      }
      try {
        return await runWithGenerationPressureRetry(modelLease, async () => {
          const result = await modelLease.state.engine.generate(linkedRequest);
          if (cancelledByPressure(result, modelLease.leases)) {
            throw modelPoolPressureError();
          }
          return result;
        });
      } finally {
        release(modelLease);
      }
    },
    async generateBatch(requests) {
      const results: (NormalizedGenerationResult | undefined)[] = [];
      for (const [model, group] of groupByModel(requests)) {
        const groupedRequests = group.map((entry) => entry.request);
        const modelLease = await acquire(model, groupedRequests);
        try {
          const groupedResults = await runWithGenerationPressureRetry(modelLease, async () => {
            const innerResults = await generateForEngine(
              modelLease.state.engine,
              modelLease.requests,
            );
            if (innerResults.some((result) => cancelledByPressure(result, modelLease.leases))) {
              throw modelPoolPressureError();
            }
            return innerResults;
          });
          for (const [groupIndex, result] of groupedResults.entries()) {
            const original = group[groupIndex];
            if (original === undefined) {
              throw new ServeError("Source model pool received an unexpected batch result.", {
                code: "invalid_engine_result",
                status: 500,
              });
            }
            results[original.index] = result;
          }
        } finally {
          release(modelLease);
        }
      }

      const completed: NormalizedGenerationResult[] = [];
      for (let index = 0; index < requests.length; index += 1) {
        const result = results[index];
        if (result === undefined) {
          throw new ServeError("Source model pool did not receive a result for every request.", {
            code: "invalid_engine_result",
            status: 500,
          });
        }
        completed.push(result);
      }
      return completed;
    },
    async stream(request) {
      const modelLease = await acquire(request.model, [request]);
      const linkedRequest = modelLease.requests[0];
      if (linkedRequest === undefined) {
        release(modelLease);
        throw new ServeError("Source model pool failed to acquire a stream lease.", {
          code: "invalid_engine_result",
          status: 500,
        });
      }
      const stream = modelLease.state.engine.stream;
      if (stream === undefined) {
        release(modelLease);
        throw new ServeError(`Model "${request.model}" does not support streaming.`, {
          code: "stream_not_supported",
          param: "stream",
        });
      }
      return startPressureAwareStream(modelLease, () => stream(linkedRequest));
    },
    [Symbol.dispose]() {
      if (disposed) {
        return;
      }
      disposed = true;
      for (const state of states.values()) {
        if (state.kind === "loaded") {
          disposeLoadedState(state);
        }
      }
      states.clear();
    },
  };
}
