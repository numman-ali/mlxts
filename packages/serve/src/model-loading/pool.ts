/**
 * Lazy source-backed model pool for multi-model serving.
 * @module
 */

import { ServeError } from "../errors";
import type {
  GenerationEngine,
  GenerationStreamEvent,
  NormalizedGenerationRequest,
  NormalizedGenerationResult,
} from "../types";

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
};

type IdleTimer = ReturnType<typeof setTimeout>;

type LoadedModelState = {
  kind: "loaded";
  entry: SourceModelPoolEntry;
  engine: GenerationEngine;
  dispose(): void;
  activeCount: number;
  idleTimer?: IdleTimer;
};

type LoadingModelState = {
  kind: "loading";
  promise: Promise<LoadedModelState>;
};

type ModelPoolState = LoadedModelState | LoadingModelState;

function requireEntries(
  entries: readonly SourceModelPoolEntry[],
): Map<string, SourceModelPoolEntry> {
  if (entries.length === 0) {
    throw new ServeError("Source model pool requires at least one model source.", {
      code: "no_models_loaded",
      status: 500,
    });
  }
  const byModelId = new Map<string, SourceModelPoolEntry>();
  for (const entry of entries) {
    if (byModelId.has(entry.modelId)) {
      throw new ServeError(`Source model pool received duplicate model id "${entry.modelId}".`, {
        code: "duplicate_model_id",
        status: 500,
      });
    }
    byModelId.set(entry.modelId, entry);
  }
  return byModelId;
}

function groupByModel(
  requests: readonly NormalizedGenerationRequest[],
): Map<string, { index: number; request: NormalizedGenerationRequest }[]> {
  const groups = new Map<string, { index: number; request: NormalizedGenerationRequest }[]>();
  requests.forEach((request, index) => {
    const group = groups.get(request.model) ?? [];
    group.push({ index, request });
    groups.set(request.model, group);
  });
  return groups;
}

async function generateForEngine(
  engine: GenerationEngine,
  requests: readonly NormalizedGenerationRequest[],
): Promise<NormalizedGenerationResult[]> {
  if (requests.length > 1 && engine.generateBatch !== undefined) {
    const results = await engine.generateBatch(requests);
    if (results.length !== requests.length) {
      throw new ServeError("Generation engine returned the wrong number of batch results.", {
        code: "invalid_engine_result",
        status: 500,
      });
    }
    return [...results];
  }

  const results: NormalizedGenerationResult[] = [];
  for (const request of requests) {
    results.push(await engine.generate(request));
  }
  return results;
}

async function* releaseAfterStream(
  stream: AsyncIterable<GenerationStreamEvent>,
  release: () => void,
) {
  try {
    yield* stream;
  } finally {
    release();
  }
}

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
    if (current !== state || state.activeCount !== 0 || state.entry.pinned === true) {
      return;
    }
    states.delete(state.entry.modelId);
    disposeLoadedState(state);
  }

  function scheduleIdleEviction(state: LoadedModelState): void {
    clearIdleTimer(state);
    if (options.idleTtlMs === undefined || state.entry.pinned === true || state.activeCount !== 0) {
      return;
    }
    state.idleTimer = setTimeout(() => {
      delete state.idleTimer;
      evictLoadedState(state);
    }, options.idleTtlMs);
  }

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
    promise = loadSequentially(entry).then(
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
          activeCount: 0,
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

  async function acquire(modelId: string, count: number): Promise<LoadedModelState> {
    const state = await ensureLoaded(entryFor(modelId));
    requireOpen();
    clearIdleTimer(state);
    state.activeCount += count;
    return state;
  }

  function release(state: LoadedModelState, count: number): void {
    state.activeCount -= count;
    scheduleIdleEviction(state);
  }

  return {
    async generate(request) {
      const state = await acquire(request.model, 1);
      try {
        return await state.engine.generate(request);
      } finally {
        release(state, 1);
      }
    },
    async generateBatch(requests) {
      const results: (NormalizedGenerationResult | undefined)[] = [];
      for (const [model, group] of groupByModel(requests)) {
        const state = await acquire(model, group.length);
        try {
          const groupedResults = await generateForEngine(
            state.engine,
            group.map((entry) => entry.request),
          );
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
          release(state, group.length);
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
      const state = await acquire(request.model, 1);
      const stream = state.engine.stream;
      if (stream === undefined) {
        release(state, 1);
        throw new ServeError(`Model "${request.model}" does not support streaming.`, {
          code: "stream_not_supported",
          param: "stream",
        });
      }
      try {
        return releaseAfterStream(await stream(request), () => release(state, 1));
      } catch (error) {
        release(state, 1);
        throw error;
      }
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
