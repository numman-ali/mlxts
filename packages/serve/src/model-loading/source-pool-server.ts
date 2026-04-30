/**
 * Lazy source-backed model server lifecycle.
 * @module
 */

import { startServeServer } from "../http/server";
import { createServeMetrics, createServeMetricsSink } from "../observability/metrics";
import type { ServedModelInfo } from "../protocols/openai-models";
import { modelAdmissionMetadata } from "../runtime/model-context";
import { createSourceModelPoolGenerationEngine, type SourceModelPoolEntry } from "./pool";
import {
  createLoadedModelEngine,
  type LoadedModelServerEntry,
  type RunningModelServer,
} from "./server";
import { type ResolvedLoadedModelsOptions, resolveRuntimeOptions } from "./server-options";
import type {
  ResolvedModelSourceEntry,
  ResolvedServeModelsOptions,
  ServeModelsRuntime,
} from "./sources";

type SourceLoadModelEntry = (
  entry: ResolvedModelSourceEntry,
  runtime: ServeModelsRuntime,
  gpuMemoryUtilization: number,
) => Promise<LoadedModelServerEntry>;

function endpointFor(server: ReturnType<typeof Bun.serve>): string {
  return `http://${server.hostname}:${server.port}`;
}

function sourceModelPoolEntries(resolved: ResolvedServeModelsOptions): SourceModelPoolEntry[] {
  const pinnedModelIds = new Set(resolved.pinnedModelIds);
  return resolved.models.map((model) => ({
    modelId: model.modelId,
    pinned: pinnedModelIds.has(model.modelId),
  }));
}

function modelInfoById(models: readonly ServedModelInfo[]): Map<string, ServedModelInfo> {
  return new Map(models.map((model) => [model.id, model]));
}

function sourceEntryById(
  models: readonly ResolvedModelSourceEntry[],
): Map<string, ResolvedModelSourceEntry> {
  return new Map(models.map((model) => [model.modelId, model]));
}

export function serveLazyModelsWithRuntime(
  resolved: ResolvedServeModelsOptions,
  runtime: ServeModelsRuntime,
  gpuMemoryUtilization: number,
  loadModelEntry: SourceLoadModelEntry,
): RunningModelServer {
  const abortController = new AbortController();
  const runtimeOptions = resolveRuntimeOptions(resolved);
  const modelIds = resolved.models.map((model) => model.modelId);
  const primaryModelId = modelIds[0];
  if (primaryModelId === undefined) {
    throw new Error("models must contain at least one model source.");
  }
  const modelInfos: ServedModelInfo[] = modelIds.map((id) => ({ id }));
  const infosById = modelInfoById(modelInfos);
  const entriesById = sourceEntryById(resolved.models);
  const metrics = createServeMetrics({ modelIds });
  const instrumentedOnEvent = createServeMetricsSink(metrics, runtimeOptions.onEvent);
  const engineOptions: ResolvedLoadedModelsOptions = {
    ...runtimeOptions,
    models: [],
    disposeModelsOnStop: true,
    onEvent: instrumentedOnEvent,
  };
  const engine = createSourceModelPoolGenerationEngine({
    entries: sourceModelPoolEntries(resolved),
    ...(resolved.modelIdleTtlMs === undefined ? {} : { idleTtlMs: resolved.modelIdleTtlMs }),
    pressurePolicy: resolved.modelPressurePolicy,
    onEvent: instrumentedOnEvent,
    async load(entry) {
      const sourceEntry = entriesById.get(entry.modelId);
      if (sourceEntry === undefined) {
        throw new Error(`Missing source entry for model "${entry.modelId}".`);
      }
      const loaded = await loadModelEntry(sourceEntry, runtime, gpuMemoryUtilization);
      try {
        const modelInfo = infosById.get(entry.modelId);
        if (modelInfo !== undefined) {
          modelInfo.admission = modelAdmissionMetadata(loaded.model, {
            maxPromptTokens: runtimeOptions.maxPromptTokens,
            maxTotalTokens: runtimeOptions.maxTotalTokens,
          });
        }
        const loadedEngine = createLoadedModelEngine(loaded, engineOptions);
        return {
          engine: loadedEngine,
          dispose() {
            loadedEngine[Symbol.dispose]?.();
            loaded.model[Symbol.dispose]();
          },
        };
      } catch (error) {
        try {
          loaded.model[Symbol.dispose]();
        } catch (cleanupError) {
          throw new AggregateError(
            [error, cleanupError],
            "serveModels: lazy model setup failed and model cleanup also failed.",
          );
        }
        throw error;
      }
    },
  });
  const server = startServeServer({
    hostname: runtimeOptions.hostname,
    port: runtimeOptions.port,
    engine,
    models: modelInfos,
    limits: {
      maxGeneratedTokens: runtimeOptions.maxGeneratedTokens,
      maxPromptTokens: runtimeOptions.maxPromptTokens,
      maxTotalTokens: runtimeOptions.maxTotalTokens,
      maxBatchSize: runtimeOptions.maxBatchSize,
      batchWindowMs: runtimeOptions.batchWindowMs,
      prefillStepSize: runtimeOptions.prefillStepSize,
      activePrefillStepSize: runtimeOptions.activePrefillStepSize,
      activeDecodeStepsPerPrefillChunk: runtimeOptions.activeDecodeStepsPerPrefillChunk,
      streamDecodeInterval: runtimeOptions.streamDecodeInterval,
      maxConcurrentRequests: runtimeOptions.maxConcurrentRequests,
      promptPrefixCacheMaxEntries: runtimeOptions.promptPrefixCacheMaxEntries,
      ...(runtimeOptions.promptPrefixCacheMaxBytes === undefined
        ? {}
        : { promptPrefixCacheMaxBytes: runtimeOptions.promptPrefixCacheMaxBytes }),
      gpuMemoryUtilization: runtimeOptions.gpuMemoryUtilization,
    },
    abortSignal: abortController.signal,
    metrics,
    ...(runtimeOptions.apiKey === undefined ? {} : { apiKey: runtimeOptions.apiKey }),
    ...(runtimeOptions.onEvent === undefined ? {} : { onEvent: runtimeOptions.onEvent }),
  });

  let stopped = false;
  function stop(closeActiveConnections = true): void {
    if (stopped) {
      return;
    }
    stopped = true;
    abortController.abort();
    server.stop(closeActiveConnections);
    engine[Symbol.dispose]?.();
  }

  return {
    endpoint: endpointFor(server),
    modelId: primaryModelId,
    modelIds,
    server,
    stop,
    [Symbol.dispose]() {
      stop(true);
    },
  };
}
