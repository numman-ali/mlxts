/**
 * Multi-source model loading for first-class serving.
 * @module
 */

import type {
  CausalLM,
  LoadSourceOptions,
  loadCausalLM,
  loadInteractionProfile,
  loadPretrainedTokenizer,
  PretrainedLoadProgressEvent,
  Qwen3_5VisionPreprocessorConfig,
  resolvePretrainedSource,
} from "@mlxts/transformers";
import { shouldLoadQwen3_5ForConditionalGeneration } from "@mlxts/transformers";
import { createQwen3_5ImageContentAdapter } from "../engine/content";
import { readGenerationMemoryUsage } from "../runtime/memory";
import { requirePositiveFraction, requirePositiveInteger } from "../runtime/strategy";
import { requireModelLoadMemoryBudget } from "./memory-preflight";
import {
  type LoadedModelServerEntry,
  type RunningModelServer,
  type ServeLoadedModelsOptions,
  serveLoadedModels,
} from "./server";
import { DEFAULT_MODEL_SERVER_GPU_MEMORY_UTILIZATION } from "./server-options";
import { serveLazyModelsWithRuntime } from "./source-pool-server";

type SourceLoadOptions = Omit<LoadSourceOptions, "onProgress">;

export type ServeModelSourceEntry = SourceLoadOptions & {
  source: string;
  modelId?: string;
  pinned?: boolean;
};

export type SourceModelLoadPolicy = "eager" | "lazy";

export type ServeModelsProgressContext = {
  index: number;
  source: string;
  modelId: string;
};

export type ServeModelsOptions = Omit<ServeLoadedModelsOptions, "models" | "disposeModelsOnStop"> &
  SourceLoadOptions & {
    models: readonly ServeModelSourceEntry[];
    modelLoadPolicy?: SourceModelLoadPolicy;
    modelIdleTtlMs?: number;
    pinnedModels?: readonly string[];
    onProgress?: (event: PretrainedLoadProgressEvent, context: ServeModelsProgressContext) => void;
  };

export type ServeModelsRuntime = {
  resolvePretrainedSource: typeof resolvePretrainedSource;
  loadCausalLM: typeof loadCausalLM;
  loadQwen3_5ForConditionalGeneration?: (
    source: string,
    options?: LoadSourceOptions,
  ) => Promise<CausalLM>;
  loadQwen3_5VisionPreprocessor?: (
    source: string,
    options?: LoadSourceOptions,
  ) => Promise<Qwen3_5VisionPreprocessorConfig>;
  loadPretrainedTokenizer: typeof loadPretrainedTokenizer;
  loadInteractionProfile: typeof loadInteractionProfile;
  serveLoadedModels: typeof serveLoadedModels;
  readGenerationMemoryUsage?: typeof readGenerationMemoryUsage;
};

export type ResolvedModelSourceEntry = {
  index: number;
  source: string;
  modelId: string;
  loadOptions: LoadSourceOptions;
  pinned: boolean;
};

export type ResolvedServeModelsOptions = Omit<
  ServeLoadedModelsOptions,
  "models" | "disposeModelsOnStop"
> & {
  models: readonly ResolvedModelSourceEntry[];
  modelLoadPolicy: SourceModelLoadPolicy;
  modelIdleTtlMs?: number;
  pinnedModelIds: readonly string[];
};

type ResolvedServeModelsRuntimeOptions = Omit<
  ResolvedServeModelsOptions,
  "models" | "modelLoadPolicy" | "modelIdleTtlMs" | "pinnedModelIds"
>;
type PromptPrefixCacheRetentionOption = Pick<
  ResolvedServeModelsRuntimeOptions,
  "promptPrefixCacheMaxEntries" | "promptPrefixCacheMaxBytes"
>;

function requireNonEmpty(name: string, value: string): string {
  if (value.trim() === "") {
    throw new Error(`${name} must be a non-empty string.`);
  }
  return value;
}

function requireSourceModels(
  models: readonly ServeModelSourceEntry[],
): readonly ServeModelSourceEntry[] {
  if (models.length === 0) {
    throw new Error("models must contain at least one model source.");
  }
  return models;
}

function requireDistinctModelIds(models: readonly ResolvedModelSourceEntry[]): void {
  const seen = new Set<string>();
  for (const model of models) {
    if (seen.has(model.modelId)) {
      throw new Error(`modelId "${model.modelId}" is duplicated.`);
    }
    seen.add(model.modelId);
  }
}

function resolveLoadOptions(
  index: number,
  entry: ServeModelSourceEntry,
  options: ServeModelsOptions,
  source: string,
  modelId: string,
): LoadSourceOptions {
  const revision = entry.revision ?? options.revision;
  const accessToken = entry.accessToken ?? options.accessToken;
  const cacheDir = entry.cacheDir ?? options.cacheDir;
  const context = { index, source, modelId };
  return {
    ...(revision === undefined ? {} : { revision }),
    ...(accessToken === undefined ? {} : { accessToken }),
    ...(cacheDir === undefined ? {} : { cacheDir }),
    localFilesOnly: entry.localFilesOnly ?? options.localFilesOnly ?? false,
    ...(options.onProgress === undefined
      ? {}
      : {
          onProgress(event: PretrainedLoadProgressEvent) {
            options.onProgress?.(event, context);
          },
        }),
  };
}

function resolveSourceEntry(
  entry: ServeModelSourceEntry,
  index: number,
  options: ServeModelsOptions,
): ResolvedModelSourceEntry {
  const source = requireNonEmpty("source", entry.source);
  const modelId = requireNonEmpty("modelId", entry.modelId ?? entry.source);
  return {
    index,
    source,
    modelId,
    loadOptions: resolveLoadOptions(index, entry, options, source, modelId),
    pinned: entry.pinned ?? false,
  };
}

function resolveModelLoadPolicy(value: SourceModelLoadPolicy | undefined): SourceModelLoadPolicy {
  if (value === undefined || value === "eager" || value === "lazy") {
    return value ?? "eager";
  }
  throw new Error(`Unknown modelLoadPolicy: ${value}`);
}

function modelIdleTtlMsOption(value: number | undefined): { modelIdleTtlMs?: number } {
  return value === undefined
    ? {}
    : { modelIdleTtlMs: requirePositiveInteger("modelIdleTtlMs", value) };
}

function resolvePinnedModelIds(
  options: ServeModelsOptions,
  models: readonly ResolvedModelSourceEntry[],
): readonly string[] {
  const available = new Set(models.map((model) => model.modelId));
  const pinned = new Set(models.filter((model) => model.pinned).map((model) => model.modelId));
  for (const modelId of options.pinnedModels ?? []) {
    const resolved = requireNonEmpty("pinnedModels", modelId);
    if (!available.has(resolved)) {
      throw new Error(`pinned model "${resolved}" is not part of this serveModels() call.`);
    }
    pinned.add(resolved);
  }
  return [...pinned];
}

function requireLazyPoolOptions(
  policy: SourceModelLoadPolicy,
  options: { modelIdleTtlMs?: number; pinnedModelIds: readonly string[] },
): void {
  if (policy === "lazy") {
    return;
  }
  if (options.modelIdleTtlMs !== undefined) {
    throw new Error('modelIdleTtlMs requires modelLoadPolicy="lazy".');
  }
  if (options.pinnedModelIds.length > 0) {
    throw new Error('pinned source models require modelLoadPolicy="lazy".');
  }
}

function promptPrefixCacheRetentionOption(
  options: ServeModelsOptions,
): PromptPrefixCacheRetentionOption {
  return {
    ...(options.promptPrefixCacheMaxEntries === undefined
      ? {}
      : { promptPrefixCacheMaxEntries: options.promptPrefixCacheMaxEntries }),
    ...(options.promptPrefixCacheMaxBytes === undefined
      ? {}
      : { promptPrefixCacheMaxBytes: options.promptPrefixCacheMaxBytes }),
  };
}

function remoteImageHostsOption(
  options: ServeModelsOptions,
): Pick<ResolvedServeModelsRuntimeOptions, "remoteImageHosts"> | Record<string, never> {
  return options.remoteImageHosts === undefined
    ? {}
    : { remoteImageHosts: options.remoteImageHosts };
}

function localImageRootsOption(
  options: ServeModelsOptions,
): Pick<ResolvedServeModelsRuntimeOptions, "localImageRoots"> | Record<string, never> {
  return options.localImageRoots === undefined ? {} : { localImageRoots: options.localImageRoots };
}

function resolveServeModelsRuntimeOptions(
  options: ServeModelsOptions,
): ResolvedServeModelsRuntimeOptions {
  return {
    ...(options.hostname === undefined ? {} : { hostname: options.hostname }),
    ...(options.port === undefined ? {} : { port: options.port }),
    ...(options.maxGeneratedTokens === undefined
      ? {}
      : { maxGeneratedTokens: options.maxGeneratedTokens }),
    ...(options.maxPromptTokens === undefined ? {} : { maxPromptTokens: options.maxPromptTokens }),
    ...(options.maxTotalTokens === undefined ? {} : { maxTotalTokens: options.maxTotalTokens }),
    ...(options.maxBatchSize === undefined ? {} : { maxBatchSize: options.maxBatchSize }),
    ...(options.batchWindowMs === undefined ? {} : { batchWindowMs: options.batchWindowMs }),
    ...(options.prefillStepSize === undefined ? {} : { prefillStepSize: options.prefillStepSize }),
    ...(options.activePrefillStepSize === undefined
      ? {}
      : { activePrefillStepSize: options.activePrefillStepSize }),
    ...(options.activeDecodeStepsPerPrefillChunk === undefined
      ? {}
      : { activeDecodeStepsPerPrefillChunk: options.activeDecodeStepsPerPrefillChunk }),
    ...(options.streamDecodeInterval === undefined
      ? {}
      : { streamDecodeInterval: options.streamDecodeInterval }),
    ...(options.maxConcurrentRequests === undefined
      ? {}
      : { maxConcurrentRequests: options.maxConcurrentRequests }),
    ...promptPrefixCacheRetentionOption(options),
    ...(options.gpuMemoryUtilization === undefined
      ? {}
      : { gpuMemoryUtilization: options.gpuMemoryUtilization }),
    ...localImageRootsOption(options),
    ...remoteImageHostsOption(options),
    ...(options.apiKey === undefined ? {} : { apiKey: options.apiKey }),
    ...(options.onEvent === undefined ? {} : { onEvent: options.onEvent }),
  };
}

function resolveServeModelsOptions(options: ServeModelsOptions): ResolvedServeModelsOptions {
  const models = requireSourceModels(options.models).map((model, index) =>
    resolveSourceEntry(model, index, options),
  );
  requireDistinctModelIds(models);
  const modelLoadPolicy = resolveModelLoadPolicy(options.modelLoadPolicy);
  const idle = modelIdleTtlMsOption(options.modelIdleTtlMs);
  const pinnedModelIds = resolvePinnedModelIds(options, models);
  requireLazyPoolOptions(modelLoadPolicy, {
    ...idle,
    pinnedModelIds,
  });
  return {
    models,
    modelLoadPolicy,
    ...idle,
    pinnedModelIds,
    ...resolveServeModelsRuntimeOptions(options),
  };
}

function resolvedGpuMemoryUtilization(options: ServeModelsOptions): number {
  return requirePositiveFraction(
    "gpuMemoryUtilization",
    options.gpuMemoryUtilization ?? DEFAULT_MODEL_SERVER_GPU_MEMORY_UTILIZATION,
  );
}

function failureWithCleanupError(error: unknown, cleanupError: unknown): AggregateError {
  return new AggregateError(
    [error, cleanupError],
    "serveModels: operation failed and model cleanup also failed.",
  );
}

function disposeLoadedModels(models: readonly LoadedModelServerEntry[]): void {
  for (const model of models) {
    model.model[Symbol.dispose]();
  }
}

function disposeLoadedModelsAfterFailure(
  models: readonly LoadedModelServerEntry[],
  error: unknown,
): never {
  try {
    disposeLoadedModels(models);
  } catch (cleanupError) {
    throw failureWithCleanupError(error, cleanupError);
  }
  throw error;
}

function disposeModelAfterFailure(model: LoadedModelServerEntry["model"], error: unknown): never {
  try {
    model[Symbol.dispose]();
  } catch (cleanupError) {
    throw failureWithCleanupError(error, cleanupError);
  }
  throw error;
}

async function loadModel(
  entry: ResolvedModelSourceEntry,
  runtime: ServeModelsRuntime,
  loadQwenConditional: boolean,
): Promise<LoadedModelServerEntry["model"]> {
  if (loadQwenConditional && runtime.loadQwen3_5ForConditionalGeneration !== undefined) {
    return runtime.loadQwen3_5ForConditionalGeneration(entry.source, entry.loadOptions);
  }
  return runtime.loadCausalLM(entry.source, entry.loadOptions);
}

async function loadContentAdapter(
  entry: ResolvedModelSourceEntry,
  runtime: ServeModelsRuntime,
  loadQwenConditional: boolean,
) {
  if (!loadQwenConditional || runtime.loadQwen3_5VisionPreprocessor === undefined) {
    return undefined;
  }
  const preprocessor = await runtime.loadQwen3_5VisionPreprocessor(entry.source, entry.loadOptions);
  return createQwen3_5ImageContentAdapter(preprocessor);
}

function canLoadQwenConditional(runtime: ServeModelsRuntime): boolean {
  return (
    runtime.loadQwen3_5ForConditionalGeneration !== undefined &&
    runtime.loadQwen3_5VisionPreprocessor !== undefined
  );
}

async function loadModelEntry(
  entry: ResolvedModelSourceEntry,
  runtime: ServeModelsRuntime,
  gpuMemoryUtilization: number,
): Promise<LoadedModelServerEntry> {
  const localSource = await runtime.resolvePretrainedSource(entry.source, entry.loadOptions);
  requireModelLoadMemoryBudget({
    modelId: entry.modelId,
    source: localSource,
    gpuMemoryUtilization,
    memory: (runtime.readGenerationMemoryUsage ?? readGenerationMemoryUsage)(),
  });
  const resolvedEntry = { ...entry, source: localSource };
  const loadQwenConditional =
    canLoadQwenConditional(runtime) &&
    (await shouldLoadQwen3_5ForConditionalGeneration(localSource));
  const model = await loadModel(resolvedEntry, runtime, loadQwenConditional);
  try {
    const tokenizer = await runtime.loadPretrainedTokenizer(localSource, entry.loadOptions);
    const interactionProfile = await runtime.loadInteractionProfile(localSource, entry.loadOptions);
    const contentAdapter = await loadContentAdapter(resolvedEntry, runtime, loadQwenConditional);
    return {
      model,
      tokenizer,
      interactionProfile,
      ...(contentAdapter === undefined ? {} : { contentAdapter }),
      modelId: entry.modelId,
    };
  } catch (error) {
    disposeModelAfterFailure(model, error);
  }
}

export async function serveModelsWithRuntime(
  options: ServeModelsOptions,
  runtime: ServeModelsRuntime,
): Promise<RunningModelServer> {
  const resolved = resolveServeModelsOptions(options);
  const gpuMemoryUtilization = resolvedGpuMemoryUtilization(options);
  if (resolved.modelLoadPolicy === "lazy") {
    return serveLazyModelsWithRuntime(resolved, runtime, gpuMemoryUtilization, loadModelEntry);
  }
  const loaded: LoadedModelServerEntry[] = [];
  try {
    for (const model of resolved.models) {
      loaded.push(await loadModelEntry(model, runtime, gpuMemoryUtilization));
    }
    return runtime.serveLoadedModels({
      ...resolved,
      models: loaded,
      disposeModelsOnStop: true,
    });
  } catch (error) {
    disposeLoadedModelsAfterFailure(loaded, error);
  }
}

/** Load and serve multiple local directories or Hugging Face model repositories. */
export async function serveModels(options: ServeModelsOptions): Promise<RunningModelServer> {
  const runtime = await import("@mlxts/transformers");
  return serveModelsWithRuntime(options, {
    resolvePretrainedSource: runtime.resolvePretrainedSource,
    loadCausalLM: runtime.loadCausalLM,
    loadQwen3_5ForConditionalGeneration: runtime.loadQwen3_5ForConditionalGeneration,
    loadQwen3_5VisionPreprocessor: runtime.loadQwen3_5VisionPreprocessor,
    loadPretrainedTokenizer: runtime.loadPretrainedTokenizer,
    loadInteractionProfile: runtime.loadInteractionProfile,
    serveLoadedModels,
  });
}
