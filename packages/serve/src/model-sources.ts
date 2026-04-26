/**
 * Multi-source model loading for first-class serving.
 * @module
 */

import type {
  LoadSourceOptions,
  loadCausalLM,
  loadInteractionProfile,
  loadPretrainedTokenizer,
  PretrainedLoadProgressEvent,
  resolvePretrainedSource,
} from "@mlxts/transformers";
import {
  type LoadedModelServerEntry,
  type RunningModelServer,
  type ServeLoadedModelsOptions,
  serveLoadedModels,
} from "./model-server";

type SourceLoadOptions = Omit<LoadSourceOptions, "onProgress">;

export type ServeModelSourceEntry = SourceLoadOptions & {
  source: string;
  modelId?: string;
};

export type ServeModelsProgressContext = {
  index: number;
  source: string;
  modelId: string;
};

export type ServeModelsOptions = Omit<ServeLoadedModelsOptions, "models" | "disposeModelsOnStop"> &
  SourceLoadOptions & {
    models: readonly ServeModelSourceEntry[];
    onProgress?: (event: PretrainedLoadProgressEvent, context: ServeModelsProgressContext) => void;
  };

export type ServeModelsRuntime = {
  resolvePretrainedSource: typeof resolvePretrainedSource;
  loadCausalLM: typeof loadCausalLM;
  loadPretrainedTokenizer: typeof loadPretrainedTokenizer;
  loadInteractionProfile: typeof loadInteractionProfile;
  serveLoadedModels: typeof serveLoadedModels;
};

type ResolvedModelSourceEntry = {
  index: number;
  source: string;
  modelId: string;
  loadOptions: LoadSourceOptions;
};

type ResolvedServeModelsOptions = Omit<
  ServeLoadedModelsOptions,
  "models" | "disposeModelsOnStop"
> & {
  models: readonly ResolvedModelSourceEntry[];
};

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
  };
}

function resolveServeModelsOptions(options: ServeModelsOptions): ResolvedServeModelsOptions {
  const models = requireSourceModels(options.models).map((model, index) =>
    resolveSourceEntry(model, index, options),
  );
  requireDistinctModelIds(models);
  return {
    models,
    ...(options.hostname === undefined ? {} : { hostname: options.hostname }),
    ...(options.port === undefined ? {} : { port: options.port }),
    ...(options.maxGeneratedTokens === undefined
      ? {}
      : { maxGeneratedTokens: options.maxGeneratedTokens }),
    ...(options.maxPromptTokens === undefined ? {} : { maxPromptTokens: options.maxPromptTokens }),
    ...(options.maxTotalTokens === undefined ? {} : { maxTotalTokens: options.maxTotalTokens }),
    ...(options.maxBatchSize === undefined ? {} : { maxBatchSize: options.maxBatchSize }),
    ...(options.batchWindowMs === undefined ? {} : { batchWindowMs: options.batchWindowMs }),
    ...(options.streamDecodeInterval === undefined
      ? {}
      : { streamDecodeInterval: options.streamDecodeInterval }),
    ...(options.maxConcurrentRequests === undefined
      ? {}
      : { maxConcurrentRequests: options.maxConcurrentRequests }),
    ...(options.gpuMemoryUtilization === undefined
      ? {}
      : { gpuMemoryUtilization: options.gpuMemoryUtilization }),
    ...(options.apiKey === undefined ? {} : { apiKey: options.apiKey }),
    ...(options.onEvent === undefined ? {} : { onEvent: options.onEvent }),
  };
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

async function loadModelEntry(
  entry: ResolvedModelSourceEntry,
  runtime: ServeModelsRuntime,
): Promise<LoadedModelServerEntry> {
  const localSource = await runtime.resolvePretrainedSource(entry.source, entry.loadOptions);
  const model = await runtime.loadCausalLM(localSource, entry.loadOptions);
  try {
    const tokenizer = await runtime.loadPretrainedTokenizer(localSource, entry.loadOptions);
    const interactionProfile = await runtime.loadInteractionProfile(localSource, entry.loadOptions);
    return {
      model,
      tokenizer,
      interactionProfile,
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
  const loaded: LoadedModelServerEntry[] = [];
  try {
    for (const model of resolved.models) {
      loaded.push(await loadModelEntry(model, runtime));
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
    loadPretrainedTokenizer: runtime.loadPretrainedTokenizer,
    loadInteractionProfile: runtime.loadInteractionProfile,
    serveLoadedModels,
  });
}
