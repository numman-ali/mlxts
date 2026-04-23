/**
 * First-class model serving helpers for local and Hugging Face checkpoints.
 * @module
 */

import type { Tokenizer } from "@mlxts/tokenizers";
import {
  type CausalLM,
  type InteractionProfile,
  type LoadSourceOptions,
  loadCausalLM,
  loadInteractionProfile,
  loadPretrainedTokenizer,
  type PretrainedLoadProgressEvent,
  resolvePretrainedSource,
} from "@mlxts/transformers";
import { createMicroBatchingGenerationEngine } from "./batching-engine";
import { createConcurrencyLimitGenerationEngine } from "./concurrency-engine";
import { createModelRouterGenerationEngine } from "./model-router";
import { createRequestLimitGenerationEngine } from "./request-limits";
import { startServeServer } from "./server";
import { createTransformersGenerationEngine } from "./transformers-engine";
import type { ServeEvent } from "./types";

export const DEFAULT_MODEL_SERVER_HOSTNAME = "127.0.0.1";
export const DEFAULT_MODEL_SERVER_PORT = 8000;
export const DEFAULT_MODEL_SERVER_MAX_GENERATED_TOKENS = 2048;
export const DEFAULT_MODEL_SERVER_MAX_TOTAL_TOKENS = 4096;
export const DEFAULT_MODEL_SERVER_MAX_BATCH_SIZE = 32;
export const DEFAULT_MODEL_SERVER_BATCH_WINDOW_MS = 1;
export const DEFAULT_MODEL_SERVER_MAX_CONCURRENT_REQUESTS = 1;
export type ModelServerRuntimeOptions = {
  hostname?: string;
  port?: number;
  maxGeneratedTokens?: number;
  maxTotalTokens?: number;
  maxBatchSize?: number;
  batchWindowMs?: number;
  maxConcurrentRequests?: number;
  apiKey?: string;
  onEvent?: (event: ServeEvent) => void;
};

export type ServeLoadedModelOptions = ModelServerRuntimeOptions & {
  model: CausalLM;
  tokenizer: Tokenizer;
  interactionProfile?: InteractionProfile;
  modelId: string;
  disposeModelOnStop?: boolean;
};

export type LoadedModelServerEntry = {
  model: CausalLM;
  tokenizer: Tokenizer;
  interactionProfile?: InteractionProfile;
  modelId: string;
};

export type ServeLoadedModelsOptions = ModelServerRuntimeOptions & {
  models: readonly LoadedModelServerEntry[];
  disposeModelsOnStop?: boolean;
};

export type ServeModelOptions = ModelServerRuntimeOptions & {
  source: string;
  modelId?: string;
  revision?: string;
  accessToken?: string;
  cacheDir?: string;
  localFilesOnly?: boolean;
  onProgress?: (event: PretrainedLoadProgressEvent) => void;
};

export type RunningModelServer = Disposable & {
  readonly endpoint: string;
  readonly modelId: string;
  readonly modelIds: readonly string[];
  readonly server: ReturnType<typeof Bun.serve>;
  stop(closeActiveConnections?: boolean): void;
};

export type ServeModelRuntime = {
  resolvePretrainedSource: typeof resolvePretrainedSource;
  loadCausalLM: typeof loadCausalLM;
  loadPretrainedTokenizer: typeof loadPretrainedTokenizer;
  loadInteractionProfile: typeof loadInteractionProfile;
  serveLoadedModel: typeof serveLoadedModel;
};

type ResolvedLoadedModelOptions = {
  model: CausalLM;
  tokenizer: Tokenizer;
  interactionProfile?: InteractionProfile;
  modelId: string;
  hostname: string;
  port: number;
  maxGeneratedTokens: number;
  maxTotalTokens: number;
  maxBatchSize: number;
  batchWindowMs: number;
  maxConcurrentRequests: number;
  apiKey?: string;
  disposeModelOnStop: boolean;
  onEvent?: (event: ServeEvent) => void;
};

type ResolvedLoadedModelEntry = {
  model: CausalLM;
  tokenizer: Tokenizer;
  interactionProfile?: InteractionProfile;
  modelId: string;
};

type ResolvedLoadedModelsOptions = {
  models: readonly ResolvedLoadedModelEntry[];
  hostname: string;
  port: number;
  maxGeneratedTokens: number;
  maxTotalTokens: number;
  maxBatchSize: number;
  batchWindowMs: number;
  maxConcurrentRequests: number;
  apiKey?: string;
  disposeModelsOnStop: boolean;
  onEvent?: (event: ServeEvent) => void;
};

type ResolvedServeModelOptions = {
  source: string;
  modelId: string;
  hostname: string;
  port: number;
  maxGeneratedTokens: number;
  maxTotalTokens: number;
  maxBatchSize: number;
  batchWindowMs: number;
  maxConcurrentRequests: number;
  revision?: string;
  accessToken?: string;
  cacheDir?: string;
  apiKey?: string;
  localFilesOnly: boolean;
  onProgress?: (event: PretrainedLoadProgressEvent) => void;
  onEvent?: (event: ServeEvent) => void;
};

function requireNonEmpty(name: string, value: string): string {
  if (value.trim() === "") {
    throw new Error(`${name} must be a non-empty string.`);
  }
  return value;
}

function requirePositiveInteger(name: string, value: number): number {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
  return value;
}

function requireNonNegativeInteger(name: string, value: number): number {
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`${name} must be a non-negative integer.`);
  }
  return value;
}

function resolveLoadedOptions(options: ServeLoadedModelOptions): ResolvedLoadedModelOptions {
  return {
    model: options.model,
    tokenizer: options.tokenizer,
    ...(options.interactionProfile === undefined
      ? {}
      : { interactionProfile: options.interactionProfile }),
    modelId: requireNonEmpty("modelId", options.modelId),
    hostname: options.hostname ?? DEFAULT_MODEL_SERVER_HOSTNAME,
    port: requireNonNegativeInteger("port", options.port ?? DEFAULT_MODEL_SERVER_PORT),
    maxGeneratedTokens: requirePositiveInteger(
      "maxGeneratedTokens",
      options.maxGeneratedTokens ?? DEFAULT_MODEL_SERVER_MAX_GENERATED_TOKENS,
    ),
    maxTotalTokens: requirePositiveInteger(
      "maxTotalTokens",
      options.maxTotalTokens ?? DEFAULT_MODEL_SERVER_MAX_TOTAL_TOKENS,
    ),
    maxBatchSize: requirePositiveInteger(
      "maxBatchSize",
      options.maxBatchSize ?? DEFAULT_MODEL_SERVER_MAX_BATCH_SIZE,
    ),
    batchWindowMs: requireNonNegativeInteger(
      "batchWindowMs",
      options.batchWindowMs ?? DEFAULT_MODEL_SERVER_BATCH_WINDOW_MS,
    ),
    maxConcurrentRequests: requirePositiveInteger(
      "maxConcurrentRequests",
      options.maxConcurrentRequests ?? DEFAULT_MODEL_SERVER_MAX_CONCURRENT_REQUESTS,
    ),
    ...(options.apiKey === undefined ? {} : { apiKey: options.apiKey }),
    disposeModelOnStop: options.disposeModelOnStop ?? false,
    ...(options.onEvent === undefined ? {} : { onEvent: options.onEvent }),
  };
}

function resolveLoadedModelEntry(entry: LoadedModelServerEntry): ResolvedLoadedModelEntry {
  return {
    model: entry.model,
    tokenizer: entry.tokenizer,
    ...(entry.interactionProfile === undefined
      ? {}
      : { interactionProfile: entry.interactionProfile }),
    modelId: requireNonEmpty("modelId", entry.modelId),
  };
}

function requireDistinctModelIds(models: readonly ResolvedLoadedModelEntry[]): void {
  const seen = new Set<string>();
  for (const model of models) {
    if (seen.has(model.modelId)) {
      throw new Error(`modelId "${model.modelId}" is duplicated.`);
    }
    seen.add(model.modelId);
  }
}

function requireLoadedModels(
  models: readonly LoadedModelServerEntry[],
): readonly ResolvedLoadedModelEntry[] {
  if (models.length === 0) {
    throw new Error("models must contain at least one loaded model.");
  }
  const resolved = models.map((model) => resolveLoadedModelEntry(model));
  requireDistinctModelIds(resolved);
  return resolved;
}

function resolveLoadedModelsOptions(
  options: ServeLoadedModelsOptions,
): ResolvedLoadedModelsOptions {
  return {
    models: requireLoadedModels(options.models),
    hostname: options.hostname ?? DEFAULT_MODEL_SERVER_HOSTNAME,
    port: requireNonNegativeInteger("port", options.port ?? DEFAULT_MODEL_SERVER_PORT),
    maxGeneratedTokens: requirePositiveInteger(
      "maxGeneratedTokens",
      options.maxGeneratedTokens ?? DEFAULT_MODEL_SERVER_MAX_GENERATED_TOKENS,
    ),
    maxTotalTokens: requirePositiveInteger(
      "maxTotalTokens",
      options.maxTotalTokens ?? DEFAULT_MODEL_SERVER_MAX_TOTAL_TOKENS,
    ),
    maxBatchSize: requirePositiveInteger(
      "maxBatchSize",
      options.maxBatchSize ?? DEFAULT_MODEL_SERVER_MAX_BATCH_SIZE,
    ),
    batchWindowMs: requireNonNegativeInteger(
      "batchWindowMs",
      options.batchWindowMs ?? DEFAULT_MODEL_SERVER_BATCH_WINDOW_MS,
    ),
    maxConcurrentRequests: requirePositiveInteger(
      "maxConcurrentRequests",
      options.maxConcurrentRequests ?? DEFAULT_MODEL_SERVER_MAX_CONCURRENT_REQUESTS,
    ),
    ...(options.apiKey === undefined ? {} : { apiKey: options.apiKey }),
    disposeModelsOnStop: options.disposeModelsOnStop ?? false,
    ...(options.onEvent === undefined ? {} : { onEvent: options.onEvent }),
  };
}

function resolveServeOptions(options: ServeModelOptions): ResolvedServeModelOptions {
  return {
    source: requireNonEmpty("source", options.source),
    modelId: requireNonEmpty("modelId", options.modelId ?? options.source),
    hostname: options.hostname ?? DEFAULT_MODEL_SERVER_HOSTNAME,
    port: requireNonNegativeInteger("port", options.port ?? DEFAULT_MODEL_SERVER_PORT),
    maxGeneratedTokens: requirePositiveInteger(
      "maxGeneratedTokens",
      options.maxGeneratedTokens ?? DEFAULT_MODEL_SERVER_MAX_GENERATED_TOKENS,
    ),
    maxTotalTokens: requirePositiveInteger(
      "maxTotalTokens",
      options.maxTotalTokens ?? DEFAULT_MODEL_SERVER_MAX_TOTAL_TOKENS,
    ),
    maxBatchSize: requirePositiveInteger(
      "maxBatchSize",
      options.maxBatchSize ?? DEFAULT_MODEL_SERVER_MAX_BATCH_SIZE,
    ),
    batchWindowMs: requireNonNegativeInteger(
      "batchWindowMs",
      options.batchWindowMs ?? DEFAULT_MODEL_SERVER_BATCH_WINDOW_MS,
    ),
    maxConcurrentRequests: requirePositiveInteger(
      "maxConcurrentRequests",
      options.maxConcurrentRequests ?? DEFAULT_MODEL_SERVER_MAX_CONCURRENT_REQUESTS,
    ),
    ...(options.revision === undefined ? {} : { revision: options.revision }),
    ...(options.accessToken === undefined ? {} : { accessToken: options.accessToken }),
    ...(options.cacheDir === undefined ? {} : { cacheDir: options.cacheDir }),
    ...(options.apiKey === undefined ? {} : { apiKey: options.apiKey }),
    localFilesOnly: options.localFilesOnly ?? false,
    ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
    ...(options.onEvent === undefined ? {} : { onEvent: options.onEvent }),
  };
}

function snapshotOptions(options: ResolvedServeModelOptions): LoadSourceOptions {
  return {
    ...(options.revision === undefined ? {} : { revision: options.revision }),
    ...(options.accessToken === undefined ? {} : { accessToken: options.accessToken }),
    ...(options.cacheDir === undefined ? {} : { cacheDir: options.cacheDir }),
    localFilesOnly: options.localFilesOnly,
    ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
  };
}

function endpointFor(server: ReturnType<typeof Bun.serve>): string {
  return `http://${server.hostname}:${server.port}`;
}

function createLoadedModelEngine(
  model: ResolvedLoadedModelEntry,
  options: ResolvedLoadedModelsOptions,
) {
  const modelEngine = createMicroBatchingGenerationEngine({
    engine: createConcurrencyLimitGenerationEngine({
      engine: createTransformersGenerationEngine({
        model: model.model,
        tokenizer: model.tokenizer,
        maxTotalTokens: options.maxTotalTokens,
        ...(model.interactionProfile === undefined
          ? {}
          : { interactionProfile: model.interactionProfile }),
        ...(options.onEvent === undefined ? {} : { onEvent: options.onEvent }),
      }),
      maxConcurrentRequests: options.maxConcurrentRequests,
    }),
    maxBatchSize: options.maxBatchSize,
    batchWindowMs: options.batchWindowMs,
    ...(options.onEvent === undefined ? {} : { onEvent: options.onEvent }),
  });
  return createRequestLimitGenerationEngine({
    engine: modelEngine,
    maxGeneratedTokens: options.maxGeneratedTokens,
  });
}

function createLoadedModelEngines(
  options: ResolvedLoadedModelsOptions,
): Record<string, ReturnType<typeof createLoadedModelEngine>> {
  const engines: Record<string, ReturnType<typeof createLoadedModelEngine>> = {};
  for (const model of options.models) {
    engines[model.modelId] = createLoadedModelEngine(model, options);
  }
  return engines;
}

function runningModelServer(
  server: ReturnType<typeof Bun.serve>,
  resolved: ResolvedLoadedModelsOptions,
): RunningModelServer {
  const modelIds = resolved.models.map((model) => model.modelId);
  const primaryModelId = modelIds[0];
  if (primaryModelId === undefined) {
    throw new Error("models must contain at least one loaded model.");
  }

  let stopped = false;
  function stop(closeActiveConnections = true): void {
    if (stopped) {
      return;
    }
    stopped = true;
    server.stop(closeActiveConnections);
    if (resolved.disposeModelsOnStop) {
      for (const model of resolved.models) {
        model.model[Symbol.dispose]();
      }
    }
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

/** Serve multiple already-loaded models and tokenizers through one OpenAI-compatible API. */
export function serveLoadedModels(options: ServeLoadedModelsOptions): RunningModelServer {
  const resolved = resolveLoadedModelsOptions(options);
  const engine = createModelRouterGenerationEngine({
    engines: createLoadedModelEngines(resolved),
  });
  const serverOptions = {
    hostname: resolved.hostname,
    port: resolved.port,
    engine,
    models: resolved.models.map((model) => ({ id: model.modelId })),
    limits: {
      maxGeneratedTokens: resolved.maxGeneratedTokens,
      maxTotalTokens: resolved.maxTotalTokens,
      maxBatchSize: resolved.maxBatchSize,
      batchWindowMs: resolved.batchWindowMs,
      maxConcurrentRequests: resolved.maxConcurrentRequests,
    },
    ...(resolved.apiKey === undefined ? {} : { apiKey: resolved.apiKey }),
    ...(resolved.onEvent === undefined ? {} : { onEvent: resolved.onEvent }),
  };
  const server = startServeServer(serverOptions);

  return runningModelServer(server, resolved);
}

/** Serve an already-loaded model and tokenizer through the OpenAI-compatible API. */
export function serveLoadedModel(options: ServeLoadedModelOptions): RunningModelServer {
  const resolved = resolveLoadedOptions(options);
  return serveLoadedModels({
    models: [
      {
        model: resolved.model,
        tokenizer: resolved.tokenizer,
        ...(resolved.interactionProfile === undefined
          ? {}
          : { interactionProfile: resolved.interactionProfile }),
        modelId: resolved.modelId,
      },
    ],
    hostname: resolved.hostname,
    port: resolved.port,
    maxGeneratedTokens: resolved.maxGeneratedTokens,
    maxTotalTokens: resolved.maxTotalTokens,
    maxBatchSize: resolved.maxBatchSize,
    batchWindowMs: resolved.batchWindowMs,
    maxConcurrentRequests: resolved.maxConcurrentRequests,
    ...(resolved.apiKey === undefined ? {} : { apiKey: resolved.apiKey }),
    disposeModelsOnStop: resolved.disposeModelOnStop,
    ...(resolved.onEvent === undefined ? {} : { onEvent: resolved.onEvent }),
  });
}

/** Load and serve one local directory or Hugging Face model repository. */
export async function serveModel(options: ServeModelOptions): Promise<RunningModelServer> {
  return serveModelWithRuntime(options, {
    resolvePretrainedSource,
    loadCausalLM,
    loadPretrainedTokenizer,
    loadInteractionProfile,
    serveLoadedModel,
  });
}

export async function serveModelWithRuntime(
  options: ServeModelOptions,
  runtime: ServeModelRuntime,
): Promise<RunningModelServer> {
  const resolved = resolveServeOptions(options);
  const loadOptions = snapshotOptions(resolved);
  const localSource = await runtime.resolvePretrainedSource(resolved.source, loadOptions);
  const model = await runtime.loadCausalLM(localSource, loadOptions);
  try {
    const tokenizer = await runtime.loadPretrainedTokenizer(localSource, loadOptions);
    const interactionProfile = await runtime.loadInteractionProfile(localSource, loadOptions);
    return runtime.serveLoadedModel({
      model,
      tokenizer,
      interactionProfile,
      modelId: resolved.modelId,
      hostname: resolved.hostname,
      port: resolved.port,
      maxGeneratedTokens: resolved.maxGeneratedTokens,
      maxTotalTokens: resolved.maxTotalTokens,
      maxBatchSize: resolved.maxBatchSize,
      batchWindowMs: resolved.batchWindowMs,
      maxConcurrentRequests: resolved.maxConcurrentRequests,
      ...(resolved.apiKey === undefined ? {} : { apiKey: resolved.apiKey }),
      ...(resolved.onEvent === undefined ? {} : { onEvent: resolved.onEvent }),
      disposeModelOnStop: true,
    });
  } catch (error) {
    model[Symbol.dispose]();
    throw error;
  }
}
