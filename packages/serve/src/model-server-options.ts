/**
 * Option resolution for model-backed serving surfaces.
 * @module
 */

import type { Tokenizer } from "@mlxts/tokenizers";
import type {
  CausalLM,
  InteractionProfile,
  LoadSourceOptions,
  PretrainedLoadProgressEvent,
} from "@mlxts/transformers";
import {
  DEFAULT_SERVE_PREFILL_STEP_SIZE,
  requireNonNegativeInteger,
  requirePositiveFraction,
  requirePositiveInteger,
} from "./serve-runtime-strategy";
import type { TransformersContentAdapter } from "./transformers-engine-content";
import { DEFAULT_STREAM_DECODE_INTERVAL } from "./transformers-engine-streaming";
import type { ServeEvent } from "./types";

export const DEFAULT_MODEL_SERVER_HOSTNAME = "127.0.0.1";
export const DEFAULT_MODEL_SERVER_PORT = 8000;
export const DEFAULT_MODEL_SERVER_MAX_GENERATED_TOKENS = 2048;
export const DEFAULT_MODEL_SERVER_MAX_PROMPT_TOKENS = 4096;
export const DEFAULT_MODEL_SERVER_MAX_TOTAL_TOKENS = 4096;
export const DEFAULT_MODEL_SERVER_MAX_BATCH_SIZE = 32;
export const DEFAULT_MODEL_SERVER_BATCH_WINDOW_MS = 1;
export const DEFAULT_MODEL_SERVER_PREFILL_STEP_SIZE = DEFAULT_SERVE_PREFILL_STEP_SIZE;
export const DEFAULT_MODEL_SERVER_ACTIVE_PREFILL_STEP_SIZE = 128;
export const DEFAULT_MODEL_SERVER_ACTIVE_DECODE_STEPS_PER_PREFILL_CHUNK = 16;
export const DEFAULT_MODEL_SERVER_STREAM_DECODE_INTERVAL = DEFAULT_STREAM_DECODE_INTERVAL;
export const DEFAULT_MODEL_SERVER_MAX_CONCURRENT_REQUESTS = 1;
export const DEFAULT_MODEL_SERVER_GPU_MEMORY_UTILIZATION = 0.9;

export type ModelServerRuntimeOptions = {
  hostname?: string;
  port?: number;
  maxGeneratedTokens?: number;
  maxPromptTokens?: number;
  maxTotalTokens?: number;
  maxBatchSize?: number;
  batchWindowMs?: number;
  prefillStepSize?: number;
  activePrefillStepSize?: number;
  activeDecodeStepsPerPrefillChunk?: number;
  streamDecodeInterval?: number;
  maxConcurrentRequests?: number;
  gpuMemoryUtilization?: number;
  apiKey?: string;
  onEvent?: (event: ServeEvent) => void;
};

export type ServeLoadedModelOptions = ModelServerRuntimeOptions & {
  model: CausalLM;
  tokenizer: Tokenizer;
  interactionProfile?: InteractionProfile;
  contentAdapter?: TransformersContentAdapter;
  modelId: string;
  disposeModelOnStop?: boolean;
};

export type LoadedModelServerEntry = {
  model: CausalLM;
  tokenizer: Tokenizer;
  interactionProfile?: InteractionProfile;
  contentAdapter?: TransformersContentAdapter;
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

export type ResolvedRuntimeOptions = {
  hostname: string;
  port: number;
  maxGeneratedTokens: number;
  maxPromptTokens: number;
  maxTotalTokens: number;
  maxBatchSize: number;
  batchWindowMs: number;
  prefillStepSize: number;
  activePrefillStepSize: number;
  activeDecodeStepsPerPrefillChunk: number;
  streamDecodeInterval: number;
  maxConcurrentRequests: number;
  gpuMemoryUtilization: number;
  apiKey?: string;
  onEvent?: (event: ServeEvent) => void;
};

export type ResolvedLoadedModelOptions = ResolvedRuntimeOptions & {
  model: CausalLM;
  tokenizer: Tokenizer;
  interactionProfile?: InteractionProfile;
  contentAdapter?: TransformersContentAdapter;
  modelId: string;
  disposeModelOnStop: boolean;
};

export type ResolvedLoadedModelEntry = {
  model: CausalLM;
  tokenizer: Tokenizer;
  interactionProfile?: InteractionProfile;
  contentAdapter?: TransformersContentAdapter;
  modelId: string;
};

export type ResolvedLoadedModelsOptions = ResolvedRuntimeOptions & {
  models: readonly ResolvedLoadedModelEntry[];
  disposeModelsOnStop: boolean;
};

export type ResolvedServeModelOptions = ResolvedRuntimeOptions & {
  source: string;
  modelId: string;
  revision?: string;
  accessToken?: string;
  cacheDir?: string;
  localFilesOnly: boolean;
  onProgress?: (event: PretrainedLoadProgressEvent) => void;
};

function requireNonEmpty(name: string, value: string): string {
  if (value.trim() === "") {
    throw new Error(`${name} must be a non-empty string.`);
  }
  return value;
}

export function resolveRuntimeOptions(options: ModelServerRuntimeOptions): ResolvedRuntimeOptions {
  const maxBatchSize = requirePositiveInteger(
    "maxBatchSize",
    options.maxBatchSize ?? DEFAULT_MODEL_SERVER_MAX_BATCH_SIZE,
  );
  const batchWindowMs = requireNonNegativeInteger(
    "batchWindowMs",
    options.batchWindowMs ?? DEFAULT_MODEL_SERVER_BATCH_WINDOW_MS,
  );
  const activePrefillStepSize = requirePositiveInteger(
    "activePrefillStepSize",
    options.activePrefillStepSize ?? DEFAULT_MODEL_SERVER_ACTIVE_PREFILL_STEP_SIZE,
  );
  const prefillStepSize = requirePositiveInteger(
    "prefillStepSize",
    options.prefillStepSize ?? DEFAULT_MODEL_SERVER_PREFILL_STEP_SIZE,
  );
  const activeDecodeStepsPerPrefillChunk = requirePositiveInteger(
    "activeDecodeStepsPerPrefillChunk",
    options.activeDecodeStepsPerPrefillChunk ??
      DEFAULT_MODEL_SERVER_ACTIVE_DECODE_STEPS_PER_PREFILL_CHUNK,
  );
  const streamDecodeInterval = requirePositiveInteger(
    "streamDecodeInterval",
    options.streamDecodeInterval ?? DEFAULT_MODEL_SERVER_STREAM_DECODE_INTERVAL,
  );
  const maxConcurrentRequests = requirePositiveInteger(
    "maxConcurrentRequests",
    options.maxConcurrentRequests ?? DEFAULT_MODEL_SERVER_MAX_CONCURRENT_REQUESTS,
  );
  const gpuMemoryUtilization = requirePositiveFraction(
    "gpuMemoryUtilization",
    options.gpuMemoryUtilization ?? DEFAULT_MODEL_SERVER_GPU_MEMORY_UTILIZATION,
  );
  return {
    hostname: options.hostname ?? DEFAULT_MODEL_SERVER_HOSTNAME,
    port: requireNonNegativeInteger("port", options.port ?? DEFAULT_MODEL_SERVER_PORT),
    maxGeneratedTokens: requirePositiveInteger(
      "maxGeneratedTokens",
      options.maxGeneratedTokens ?? DEFAULT_MODEL_SERVER_MAX_GENERATED_TOKENS,
    ),
    maxPromptTokens: requirePositiveInteger(
      "maxPromptTokens",
      options.maxPromptTokens ?? DEFAULT_MODEL_SERVER_MAX_PROMPT_TOKENS,
    ),
    maxTotalTokens: requirePositiveInteger(
      "maxTotalTokens",
      options.maxTotalTokens ?? DEFAULT_MODEL_SERVER_MAX_TOTAL_TOKENS,
    ),
    maxBatchSize,
    batchWindowMs,
    prefillStepSize,
    activePrefillStepSize,
    activeDecodeStepsPerPrefillChunk,
    streamDecodeInterval,
    maxConcurrentRequests,
    gpuMemoryUtilization,
    ...(options.apiKey === undefined ? {} : { apiKey: options.apiKey }),
    ...(options.onEvent === undefined ? {} : { onEvent: options.onEvent }),
  };
}

export function resolveLoadedOptions(options: ServeLoadedModelOptions): ResolvedLoadedModelOptions {
  return {
    ...resolveRuntimeOptions(options),
    model: options.model,
    tokenizer: options.tokenizer,
    ...(options.interactionProfile === undefined
      ? {}
      : { interactionProfile: options.interactionProfile }),
    ...(options.contentAdapter === undefined ? {} : { contentAdapter: options.contentAdapter }),
    modelId: requireNonEmpty("modelId", options.modelId),
    disposeModelOnStop: options.disposeModelOnStop ?? false,
  };
}

function resolveLoadedModelEntry(entry: LoadedModelServerEntry): ResolvedLoadedModelEntry {
  return {
    model: entry.model,
    tokenizer: entry.tokenizer,
    ...(entry.interactionProfile === undefined
      ? {}
      : { interactionProfile: entry.interactionProfile }),
    ...(entry.contentAdapter === undefined ? {} : { contentAdapter: entry.contentAdapter }),
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

export function resolveLoadedModelsOptions(
  options: ServeLoadedModelsOptions,
): ResolvedLoadedModelsOptions {
  return {
    ...resolveRuntimeOptions(options),
    models: requireLoadedModels(options.models),
    disposeModelsOnStop: options.disposeModelsOnStop ?? false,
  };
}

export function resolveServeOptions(options: ServeModelOptions): ResolvedServeModelOptions {
  return {
    ...resolveRuntimeOptions(options),
    source: requireNonEmpty("source", options.source),
    modelId: requireNonEmpty("modelId", options.modelId ?? options.source),
    ...(options.revision === undefined ? {} : { revision: options.revision }),
    ...(options.accessToken === undefined ? {} : { accessToken: options.accessToken }),
    ...(options.cacheDir === undefined ? {} : { cacheDir: options.cacheDir }),
    localFilesOnly: options.localFilesOnly ?? false,
    ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
  };
}

export function snapshotOptions(options: ResolvedServeModelOptions): LoadSourceOptions {
  return {
    ...(options.revision === undefined ? {} : { revision: options.revision }),
    ...(options.accessToken === undefined ? {} : { accessToken: options.accessToken }),
    ...(options.cacheDir === undefined ? {} : { cacheDir: options.cacheDir }),
    localFilesOnly: options.localFilesOnly,
    ...(options.onProgress === undefined ? {} : { onProgress: options.onProgress }),
  };
}

export function runtimeServeOptions(options: ResolvedRuntimeOptions): ModelServerRuntimeOptions {
  return {
    hostname: options.hostname,
    port: options.port,
    maxGeneratedTokens: options.maxGeneratedTokens,
    maxPromptTokens: options.maxPromptTokens,
    maxTotalTokens: options.maxTotalTokens,
    maxBatchSize: options.maxBatchSize,
    batchWindowMs: options.batchWindowMs,
    prefillStepSize: options.prefillStepSize,
    activePrefillStepSize: options.activePrefillStepSize,
    activeDecodeStepsPerPrefillChunk: options.activeDecodeStepsPerPrefillChunk,
    streamDecodeInterval: options.streamDecodeInterval,
    maxConcurrentRequests: options.maxConcurrentRequests,
    gpuMemoryUtilization: options.gpuMemoryUtilization,
    ...(options.apiKey === undefined ? {} : { apiKey: options.apiKey }),
    ...(options.onEvent === undefined ? {} : { onEvent: options.onEvent }),
  };
}
