/**
 * First-class model serving helpers for local and Hugging Face checkpoints.
 * @module
 */

import {
  type CausalLM,
  type LoadSourceOptions,
  loadCausalLM,
  loadInteractionProfile,
  loadPretrainedTokenizer,
  loadQwen3_5ForConditionalGeneration,
  loadQwen3_5VisionPreprocessor,
  type Qwen3_5VisionPreprocessorConfig,
  resolvePretrainedSource,
  shouldLoadQwen3_5ForConditionalGeneration,
} from "@mlxts/transformers";
import { createRequestLimitGenerationEngine } from "../admission/request-limits";
import { createQwen3_5ImageContentAdapter } from "../engine/content";
import { createTransformersGenerationEngine } from "../engine/index";
import { startServeServer } from "../http/server";
import { createServeMetrics, createServeMetricsSink } from "../observability/metrics";
import { readGenerationMemoryUsage } from "../runtime/memory";
import { modelAdmissionMetadata } from "../runtime/model-context";
import type { GenerationEngine } from "../types";
import { requireModelLoadMemoryBudget } from "./memory-preflight";
import { createModelRouterGenerationEngine } from "./router";
import {
  type ResolvedLoadedModelEntry,
  type ResolvedLoadedModelsOptions,
  resolveLoadedModelsOptions,
  resolveLoadedOptions,
  resolveServeOptions,
  runtimeServeOptions,
  type ServeLoadedModelOptions,
  type ServeLoadedModelsOptions,
  type ServeModelOptions,
  snapshotOptions,
} from "./server-options";

export {
  DEFAULT_MODEL_SERVER_ACTIVE_DECODE_STEPS_PER_PREFILL_CHUNK,
  DEFAULT_MODEL_SERVER_ACTIVE_PREFILL_STEP_SIZE,
  DEFAULT_MODEL_SERVER_BATCH_WINDOW_MS,
  DEFAULT_MODEL_SERVER_GPU_MEMORY_UTILIZATION,
  DEFAULT_MODEL_SERVER_HOSTNAME,
  DEFAULT_MODEL_SERVER_MAX_BATCH_SIZE,
  DEFAULT_MODEL_SERVER_MAX_CONCURRENT_REQUESTS,
  DEFAULT_MODEL_SERVER_MAX_GENERATED_TOKENS,
  DEFAULT_MODEL_SERVER_MAX_PROMPT_TOKENS,
  DEFAULT_MODEL_SERVER_MAX_TOTAL_TOKENS,
  DEFAULT_MODEL_SERVER_PORT,
  DEFAULT_MODEL_SERVER_PREFILL_STEP_SIZE,
  DEFAULT_MODEL_SERVER_PROMPT_PREFIX_CACHE_MAX_ENTRIES,
  DEFAULT_MODEL_SERVER_STREAM_DECODE_INTERVAL,
  type LoadedModelServerEntry,
  type ModelServerRuntimeOptions,
  type ServeLoadedModelOptions,
  type ServeLoadedModelsOptions,
  type ServeModelOptions,
} from "./server-options";

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
  serveLoadedModel: typeof serveLoadedModel;
  readGenerationMemoryUsage?: typeof readGenerationMemoryUsage;
};

function endpointFor(server: ReturnType<typeof Bun.serve>): string {
  return `http://${server.hostname}:${server.port}`;
}

function createLoadedModelEngine(
  model: ResolvedLoadedModelEntry,
  options: ResolvedLoadedModelsOptions,
) {
  const modelEngine = createTransformersGenerationEngine({
    model: model.model,
    tokenizer: model.tokenizer,
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
    promptPrefixCacheMaxEntries: options.promptPrefixCacheMaxEntries,
    ...(options.promptPrefixCacheMaxBytes === undefined
      ? {}
      : { promptPrefixCacheMaxBytes: options.promptPrefixCacheMaxBytes }),
    gpuMemoryUtilization: options.gpuMemoryUtilization,
    remoteImageHosts: options.remoteImageHosts,
    ...(model.interactionProfile === undefined
      ? {}
      : { interactionProfile: model.interactionProfile }),
    ...(model.contentAdapter === undefined ? {} : { contentAdapter: model.contentAdapter }),
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
  abortController: AbortController,
  engine: GenerationEngine,
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
    abortController.abort();
    server.stop(closeActiveConnections);
    engine[Symbol.dispose]?.();
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
  const abortController = new AbortController();
  const modelIds = resolved.models.map((model) => model.modelId);
  const metrics = createServeMetrics({ modelIds });
  const instrumentedOnEvent = createServeMetricsSink(metrics, resolved.onEvent);
  const engineOptions: ResolvedLoadedModelsOptions = {
    ...resolved,
    onEvent: instrumentedOnEvent,
  };
  const engine = createModelRouterGenerationEngine({
    engines: createLoadedModelEngines(engineOptions),
  });
  const serverOptions = {
    hostname: resolved.hostname,
    port: resolved.port,
    engine,
    models: resolved.models.map((model) => ({
      id: model.modelId,
      admission: modelAdmissionMetadata(model.model, {
        maxPromptTokens: resolved.maxPromptTokens,
        maxTotalTokens: resolved.maxTotalTokens,
      }),
    })),
    limits: {
      maxGeneratedTokens: resolved.maxGeneratedTokens,
      maxPromptTokens: resolved.maxPromptTokens,
      maxTotalTokens: resolved.maxTotalTokens,
      maxBatchSize: resolved.maxBatchSize,
      batchWindowMs: resolved.batchWindowMs,
      prefillStepSize: resolved.prefillStepSize,
      activePrefillStepSize: resolved.activePrefillStepSize,
      activeDecodeStepsPerPrefillChunk: resolved.activeDecodeStepsPerPrefillChunk,
      streamDecodeInterval: resolved.streamDecodeInterval,
      maxConcurrentRequests: resolved.maxConcurrentRequests,
      promptPrefixCacheMaxEntries: resolved.promptPrefixCacheMaxEntries,
      ...(resolved.promptPrefixCacheMaxBytes === undefined
        ? {}
        : { promptPrefixCacheMaxBytes: resolved.promptPrefixCacheMaxBytes }),
      gpuMemoryUtilization: resolved.gpuMemoryUtilization,
    },
    abortSignal: abortController.signal,
    metrics,
    ...(resolved.apiKey === undefined ? {} : { apiKey: resolved.apiKey }),
    ...(resolved.onEvent === undefined ? {} : { onEvent: resolved.onEvent }),
  };
  const server = startServeServer(serverOptions);

  return runningModelServer(server, resolved, abortController, engine);
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
        ...(resolved.contentAdapter === undefined
          ? {}
          : { contentAdapter: resolved.contentAdapter }),
        modelId: resolved.modelId,
      },
    ],
    ...runtimeServeOptions(resolved),
    disposeModelsOnStop: resolved.disposeModelOnStop,
  });
}

/** Load and serve one local directory or Hugging Face model repository. */
export async function serveModel(options: ServeModelOptions): Promise<RunningModelServer> {
  return serveModelWithRuntime(options, {
    resolvePretrainedSource,
    loadCausalLM,
    loadQwen3_5ForConditionalGeneration,
    loadQwen3_5VisionPreprocessor,
    loadPretrainedTokenizer,
    loadInteractionProfile,
    serveLoadedModel,
  });
}

function canLoadQwenConditional(runtime: ServeModelRuntime): boolean {
  return (
    runtime.loadQwen3_5ForConditionalGeneration !== undefined &&
    runtime.loadQwen3_5VisionPreprocessor !== undefined
  );
}

async function loadModelForSource(
  source: string,
  loadOptions: ReturnType<typeof snapshotOptions>,
  runtime: ServeModelRuntime,
  loadQwenConditional: boolean,
) {
  if (loadQwenConditional && runtime.loadQwen3_5ForConditionalGeneration !== undefined) {
    return runtime.loadQwen3_5ForConditionalGeneration(source, loadOptions);
  }
  return runtime.loadCausalLM(source, loadOptions);
}

async function loadContentAdapterForSource(
  source: string,
  loadOptions: ReturnType<typeof snapshotOptions>,
  runtime: ServeModelRuntime,
  loadQwenConditional: boolean,
) {
  if (!loadQwenConditional || runtime.loadQwen3_5VisionPreprocessor === undefined) {
    return undefined;
  }
  const preprocessor = await runtime.loadQwen3_5VisionPreprocessor(source, loadOptions);
  return createQwen3_5ImageContentAdapter(preprocessor);
}

function readModelLoadMemory(runtime: ServeModelRuntime) {
  return (runtime.readGenerationMemoryUsage ?? readGenerationMemoryUsage)();
}

export async function serveModelWithRuntime(
  options: ServeModelOptions,
  runtime: ServeModelRuntime,
): Promise<RunningModelServer> {
  const resolved = resolveServeOptions(options);
  const loadOptions = snapshotOptions(resolved);
  const localSource = await runtime.resolvePretrainedSource(resolved.source, loadOptions);
  requireModelLoadMemoryBudget({
    modelId: resolved.modelId,
    source: localSource,
    gpuMemoryUtilization: resolved.gpuMemoryUtilization,
    memory: readModelLoadMemory(runtime),
  });
  const loadQwenConditional =
    canLoadQwenConditional(runtime) &&
    (await shouldLoadQwen3_5ForConditionalGeneration(localSource));
  const model = await loadModelForSource(localSource, loadOptions, runtime, loadQwenConditional);
  try {
    const tokenizer = await runtime.loadPretrainedTokenizer(localSource, loadOptions);
    const interactionProfile = await runtime.loadInteractionProfile(localSource, loadOptions);
    const contentAdapter = await loadContentAdapterForSource(
      localSource,
      loadOptions,
      runtime,
      loadQwenConditional,
    );
    return runtime.serveLoadedModel({
      model,
      tokenizer,
      interactionProfile,
      ...(contentAdapter === undefined ? {} : { contentAdapter }),
      modelId: resolved.modelId,
      ...runtimeServeOptions(resolved),
      disposeModelOnStop: true,
    });
  } catch (error) {
    model[Symbol.dispose]();
    throw error;
  }
}
