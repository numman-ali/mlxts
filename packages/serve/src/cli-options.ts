import { readIntegerFlag, readNumberFlag, readStringFlag } from "./cli-flag-readers";
import {
  parseModelFlagValue,
  requireDistinctModelIds,
  requirePinnedModelsExist,
  type ServeCliModelOption,
} from "./cli-model-options";
import {
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
} from "./model-loading/server";
import type { SourceModelPressurePolicy } from "./model-loading/sources";

export type { ServeCliModelOption };

export type ServeCliOptions = {
  source: string;
  modelId: string;
  models: readonly ServeCliModelOption[];
  modelRoots: readonly string[];
  modelLoadPolicy: "eager" | "lazy";
  modelPressurePolicy: SourceModelPressurePolicy;
  modelPressureReleaseTimeoutMs?: number;
  modelIdleTtlMs?: number;
  pinnedModels: readonly string[];
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
  promptPrefixCacheMaxEntries: number;
  promptPrefixCacheMaxBytes?: number;
  gpuMemoryUtilization: number;
  localImageRoots: readonly string[];
  remoteImageHosts: readonly string[];
  revision?: string;
  accessToken?: string;
  cacheDir?: string;
  apiKey?: string;
  localFilesOnly: boolean;
  verbose: boolean;
};

export type ServeCliParseResult =
  | {
      kind: "serve";
      options: ServeCliOptions;
    }
  | {
      kind: "help";
      exitCode: number;
      message?: string;
    };

type ParseState = {
  source?: string;
  modelId?: string;
  models: ServeCliModelOption[];
  modelRoots: string[];
  modelLoadPolicy: "eager" | "lazy";
  modelLoadPolicyExplicit: boolean;
  modelPressurePolicy: SourceModelPressurePolicy;
  modelPressureReleaseTimeoutMs?: number;
  modelIdleTtlMs?: number;
  pinnedModels: string[];
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
  promptPrefixCacheMaxEntries: number;
  promptPrefixCacheMaxBytes?: number;
  gpuMemoryUtilization: number;
  localImageRoots: string[];
  remoteImageHosts: string[];
  revision?: string;
  accessToken?: string;
  cacheDir?: string;
  apiKey?: string;
  localFilesOnly: boolean;
  verbose: boolean;
};

function readModelLoadPolicy(value: string): "eager" | "lazy" {
  if (value === "eager" || value === "lazy") {
    return value;
  }
  throw new Error(`Unknown model load policy: ${value}.`);
}

function readModelPressurePolicy(value: string): SourceModelPressurePolicy {
  if (value === "reject" || value === "shed_non_pinned") {
    return value;
  }
  throw new Error(`Unknown model pressure policy: ${value}.`);
}

function readPositiveIntegerFlag(arg: string, value: string | undefined): number {
  return readIntegerFlag(arg, value, (candidate) => candidate > 0, "a positive integer");
}

function createParseState(): ParseState {
  return {
    models: [],
    modelRoots: [],
    modelLoadPolicy: "eager",
    modelLoadPolicyExplicit: false,
    modelPressurePolicy: "reject",
    pinnedModels: [],
    hostname: DEFAULT_MODEL_SERVER_HOSTNAME,
    port: DEFAULT_MODEL_SERVER_PORT,
    maxGeneratedTokens: DEFAULT_MODEL_SERVER_MAX_GENERATED_TOKENS,
    maxPromptTokens: DEFAULT_MODEL_SERVER_MAX_PROMPT_TOKENS,
    maxTotalTokens: DEFAULT_MODEL_SERVER_MAX_TOTAL_TOKENS,
    maxBatchSize: DEFAULT_MODEL_SERVER_MAX_BATCH_SIZE,
    batchWindowMs: DEFAULT_MODEL_SERVER_BATCH_WINDOW_MS,
    prefillStepSize: DEFAULT_MODEL_SERVER_PREFILL_STEP_SIZE,
    activePrefillStepSize: DEFAULT_MODEL_SERVER_ACTIVE_PREFILL_STEP_SIZE,
    activeDecodeStepsPerPrefillChunk: DEFAULT_MODEL_SERVER_ACTIVE_DECODE_STEPS_PER_PREFILL_CHUNK,
    streamDecodeInterval: DEFAULT_MODEL_SERVER_STREAM_DECODE_INTERVAL,
    maxConcurrentRequests: DEFAULT_MODEL_SERVER_MAX_CONCURRENT_REQUESTS,
    promptPrefixCacheMaxEntries: DEFAULT_MODEL_SERVER_PROMPT_PREFIX_CACHE_MAX_ENTRIES,
    gpuMemoryUtilization: DEFAULT_MODEL_SERVER_GPU_MEMORY_UTILIZATION,
    localImageRoots: [],
    remoteImageHosts: [],
    localFilesOnly: false,
    verbose: false,
  };
}

function applyPositionalArg(state: ParseState, arg: string): void {
  if (state.source !== undefined) {
    throw new Error(`Unexpected extra positional argument: ${arg}`);
  }
  state.source = arg;
}

function applyFlag(state: ParseState, argv: readonly string[], index: number): number {
  const arg = argv[index];
  switch (arg) {
    case "--model":
      state.models.push(parseModelFlagValue(readStringFlag(arg, argv[index + 1])));
      return index + 1;
    case "--model-root":
      state.modelRoots.push(readStringFlag(arg, argv[index + 1]));
      return index + 1;
    case "--model-load-policy":
      state.modelLoadPolicy = readModelLoadPolicy(readStringFlag(arg, argv[index + 1]));
      state.modelLoadPolicyExplicit = true;
      return index + 1;
    case "--model-idle-ttl-ms":
      state.modelIdleTtlMs = readPositiveIntegerFlag(arg, argv[index + 1]);
      return index + 1;
    case "--model-pressure-policy":
      state.modelPressurePolicy = readModelPressurePolicy(readStringFlag(arg, argv[index + 1]));
      return index + 1;
    case "--model-pressure-release-timeout-ms":
      state.modelPressureReleaseTimeoutMs = readPositiveIntegerFlag(arg, argv[index + 1]);
      return index + 1;
    case "--pin-model":
      state.pinnedModels.push(readStringFlag(arg, argv[index + 1]));
      return index + 1;
    case "--model-id":
    case "--served-model-name":
      state.modelId = readStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--host":
      state.hostname = readStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--port":
      state.port = readIntegerFlag(
        arg,
        argv[index + 1],
        (value) => value >= 0,
        "a non-negative integer",
      );
      return index + 1;
    case "--max-generated-tokens":
      state.maxGeneratedTokens = readPositiveIntegerFlag(arg, argv[index + 1]);
      return index + 1;
    case "--max-prompt-tokens":
      state.maxPromptTokens = readPositiveIntegerFlag(arg, argv[index + 1]);
      return index + 1;
    case "--max-total-tokens":
      state.maxTotalTokens = readPositiveIntegerFlag(arg, argv[index + 1]);
      return index + 1;
    case "--max-batch-size":
      state.maxBatchSize = readPositiveIntegerFlag(arg, argv[index + 1]);
      return index + 1;
    case "--batch-window-ms":
      state.batchWindowMs = readIntegerFlag(
        arg,
        argv[index + 1],
        (value) => value >= 0,
        "a non-negative integer",
      );
      return index + 1;
    case "--active-prefill-step-size":
      state.activePrefillStepSize = readPositiveIntegerFlag(arg, argv[index + 1]);
      return index + 1;
    case "--prefill-step-size":
      state.prefillStepSize = readPositiveIntegerFlag(arg, argv[index + 1]);
      return index + 1;
    case "--active-decode-steps-per-prefill-chunk":
      state.activeDecodeStepsPerPrefillChunk = readPositiveIntegerFlag(arg, argv[index + 1]);
      return index + 1;
    case "--stream-decode-interval":
      state.streamDecodeInterval = readPositiveIntegerFlag(arg, argv[index + 1]);
      return index + 1;
    case "--max-concurrent-requests":
      state.maxConcurrentRequests = readPositiveIntegerFlag(arg, argv[index + 1]);
      return index + 1;
    case "--prompt-prefix-cache-max-entries":
      state.promptPrefixCacheMaxEntries = readPositiveIntegerFlag(arg, argv[index + 1]);
      return index + 1;
    case "--prompt-prefix-cache-max-bytes":
      state.promptPrefixCacheMaxBytes = readPositiveIntegerFlag(arg, argv[index + 1]);
      return index + 1;
    case "--gpu-memory-utilization":
      state.gpuMemoryUtilization = readNumberFlag(
        arg,
        argv[index + 1],
        (value) => value > 0 && value <= 1,
        "a number greater than 0 and less than or equal to 1",
      );
      return index + 1;
    case "--remote-image-host":
      state.remoteImageHosts.push(readStringFlag(arg, argv[index + 1]).toLowerCase());
      return index + 1;
    case "--local-image-root":
      state.localImageRoots.push(readStringFlag(arg, argv[index + 1]));
      return index + 1;
    case "--revision":
      state.revision = readStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--access-token":
      state.accessToken = readStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--cache-dir":
      state.cacheDir = readStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--api-key":
      state.apiKey = readStringFlag(arg, argv[index + 1]);
      return index + 1;
    case "--local-files-only":
      state.localFilesOnly = true;
      return index;
    case "--verbose":
      state.verbose = true;
      return index;
    default:
      if (arg?.startsWith("--")) {
        throw new Error(`Unknown argument: ${arg}`);
      }
      return index;
  }
}

function parseServeState(argv: readonly string[]): ParseState {
  const state = createParseState();
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === undefined) {
      continue;
    }
    const nextIndex = applyFlag(state, argv, index);
    if (nextIndex !== index || arg.startsWith("--")) {
      index = nextIndex;
      continue;
    }
    applyPositionalArg(state, arg);
  }
  return state;
}

function modelOptionsFromState(state: ParseState): readonly ServeCliModelOption[] {
  if (state.source !== undefined && state.models.length > 0) {
    throw new Error("Cannot mix a positional model source with --model entries.");
  }
  if (state.modelRoots.length > 0 && state.source !== undefined) {
    throw new Error("Cannot mix a positional model source with --model-root.");
  }
  if (state.modelRoots.length > 0 && state.modelId !== undefined) {
    throw new Error(
      "Cannot use --model-id with --model-root; use --model <model-id=source> for explicit additions.",
    );
  }
  if (state.models.length > 0) {
    if (state.modelId !== undefined) {
      throw new Error("Cannot use --model-id with --model; use --model <model-id=source>.");
    }
    requireDistinctModelIds(state.models);
    return state.models;
  }
  if (state.modelRoots.length > 0) {
    return [];
  }
  if (state.source === undefined || state.source.trim() === "") {
    throw new Error("Missing model path or Hugging Face repo id.");
  }
  return [
    {
      source: state.source,
      modelId: state.modelId ?? state.source,
    },
  ];
}

function requireLazyModelPoolOptions(
  modelLoadPolicy: "eager" | "lazy",
  modelIdleTtlMs: number | undefined,
  modelPressurePolicy: SourceModelPressurePolicy,
  modelPressureReleaseTimeoutMs: number | undefined,
  pinnedModels: readonly string[],
): void {
  if (modelLoadPolicy === "lazy") {
    return;
  }
  if (modelIdleTtlMs !== undefined) {
    throw new Error("--model-idle-ttl-ms requires --model-load-policy lazy.");
  }
  if (modelPressurePolicy !== "reject") {
    throw new Error("--model-pressure-policy requires --model-load-policy lazy.");
  }
  if (modelPressureReleaseTimeoutMs !== undefined) {
    throw new Error("--model-pressure-release-timeout-ms requires --model-load-policy lazy.");
  }
  if (pinnedModels.length > 0) {
    throw new Error("--pin-model requires --model-load-policy lazy.");
  }
}

function modelPressureReleaseTimeoutOption(
  state: ParseState,
): Pick<ServeCliOptions, "modelPressureReleaseTimeoutMs"> | Record<string, never> {
  return state.modelPressureReleaseTimeoutMs === undefined
    ? {}
    : { modelPressureReleaseTimeoutMs: state.modelPressureReleaseTimeoutMs };
}

function stateToOptions(state: ParseState): ServeCliParseResult {
  const models = modelOptionsFromState(state);
  const pinnedModels = [...new Set(state.pinnedModels)];
  const modelLoadPolicy =
    state.modelRoots.length > 0 && !state.modelLoadPolicyExplicit ? "lazy" : state.modelLoadPolicy;
  if (
    state.modelRoots.length > 0 &&
    state.modelLoadPolicyExplicit &&
    state.modelLoadPolicy !== "lazy"
  ) {
    throw new Error("--model-root requires --model-load-policy lazy.");
  }
  if (state.modelRoots.length === 0) {
    requirePinnedModelsExist(models, pinnedModels);
  }
  requireLazyModelPoolOptions(
    modelLoadPolicy,
    state.modelIdleTtlMs,
    state.modelPressurePolicy,
    state.modelPressureReleaseTimeoutMs,
    pinnedModels,
  );
  const [primaryModel] = models;
  const fallbackRoot = state.modelRoots[0];
  const source = primaryModel?.source ?? fallbackRoot;
  const modelId = primaryModel?.modelId ?? fallbackRoot;
  if (source === undefined || modelId === undefined) {
    throw new Error("Missing model path or Hugging Face repo id.");
  }

  return {
    kind: "serve",
    options: {
      source,
      modelId,
      models,
      modelRoots: [...new Set(state.modelRoots)],
      modelLoadPolicy,
      modelPressurePolicy: state.modelPressurePolicy,
      ...modelPressureReleaseTimeoutOption(state),
      ...(state.modelIdleTtlMs === undefined ? {} : { modelIdleTtlMs: state.modelIdleTtlMs }),
      pinnedModels,
      hostname: state.hostname,
      port: state.port,
      maxGeneratedTokens: state.maxGeneratedTokens,
      maxPromptTokens: state.maxPromptTokens,
      maxTotalTokens: state.maxTotalTokens,
      maxBatchSize: state.maxBatchSize,
      batchWindowMs: state.batchWindowMs,
      prefillStepSize: state.prefillStepSize,
      activePrefillStepSize: state.activePrefillStepSize,
      activeDecodeStepsPerPrefillChunk: state.activeDecodeStepsPerPrefillChunk,
      streamDecodeInterval: state.streamDecodeInterval,
      maxConcurrentRequests: state.maxConcurrentRequests,
      promptPrefixCacheMaxEntries: state.promptPrefixCacheMaxEntries,
      ...(state.promptPrefixCacheMaxBytes === undefined
        ? {}
        : { promptPrefixCacheMaxBytes: state.promptPrefixCacheMaxBytes }),
      gpuMemoryUtilization: state.gpuMemoryUtilization,
      localImageRoots: [...new Set(state.localImageRoots)],
      remoteImageHosts: [...new Set(state.remoteImageHosts)],
      ...(state.revision === undefined ? {} : { revision: state.revision }),
      ...(state.accessToken === undefined ? {} : { accessToken: state.accessToken }),
      ...(state.cacheDir === undefined ? {} : { cacheDir: state.cacheDir }),
      ...(state.apiKey === undefined ? {} : { apiKey: state.apiKey }),
      localFilesOnly: state.localFilesOnly,
      verbose: state.verbose,
    },
  };
}

export function parseServeArgs(argv: readonly string[]): ServeCliParseResult {
  if (argv.includes("--help")) {
    return { kind: "help", exitCode: 0 };
  }

  try {
    return stateToOptions(parseServeState(argv));
  } catch (error) {
    return {
      kind: "help",
      exitCode: 1,
      message: error instanceof Error ? error.message : String(error),
    };
  }
}
