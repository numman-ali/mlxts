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

export type ServeCliModelOption = {
  source: string;
  modelId: string;
};

export type ServeCliOptions = {
  source: string;
  modelId: string;
  models: readonly ServeCliModelOption[];
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
  remoteImageHosts: string[];
  revision?: string;
  accessToken?: string;
  cacheDir?: string;
  apiKey?: string;
  localFilesOnly: boolean;
  verbose: boolean;
};

function readStringFlag(flag: string, value: string | undefined): string {
  if (value === undefined || value.trim() === "") {
    throw new Error(`Missing value for ${flag}.`);
  }
  return value;
}

function readIntegerFlag(
  flag: string,
  value: string | undefined,
  isValid: (value: number) => boolean,
  description: string,
): number {
  const raw = readStringFlag(flag, value);
  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || !isValid(parsed)) {
    throw new Error(`Expected ${flag} to be ${description}, got "${raw}".`);
  }
  return parsed;
}

function readNumberFlag(
  flag: string,
  value: string | undefined,
  isValid: (value: number) => boolean,
  description: string,
): number {
  const raw = readStringFlag(flag, value);
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || !isValid(parsed)) {
    throw new Error(`Expected ${flag} to be ${description}, got "${raw}".`);
  }
  return parsed;
}

function requireNonEmptyModelPart(part: string, value: string): string {
  if (value.trim() === "") {
    throw new Error(`Expected --model to include a non-empty ${part}.`);
  }
  return value;
}

function parseModelFlagValue(rawValue: string): ServeCliModelOption {
  const raw = rawValue.trim();
  const separator = raw.indexOf("=");
  if (separator === -1) {
    return {
      source: requireNonEmptyModelPart("source", raw),
      modelId: raw,
    };
  }

  const modelId = requireNonEmptyModelPart("model id", raw.slice(0, separator).trim());
  const source = requireNonEmptyModelPart("source", raw.slice(separator + 1).trim());
  return { source, modelId };
}

function createParseState(): ParseState {
  return {
    models: [],
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
      state.maxGeneratedTokens = readIntegerFlag(
        arg,
        argv[index + 1],
        (value) => value > 0,
        "a positive integer",
      );
      return index + 1;
    case "--max-prompt-tokens":
      state.maxPromptTokens = readIntegerFlag(
        arg,
        argv[index + 1],
        (value) => value > 0,
        "a positive integer",
      );
      return index + 1;
    case "--max-total-tokens":
      state.maxTotalTokens = readIntegerFlag(
        arg,
        argv[index + 1],
        (value) => value > 0,
        "a positive integer",
      );
      return index + 1;
    case "--max-batch-size":
      state.maxBatchSize = readIntegerFlag(
        arg,
        argv[index + 1],
        (value) => value > 0,
        "a positive integer",
      );
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
      state.activePrefillStepSize = readIntegerFlag(
        arg,
        argv[index + 1],
        (value) => value > 0,
        "a positive integer",
      );
      return index + 1;
    case "--prefill-step-size":
      state.prefillStepSize = readIntegerFlag(
        arg,
        argv[index + 1],
        (value) => value > 0,
        "a positive integer",
      );
      return index + 1;
    case "--active-decode-steps-per-prefill-chunk":
      state.activeDecodeStepsPerPrefillChunk = readIntegerFlag(
        arg,
        argv[index + 1],
        (value) => value > 0,
        "a positive integer",
      );
      return index + 1;
    case "--stream-decode-interval":
      state.streamDecodeInterval = readIntegerFlag(
        arg,
        argv[index + 1],
        (value) => value > 0,
        "a positive integer",
      );
      return index + 1;
    case "--max-concurrent-requests":
      state.maxConcurrentRequests = readIntegerFlag(
        arg,
        argv[index + 1],
        (value) => value > 0,
        "a positive integer",
      );
      return index + 1;
    case "--prompt-prefix-cache-max-entries":
      state.promptPrefixCacheMaxEntries = readIntegerFlag(
        arg,
        argv[index + 1],
        (value) => value > 0,
        "a positive integer",
      );
      return index + 1;
    case "--prompt-prefix-cache-max-bytes":
      state.promptPrefixCacheMaxBytes = readIntegerFlag(
        arg,
        argv[index + 1],
        (value) => value > 0,
        "a positive integer",
      );
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

function requireDistinctModelIds(models: readonly ServeCliModelOption[]): void {
  const seen = new Set<string>();
  for (const model of models) {
    if (seen.has(model.modelId)) {
      throw new Error(`model id "${model.modelId}" is duplicated.`);
    }
    seen.add(model.modelId);
  }
}

function modelOptionsFromState(state: ParseState): readonly ServeCliModelOption[] {
  if (state.source !== undefined && state.models.length > 0) {
    throw new Error("Cannot mix a positional model source with --model entries.");
  }
  if (state.models.length > 0) {
    if (state.modelId !== undefined) {
      throw new Error("Cannot use --model-id with --model; use --model <model-id=source>.");
    }
    requireDistinctModelIds(state.models);
    return state.models;
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

function stateToOptions(state: ParseState): ServeCliParseResult {
  const models = modelOptionsFromState(state);
  const [primaryModel] = models;
  if (primaryModel === undefined) {
    throw new Error("Missing model path or Hugging Face repo id.");
  }

  return {
    kind: "serve",
    options: {
      source: primaryModel.source,
      modelId: primaryModel.modelId,
      models,
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

export function formatServeUsage(): string {
  return [
    "Serve one or more local or Hugging Face models through the @mlxts/serve OpenAI-compatible API.",
    "",
    "Usage:",
    "  mlxts-serve <model-path-or-repo-id> [options]",
    "  mlxts-serve --model <model-path-or-repo-id> [--model <model-id=path-or-repo-id>] [options]",
    "  bunx @mlxts/serve <model-path-or-repo-id> [options]",
    "",
    "Options:",
    "  --model <source|id=source>  Add a model source; repeat for multi-model serving",
    "  --model-id <id>             Served model id for positional single-model usage",
    "  --served-model-name <id>    Alias for --model-id",
    `  --host <host>               Hostname to bind (default: ${DEFAULT_MODEL_SERVER_HOSTNAME})`,
    `  --port <port>               Port to bind (default: ${DEFAULT_MODEL_SERVER_PORT})`,
    `  --max-generated-tokens <n>  Reject requests above this max_tokens cap (default: ${DEFAULT_MODEL_SERVER_MAX_GENERATED_TOKENS})`,
    `  --max-prompt-tokens <n>     Reject prompts above this tokenized prompt cap (default: ${DEFAULT_MODEL_SERVER_MAX_PROMPT_TOKENS})`,
    `  --max-total-tokens <n>      Reject prompt_tokens + max_tokens above this cap (default: ${DEFAULT_MODEL_SERVER_MAX_TOTAL_TOKENS})`,
    `  --max-batch-size <n>        Admission micro-batch size per model instance (default: ${DEFAULT_MODEL_SERVER_MAX_BATCH_SIZE})`,
    `  --batch-window-ms <n>       Wait window before flushing a micro-batch (default: ${DEFAULT_MODEL_SERVER_BATCH_WINDOW_MS})`,
    `  --prefill-step-size <n>     Cold prompt-prefill chunk size (default: ${DEFAULT_MODEL_SERVER_PREFILL_STEP_SIZE})`,
    [
      "  --active-prefill-step-size <n>",
      `Prompt-prefill chunk size while rows are decoding (default: ${DEFAULT_MODEL_SERVER_ACTIVE_PREFILL_STEP_SIZE})`,
    ].join("  "),
    [
      "  --active-decode-steps-per-prefill-chunk <n>",
      `Decode-step quantum before long prompt-prefill work resumes (default: ${DEFAULT_MODEL_SERVER_ACTIVE_DECODE_STEPS_PER_PREFILL_CHUNK})`,
    ].join("  "),
    `  --stream-decode-interval <n>  Decode/flush streaming text every n generated token(s) (default: ${DEFAULT_MODEL_SERVER_STREAM_DECODE_INTERVAL})`,
    `  --max-concurrent-requests <n>  Max in-flight jobs per served model (default: ${DEFAULT_MODEL_SERVER_MAX_CONCURRENT_REQUESTS})`,
    `  --prompt-prefix-cache-max-entries <n>  Retained prompt-prefix snapshots per served model (default: ${DEFAULT_MODEL_SERVER_PROMPT_PREFIX_CACHE_MAX_ENTRIES})`,
    "  --prompt-prefix-cache-max-bytes <n>  Estimated retained prompt-prefix snapshot bytes per served model",
    `  --gpu-memory-utilization <f>   Reject estimated requests above this fraction of MLX memory limit (default: ${DEFAULT_MODEL_SERVER_GPU_MEMORY_UTILIZATION})`,
    "  --remote-image-host <host>      Allow remote image URLs from this exact host; repeat for redirects/CDNs",
    "  --revision <ref>            Hugging Face revision when source is a repo id",
    "  --access-token <token>      Hugging Face access token for private or gated repos",
    "  --cache-dir <path>          Hugging Face cache directory",
    "  --api-key <key>             Require Authorization: Bearer <key> for /v1 routes",
    "  --local-files-only          Use only already-cached/local files",
    "  --verbose                   Log request and generation lifecycle events",
    "  --help                      Show this help",
  ].join("\n");
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
