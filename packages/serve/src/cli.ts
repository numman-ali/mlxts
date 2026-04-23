#!/usr/bin/env bun

import type { PretrainedLoadProgressEvent } from "@mlxts/transformers";
import {
  DEFAULT_MODEL_SERVER_BATCH_WINDOW_MS,
  DEFAULT_MODEL_SERVER_HOSTNAME,
  DEFAULT_MODEL_SERVER_MAX_BATCH_SIZE,
  DEFAULT_MODEL_SERVER_MAX_GENERATED_TOKENS,
  DEFAULT_MODEL_SERVER_MAX_TOTAL_TOKENS,
  DEFAULT_MODEL_SERVER_PORT,
  type RunningModelServer,
  type ServeModelOptions,
  serveModel,
} from "./model-server";
import type { GenerationMemoryUsage, ServeEvent } from "./types";

export type ServeCliOptions = {
  source: string;
  modelId: string;
  hostname: string;
  port: number;
  maxGeneratedTokens: number;
  maxTotalTokens: number;
  maxBatchSize: number;
  batchWindowMs: number;
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

export type ServeCliRuntime = {
  serveModel?: (options: ServeModelOptions) => Promise<RunningModelServer>;
  log?: (message: string) => void;
  error?: (message: string) => void;
  exit?: (code: number) => void;
  waitForShutdown?: (running: RunningModelServer) => Promise<void>;
};

type ParseState = {
  source?: string;
  modelId?: string;
  hostname: string;
  port: number;
  maxGeneratedTokens: number;
  maxTotalTokens: number;
  maxBatchSize: number;
  batchWindowMs: number;
  revision?: string;
  accessToken?: string;
  cacheDir?: string;
  apiKey?: string;
  localFilesOnly: boolean;
  verbose: boolean;
};

function formatBytes(bytes: number): string {
  if (bytes >= 1e9) {
    return `${(bytes / 1e9).toFixed(1)} GB`;
  }
  if (bytes >= 1e6) {
    return `${(bytes / 1e6).toFixed(1)} MB`;
  }
  if (bytes >= 1e3) {
    return `${(bytes / 1e3).toFixed(1)} KB`;
  }
  return `${bytes} B`;
}

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

function createParseState(): ParseState {
  return {
    hostname: DEFAULT_MODEL_SERVER_HOSTNAME,
    port: DEFAULT_MODEL_SERVER_PORT,
    maxGeneratedTokens: DEFAULT_MODEL_SERVER_MAX_GENERATED_TOKENS,
    maxTotalTokens: DEFAULT_MODEL_SERVER_MAX_TOTAL_TOKENS,
    maxBatchSize: DEFAULT_MODEL_SERVER_MAX_BATCH_SIZE,
    batchWindowMs: DEFAULT_MODEL_SERVER_BATCH_WINDOW_MS,
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

function stateToOptions(state: ParseState): ServeCliParseResult {
  if (state.source === undefined || state.source.trim() === "") {
    return {
      kind: "help",
      exitCode: 1,
      message: "Missing model path or Hugging Face repo id.",
    };
  }

  return {
    kind: "serve",
    options: {
      source: state.source,
      modelId: state.modelId ?? state.source,
      hostname: state.hostname,
      port: state.port,
      maxGeneratedTokens: state.maxGeneratedTokens,
      maxTotalTokens: state.maxTotalTokens,
      maxBatchSize: state.maxBatchSize,
      batchWindowMs: state.batchWindowMs,
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
    "Serve one local or Hugging Face model through the @mlxts/serve OpenAI-compatible API.",
    "",
    "Usage:",
    "  mlxts-serve <model-path-or-repo-id> [options]",
    "  bunx @mlxts/serve <model-path-or-repo-id> [options]",
    "",
    "Options:",
    "  --model-id <id>              Served model id returned by /v1/models (default: source)",
    "  --served-model-name <id>     Alias for --model-id",
    `  --host <host>                Hostname to bind (default: ${DEFAULT_MODEL_SERVER_HOSTNAME})`,
    `  --port <port>                Port to bind (default: ${DEFAULT_MODEL_SERVER_PORT})`,
    `  --max-generated-tokens <n>   Reject requests above this max_tokens cap (default: ${DEFAULT_MODEL_SERVER_MAX_GENERATED_TOKENS})`,
    `  --max-total-tokens <n>       Reject prompt_tokens + max_tokens above this cap (default: ${DEFAULT_MODEL_SERVER_MAX_TOTAL_TOKENS})`,
    `  --max-batch-size <n>         Admission micro-batch size for one model instance (default: ${DEFAULT_MODEL_SERVER_MAX_BATCH_SIZE})`,
    `  --batch-window-ms <n>        Wait window before flushing a micro-batch (default: ${DEFAULT_MODEL_SERVER_BATCH_WINDOW_MS})`,
    "  --revision <ref>             Hugging Face revision when source is a repo id",
    "  --access-token <token>       Hugging Face access token for private or gated repos",
    "  --cache-dir <path>           Hugging Face cache directory",
    "  --api-key <key>              Require Authorization: Bearer <key> for /v1 routes",
    "  --local-files-only           Use only already-cached/local files",
    "  --verbose                    Log request and generation lifecycle events",
    "  --help                       Show this help",
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

export function formatPretrainedLoadProgress(event: PretrainedLoadProgressEvent): string {
  switch (event.stage) {
    case "resolve":
      if (event.status === "start") {
        return `[resolve] resolving ${event.source}`;
      }
      return `[resolve] ${event.directory} (${event.fileCount} files, ${formatBytes(event.totalBytes)})`;
    case "download":
      return `[download] ${event.index}/${event.totalFiles} ${event.status} ${event.relativePath} ${formatBytes(event.completedBytes)} / ${formatBytes(event.totalBytes)}`;
    case "model":
      return event.status === "weights-start"
        ? `[model] loading ${event.shardCount} safetensor shard(s)`
        : `[model] loaded ${event.shardCount} safetensor shard(s)`;
    case "tokenizer":
      return event.status === "start"
        ? `[tokenizer] loading from ${event.directory}`
        : "[tokenizer] ready";
  }
}

function authHeader(options: ServeCliOptions): string[] {
  return options.apiKey === undefined ? [] : ["-H 'authorization: Bearer <your-api-key>'"];
}

export function formatServeReady(endpoint: string, options: ServeCliOptions): string {
  return [
    "",
    `Serving ${options.modelId} at ${endpoint}`,
    `Generated-token limit: ${options.maxGeneratedTokens}`,
    `Total-token limit: ${options.maxTotalTokens}`,
    `Micro-batching: max_batch=${options.maxBatchSize} window_ms=${options.batchWindowMs}`,
    "",
    "Try:",
    [
      "curl",
      "-s",
      `${endpoint}/v1/completions`,
      ...authHeader(options),
      "-H 'content-type: application/json'",
      "-d",
      `'${JSON.stringify({
        model: options.modelId,
        prompt: "Write one crisp sentence about Apple Silicon ML.",
        max_tokens: 64,
      })}'`,
    ].join(" \\\n  "),
    "",
  ].join("\n");
}

export function publicBindWarning(options: ServeCliOptions): string | null {
  if (options.hostname !== "0.0.0.0" || options.apiKey !== undefined) {
    return null;
  }
  return "[warning] Binding to 0.0.0.0 without --api-key exposes the endpoint to your network.";
}

function formatDuration(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms.toFixed(1)}ms`;
}

function formatMemoryUsage(memory: GenerationMemoryUsage | undefined): string {
  if (memory === undefined) {
    return "";
  }
  return ` active=${formatBytes(memory.activeBytes)} cache=${formatBytes(memory.cacheBytes)} peak=${formatBytes(memory.peakBytes)}`;
}

export function formatServeEvent(event: ServeEvent): string {
  switch (event.type) {
    case "request_start":
      return `[request] ${event.method} ${event.path} started`;
    case "request_complete":
      return `[request] ${event.method} ${event.path} -> ${event.status} in ${formatDuration(event.durationMs)}`;
    case "request_error":
      return `[error] ${event.method} ${event.path} -> ${event.status} ${event.code}: ${event.message} in ${formatDuration(event.durationMs)}`;
    case "generation_start":
      return `[generation] ${event.id} ${event.protocol} model=${event.model} input=${event.inputKind} max_tokens=${event.maxTokens} started`;
    case "generation_progress":
      return `[generation] ${event.id} progress prompt_tokens=${event.promptTokens} completion_tokens=${event.completionTokens}/${event.maxTokens}${formatMemoryUsage(event.memory)}`;
    case "generation_complete":
      return `[generation] ${event.id} ${event.finishReason}${event.promptTokens === undefined ? "" : ` prompt_tokens=${event.promptTokens}`} tokens=${event.completionTokens ?? "?"}${event.totalTokens === undefined ? "" : ` total_tokens=${event.totalTokens}`} in ${formatDuration(event.durationMs)}${formatMemoryUsage(event.memory)}`;
  }
}

export function shouldLogServeEvent(event: ServeEvent, verbose: boolean): boolean {
  switch (event.type) {
    case "generation_start":
    case "generation_progress":
    case "generation_complete":
    case "request_error":
      return true;
    case "request_start":
    case "request_complete":
      return verbose;
  }
}

function waitForShutdown(running: RunningModelServer): Promise<void> {
  return new Promise((resolve) => {
    let closed = false;
    const close = () => {
      if (closed) {
        return;
      }
      closed = true;
      console.log("\nStopping server...");
      running.stop(true);
      resolve();
    };
    process.once("SIGINT", close);
    process.once("SIGTERM", close);
  });
}

function toServeModelOptions(options: ServeCliOptions): ServeModelOptions {
  return {
    source: options.source,
    modelId: options.modelId,
    hostname: options.hostname,
    port: options.port,
    maxGeneratedTokens: options.maxGeneratedTokens,
    maxTotalTokens: options.maxTotalTokens,
    maxBatchSize: options.maxBatchSize,
    batchWindowMs: options.batchWindowMs,
    ...(options.revision === undefined ? {} : { revision: options.revision }),
    ...(options.accessToken === undefined ? {} : { accessToken: options.accessToken }),
    ...(options.cacheDir === undefined ? {} : { cacheDir: options.cacheDir }),
    ...(options.apiKey === undefined ? {} : { apiKey: options.apiKey }),
    localFilesOnly: options.localFilesOnly,
  };
}

export async function runServeCli(
  argv: readonly string[] = Bun.argv.slice(2),
  runtime: ServeCliRuntime = {},
): Promise<void> {
  const log = runtime.log ?? console.log;
  const error = runtime.error ?? console.error;
  const exit = runtime.exit ?? process.exit;
  const startModelServer = runtime.serveModel ?? serveModel;
  const shutdown = runtime.waitForShutdown ?? waitForShutdown;
  const parsed = parseServeArgs(argv);
  if (parsed.kind === "help") {
    if (parsed.message !== undefined) {
      error(parsed.message);
      error("");
    }
    error(formatServeUsage());
    exit(parsed.exitCode);
    return;
  }

  const running = await startModelServer({
    ...toServeModelOptions(parsed.options),
    onProgress: (event) => log(formatPretrainedLoadProgress(event)),
    onEvent: (event) => {
      if (shouldLogServeEvent(event, parsed.options.verbose)) {
        log(formatServeEvent(event));
      }
    },
  });
  const warning = publicBindWarning(parsed.options);
  if (warning !== null) {
    log(warning);
  }
  log(formatServeReady(running.endpoint, parsed.options));
  await shutdown(running);
}

export async function main(argv: readonly string[] = Bun.argv.slice(2)): Promise<void> {
  await runServeCli(argv);
}

if (import.meta.main) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  });
}
