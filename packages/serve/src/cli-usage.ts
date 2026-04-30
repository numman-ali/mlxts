import type { PretrainedLoadProgressEvent } from "@mlxts/transformers";
import type { ServeCliOptions } from "./cli-options";
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

export function formatBytes(bytes: number): string {
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

export function formatDuration(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms.toFixed(1)}ms`;
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

export function formatModelLoadProgress(
  event: PretrainedLoadProgressEvent,
  index: number,
  total: number,
  modelId: string,
): string {
  return `[load ${index + 1}/${total} ${modelId}] ${formatPretrainedLoadProgress(event)}`;
}

function authHeader(options: ServeCliOptions): string[] {
  return options.apiKey === undefined ? [] : ["-H 'authorization: Bearer <your-api-key>'"];
}

export function formatServeReady(endpoint: string, options: ServeCliOptions): string {
  const modelIds = options.models.map((model) => model.modelId);
  const servingLine =
    modelIds.length === 1
      ? `Serving ${options.modelId} at ${endpoint}`
      : `Serving ${modelIds.length} models at ${endpoint}`;
  return [
    "",
    servingLine,
    `Models: ${modelIds.join(", ")}`,
    `Generated-token limit: ${options.maxGeneratedTokens}`,
    `Prompt-token limit: ${options.maxPromptTokens}`,
    `Total-token limit: ${options.maxTotalTokens}`,
    `GPU memory budget: ${Math.round(options.gpuMemoryUtilization * 100)}%`,
    `Model load policy: ${
      options.modelLoadPolicy === "lazy"
        ? `lazy${options.modelIdleTtlMs === undefined ? "" : ` idle_ttl=${formatDuration(options.modelIdleTtlMs)}`}`
        : "eager"
    }`,
    `Model pressure policy: ${options.modelPressurePolicy}`,
    `Pinned models: ${
      options.pinnedModels.length === 0 ? "none" : options.pinnedModels.join(", ")
    }`,
    [
      `Batch scheduler: max_batch=${options.maxBatchSize}`,
      `window_ms=${options.batchWindowMs}`,
      `prefill=${options.prefillStepSize}`,
      `active_prefill=${options.activePrefillStepSize}`,
      `active_decode_quantum=${options.activeDecodeStepsPerPrefillChunk}`,
    ].join(" "),
    `Streaming decode interval: ${options.streamDecodeInterval} token(s)`,
    `Model execution lanes: max_in_flight=${options.maxConcurrentRequests}`,
    `Prompt-prefix cache entries: ${options.promptPrefixCacheMaxEntries}`,
    `Prompt-prefix cache bytes: ${
      options.promptPrefixCacheMaxBytes === undefined
        ? "unbounded"
        : formatBytes(options.promptPrefixCacheMaxBytes)
    }`,
    `Remote image hosts: ${
      options.remoteImageHosts.length === 0 ? "disabled" : options.remoteImageHosts.join(", ")
    }`,
    `Local image roots: ${
      options.localImageRoots.length === 0 ? "disabled" : options.localImageRoots.join(", ")
    }`,
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

export function formatServeUsage(): string {
  return [
    "description: Serve local or Hugging Face autoregressive models through @mlxts/serve",
    "usage[6]:",
    "  mlxts-serve <model-path-or-repo-id> [options]",
    "  mlxts-serve --model <model-path-or-repo-id> [--model <model-id=path-or-repo-id>] [options]",
    "  mlxts-serve --model-root <directory> [--model-root <directory>] [options]",
    "  mlxts-serve discover --model-root <directory> [--full]",
    "  mlxts-serve status [--base-url <url>] [--api-key <key>] [--timeout-ms <n>] [--full]",
    "  bunx @mlxts/serve <model-path-or-repo-id> [options]",
    "options[33]{flag,description}:",
    '  "--model <source|id=source>","Add a model source; repeat for multi-model serving"',
    '  "--model-root <directory>","Discover supported autoregressive checkpoints under root and org/model folders"',
    '  "--model-load-policy <eager|lazy>","Load all models at startup or on first request"',
    '  "--model-idle-ttl-ms <n>","Evict lazy-loaded idle models after n milliseconds"',
    '  "--model-pressure-policy <reject|shed_non_pinned>","Lazy-pool response when memory pressure blocks a load or request"',
    '  "--model-pressure-release-timeout-ms <n>","Wait up to n ms for pressure-cancelled lazy requests to release"',
    '  "--pin-model <id>","Keep a lazy-loaded model resident; repeat as needed"',
    '  "--model-id <id>","Served model id for positional single-model usage"',
    '  "--served-model-name <id>","Alias for --model-id"',
    `  "--host <host>","Hostname to bind; default ${DEFAULT_MODEL_SERVER_HOSTNAME}"`,
    `  "--port <port>","Port to bind; default ${DEFAULT_MODEL_SERVER_PORT}"`,
    `  "--max-generated-tokens <n>","Reject requests above this max_tokens cap; default ${DEFAULT_MODEL_SERVER_MAX_GENERATED_TOKENS}"`,
    `  "--max-prompt-tokens <n>","Reject prompts above this tokenized prompt cap; default ${DEFAULT_MODEL_SERVER_MAX_PROMPT_TOKENS}"`,
    `  "--max-total-tokens <n>","Reject prompt_tokens + max_tokens above this cap; default ${DEFAULT_MODEL_SERVER_MAX_TOTAL_TOKENS}"`,
    `  "--max-batch-size <n>","Admission micro-batch size per model instance; default ${DEFAULT_MODEL_SERVER_MAX_BATCH_SIZE}"`,
    `  "--batch-window-ms <n>","Wait window before flushing a micro-batch; default ${DEFAULT_MODEL_SERVER_BATCH_WINDOW_MS}"`,
    `  "--prefill-step-size <n>","Cold prompt-prefill chunk size; default ${DEFAULT_MODEL_SERVER_PREFILL_STEP_SIZE}"`,
    `  "--active-prefill-step-size <n>","Prompt-prefill chunk size while rows are decoding; default ${DEFAULT_MODEL_SERVER_ACTIVE_PREFILL_STEP_SIZE}"`,
    `  "--active-decode-steps-per-prefill-chunk <n>","Decode-step quantum before long prompt-prefill work resumes; default ${DEFAULT_MODEL_SERVER_ACTIVE_DECODE_STEPS_PER_PREFILL_CHUNK}"`,
    `  "--stream-decode-interval <n>","Decode/flush streaming text every n generated token; default ${DEFAULT_MODEL_SERVER_STREAM_DECODE_INTERVAL}"`,
    `  "--max-concurrent-requests <n>","Max in-flight jobs per served model; default ${DEFAULT_MODEL_SERVER_MAX_CONCURRENT_REQUESTS}"`,
    `  "--prompt-prefix-cache-max-entries <n>","Retained prompt-prefix snapshots per served model; default ${DEFAULT_MODEL_SERVER_PROMPT_PREFIX_CACHE_MAX_ENTRIES}"`,
    '  "--prompt-prefix-cache-max-bytes <n>","Estimated retained prompt-prefix snapshot bytes per served model"',
    `  "--gpu-memory-utilization <f>","Reject estimated requests above this fraction of MLX memory limit; default ${DEFAULT_MODEL_SERVER_GPU_MEMORY_UTILIZATION}"`,
    '  "--local-image-root <directory>","Allow image file_id values as relative image paths under this root; repeat as needed"',
    '  "--remote-image-host <host>","Allow remote image URLs from this exact host; repeat for redirects/CDNs"',
    '  "--revision <ref>","Hugging Face revision when source is a repo id"',
    '  "--access-token <token>","Hugging Face access token for private or gated repos"',
    '  "--cache-dir <path>","Hugging Face cache directory"',
    '  "--api-key <key>","Require Authorization: Bearer <key> for /v1 routes"',
    '  "--local-files-only","Use only already-cached/local files"',
    '  "--verbose","Log request and generation lifecycle events"',
    '  "--help","Show this help"',
  ].join("\n");
}

function toon(value: string): string {
  return JSON.stringify(value);
}

export function formatServeCliError(message: string, code = "usage"): string {
  return [
    "error:",
    `  code: ${toon(code)}`,
    `  message: ${toon(message)}`,
    "help[2]:",
    "  Run `mlxts-serve --help` for serve options",
    "  Run `mlxts-serve status --base-url <url>` to inspect a running endpoint",
  ].join("\n");
}
