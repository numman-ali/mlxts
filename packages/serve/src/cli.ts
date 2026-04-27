#!/usr/bin/env bun

import type { PretrainedLoadProgressEvent } from "@mlxts/transformers";
import {
  formatServeUsage,
  parseServeArgs,
  type ServeCliOptions,
  type ServeCliParseResult,
} from "./cli-options";
import { type RunningModelServer, type ServeModelOptions, serveModel } from "./model-server";
import { type ServeModelsOptions, serveModels } from "./model-sources";
import type { GenerationMemoryUsage, ServeEvent } from "./types";

export type { ServeCliOptions, ServeCliParseResult };
export { formatServeUsage, parseServeArgs };

export type ServeCliRuntime = {
  serveModel?: (options: ServeModelOptions) => Promise<RunningModelServer>;
  serveModels?: (options: ServeModelsOptions) => Promise<RunningModelServer>;
  log?: (message: string) => void;
  error?: (message: string) => void;
  exit?: (code: number) => void;
  waitForShutdown?: (running: RunningModelServer) => Promise<void>;
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

function formatModelLoadProgress(
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
    `Batch scheduler: max_batch=${options.maxBatchSize} window_ms=${options.batchWindowMs}`,
    `Streaming decode interval: ${options.streamDecodeInterval} token(s)`,
    `Model execution lanes: max_in_flight=${options.maxConcurrentRequests}`,
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

function formatSchedulerCounts(
  event: Extract<ServeEvent, { type: "generation_scheduler_phase" }>,
): string {
  return `waiting=${event.waiting} prefilling=${event.prefilling} active=${event.active}/${event.maxBatchSize}`;
}

function formatSchedulerPhase(
  event: Extract<ServeEvent, { type: "generation_scheduler_phase" }>,
): string {
  switch (event.phase) {
    case "queued":
      return `[scheduler] ${event.mode} ${event.id} queued queued_ahead=${event.queuedAhead} prompt_tokens=${event.promptTokens} max_tokens=${event.maxTokens} ${formatSchedulerCounts(event)}`;
    case "prefill_start":
      return `[scheduler] ${event.mode} ${event.id} prefill_start queued=${formatDuration(event.queuedMs)} prompt_tokens=${event.promptTokens} max_tokens=${event.maxTokens} ${formatSchedulerCounts(event)}`;
    case "admitted":
      return `[scheduler] ${event.mode} admitted size=${event.batchSize} wait_ms=${event.queuedMsByRequest.map((ms) => ms.toFixed(1)).join(",")} max_tokens=${event.maxTokens} per_request=${event.maxTokensByRequest.join(",")} ids=${event.ids.join(",")} ${formatSchedulerCounts(event)}`;
    case "first_token":
      return `[scheduler] ${event.mode} ${event.id} first_token at=${formatDuration(event.schedulerMs)} queued=${formatDuration(event.queuedMs)} completion_tokens=${event.completionTokens} ${formatSchedulerCounts(event)}`;
    case "finished":
      return `[scheduler] ${event.mode} ${event.id} finished reason=${event.finishReason} elapsed=${formatDuration(event.schedulerMs)} completion_tokens=${event.completionTokens} ${formatSchedulerCounts(event)}`;
    case "cancelled":
      return `[scheduler] ${event.mode} ${event.id} cancelled elapsed=${formatDuration(event.schedulerMs)} completion_tokens=${event.completionTokens} ${formatSchedulerCounts(event)}`;
  }
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
    case "generation_route_decision":
      return `[route] ${event.id} model=${event.model} route=${event.route} eligible=${event.eligible ? "yes" : "no"} reason=${event.reason} model_type=${event.modelType} scheduler=${event.schedulerMode} cache=${event.cacheBackend} attention=${event.attentionBackend} decoding=${event.decodingBackend} max_batch_size=${event.maxBatchSize}`;
    case "generation_model_lane_wait":
      return `[queue] ${event.id} model=${event.model} lane=${event.lane} wait=${formatDuration(event.waitMs)} queued_ahead=${event.queuedAhead} in_flight_at_queue=${event.inFlightAtQueue} in_flight_at_dispatch=${event.inFlightAtDispatch} max_in_flight=${event.maxConcurrentJobs}`;
    case "generation_progress":
      return `[generation] ${event.id} progress prompt_tokens=${event.promptTokens} completion_tokens=${event.completionTokens}/${event.maxTokens}${formatMemoryUsage(event.memory)}`;
    case "generation_prefill_progress":
      return `[generation] ${event.id} prefill prompt_tokens=${event.promptTokens} prefill_tokens=${event.processedPrefillTokens}/${event.totalPrefillTokens} chunk_tokens=${event.chunkTokens}${formatMemoryUsage(event.memory)}`;
    case "generation_stream_chunk":
      return `[stream] ${event.id} chunk=${event.chunkIndex} bytes=${event.bytes} at=${formatDuration(event.elapsedMs)}`;
    case "generation_stream_end":
      return `[stream] ${event.id} ${event.result} reason=${event.finishReason} chunks=${event.chunks} output_chunks=${event.outputChunks} bytes=${event.bytes} output_bytes=${event.outputBytes}${event.ttftMs === undefined ? "" : ` ttft=${formatDuration(event.ttftMs)}`} in ${formatDuration(event.durationMs)}`;
    case "generation_batch_start":
      return `[batch] ${event.mode} model=${event.model} size=${event.batchSize} max_tokens=${event.maxTokens} per_request=${event.maxTokensByRequest.join(",")} ids=${event.ids.join(",")} started`;
    case "generation_scheduler_phase":
      return formatSchedulerPhase(event);
    case "generation_admission_batch":
      return `[batch] ${event.mode} model=${event.model} size=${event.batchSize} engine=${event.engineMode} max_tokens=${event.maxTokens} per_request=${event.maxTokensByRequest.join(",")} ids=${event.ids.join(",")} admitted`;
    case "generation_complete":
      return `[generation] ${event.id} ${event.finishReason}${event.promptTokens === undefined ? "" : ` prompt_tokens=${event.promptTokens}`} tokens=${event.completionTokens ?? "?"}${event.totalTokens === undefined ? "" : ` total_tokens=${event.totalTokens}`} in ${formatDuration(event.durationMs)}${formatMemoryUsage(event.memory)}`;
    case "generation_error":
      return `[generation:error] ${event.id} ${event.protocol} model=${event.model} ${event.code}: ${event.message} in ${formatDuration(event.durationMs)}`;
  }
}

export function shouldLogServeEvent(event: ServeEvent, verbose: boolean): boolean {
  switch (event.type) {
    case "generation_start":
    case "generation_route_decision":
    case "generation_model_lane_wait":
    case "generation_progress":
    case "generation_prefill_progress":
    case "generation_stream_end":
    case "generation_batch_start":
    case "generation_scheduler_phase":
      return true;
    case "generation_stream_chunk":
      return verbose;
    case "generation_admission_batch":
    case "generation_complete":
    case "generation_error":
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
    maxPromptTokens: options.maxPromptTokens,
    maxTotalTokens: options.maxTotalTokens,
    maxBatchSize: options.maxBatchSize,
    batchWindowMs: options.batchWindowMs,
    streamDecodeInterval: options.streamDecodeInterval,
    maxConcurrentRequests: options.maxConcurrentRequests,
    gpuMemoryUtilization: options.gpuMemoryUtilization,
    ...(options.revision === undefined ? {} : { revision: options.revision }),
    ...(options.accessToken === undefined ? {} : { accessToken: options.accessToken }),
    ...(options.cacheDir === undefined ? {} : { cacheDir: options.cacheDir }),
    ...(options.apiKey === undefined ? {} : { apiKey: options.apiKey }),
    localFilesOnly: options.localFilesOnly,
  };
}

function toServeModelsOptions(options: ServeCliOptions): ServeModelsOptions {
  return {
    models: options.models.map((model) => ({
      source: model.source,
      modelId: model.modelId,
    })),
    hostname: options.hostname,
    port: options.port,
    maxGeneratedTokens: options.maxGeneratedTokens,
    maxPromptTokens: options.maxPromptTokens,
    maxTotalTokens: options.maxTotalTokens,
    maxBatchSize: options.maxBatchSize,
    batchWindowMs: options.batchWindowMs,
    streamDecodeInterval: options.streamDecodeInterval,
    maxConcurrentRequests: options.maxConcurrentRequests,
    gpuMemoryUtilization: options.gpuMemoryUtilization,
    ...(options.revision === undefined ? {} : { revision: options.revision }),
    ...(options.accessToken === undefined ? {} : { accessToken: options.accessToken }),
    ...(options.cacheDir === undefined ? {} : { cacheDir: options.cacheDir }),
    ...(options.apiKey === undefined ? {} : { apiKey: options.apiKey }),
    localFilesOnly: options.localFilesOnly,
  };
}

async function startParsedServer(
  options: ServeCliOptions,
  runtime: ServeCliRuntime,
  log: (message: string) => void,
): Promise<RunningModelServer> {
  const onEvent = (event: ServeEvent) => {
    if (shouldLogServeEvent(event, options.verbose)) {
      log(formatServeEvent(event));
    }
  };

  if (options.models.length === 1) {
    const startModelServer = runtime.serveModel ?? serveModel;
    return startModelServer({
      ...toServeModelOptions(options),
      onProgress: (event) => log(formatPretrainedLoadProgress(event)),
      onEvent,
    });
  }

  const startModelsServer = runtime.serveModels ?? serveModels;
  return startModelsServer({
    ...toServeModelsOptions(options),
    onProgress: (event, context) =>
      log(formatModelLoadProgress(event, context.index, options.models.length, context.modelId)),
    onEvent,
  });
}

export async function runServeCli(
  argv: readonly string[] = Bun.argv.slice(2),
  runtime: ServeCliRuntime = {},
): Promise<void> {
  const log = runtime.log ?? console.log;
  const error = runtime.error ?? console.error;
  const exit = runtime.exit ?? process.exit;
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

  const running = await startParsedServer(parsed.options, runtime, log);
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
