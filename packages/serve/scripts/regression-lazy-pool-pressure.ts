#!/usr/bin/env bun

import { clearMemoryCache } from "@mlxts/core";
import { mkdirSync, readFileSync } from "fs";
import { dirname, join } from "path";
import { acquireRuntimeCommandLock } from "../../../scripts/runtime-command-lock";
import { resolvePretrainedSource } from "../../transformers/src";
import { estimateModelLoadMemory } from "../src/model-loading/memory-preflight";
import { serveModels } from "../src/model-loading/sources";
import { readGenerationMemoryUsage } from "../src/runtime/memory";
import type { GenerationMemoryUsage, ServeEvent } from "../src/types";

type CliOptions = {
  qwenModel: string;
  gemma4Model: string;
  reportDir: string;
  requestTimeoutMs: number;
  activeMaxTokens: number;
  blockedMaxTokens: number;
  pressureReleaseTimeoutMs: number;
  budgetMultiplier: number;
  allowDownload: boolean;
};

type RecordedServeEvent = ServeEvent & { observedAtMs: number };

type ModelMemoryEstimate = {
  source: string;
  snapshotPath: string;
  safetensorBytes: number;
  estimatedBytes: number;
};

export type LazyPoolPressureReport = {
  createdAt: string;
  activeModel: string;
  blockedModel: string;
  activeModelId: string;
  blockedModelId: string;
  gpuMemoryUtilization: number;
  memory: GenerationMemoryUsage;
  estimates: {
    active: ModelMemoryEstimate;
    blocked: ModelMemoryEstimate;
  };
  requests: {
    active: StreamProbeSummary;
    blocked: ChatProbeSummary;
  };
  pressure: {
    events: number;
    abortActiveEvents: number;
    abortedRequestIds: readonly string[];
    actions: readonly PressureActionSummary[];
  };
  metrics: {
    pressureLines: readonly string[];
  };
};

type StreamProbeSummary = {
  id?: string;
  status: number;
  chunks: number;
  bytes: number;
  firstChunkMs: number | null;
  done: boolean;
  errorCode?: string;
  error?: string;
};

type ChatProbeSummary = {
  status: number;
  outputChars: number;
  finishReason: string;
  durationMs: number;
};

type PressureActionSummary = {
  targetModel: string;
  action: "evict_idle" | "abort_active";
  reason: "model_load_memory_exceeded" | "memory_budget_exceeded";
  evictedModels: readonly string[];
  abortedRequestIds: readonly string[];
};

type FirstChunkLatch = {
  promise: Promise<void>;
  resolve(): void;
  reject(error: Error): void;
};

type StreamProbeState = {
  started: number;
  id?: string;
  errorCode?: string;
  firstChunkMs: number | null;
  chunks: number;
  bytes: number;
  buffer: string;
  resolveFirstChunk(): void;
  rejectFirstChunk(error: Error): void;
};

const ACTIVE_MODEL_ID = "gemma-pressure-active";
const BLOCKED_MODEL_ID = "qwen-pressure-blocked";
const MIN_BUDGET_MARGIN_BYTES = 512 * 1024 * 1024;

function usage(): string {
  return [
    "Usage: bun run regression:lazy-pool-pressure -- [options]",
    "",
    "Options:",
    "  --qwen-model <id>                    Qwen model id/path used as the blocked load.",
    "  --gemma4-model <id>                  Gemma model id/path used as the active request.",
    "  --report-dir <path>                  Directory for JSON evidence.",
    "  --request-timeout-ms <n>             Timeout for endpoint requests.",
    "  --active-max-tokens <n>              Tokens for the active streaming request.",
    "  --blocked-max-tokens <n>             Tokens for the blocked retry request.",
    "  --pressure-release-timeout-ms <n>    Lazy-pool release wait.",
    "  --budget-multiplier <n>              Multiplier over the larger model-load estimate.",
    "  --allow-download                     Allow Hub downloads; default is cached/local only.",
  ].join("\n");
}

function defaultOptions(): CliOptions {
  return {
    qwenModel: "mlx-community/Qwen3.6-27B-4bit",
    gemma4Model: "google/gemma-4-E2B-it",
    reportDir: ".tmp/lazy-pool-pressure",
    requestTimeoutMs: 3_600_000,
    activeMaxTokens: 2048,
    blockedMaxTokens: 8,
    pressureReleaseTimeoutMs: 120_000,
    budgetMultiplier: 1.05,
    allowDownload: false,
  };
}

function readStringFlag(args: readonly string[], index: number, flag: string): string {
  const value = args[index + 1];
  if (value === undefined || value.startsWith("--")) {
    throw new Error(`${flag} requires a value.\n\n${usage()}`);
  }
  return value;
}

function readPositiveIntegerFlag(args: readonly string[], index: number, flag: string): number {
  const raw = readStringFlag(args, index, flag);
  if (!/^[1-9]\d*$/.test(raw)) {
    throw new Error(`${flag} must be a positive integer.`);
  }
  return Number.parseInt(raw, 10);
}

function readPositiveNumberFlag(args: readonly string[], index: number, flag: string): number {
  const raw = readStringFlag(args, index, flag);
  if (!/^(?:[1-9]\d*|0?\.\d+|[1-9]\d*\.\d+)$/.test(raw)) {
    throw new Error(`${flag} must be a positive number.`);
  }
  const value = Number.parseFloat(raw);
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${flag} must be a positive number.`);
  }
  return value;
}

export function parseLazyPoolPressureArgs(argv: readonly string[]): CliOptions {
  const options = defaultOptions();

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
      case "--help":
      case "-h":
        console.log(usage());
        process.exit(0);
        return options;
      case "--qwen-model":
        options.qwenModel = readStringFlag(argv, index, arg);
        index += 1;
        break;
      case "--gemma4-model":
        options.gemma4Model = readStringFlag(argv, index, arg);
        index += 1;
        break;
      case "--report-dir":
        options.reportDir = readStringFlag(argv, index, arg);
        index += 1;
        break;
      case "--request-timeout-ms":
        options.requestTimeoutMs = readPositiveIntegerFlag(argv, index, arg);
        index += 1;
        break;
      case "--active-max-tokens":
        options.activeMaxTokens = readPositiveIntegerFlag(argv, index, arg);
        index += 1;
        break;
      case "--blocked-max-tokens":
        options.blockedMaxTokens = readPositiveIntegerFlag(argv, index, arg);
        index += 1;
        break;
      case "--pressure-release-timeout-ms":
        options.pressureReleaseTimeoutMs = readPositiveIntegerFlag(argv, index, arg);
        index += 1;
        break;
      case "--budget-multiplier":
        options.budgetMultiplier = readPositiveNumberFlag(argv, index, arg);
        index += 1;
        break;
      case "--allow-download":
        options.allowDownload = true;
        break;
      default:
        throw new Error(`Unknown option: ${arg}\n\n${usage()}`);
    }
  }

  return options;
}

export function pressureGpuMemoryUtilization(options: {
  memory: GenerationMemoryUsage;
  activeEstimatedBytes: number;
  blockedEstimatedBytes: number;
  multiplier: number;
}): number {
  const largestEstimate = Math.max(options.activeEstimatedBytes, options.blockedEstimatedBytes);
  const margin = Math.max(MIN_BUDGET_MARGIN_BYTES, Math.ceil(largestEstimate * 0.02));
  const budgetBytes = Math.ceil(largestEstimate * options.multiplier + margin);
  if (budgetBytes >= options.memory.limitBytes) {
    throw new Error(
      [
        "lazy-pool-pressure: model estimates exceed the available MLX memory limit.",
        `budget=${formatBytes(budgetBytes)}`,
        `limit=${formatBytes(options.memory.limitBytes)}`,
      ].join(" "),
    );
  }
  return Number((budgetBytes / options.memory.limitBytes).toFixed(6));
}

async function modelEstimate(source: string, allowDownload: boolean): Promise<ModelMemoryEstimate> {
  const snapshotPath = await resolvePretrainedSource(source, { localFilesOnly: !allowDownload });
  const estimate = estimateModelLoadMemory(snapshotPath);
  if (estimate === undefined) {
    throw new Error(`lazy-pool-pressure: cannot estimate model-load memory for ${snapshotPath}.`);
  }
  return {
    source,
    snapshotPath,
    safetensorBytes: estimate.safetensorBytes,
    estimatedBytes: estimate.estimatedBytes,
  };
}

function chatBody(model: string, maxTokens: number, stream: boolean) {
  return {
    model,
    messages: [
      {
        role: "user",
        content: "Continue with a concise numbered list about local inference reliability.",
      },
    ],
    max_tokens: maxTokens,
    temperature: 0,
    ignore_eos: true,
    chat_template_kwargs: { enable_thinking: false },
    ...(stream ? { stream: true, stream_options: { include_usage: true } } : {}),
  };
}

function ssePayloads(frame: string): string[] {
  return frame
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice("data:".length).trimStart())
    .filter((payload) => payload !== "" && payload !== "[DONE]");
}

function consumeSseFrames(buffer: string, onFrame: (frame: string) => void): { remainder: string } {
  let cursor = 0;
  while (true) {
    const next = buffer.indexOf("\n\n", cursor);
    if (next === -1) {
      return { remainder: buffer.slice(cursor) };
    }
    onFrame(buffer.slice(cursor, next));
    cursor = next + 2;
  }
}

function createFirstChunkLatch(): FirstChunkLatch {
  let resolveFirstChunk: () => void = () => {};
  let rejectFirstChunk: (error: Error) => void = () => {};
  let settled = false;
  const firstChunk = new Promise<void>((resolve) => {
    resolveFirstChunk = () => {
      if (!settled) {
        settled = true;
        resolve();
      }
    };
  });
  const firstChunkFailure = new Promise<never>((_, reject) => {
    rejectFirstChunk = (error: Error) => {
      if (!settled) {
        settled = true;
        reject(error);
      }
    };
  });
  return {
    promise: Promise.race([firstChunk, firstChunkFailure]),
    resolve: resolveFirstChunk,
    reject: rejectFirstChunk,
  };
}

function recordStreamPayload(state: StreamProbeState, payload: string): void {
  const parsed = JSON.parse(payload) as {
    id?: string;
    choices?: Array<{ delta?: { content?: string; reasoning_content?: string } }>;
    error?: { code?: string };
  };
  if (state.id === undefined && typeof parsed.id === "string") {
    state.id = parsed.id;
  }
  if (typeof parsed.error?.code === "string") {
    state.errorCode = parsed.error.code;
    return;
  }
  const delta = parsed.choices?.[0]?.delta;
  const text = delta?.content ?? delta?.reasoning_content;
  if (typeof text !== "string" || text === "") {
    return;
  }
  state.chunks += 1;
  state.firstChunkMs ??= performance.now() - state.started;
  state.resolveFirstChunk();
}

function consumeStreamProbeFrames(state: StreamProbeState): void {
  const consumed = consumeSseFrames(state.buffer, (frame) => {
    for (const payload of ssePayloads(frame)) {
      recordStreamPayload(state, payload);
    }
  });
  state.buffer = consumed.remainder;
}

function streamProbeSummary(
  status: number,
  state: StreamProbeState,
  done: boolean,
  error?: string,
): StreamProbeSummary {
  return {
    ...(state.id === undefined ? {} : { id: state.id }),
    status,
    chunks: state.chunks,
    bytes: state.bytes,
    firstChunkMs: state.firstChunkMs,
    done,
    ...(state.errorCode === undefined ? {} : { errorCode: state.errorCode }),
    ...(error === undefined ? {} : { error }),
  };
}

async function readStreamingProbe(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  decoder: TextDecoder,
  responseStatus: number,
  state: StreamProbeState,
): Promise<StreamProbeSummary> {
  try {
    while (true) {
      const read = await reader.read();
      if (read.done) {
        break;
      }
      state.bytes += read.value.byteLength;
      state.buffer += decoder.decode(read.value, { stream: true });
      consumeStreamProbeFrames(state);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    state.rejectFirstChunk(new Error(`lazy-pool-pressure: active stream failed: ${message}`));
    return streamProbeSummary(responseStatus, state, false, message);
  }
  if (state.chunks === 0) {
    state.rejectFirstChunk(
      new Error("lazy-pool-pressure: active stream ended before first chunk."),
    );
  }
  return streamProbeSummary(responseStatus, state, true);
}

async function startStreamingProbe(
  endpoint: string,
  modelId: string,
  maxTokens: number,
  requestTimeoutMs: number,
): Promise<{ firstChunk: Promise<void>; done: Promise<StreamProbeSummary> }> {
  const started = performance.now();
  const response = await fetch(`${endpoint}/v1/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    signal: AbortSignal.timeout(requestTimeoutMs),
    body: JSON.stringify(chatBody(modelId, maxTokens, true)),
  });
  if (!response.ok) {
    throw new Error(
      `lazy-pool-pressure: active stream failed before SSE: ${await response.text()}`,
    );
  }
  if (response.body === null) {
    throw new Error("lazy-pool-pressure: active stream response had no body.");
  }

  const firstChunk = createFirstChunkLatch();
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const state: StreamProbeState = {
    started,
    firstChunkMs: null,
    chunks: 0,
    bytes: 0,
    buffer: "",
    resolveFirstChunk: firstChunk.resolve,
    rejectFirstChunk: firstChunk.reject,
  };
  const done = readStreamingProbe(reader, decoder, response.status, state);

  return { firstChunk: firstChunk.promise, done };
}

async function runChatProbe(
  endpoint: string,
  modelId: string,
  maxTokens: number,
  requestTimeoutMs: number,
): Promise<ChatProbeSummary> {
  const started = performance.now();
  const response = await fetch(`${endpoint}/v1/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    signal: AbortSignal.timeout(requestTimeoutMs),
    body: JSON.stringify(chatBody(modelId, maxTokens, false)),
  });
  const body = (await response.json()) as {
    choices?: Array<{ message?: { content?: string | null }; finish_reason?: string | null }>;
    error?: unknown;
  };
  if (!response.ok) {
    throw new Error(`lazy-pool-pressure: request failed: ${JSON.stringify(body)}`);
  }
  const choice = body.choices?.[0];
  return {
    status: response.status,
    outputChars: choice?.message?.content?.length ?? 0,
    finishReason: choice?.finish_reason ?? "unknown",
    durationMs: performance.now() - started,
  };
}

async function pressureMetricLines(endpoint: string): Promise<string[]> {
  const response = await fetch(`${endpoint}/metrics`);
  const text = await response.text();
  return text.split("\n").filter((line) => line.startsWith("mlxts_serve_model_pool_pressure_"));
}

export function assertLazyPoolPressureReport(report: LazyPoolPressureReport): void {
  if (report.pressure.abortActiveEvents < 1) {
    throw new Error("lazy-pool-pressure: expected at least one active-request pressure abort.");
  }
  if (report.pressure.abortedRequestIds.length < 1) {
    throw new Error("lazy-pool-pressure: pressure abort did not name an aborted request.");
  }
  if (
    report.requests.active.id === undefined ||
    !report.pressure.abortedRequestIds.includes(report.requests.active.id)
  ) {
    throw new Error("lazy-pool-pressure: pressure abort did not target the active stream id.");
  }
  const abortAction = report.pressure.actions.find((action) => action.action === "abort_active");
  if (
    abortAction === undefined ||
    abortAction.targetModel !== report.blockedModelId ||
    abortAction.reason !== "model_load_memory_exceeded"
  ) {
    throw new Error("lazy-pool-pressure: pressure abort action did not target the blocked load.");
  }
  if (report.requests.blocked.status !== 200 || report.requests.blocked.outputChars <= 0) {
    throw new Error("lazy-pool-pressure: blocked request did not complete after pressure relief.");
  }
  if (report.metrics.pressureLines.length === 0) {
    throw new Error("lazy-pool-pressure: metrics did not expose model-pool pressure counters.");
  }
}

function pressureEvents(events: readonly RecordedServeEvent[]) {
  return events.filter((event) => event.type === "model_pool_pressure");
}

function pressureActionSummaries(
  events: readonly Extract<RecordedServeEvent, { type: "model_pool_pressure" }>[],
): PressureActionSummary[] {
  return events.map((event) => ({
    targetModel: event.targetModel,
    action: event.action,
    reason: event.reason,
    evictedModels: [...event.evictedModels],
    abortedRequestIds: [...event.abortedRequestIds],
  }));
}

function formatBytes(bytes: number): string {
  return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(2)}GB` : `${(bytes / 1e6).toFixed(2)}MB`;
}

async function writeReport(reportDir: string, report: LazyPoolPressureReport): Promise<string> {
  const path = join(reportDir, "lazy-pool-pressure.json");
  mkdirSync(dirname(path), { recursive: true });
  await Bun.write(path, `${JSON.stringify(report, null, 2)}\n`);
  return path;
}

function formatSuccess(reportPath: string, report: LazyPoolPressureReport): string {
  return [
    "lazy_pool_pressure:",
    `  report: ${reportPath}`,
    `  active_model: ${report.activeModel}`,
    `  blocked_model: ${report.blockedModel}`,
    `  gpu_memory_utilization: ${report.gpuMemoryUtilization}`,
    `  pressure_events: ${report.pressure.events}`,
    `  aborted_requests: ${report.pressure.abortedRequestIds.length}`,
    `  blocked_output_chars: ${report.requests.blocked.outputChars}`,
    `  metrics_lines: ${report.metrics.pressureLines.length}`,
  ].join("\n");
}

function formatError(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  return [`error: ${message}`, "help: bun run regression:lazy-pool-pressure -- --help"].join("\n");
}

export function readLazyPoolPressureReport(path: string): LazyPoolPressureReport {
  return JSON.parse(readFileSync(path, "utf8")) as LazyPoolPressureReport;
}

export async function runLazyPoolPressureRegression(
  options: CliOptions,
): Promise<{ path: string; report: LazyPoolPressureReport }> {
  clearMemoryCache();
  const memory = readGenerationMemoryUsage();
  if (memory === undefined) {
    throw new Error("lazy-pool-pressure: MLX memory telemetry is unavailable.");
  }
  const [activeEstimate, blockedEstimate] = await Promise.all([
    modelEstimate(options.gemma4Model, options.allowDownload),
    modelEstimate(options.qwenModel, options.allowDownload),
  ]);
  const gpuMemoryUtilization = pressureGpuMemoryUtilization({
    memory,
    activeEstimatedBytes: activeEstimate.estimatedBytes,
    blockedEstimatedBytes: blockedEstimate.estimatedBytes,
    multiplier: options.budgetMultiplier,
  });

  const events: RecordedServeEvent[] = [];
  const server = await serveModels({
    models: [
      { source: activeEstimate.snapshotPath, modelId: ACTIVE_MODEL_ID, localFilesOnly: true },
      { source: blockedEstimate.snapshotPath, modelId: BLOCKED_MODEL_ID, localFilesOnly: true },
    ],
    modelLoadPolicy: "lazy",
    modelPressurePolicy: "shed_non_pinned",
    modelPressureReleaseTimeoutMs: options.pressureReleaseTimeoutMs,
    gpuMemoryUtilization,
    maxGeneratedTokens: Math.max(options.activeMaxTokens, options.blockedMaxTokens),
    maxPromptTokens: 1024,
    maxTotalTokens: 1024 + Math.max(options.activeMaxTokens, options.blockedMaxTokens),
    maxBatchSize: 1,
    port: 0,
    onEvent(event) {
      events.push({ ...event, observedAtMs: performance.now() });
    },
  });

  try {
    console.error("lazy-pool-pressure: starting active Gemma stream");
    const active = await startStreamingProbe(
      server.endpoint,
      ACTIVE_MODEL_ID,
      options.activeMaxTokens,
      options.requestTimeoutMs,
    );
    await active.firstChunk;
    console.error("lazy-pool-pressure: triggering blocked Qwen load");
    const blocked = await runChatProbe(
      server.endpoint,
      BLOCKED_MODEL_ID,
      options.blockedMaxTokens,
      options.requestTimeoutMs,
    );
    const activeSummary = await active.done;
    const metricLines = await pressureMetricLines(server.endpoint);
    const pressure = pressureEvents(events);
    const report: LazyPoolPressureReport = {
      createdAt: new Date().toISOString(),
      activeModel: options.gemma4Model,
      blockedModel: options.qwenModel,
      activeModelId: ACTIVE_MODEL_ID,
      blockedModelId: BLOCKED_MODEL_ID,
      gpuMemoryUtilization,
      memory,
      estimates: { active: activeEstimate, blocked: blockedEstimate },
      requests: { active: activeSummary, blocked },
      pressure: {
        events: pressure.length,
        abortActiveEvents: pressure.filter((event) => event.action === "abort_active").length,
        abortedRequestIds: pressure.flatMap((event) => [...event.abortedRequestIds]),
        actions: pressureActionSummaries(pressure),
      },
      metrics: { pressureLines: metricLines },
    };
    assertLazyPoolPressureReport(report);
    const path = await writeReport(options.reportDir, report);
    return { path, report };
  } finally {
    server.stop(true);
  }
}

async function runLockedLazyPoolPressureRegression(argv: readonly string[]) {
  const options = parseLazyPoolPressureArgs(argv);
  using _runtimeLock = acquireRuntimeCommandLock("regression:lazy-pool-pressure");
  return runLazyPoolPressureRegression(options);
}

if (import.meta.main) {
  runLockedLazyPoolPressureRegression(Bun.argv.slice(2))
    .then(({ path, report }) => console.log(formatSuccess(path, report)))
    .catch((error) => {
      console.log(formatError(error));
      process.exit(1);
    });
}
