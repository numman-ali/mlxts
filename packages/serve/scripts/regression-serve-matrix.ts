#!/usr/bin/env bun

import { existsSync, mkdirSync, readFileSync } from "fs";
import { join } from "path";
import { acquireRuntimeCommandLock } from "../../../scripts/runtime-command-lock";
import type { BenchmarkReport, TrialMetrics } from "./benchmark-serve";
import {
  expectedCompletionTokensForRung,
  formatServeBenchmarkRung,
  type ProtocolMode,
  rungConcurrency,
} from "./benchmark-serve-options";

type CliOptions = {
  realModels: boolean;
  fairnessSmoke: boolean;
  capabilitySmoke: boolean;
  qwenModel: string;
  gemma4Model: string;
  reportDir: string;
  allowDownload: boolean;
  requestTimeoutMs: number;
};

type ServeRegressionCommand = { kind: "help" } | { kind: "run"; options: CliOptions };

type ServeRegressionRunReport = {
  label: string;
  modelId: string;
  rung: string;
  protocol: ProtocolMode;
  stream: boolean;
  reportPath: string;
};

type ServeRegressionResult = {
  focusedChecks: "passed";
  realModels: boolean;
  reports: ServeRegressionRunReport[];
};

type ServeRegressionRuntime = {
  stdout?: (text: string) => void;
  stderr?: (text: string) => void;
  runRegression?: (
    options: CliOptions,
    progress: (text: string) => void,
  ) => Promise<ServeRegressionResult>;
};

export type ServeRegressionBudget = {
  minCompletionTps: number;
  minPostTtftCompletionTps?: number;
  maxPeakMemoryGb: number;
  maxActiveDeltaGb: number;
  minCompletionTokenRatio: number;
  minStreamChunks?: number;
  minStreamBytes?: number;
  expectEveryRequestStreamed?: boolean;
  expectEveryRequestOutputStreamed?: boolean;
  expectEveryServerRequestStreamed?: boolean;
  expectEveryServerRequestOutputStreamed?: boolean;
  maxMeanTtftMs?: number;
  maxClientRequestTtftMs?: number;
  maxObservedStreamChunkGapMs?: number;
  maxServerSchedulerQueuedMs?: number;
  expectedRoute?: string;
  expectedReason?: string;
  minRouteDecisions?: number;
  minServerRequests?: number;
  expectedAdmissionBatches?: number;
  expectedStaticBatches?: number;
  expectedStaticBatchRows?: number;
  minContinuousAdmissions?: number;
  expectedContinuousAdmissions?: number;
  minContinuousAdmissionRows?: number;
  expectedContinuousAdmissionRows?: number;
  minContinuousSchedulerPhases?: number;
  expectedContinuousSchedulerPhases?: number;
  expectedMaxGenerationBatchSize?: number;
  expectSchedulerTokenPressure?: boolean;
  minPromptCacheHits?: number;
  minPromptCacheReadTokens?: number;
  minPromptCacheWrites?: number;
  minPromptCacheWriteTokens?: number;
  minModelLaneWaitEvents?: number;
  minModelLaneBusyWaitEvents?: number;
  requestBudgets?: ServeRequestBudget[];
};

export type ServeRequestBudget = {
  label?: string;
  index?: number;
  promptTokens?: number;
  completionTokens?: number;
  maxClientTtftMs?: number;
  maxClientStreamChunkGapMs?: number;
  maxServerSchedulerQueuedMs?: number;
  maxServerStreamTtftMs?: number;
  maxServerSilentEventGapMs?: number;
  maxServerFirstPrefillProgressMs?: number;
  minServerPrefillEvents?: number;
};

type ServeRegressionSpec = {
  label: string;
  model: string;
  modelId: string;
  rungs?: string;
  mixedRungs?: string;
  protocol?: ProtocolMode;
  maxPromptTokens?: number;
  maxTotalTokens?: number;
  stream: boolean;
  ignoreEos: boolean;
  greedy?: boolean;
  requestStaggerMs?: number;
  budget: ServeRegressionBudget;
};

const FOCUSED_TESTS = [
  "packages/serve/src/http/server.test.ts",
  "packages/serve/src/streaming/writer-openai.test.ts",
  "packages/serve/src/engine/engine.test.ts",
  "packages/serve/src/model-loading/server.test.ts",
  "packages/serve/src/admission/batching.test.ts",
  "packages/serve/src/protocols/openai-completions.test.ts",
  "packages/serve/src/protocols/openai-chat-completions.test.ts",
  "packages/serve/src/protocols/openai-responses.test.ts",
  "packages/serve/src/protocols/anthropic-messages.test.ts",
  "packages/serve/scripts/benchmark-serve-options.test.ts",
  "packages/serve/scripts/benchmark-serve-completions.test.ts",
  "packages/serve/scripts/benchmark-serve.test.ts",
  "packages/serve/scripts/regression-agent-cache.test.ts",
  "packages/serve/scripts/regression-serve-matrix.test.ts",
];

class ServeRegressionUsageError extends Error {}

export function formatServeRegressionUsage(): string {
  return [
    "description: Run the @mlxts/serve regression matrix and optional real-model smoke benchmarks",
    "usage[3]:",
    "  bun run --filter '@mlxts/serve' regression:serve",
    "  bun run packages/serve/scripts/regression-serve-matrix.ts --real-models",
    "  bun run packages/serve/scripts/regression-serve-matrix.ts --capability-smoke",
    "options[9]{flag,description}:",
    '  "--real-models","Run cached Qwen/Gemma endpoint smoke benchmarks"',
    '  "--fairness-smoke","Add mixed long-prefill/short-arrival guardrails; implies --real-models"',
    '  "--capability-smoke","Add longer output/context rungs; implies --real-models"',
    '  "--qwen-model <id>","Qwen model id/path"',
    '  "--gemma4-model <id>","Gemma 4 model id/path"',
    '  "--report-dir <path>","Directory for benchmark JSON evidence"',
    '  "--request-timeout-ms <n>","Client timeout per benchmark request; default 3600000"',
    '  "--allow-download","Allow Hub downloads; default is cached/local only"',
    '  "--help","Show this help"',
    "exit_codes[3]{code,meaning}:",
    '  0,"regression passed"',
    '  1,"runtime or regression failure"',
    '  2,"usage error"',
  ].join("\n");
}

function defaultOptions(): CliOptions {
  return {
    realModels: false,
    fairnessSmoke: false,
    capabilitySmoke: false,
    qwenModel: "mlx-community/Qwen3.6-27B-4bit",
    gemma4Model: "google/gemma-4-E2B-it",
    reportDir: ".tmp/serve-regression",
    allowDownload: false,
    requestTimeoutMs: 3_600_000,
  };
}

function readStringFlag(args: readonly string[], index: number, flag: string): string {
  const value = args[index + 1];
  if (value === undefined || value.trim() === "" || value.startsWith("-")) {
    throw new ServeRegressionUsageError(`${flag} requires a value.`);
  }
  return value;
}

function readPositiveIntegerFlag(args: readonly string[], index: number, flag: string): number {
  const rawValue = args[index + 1]?.trim();
  if (rawValue === undefined || rawValue === "" || rawValue.startsWith("--")) {
    throw new ServeRegressionUsageError(`${flag} requires a value.`);
  }
  const value = /^\d+$/.test(rawValue) ? Number(rawValue) : Number.NaN;
  if (!Number.isInteger(value) || value <= 0) {
    throw new ServeRegressionUsageError(`${flag} must be a positive integer.`);
  }
  return value;
}

export function parseServeRegressionArgs(argv: readonly string[]): ServeRegressionCommand {
  if (argv.length === 1 && (argv[0] === "--help" || argv[0] === "-h")) {
    return { kind: "help" };
  }
  const options = defaultOptions();

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === undefined) {
      throw new ServeRegressionUsageError("argument parsing reached an empty slot.");
    }
    switch (arg) {
      case "--help":
      case "-h":
        return { kind: "help" };
      case "--real-models":
        options.realModels = true;
        break;
      case "--fairness-smoke":
        options.fairnessSmoke = true;
        options.realModels = true;
        break;
      case "--capability-smoke":
        options.capabilitySmoke = true;
        options.fairnessSmoke = true;
        options.realModels = true;
        break;
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
      case "--allow-download":
        options.allowDownload = true;
        break;
      default:
        throw new ServeRegressionUsageError(
          arg.startsWith("-") ? `unknown option "${arg}".` : `unexpected argument "${arg}".`,
        );
    }
  }

  return { kind: "run", options };
}

function inheritedStringEnv(): Record<string, string> {
  return Object.fromEntries(
    Object.entries(process.env).filter((entry): entry is [string, string] => {
      const value = entry[1];
      return typeof value === "string";
    }),
  );
}

async function pipeReadableToProgress(
  stream: ReadableStream<Uint8Array> | null,
  progress: (text: string) => void,
): Promise<void> {
  if (stream === null) {
    return;
  }
  const decoder = new TextDecoder();
  let pending = "";
  for await (const chunk of stream) {
    pending += decoder.decode(chunk, { stream: true });
    let newline = pending.indexOf("\n");
    while (newline !== -1) {
      const line = pending.slice(0, newline);
      if (line !== "") {
        progress(line);
      }
      pending = pending.slice(newline + 1);
      newline = pending.indexOf("\n");
    }
  }
  pending += decoder.decode();
  if (pending !== "") {
    progress(pending);
  }
}

async function runCommand(
  label: string,
  args: readonly string[],
  progress: (text: string) => void,
): Promise<void> {
  progress(`[serve-regression] ${label}: ${args.join(" ")}`);
  const child = Bun.spawn([...args], {
    cwd: new URL("../../..", import.meta.url).pathname,
    env: inheritedStringEnv(),
    stdout: "pipe",
    stderr: "pipe",
  });
  const stdout = pipeReadableToProgress(child.stdout, progress);
  const stderr = pipeReadableToProgress(child.stderr, progress);
  const exitCode = await child.exited;
  await Promise.all([stdout, stderr]);
  if (exitCode !== 0) {
    throw new Error(`[serve-regression] ${label} failed with exit code ${exitCode}.`);
  }
}

async function runFocusedUnitChecks(progress: (text: string) => void): Promise<void> {
  await runCommand("focused unit checks", ["bun", "test", ...FOCUSED_TESTS], progress);
}

function sanitizeLabel(label: string): string {
  return label.replace(/[^a-zA-Z0-9_.-]+/g, "-").replace(/^-|-$/g, "");
}

function readBenchmarkReport(path: string): BenchmarkReport {
  const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
  if (typeof parsed !== "object" || parsed === null || !("rungs" in parsed)) {
    throw new Error(`[serve-regression] malformed benchmark report: ${path}`);
  }
  return parsed as BenchmarkReport;
}

function isAllowedFinishReason(reason: string | undefined, protocolMode: ProtocolMode): boolean {
  if (reason === "length" || reason === "stop" || reason === "eos") {
    return true;
  }
  if (protocolMode !== "anthropic") {
    return false;
  }
  return reason === "end_turn" || reason === "max_tokens" || reason === "stop_sequence";
}

function assertFinishReasons(
  label: string,
  metrics: TrialMetrics,
  protocolMode: ProtocolMode,
): void {
  const badReasons = metrics.finishReasons.filter(
    (reason) => !isAllowedFinishReason(reason, protocolMode),
  );
  if (badReasons.length > 0) {
    throw new Error(
      `[serve-regression] ${label} reported unexpected finish reasons: ${badReasons.join(",")}`,
    );
  }
}

function throughputFailures(metrics: TrialMetrics, budget: ServeRegressionBudget): string[] {
  const failures: string[] = [];
  if (metrics.completionTps < budget.minCompletionTps) {
    failures.push(
      `completion_tps ${metrics.completionTps.toFixed(3)} < ${budget.minCompletionTps.toFixed(3)}`,
    );
  }
  if (budget.minPostTtftCompletionTps === undefined) {
    return failures;
  }
  const postTtftTps = metrics.meanPostTtftCompletionTps;
  if (postTtftTps === null || postTtftTps < budget.minPostTtftCompletionTps) {
    failures.push(
      `post_ttft_completion_tps ${postTtftTps?.toFixed(3) ?? "n/a"} < ${budget.minPostTtftCompletionTps.toFixed(
        3,
      )}`,
    );
  }
  return failures;
}

function memoryFailures(metrics: TrialMetrics, budget: ServeRegressionBudget): string[] {
  const failures: string[] = [];
  if (metrics.peakMemoryGb > budget.maxPeakMemoryGb) {
    failures.push(
      `peak_memory ${metrics.peakMemoryGb.toFixed(3)}GB > ${budget.maxPeakMemoryGb.toFixed(3)}GB`,
    );
  }
  if (metrics.activeDeltaGb > budget.maxActiveDeltaGb) {
    failures.push(
      `active_delta ${metrics.activeDeltaGb.toFixed(3)}GB > ${budget.maxActiveDeltaGb.toFixed(
        3,
      )}GB`,
    );
  }
  return failures;
}

function tokenFailures(
  metrics: TrialMetrics,
  expectedCompletionTokens: number,
  budget: ServeRegressionBudget,
): string[] {
  const minCompletionTokens = expectedCompletionTokens * budget.minCompletionTokenRatio;
  return metrics.completionTokens < minCompletionTokens
    ? [
        `completion_tokens ${metrics.completionTokens.toFixed(0)} < ${minCompletionTokens.toFixed(
          0,
        )}`,
      ]
    : [];
}

function perRequestStreamLifecycleFailures(
  metrics: TrialMetrics,
  concurrency: number,
  protocolMode: ProtocolMode,
): string[] {
  const failures: string[] = [];
  if (metrics.requests.length < concurrency) {
    failures.push(`requests ${metrics.requests.length} < concurrency ${concurrency}`);
  }
  const nonStreamingRequests = metrics.requests.filter(
    (request) =>
      request.streamBytes <= 0 ||
      request.completionTokens <= 0 ||
      !isAllowedFinishReason(request.finishReason, protocolMode),
  );
  if (nonStreamingRequests.length > 0) {
    const ids = nonStreamingRequests.map((request) => request.id ?? `index:${request.index}`);
    failures.push(`requests missing per-request SSE lifecycle evidence: ${ids.join(",")}`);
  }
  return failures;
}

function perRequestOutputStreamFailures(metrics: TrialMetrics): string[] {
  const missingOutput = metrics.requests.filter(
    (request) => request.streamChunks <= 0 || request.ttftMs === null,
  );
  if (missingOutput.length === 0) {
    return [];
  }
  const ids = missingOutput.map((request) => request.id ?? `index:${request.index}`);
  return [`requests missing per-request output SSE evidence: ${ids.join(",")}`];
}

function serverRequestStreamLifecycleFailures(
  metrics: TrialMetrics,
  concurrency: number,
  protocolMode: ProtocolMode,
): string[] {
  const failures: string[] = [];
  if (metrics.serverRequests.length < concurrency) {
    failures.push(`server_requests ${metrics.serverRequests.length} < concurrency ${concurrency}`);
  }
  const missingLifecycleEvidence = metrics.serverRequests.filter(
    (request) =>
      request.serverStreamEndEvents !== 1 ||
      (request.serverStreamChunks ?? 0) <= 0 ||
      (request.serverStreamBytes ?? 0) <= 0 ||
      request.serverStreamDurationMs === null ||
      request.serverStreamResult !== "completed" ||
      !isAllowedFinishReason(request.serverStreamFinishReason, protocolMode),
  );
  if (missingLifecycleEvidence.length > 0) {
    const ids = missingLifecycleEvidence.map((request) => request.id);
    failures.push(
      `server_requests missing server-side stream lifecycle evidence: ${ids.join(",")}`,
    );
  }
  return failures;
}

function serverRequestOutputStreamFailures(metrics: TrialMetrics): string[] {
  const missingOutputEvidence = metrics.serverRequests.filter(
    (request) =>
      request.serverStreamChunkEvents <= 0 ||
      (request.serverStreamOutputChunks ?? 0) <= 0 ||
      (request.serverStreamOutputBytes ?? 0) <= 0 ||
      request.serverStreamTtftMs === null,
  );
  if (missingOutputEvidence.length === 0) {
    return [];
  }
  const ids = missingOutputEvidence.map((request) => request.id);
  return [`server_requests missing server-side output stream evidence: ${ids.join(",")}`];
}

function aggregateStreamFailures(metrics: TrialMetrics, budget: ServeRegressionBudget): string[] {
  const failures: string[] = [];
  if (budget.minStreamChunks !== undefined && metrics.streamChunks < budget.minStreamChunks) {
    failures.push(
      `stream_chunks ${metrics.streamChunks.toFixed(0)} < ${budget.minStreamChunks.toFixed(0)}`,
    );
  }
  if (budget.minStreamBytes !== undefined && metrics.streamBytes < budget.minStreamBytes) {
    failures.push(
      `stream_bytes ${metrics.streamBytes.toFixed(0)} < ${budget.minStreamBytes.toFixed(0)}`,
    );
  }
  return failures;
}

function streamTimingFailures(metrics: TrialMetrics, budget: ServeRegressionBudget): string[] {
  const failures: string[] = [];
  if (budget.maxMeanTtftMs !== undefined) {
    const ttftMs = metrics.meanTtftMs;
    if (ttftMs === null || ttftMs > budget.maxMeanTtftMs) {
      failures.push(
        `mean_ttft_ms ${ttftMs?.toFixed(1) ?? "n/a"} > ${budget.maxMeanTtftMs.toFixed(1)}`,
      );
    }
  }
  if (budget.maxObservedStreamChunkGapMs !== undefined) {
    const gapMs = metrics.maxStreamChunkGapMs;
    if (gapMs === null || gapMs > budget.maxObservedStreamChunkGapMs) {
      failures.push(
        `max_stream_chunk_gap_ms ${gapMs?.toFixed(1) ?? "n/a"} > ${budget.maxObservedStreamChunkGapMs.toFixed(
          1,
        )}`,
      );
    }
  }
  if (budget.maxClientRequestTtftMs !== undefined) {
    const maxClientRequestTtftMs = budget.maxClientRequestTtftMs;
    const slowRequests = metrics.requests.filter(
      (request) => request.ttftMs === null || request.ttftMs > maxClientRequestTtftMs,
    );
    if (slowRequests.length > 0) {
      const ids = slowRequests.map((request) => request.id ?? `index:${request.index}`);
      failures.push(
        `client_request_ttft_ms exceeded ${maxClientRequestTtftMs.toFixed(1)} for ${ids.join(",")}`,
      );
    }
  }
  return failures;
}

function streamFailures(
  metrics: TrialMetrics,
  concurrency: number,
  budget: ServeRegressionBudget,
  protocolMode: ProtocolMode,
): string[] {
  const failures = [
    ...aggregateStreamFailures(metrics, budget),
    ...streamTimingFailures(metrics, budget),
  ];
  if (budget.expectEveryRequestStreamed) {
    failures.push(...perRequestStreamLifecycleFailures(metrics, concurrency, protocolMode));
  }
  if (budget.expectEveryRequestOutputStreamed) {
    failures.push(...perRequestOutputStreamFailures(metrics));
  }
  if (budget.expectEveryServerRequestStreamed) {
    failures.push(...serverRequestStreamLifecycleFailures(metrics, concurrency, protocolMode));
  }
  if (budget.expectEveryServerRequestOutputStreamed) {
    failures.push(...serverRequestOutputStreamFailures(metrics));
  }
  return failures;
}

function routeFailures(metrics: TrialMetrics, budget: ServeRegressionBudget): string[] {
  const failures: string[] = [];
  if (
    budget.minRouteDecisions !== undefined &&
    metrics.routeDecisions.length < budget.minRouteDecisions
  ) {
    failures.push(`route_decisions ${metrics.routeDecisions.length} < ${budget.minRouteDecisions}`);
  }
  if (budget.expectedRoute !== undefined) {
    const unexpectedRoutes = metrics.routeDecisions.filter(
      (decision) => decision.route !== budget.expectedRoute,
    );
    if (unexpectedRoutes.length > 0 || metrics.routeDecisions.length === 0) {
      failures.push(`route_decisions did not stay on ${budget.expectedRoute}`);
    }
  }
  if (budget.expectedReason !== undefined) {
    const unexpectedReasons = metrics.routeDecisions.filter(
      (decision) => decision.reason !== budget.expectedReason,
    );
    if (unexpectedReasons.length > 0 || metrics.routeDecisions.length === 0) {
      failures.push(`route_reasons did not stay on ${budget.expectedReason}`);
    }
  }
  return failures;
}

function evidenceFailures(metrics: TrialMetrics, budget: ServeRegressionBudget): string[] {
  const failures: string[] = [];
  if (
    budget.minServerRequests !== undefined &&
    metrics.serverRequests.length < budget.minServerRequests
  ) {
    failures.push(`server_requests ${metrics.serverRequests.length} < ${budget.minServerRequests}`);
  }
  if (budget.expectSchedulerTokenPressure) {
    const missingTokenPressure = metrics.serverRequests.filter(
      (request) =>
        request.route === "continuous" &&
        (request.schedulerScheduledPromptTokens === null ||
          request.schedulerMaxScheduledPromptTokens === null ||
          request.schedulerScheduledCompletionTokens === null ||
          request.schedulerMaxScheduledCompletionTokens === null ||
          request.schedulerScheduledTotalTokens === null ||
          request.schedulerMaxScheduledTotalTokens === null),
    );
    if (missingTokenPressure.length > 0) {
      failures.push(
        `server_requests missing scheduler token pressure: ${missingTokenPressure
          .map((request) => request.id)
          .join(",")}`,
      );
    }
    const missingMemoryPressure = metrics.serverRequests.filter(
      (request) =>
        request.route === "continuous" &&
        (request.schedulerScheduledMemoryBytes === null ||
          request.schedulerMaxScheduledMemoryBytes === null),
    );
    if (missingMemoryPressure.length > 0) {
      failures.push(
        `server_requests missing scheduler memory pressure: ${missingMemoryPressure
          .map((request) => request.id)
          .join(",")}`,
      );
    }
  }
  if (budget.maxServerSchedulerQueuedMs !== undefined) {
    const maxServerSchedulerQueuedMs = budget.maxServerSchedulerQueuedMs;
    const slowSchedulerRequests = metrics.serverRequests.filter(
      (request) =>
        request.route === "continuous" &&
        (request.schedulerQueuedMs === null ||
          request.schedulerQueuedMs > maxServerSchedulerQueuedMs),
    );
    if (slowSchedulerRequests.length > 0) {
      failures.push(
        `server_requests exceeded scheduler queued budget ${maxServerSchedulerQueuedMs.toFixed(
          1,
        )}ms: ${slowSchedulerRequests.map((request) => request.id).join(",")}`,
      );
    }
  }
  return failures;
}

function requestMatchesBudget(
  request: TrialMetrics["requests"][number],
  budget: ServeRequestBudget,
): boolean {
  if (budget.index !== undefined && request.index !== budget.index) {
    return false;
  }
  if (budget.promptTokens !== undefined && request.promptTokens !== budget.promptTokens) {
    return false;
  }
  if (
    budget.completionTokens !== undefined &&
    request.completionTokens !== budget.completionTokens
  ) {
    return false;
  }
  return true;
}

function requestBudgetLabel(budget: ServeRequestBudget): string {
  if (budget.label !== undefined) {
    return budget.label;
  }
  const selectors = [
    budget.index === undefined ? undefined : `index:${budget.index}`,
    budget.promptTokens === undefined ? undefined : `prompt:${budget.promptTokens}`,
    budget.completionTokens === undefined ? undefined : `completion:${budget.completionTokens}`,
  ].filter((selector): selector is string => selector !== undefined);
  return selectors.length === 0 ? "unscoped" : selectors.join(",");
}

function requestBudgetHasSelector(budget: ServeRequestBudget): boolean {
  return (
    budget.index !== undefined ||
    budget.promptTokens !== undefined ||
    budget.completionTokens !== undefined
  );
}

function requestBudgetNeedsServerRequest(budget: ServeRequestBudget): boolean {
  return (
    budget.maxServerSchedulerQueuedMs !== undefined ||
    budget.maxServerStreamTtftMs !== undefined ||
    budget.maxServerSilentEventGapMs !== undefined ||
    budget.maxServerFirstPrefillProgressMs !== undefined ||
    budget.minServerPrefillEvents !== undefined
  );
}

function metricBudgetFailure(
  label: string,
  metricName: string,
  actual: number | null | undefined,
  expectedMax: number | undefined,
  id: string,
): string | null {
  if (expectedMax === undefined) {
    return null;
  }
  if (actual !== null && actual !== undefined && actual <= expectedMax) {
    return null;
  }
  return `request_budget ${label} ${metricName} ${actual?.toFixed(1) ?? "n/a"}ms > ${expectedMax.toFixed(
    1,
  )}ms for ${id}`;
}

function clientRequestBudgetFailures(
  label: string,
  request: TrialMetrics["requests"][number],
  budget: ServeRequestBudget,
): string[] {
  const id = request.id ?? `index:${request.index}`;
  return [
    metricBudgetFailure(label, "client ttft", request.ttftMs, budget.maxClientTtftMs, id),
    metricBudgetFailure(
      label,
      "client stream gap",
      request.maxStreamChunkGapMs,
      budget.maxClientStreamChunkGapMs,
      id,
    ),
  ].filter((failure): failure is string => failure !== null);
}

function serverRequestBudgetFailures(
  label: string,
  request: TrialMetrics["requests"][number],
  serverRequest: TrialMetrics["serverRequests"][number] | undefined,
  budget: ServeRequestBudget,
): string[] {
  const id = request.id ?? `index:${request.index}`;
  if (
    requestBudgetNeedsServerRequest(budget) &&
    (request.id === undefined || serverRequest === undefined)
  ) {
    return [`request_budget ${label} found no matching server request for ${id}`];
  }
  const failures = [
    metricBudgetFailure(
      label,
      "server scheduler queued",
      serverRequest?.schedulerQueuedMs,
      budget.maxServerSchedulerQueuedMs,
      id,
    ),
    metricBudgetFailure(
      label,
      "server stream ttft",
      serverRequest?.serverStreamTtftMs,
      budget.maxServerStreamTtftMs,
      id,
    ),
    metricBudgetFailure(
      label,
      "server silent gap",
      serverRequest?.maxSilentEventGapMs,
      budget.maxServerSilentEventGapMs,
      id,
    ),
    metricBudgetFailure(
      label,
      "server first prefill progress",
      serverRequest?.firstPrefillProgressMs,
      budget.maxServerFirstPrefillProgressMs,
      id,
    ),
  ].filter((failure): failure is string => failure !== null);
  const minServerPrefillEvents = budget.minServerPrefillEvents;
  if (
    minServerPrefillEvents !== undefined &&
    (serverRequest?.prefillEvents ?? 0) < minServerPrefillEvents
  ) {
    const actual = serverRequest?.prefillEvents ?? 0;
    return [
      ...failures,
      `request_budget ${label} server prefill events ${actual.toFixed(0)} < ${minServerPrefillEvents.toFixed(
        0,
      )} for ${id}`,
    ];
  }
  return failures;
}

function failuresForMatchedRequest(
  metrics: TrialMetrics,
  label: string,
  request: TrialMetrics["requests"][number],
  budget: ServeRequestBudget,
): string[] {
  const serverRequest = metrics.serverRequests.find((server) => server.id === request.id);
  return [
    ...clientRequestBudgetFailures(label, request, budget),
    ...serverRequestBudgetFailures(label, request, serverRequest, budget),
  ];
}

function requestBudgetFailures(metrics: TrialMetrics, budget: ServeRegressionBudget): string[] {
  const failures: string[] = [];
  for (const requestBudget of budget.requestBudgets ?? []) {
    const label = requestBudgetLabel(requestBudget);
    if (!requestBudgetHasSelector(requestBudget)) {
      failures.push(`request_budget ${label} must set at least one selector`);
      continue;
    }

    const requests = metrics.requests.filter((request) =>
      requestMatchesBudget(request, requestBudget),
    );
    if (requests.length === 0) {
      failures.push(`request_budget ${label} matched no client requests`);
      continue;
    }

    for (const request of requests) {
      failures.push(...failuresForMatchedRequest(metrics, label, request, requestBudget));
    }
  }
  return failures;
}

type CounterExpectation = {
  name: string;
  value: number;
  minimum?: number | undefined;
  expected?: number | undefined;
};

function counterFailures(expectations: readonly CounterExpectation[]): string[] {
  const failures: string[] = [];
  for (const expectation of expectations) {
    if (expectation.minimum !== undefined && expectation.value < expectation.minimum) {
      failures.push(
        `${expectation.name} ${expectation.value.toFixed(0)} < ${expectation.minimum.toFixed(0)}`,
      );
    }
    if (expectation.expected !== undefined && expectation.value !== expectation.expected) {
      failures.push(
        `${expectation.name} ${expectation.value.toFixed(0)} != ${expectation.expected.toFixed(0)}`,
      );
    }
  }
  return failures;
}

function batchCounterFailures(metrics: TrialMetrics, budget: ServeRegressionBudget): string[] {
  return counterFailures([
    {
      name: "admission_batches",
      value: metrics.admissionBatches,
      expected: budget.expectedAdmissionBatches,
    },
    {
      name: "static_batches",
      value: metrics.staticBatches,
      expected: budget.expectedStaticBatches,
    },
    {
      name: "static_batch_rows",
      value: metrics.staticBatchRows,
      expected: budget.expectedStaticBatchRows,
    },
    {
      name: "continuous_admissions",
      value: metrics.continuousAdmissions,
      minimum: budget.minContinuousAdmissions,
      expected: budget.expectedContinuousAdmissions,
    },
    {
      name: "continuous_admission_rows",
      value: metrics.continuousAdmissionRows,
      minimum: budget.minContinuousAdmissionRows,
      expected: budget.expectedContinuousAdmissionRows,
    },
    {
      name: "continuous_scheduler_phases",
      value: metrics.continuousSchedulerPhases,
      minimum: budget.minContinuousSchedulerPhases,
      expected: budget.expectedContinuousSchedulerPhases,
    },
    {
      name: "max_generation_batch",
      value: metrics.maxGenerationBatchSize,
      expected: budget.expectedMaxGenerationBatchSize,
    },
  ]);
}

function modelLaneWaitFailures(metrics: TrialMetrics, budget: ServeRegressionBudget): string[] {
  const failures: string[] = [];
  const waitEvents = metrics.serverRequests.filter((request) => request.modelLaneWaitMs !== null);
  if (
    budget.minModelLaneWaitEvents !== undefined &&
    waitEvents.length < budget.minModelLaneWaitEvents
  ) {
    failures.push(`model_lane_wait_events ${waitEvents.length} < ${budget.minModelLaneWaitEvents}`);
  }

  const busyWaitEvents = waitEvents.filter(
    (request) => (request.modelLaneInFlightAtQueue ?? 0) > 0,
  );
  if (
    budget.minModelLaneBusyWaitEvents !== undefined &&
    busyWaitEvents.length < budget.minModelLaneBusyWaitEvents
  ) {
    failures.push(
      `model_lane_busy_wait_events ${busyWaitEvents.length} < ${budget.minModelLaneBusyWaitEvents}`,
    );
  }
  return failures;
}

function promptCacheFailures(metrics: TrialMetrics, budget: ServeRegressionBudget): string[] {
  return counterFailures([
    {
      name: "prompt_cache_hits",
      value: metrics.promptCacheHits,
      minimum: budget.minPromptCacheHits,
    },
    {
      name: "prompt_cache_read_tokens",
      value: metrics.promptCacheReadTokens,
      minimum: budget.minPromptCacheReadTokens,
    },
    {
      name: "prompt_cache_writes",
      value: metrics.promptCacheWrites,
      minimum: budget.minPromptCacheWrites,
    },
    {
      name: "prompt_cache_write_tokens",
      value: metrics.promptCacheWriteTokens,
      minimum: budget.minPromptCacheWriteTokens,
    },
  ]);
}

export function assertServeReportBudget(
  label: string,
  report: BenchmarkReport,
  budget: ServeRegressionBudget,
): void {
  for (const rungReport of report.rungs) {
    const { rung, averages } = rungReport;
    const rungLabel = formatServeBenchmarkRung(rung);
    const concurrency = rungConcurrency(rung);
    const failures = [
      ...throughputFailures(averages, budget),
      ...memoryFailures(averages, budget),
      ...tokenFailures(averages, expectedCompletionTokensForRung(rung), budget),
      ...streamFailures(averages, concurrency, budget, report.protocolMode),
      ...routeFailures(averages, budget),
      ...evidenceFailures(averages, budget),
      ...batchCounterFailures(averages, budget),
      ...promptCacheFailures(averages, budget),
      ...modelLaneWaitFailures(averages, budget),
      ...requestBudgetFailures(averages, budget),
    ];

    assertFinishReasons(`${label} ${rungLabel}`, averages, report.protocolMode);
    if (failures.length > 0) {
      throw new Error(`[serve-regression] ${label} ${rungLabel} failed: ${failures.join("; ")}.`);
    }
  }
}

function baseSpecs(options: CliOptions): ServeRegressionSpec[] {
  return [
    {
      label: "qwen36-completions-stream",
      model: options.qwenModel,
      modelId: options.qwenModel,
      rungs: "1024x128@1",
      stream: true,
      ignoreEos: true,
      budget: {
        minCompletionTps: 12,
        minPostTtftCompletionTps: 20,
        maxPeakMemoryGb: 22,
        maxActiveDeltaGb: 1,
        minCompletionTokenRatio: 0.98,
        minStreamChunks: 1,
        minStreamBytes: 1,
        expectEveryRequestStreamed: true,
        expectEveryRequestOutputStreamed: true,
        expectEveryServerRequestStreamed: true,
        expectEveryServerRequestOutputStreamed: true,
        maxMeanTtftMs: 8_000,
        maxObservedStreamChunkGapMs: 1_000,
        expectedRoute: "continuous",
        expectedReason: "eligible",
        minRouteDecisions: 1,
        minServerRequests: 1,
        expectedAdmissionBatches: 0,
        expectedStaticBatches: 0,
        minContinuousAdmissions: 1,
        minContinuousAdmissionRows: 1,
        minContinuousSchedulerPhases: 4,
        expectedMaxGenerationBatchSize: 1,
        expectSchedulerTokenPressure: true,
        minModelLaneWaitEvents: 0,
      },
    },
    {
      label: "qwen36-completions-continuous",
      model: options.qwenModel,
      modelId: options.qwenModel,
      rungs: "128x32@2",
      stream: false,
      ignoreEos: true,
      budget: {
        minCompletionTps: 8,
        maxPeakMemoryGb: 22,
        maxActiveDeltaGb: 1,
        minCompletionTokenRatio: 0.98,
        expectedRoute: "continuous",
        expectedReason: "eligible",
        minRouteDecisions: 2,
        minServerRequests: 2,
        expectedAdmissionBatches: 0,
        expectedStaticBatches: 0,
        minContinuousAdmissions: 1,
        minContinuousAdmissionRows: 2,
        minContinuousSchedulerPhases: 7,
        expectedMaxGenerationBatchSize: 2,
        expectSchedulerTokenPressure: true,
        minModelLaneWaitEvents: 0,
      },
    },
    {
      label: "qwen36-completions-stream-continuous",
      model: options.qwenModel,
      modelId: options.qwenModel,
      rungs: "128x32@2",
      stream: true,
      ignoreEos: true,
      budget: {
        minCompletionTps: 12,
        minPostTtftCompletionTps: 20,
        maxPeakMemoryGb: 22,
        maxActiveDeltaGb: 1,
        minCompletionTokenRatio: 0.98,
        minStreamChunks: 8,
        minStreamBytes: 1,
        expectEveryRequestStreamed: true,
        expectEveryRequestOutputStreamed: true,
        expectEveryServerRequestStreamed: true,
        expectEveryServerRequestOutputStreamed: true,
        maxMeanTtftMs: 5_000,
        maxObservedStreamChunkGapMs: 1_000,
        expectedRoute: "continuous",
        expectedReason: "eligible",
        minRouteDecisions: 2,
        minServerRequests: 2,
        expectedAdmissionBatches: 0,
        expectedStaticBatches: 0,
        expectedContinuousAdmissions: 1,
        expectedContinuousAdmissionRows: 2,
        expectedContinuousSchedulerPhases: 7,
        expectedMaxGenerationBatchSize: 2,
        expectSchedulerTokenPressure: true,
        minModelLaneWaitEvents: 0,
      },
    },
    {
      label: "qwen36-completions-stream-continuous-at4",
      model: options.qwenModel,
      modelId: options.qwenModel,
      rungs: "128x16@4",
      stream: true,
      ignoreEos: true,
      budget: {
        minCompletionTps: 10,
        minPostTtftCompletionTps: 6,
        maxPeakMemoryGb: 22,
        maxActiveDeltaGb: 1,
        minCompletionTokenRatio: 0.98,
        minStreamChunks: 16,
        minStreamBytes: 1,
        expectEveryRequestStreamed: true,
        expectEveryRequestOutputStreamed: true,
        expectEveryServerRequestStreamed: true,
        expectEveryServerRequestOutputStreamed: true,
        maxMeanTtftMs: 6_000,
        maxClientRequestTtftMs: 8_000,
        maxObservedStreamChunkGapMs: 1_500,
        maxServerSchedulerQueuedMs: 8_000,
        expectedRoute: "continuous",
        expectedReason: "eligible",
        minRouteDecisions: 4,
        minServerRequests: 4,
        expectedAdmissionBatches: 0,
        expectedStaticBatches: 0,
        minContinuousAdmissions: 1,
        minContinuousAdmissionRows: 4,
        minContinuousSchedulerPhases: 13,
        expectedMaxGenerationBatchSize: 4,
        expectSchedulerTokenPressure: true,
        minModelLaneWaitEvents: 0,
      },
    },
    {
      label: "qwen36-completions-stream-continuous-at8",
      model: options.qwenModel,
      modelId: options.qwenModel,
      rungs: "128x16@8",
      stream: true,
      ignoreEos: true,
      budget: {
        minCompletionTps: 10,
        minPostTtftCompletionTps: 6,
        maxPeakMemoryGb: 23,
        maxActiveDeltaGb: 1,
        minCompletionTokenRatio: 0.98,
        minStreamChunks: 32,
        minStreamBytes: 1,
        expectEveryRequestStreamed: true,
        expectEveryRequestOutputStreamed: true,
        expectEveryServerRequestStreamed: true,
        expectEveryServerRequestOutputStreamed: true,
        maxMeanTtftMs: 9_000,
        maxClientRequestTtftMs: 10_000,
        maxObservedStreamChunkGapMs: 1_500,
        maxServerSchedulerQueuedMs: 10_000,
        expectedRoute: "continuous",
        expectedReason: "eligible",
        minRouteDecisions: 8,
        minServerRequests: 8,
        expectedAdmissionBatches: 0,
        expectedStaticBatches: 0,
        minContinuousAdmissions: 1,
        minContinuousAdmissionRows: 8,
        minContinuousSchedulerPhases: 20,
        expectedMaxGenerationBatchSize: 8,
        expectSchedulerTokenPressure: true,
        minModelLaneWaitEvents: 0,
      },
    },
    {
      label: "qwen36-completions-staggered-continuous",
      model: options.qwenModel,
      modelId: options.qwenModel,
      rungs: "128x32@2",
      stream: false,
      ignoreEos: true,
      requestStaggerMs: 100,
      budget: {
        minCompletionTps: 8,
        maxPeakMemoryGb: 22,
        maxActiveDeltaGb: 1,
        minCompletionTokenRatio: 0.98,
        expectedRoute: "continuous",
        expectedReason: "eligible",
        minRouteDecisions: 2,
        minServerRequests: 2,
        expectedAdmissionBatches: 0,
        expectedStaticBatches: 0,
        expectedContinuousAdmissions: 2,
        expectedContinuousAdmissionRows: 3,
        expectedContinuousSchedulerPhases: 9,
        expectedMaxGenerationBatchSize: 2,
        expectSchedulerTokenPressure: true,
        minModelLaneWaitEvents: 0,
      },
    },
    {
      label: "qwen36-completions-model-defaults-continuous",
      model: options.qwenModel,
      modelId: options.qwenModel,
      rungs: "128x16@2",
      stream: false,
      ignoreEos: true,
      greedy: false,
      budget: {
        minCompletionTps: 4,
        maxPeakMemoryGb: 22,
        maxActiveDeltaGb: 1,
        minCompletionTokenRatio: 0.98,
        expectedRoute: "continuous",
        expectedReason: "eligible",
        minRouteDecisions: 2,
        minServerRequests: 2,
        expectedAdmissionBatches: 0,
        expectedStaticBatches: 0,
        expectedContinuousAdmissions: 1,
        expectedContinuousAdmissionRows: 2,
        expectedContinuousSchedulerPhases: 7,
        expectedMaxGenerationBatchSize: 2,
        expectSchedulerTokenPressure: true,
        minModelLaneWaitEvents: 0,
      },
    },
    {
      label: "qwen36-completions-model-defaults-stream-continuous",
      model: options.qwenModel,
      modelId: options.qwenModel,
      rungs: "128x16@2",
      stream: true,
      ignoreEos: true,
      greedy: false,
      budget: {
        minCompletionTps: 4,
        minPostTtftCompletionTps: 20,
        maxPeakMemoryGb: 22,
        maxActiveDeltaGb: 1,
        minCompletionTokenRatio: 0.98,
        minStreamChunks: 1,
        minStreamBytes: 1,
        expectEveryRequestStreamed: true,
        expectEveryServerRequestStreamed: true,
        maxMeanTtftMs: 5_000,
        maxObservedStreamChunkGapMs: 1_000,
        expectedRoute: "continuous",
        expectedReason: "eligible",
        minRouteDecisions: 2,
        minServerRequests: 2,
        expectedAdmissionBatches: 0,
        expectedStaticBatches: 0,
        expectedContinuousAdmissions: 1,
        expectedContinuousAdmissionRows: 2,
        expectedContinuousSchedulerPhases: 7,
        expectedMaxGenerationBatchSize: 2,
        expectSchedulerTokenPressure: true,
        minModelLaneWaitEvents: 0,
      },
    },
    {
      label: "gemma4-completions-stream",
      model: options.gemma4Model,
      modelId: "gemma-local",
      rungs: "1024x128@1",
      stream: true,
      ignoreEos: true,
      budget: {
        minCompletionTps: 45,
        minPostTtftCompletionTps: 60,
        maxPeakMemoryGb: 13,
        maxActiveDeltaGb: 1,
        minCompletionTokenRatio: 0.98,
        minStreamChunks: 1,
        minStreamBytes: 1,
        expectEveryRequestStreamed: true,
        expectEveryRequestOutputStreamed: true,
        expectEveryServerRequestStreamed: true,
        expectEveryServerRequestOutputStreamed: true,
        maxMeanTtftMs: 1_000,
        maxObservedStreamChunkGapMs: 1_000,
        expectedRoute: "continuous",
        expectedReason: "eligible",
        minRouteDecisions: 1,
        minServerRequests: 1,
        expectedAdmissionBatches: 0,
        expectedStaticBatches: 0,
        minContinuousAdmissions: 1,
        minContinuousAdmissionRows: 1,
        minContinuousSchedulerPhases: 4,
        expectedMaxGenerationBatchSize: 1,
        expectSchedulerTokenPressure: true,
        minModelLaneWaitEvents: 0,
      },
    },
    {
      label: "gemma4-completions-model-defaults-continuous",
      model: options.gemma4Model,
      modelId: "gemma-local",
      rungs: "128x16@2",
      stream: false,
      ignoreEos: true,
      greedy: false,
      budget: {
        minCompletionTps: 10,
        maxPeakMemoryGb: 13,
        maxActiveDeltaGb: 1,
        minCompletionTokenRatio: 0.98,
        expectedRoute: "continuous",
        expectedReason: "eligible",
        minRouteDecisions: 2,
        minServerRequests: 2,
        expectedAdmissionBatches: 0,
        expectedStaticBatches: 0,
        expectedContinuousAdmissions: 1,
        expectedContinuousAdmissionRows: 2,
        expectedContinuousSchedulerPhases: 7,
        expectedMaxGenerationBatchSize: 2,
        expectSchedulerTokenPressure: true,
        minModelLaneWaitEvents: 0,
      },
    },
    {
      label: "gemma4-completions-model-defaults-stream-continuous",
      model: options.gemma4Model,
      modelId: "gemma-local",
      rungs: "128x16@2",
      stream: true,
      ignoreEos: true,
      greedy: false,
      budget: {
        minCompletionTps: 10,
        minPostTtftCompletionTps: 20,
        maxPeakMemoryGb: 13,
        maxActiveDeltaGb: 1,
        minCompletionTokenRatio: 0.98,
        minStreamChunks: 1,
        minStreamBytes: 1,
        expectEveryRequestStreamed: true,
        expectEveryServerRequestStreamed: true,
        maxMeanTtftMs: 1_000,
        maxObservedStreamChunkGapMs: 1_000,
        expectedRoute: "continuous",
        expectedReason: "eligible",
        minRouteDecisions: 2,
        minServerRequests: 2,
        expectedAdmissionBatches: 0,
        expectedStaticBatches: 0,
        expectedContinuousAdmissions: 1,
        expectedContinuousAdmissionRows: 2,
        expectedContinuousSchedulerPhases: 7,
        expectedMaxGenerationBatchSize: 2,
        expectSchedulerTokenPressure: true,
        minModelLaneWaitEvents: 0,
      },
    },
    {
      label: "gemma4-completions-stream-continuous",
      model: options.gemma4Model,
      modelId: "gemma-local",
      rungs: "128x32@2",
      stream: true,
      ignoreEos: true,
      budget: {
        minCompletionTps: 30,
        minPostTtftCompletionTps: 20,
        maxPeakMemoryGb: 13,
        maxActiveDeltaGb: 1,
        minCompletionTokenRatio: 0.98,
        minStreamChunks: 8,
        minStreamBytes: 1,
        expectEveryRequestStreamed: true,
        expectEveryRequestOutputStreamed: true,
        expectEveryServerRequestStreamed: true,
        expectEveryServerRequestOutputStreamed: true,
        maxMeanTtftMs: 1_000,
        maxObservedStreamChunkGapMs: 1_000,
        expectedRoute: "continuous",
        expectedReason: "eligible",
        minRouteDecisions: 2,
        minServerRequests: 2,
        expectedAdmissionBatches: 0,
        expectedStaticBatches: 0,
        expectedContinuousAdmissions: 1,
        expectedContinuousAdmissionRows: 2,
        expectedContinuousSchedulerPhases: 7,
        expectedMaxGenerationBatchSize: 2,
        expectSchedulerTokenPressure: true,
        minModelLaneWaitEvents: 0,
      },
    },
    {
      label: "gemma4-completions-stream-continuous-at4",
      model: options.gemma4Model,
      modelId: "gemma-local",
      rungs: "128x16@4",
      stream: true,
      ignoreEos: true,
      budget: {
        minCompletionTps: 40,
        minPostTtftCompletionTps: 20,
        maxPeakMemoryGb: 13,
        maxActiveDeltaGb: 1,
        minCompletionTokenRatio: 0.98,
        minStreamChunks: 16,
        minStreamBytes: 1,
        expectEveryRequestStreamed: true,
        expectEveryRequestOutputStreamed: true,
        expectEveryServerRequestStreamed: true,
        expectEveryServerRequestOutputStreamed: true,
        maxMeanTtftMs: 1_500,
        maxClientRequestTtftMs: 2_000,
        maxObservedStreamChunkGapMs: 1_000,
        maxServerSchedulerQueuedMs: 2_000,
        expectedRoute: "continuous",
        expectedReason: "eligible",
        minRouteDecisions: 4,
        minServerRequests: 4,
        expectedAdmissionBatches: 0,
        expectedStaticBatches: 0,
        minContinuousAdmissions: 1,
        minContinuousAdmissionRows: 4,
        minContinuousSchedulerPhases: 13,
        expectedMaxGenerationBatchSize: 4,
        expectSchedulerTokenPressure: true,
        minModelLaneWaitEvents: 0,
      },
    },
    {
      label: "gemma4-completions-stream-continuous-at8",
      model: options.gemma4Model,
      modelId: "gemma-local",
      rungs: "128x16@8",
      stream: true,
      ignoreEos: true,
      budget: {
        minCompletionTps: 60,
        minPostTtftCompletionTps: 20,
        maxPeakMemoryGb: 13,
        maxActiveDeltaGb: 1,
        minCompletionTokenRatio: 0.98,
        minStreamChunks: 32,
        minStreamBytes: 1,
        expectEveryRequestStreamed: true,
        expectEveryRequestOutputStreamed: true,
        expectEveryServerRequestStreamed: true,
        expectEveryServerRequestOutputStreamed: true,
        maxMeanTtftMs: 2_000,
        maxClientRequestTtftMs: 2_500,
        maxObservedStreamChunkGapMs: 1_000,
        maxServerSchedulerQueuedMs: 2_500,
        expectedRoute: "continuous",
        expectedReason: "eligible",
        minRouteDecisions: 8,
        minServerRequests: 8,
        expectedAdmissionBatches: 0,
        expectedStaticBatches: 0,
        minContinuousAdmissions: 1,
        minContinuousAdmissionRows: 8,
        minContinuousSchedulerPhases: 20,
        expectedMaxGenerationBatchSize: 8,
        expectSchedulerTokenPressure: true,
        minModelLaneWaitEvents: 0,
      },
    },
    {
      label: "gemma4-completions-continuous",
      model: options.gemma4Model,
      modelId: "gemma-local",
      rungs: "128x32@2",
      stream: false,
      ignoreEos: true,
      budget: {
        minCompletionTps: 20,
        maxPeakMemoryGb: 13,
        maxActiveDeltaGb: 1,
        minCompletionTokenRatio: 0.98,
        expectedRoute: "continuous",
        expectedReason: "eligible",
        minRouteDecisions: 2,
        minServerRequests: 2,
        expectedAdmissionBatches: 0,
        expectedStaticBatches: 0,
        expectedContinuousAdmissions: 1,
        expectedContinuousSchedulerPhases: 7,
        expectedMaxGenerationBatchSize: 2,
        expectSchedulerTokenPressure: true,
        minModelLaneWaitEvents: 0,
      },
    },
  ];
}

export function protocolHealthBudget(model: "qwen" | "gemma"): ServeRegressionBudget {
  return {
    minCompletionTps: model === "qwen" ? 0.5 : 2,
    maxPeakMemoryGb: model === "qwen" ? 22 : 13,
    maxActiveDeltaGb: 1,
    minCompletionTokenRatio: 0.05,
    minStreamChunks: 1,
    minStreamBytes: 1,
    expectEveryRequestStreamed: true,
    expectEveryServerRequestStreamed: true,
    maxMeanTtftMs: model === "qwen" ? 8_000 : 2_000,
    maxObservedStreamChunkGapMs: 1_500,
    expectedRoute: "continuous",
    expectedReason: "eligible",
    minRouteDecisions: 1,
    minServerRequests: 1,
    expectedAdmissionBatches: 0,
    expectedStaticBatches: 0,
    expectedContinuousAdmissions: 1,
    expectedContinuousAdmissionRows: 1,
    expectedContinuousSchedulerPhases: 5,
    expectedMaxGenerationBatchSize: 1,
    expectSchedulerTokenPressure: true,
    minPromptCacheHits: 1,
    minPromptCacheReadTokens: 1,
    minModelLaneWaitEvents: 0,
  };
}

function protocolHealthSpecs(options: CliOptions): ServeRegressionSpec[] {
  const protocols: Array<Exclude<ProtocolMode, "completions">> = ["chat", "responses", "anthropic"];
  const textProtocolAdmission = {
    maxPromptTokens: 512,
    maxTotalTokens: 1024,
  };
  return [
    ...protocols.map((protocol) => ({
      label: `qwen36-${protocol}-stream`,
      model: options.qwenModel,
      modelId: options.qwenModel,
      protocol,
      ...textProtocolAdmission,
      rungs: "128x16@1",
      stream: true,
      ignoreEos: false,
      budget: protocolHealthBudget("qwen"),
    })),
    ...protocols.map((protocol) => ({
      label: `gemma4-${protocol}-stream`,
      model: options.gemma4Model,
      modelId: "gemma-local",
      protocol,
      ...textProtocolAdmission,
      rungs: "128x16@1",
      stream: true,
      ignoreEos: false,
      budget: protocolHealthBudget("gemma"),
    })),
  ];
}

function capabilitySpecs(options: CliOptions): ServeRegressionSpec[] {
  return [
    {
      label: "qwen36-long-output-stream",
      model: options.qwenModel,
      modelId: options.qwenModel,
      rungs: "1024x1024@1",
      stream: true,
      ignoreEos: true,
      budget: {
        minCompletionTps: 18,
        minPostTtftCompletionTps: 20,
        maxPeakMemoryGb: 23,
        maxActiveDeltaGb: 1.5,
        minCompletionTokenRatio: 0.98,
        minStreamChunks: 1,
        minStreamBytes: 1,
        expectEveryRequestStreamed: true,
        expectEveryRequestOutputStreamed: true,
        expectEveryServerRequestStreamed: true,
        expectEveryServerRequestOutputStreamed: true,
        maxMeanTtftMs: 8_000,
        maxObservedStreamChunkGapMs: 1_000,
        expectedRoute: "continuous",
        expectedReason: "eligible",
        minRouteDecisions: 1,
        minServerRequests: 1,
        expectedAdmissionBatches: 0,
        expectedStaticBatches: 0,
        minContinuousAdmissions: 1,
        minContinuousAdmissionRows: 1,
        minContinuousSchedulerPhases: 4,
        expectedMaxGenerationBatchSize: 1,
        expectSchedulerTokenPressure: true,
        minModelLaneWaitEvents: 0,
      },
    },
    {
      label: "qwen36-long-context-stream",
      model: options.qwenModel,
      modelId: options.qwenModel,
      rungs: "32768x128@1",
      stream: true,
      ignoreEos: true,
      budget: {
        minCompletionTps: 0.1,
        minPostTtftCompletionTps: 9,
        maxPeakMemoryGb: 32,
        maxActiveDeltaGb: 1,
        minCompletionTokenRatio: 0.98,
        minStreamChunks: 1,
        minStreamBytes: 1,
        expectEveryRequestStreamed: true,
        expectEveryRequestOutputStreamed: true,
        expectEveryServerRequestStreamed: true,
        expectEveryServerRequestOutputStreamed: true,
        maxObservedStreamChunkGapMs: 2_000,
        expectedRoute: "continuous",
        expectedReason: "eligible",
        minRouteDecisions: 1,
        minServerRequests: 1,
        expectedAdmissionBatches: 0,
        expectedStaticBatches: 0,
        minContinuousAdmissions: 1,
        minContinuousAdmissionRows: 1,
        minContinuousSchedulerPhases: 4,
        expectedMaxGenerationBatchSize: 1,
        expectSchedulerTokenPressure: true,
        minModelLaneWaitEvents: 0,
      },
    },
  ];
}

function fairnessSpecs(options: CliOptions): ServeRegressionSpec[] {
  return [
    {
      label: "qwen36-mixed-long-short-staggered-stream",
      model: options.qwenModel,
      modelId: options.qwenModel,
      mixedRungs: "32768x128+128x32",
      stream: true,
      ignoreEos: true,
      requestStaggerMs: 100,
      budget: {
        minCompletionTps: 0.1,
        minPostTtftCompletionTps: 9,
        maxPeakMemoryGb: 33,
        maxActiveDeltaGb: 1.5,
        minCompletionTokenRatio: 0.98,
        minStreamChunks: 2,
        minStreamBytes: 1,
        expectEveryRequestStreamed: true,
        expectEveryRequestOutputStreamed: true,
        expectEveryServerRequestStreamed: true,
        expectEveryServerRequestOutputStreamed: true,
        maxObservedStreamChunkGapMs: 1_000,
        expectedRoute: "continuous",
        expectedReason: "eligible",
        minRouteDecisions: 2,
        minServerRequests: 2,
        expectedAdmissionBatches: 0,
        expectedStaticBatches: 0,
        minContinuousAdmissions: 2,
        minContinuousAdmissionRows: 2,
        minContinuousSchedulerPhases: 8,
        expectSchedulerTokenPressure: true,
        minModelLaneWaitEvents: 0,
        requestBudgets: [
          {
            label: "long 32768x128",
            promptTokens: 32768,
            completionTokens: 128,
            maxServerFirstPrefillProgressMs: 6_000,
            maxServerSilentEventGapMs: 6_000,
            minServerPrefillEvents: 8,
          },
          {
            label: "short 128x32",
            promptTokens: 128,
            completionTokens: 32,
            maxClientTtftMs: 10_000,
            maxClientStreamChunkGapMs: 1_000,
            maxServerSchedulerQueuedMs: 5_000,
            maxServerStreamTtftMs: 6_000,
            maxServerSilentEventGapMs: 6_000,
          },
        ],
      },
    },
    {
      label: "gemma4-mixed-long-short-staggered-stream",
      model: options.gemma4Model,
      modelId: "gemma-local",
      mixedRungs: "5000x128+128x32",
      stream: true,
      ignoreEos: true,
      requestStaggerMs: 100,
      budget: {
        minCompletionTps: 2,
        minPostTtftCompletionTps: 20,
        maxPeakMemoryGb: 14,
        maxActiveDeltaGb: 1.5,
        minCompletionTokenRatio: 0.98,
        minStreamChunks: 2,
        minStreamBytes: 1,
        expectEveryRequestStreamed: true,
        expectEveryRequestOutputStreamed: true,
        expectEveryServerRequestStreamed: true,
        expectEveryServerRequestOutputStreamed: true,
        maxObservedStreamChunkGapMs: 1_500,
        expectedRoute: "continuous",
        expectedReason: "eligible",
        minRouteDecisions: 2,
        minServerRequests: 2,
        expectedAdmissionBatches: 0,
        expectedStaticBatches: 0,
        minContinuousAdmissions: 2,
        minContinuousAdmissionRows: 2,
        minContinuousSchedulerPhases: 8,
        expectSchedulerTokenPressure: true,
        minModelLaneWaitEvents: 0,
        requestBudgets: [
          {
            label: "long 5000x128",
            promptTokens: 5000,
            completionTokens: 128,
            maxServerFirstPrefillProgressMs: 2_500,
            maxServerSilentEventGapMs: 2_500,
            minServerPrefillEvents: 5,
          },
          {
            label: "short 128x32",
            promptTokens: 128,
            completionTokens: 32,
            maxClientTtftMs: 2_500,
            maxClientStreamChunkGapMs: 1_500,
            maxServerSchedulerQueuedMs: 2_500,
            maxServerStreamTtftMs: 2_500,
            maxServerSilentEventGapMs: 2_500,
          },
        ],
      },
    },
  ];
}

function benchmarkRungArgs(spec: ServeRegressionSpec): [string, string] {
  if (spec.rungs !== undefined && spec.mixedRungs !== undefined) {
    throw new Error(`[serve-regression] ${spec.label} cannot set both rungs and mixedRungs.`);
  }
  if (spec.mixedRungs !== undefined) {
    return ["--mixed-rungs", spec.mixedRungs];
  }
  if (spec.rungs !== undefined) {
    return ["--rungs", spec.rungs];
  }
  throw new Error(`[serve-regression] ${spec.label} must set rungs or mixedRungs.`);
}

async function runServeBenchmark(
  spec: ServeRegressionSpec,
  options: CliOptions,
  progress: (text: string) => void,
): Promise<ServeRegressionRunReport> {
  mkdirSync(options.reportDir, { recursive: true });
  const reportPath = join(options.reportDir, `${sanitizeLabel(spec.label)}.json`);
  const [rungFlag, rungSpec] = benchmarkRungArgs(spec);
  const args = [
    "bun",
    "run",
    "bench:serve",
    "--model",
    spec.model,
    "--model-id",
    spec.modelId,
    rungFlag,
    rungSpec,
    "--trials",
    "1",
    "--report-json",
    reportPath,
    "--request-timeout-ms",
    String(options.requestTimeoutMs),
    "--max-concurrent-requests",
    "1",
    "--max-batch-size",
    "8",
    "--batch-window-ms",
    "2",
  ];
  if (spec.protocol !== undefined) {
    args.push("--protocol", spec.protocol);
  }
  if (spec.maxPromptTokens !== undefined) {
    args.push("--max-prompt-tokens", String(spec.maxPromptTokens));
  }
  if (spec.maxTotalTokens !== undefined) {
    args.push("--max-total-tokens", String(spec.maxTotalTokens));
  }
  if (spec.greedy ?? true) {
    args.push("--greedy");
  }
  if (spec.requestStaggerMs !== undefined) {
    args.push("--request-stagger-ms", String(spec.requestStaggerMs));
  }
  if (spec.stream) {
    args.push("--stream");
  }
  if (spec.ignoreEos) {
    args.push("--ignore-eos");
  }
  if (options.allowDownload) {
    args.push("--allow-download");
  }

  await runCommand(spec.label, args, progress);
  if (!existsSync(reportPath)) {
    throw new Error(`[serve-regression] ${spec.label} did not write ${reportPath}.`);
  }
  assertServeReportBudget(spec.label, readBenchmarkReport(reportPath), spec.budget);
  progress(`[serve-regression] ${spec.label} passed budgets report=${reportPath}`);
  return {
    label: spec.label,
    modelId: spec.modelId,
    rung: spec.mixedRungs ?? spec.rungs ?? "unknown",
    protocol: spec.protocol ?? "completions",
    stream: spec.stream,
    reportPath,
  };
}

async function runRealModelSmoke(
  options: CliOptions,
  progress: (text: string) => void,
): Promise<ServeRegressionRunReport[]> {
  const specs = [
    ...baseSpecs(options),
    ...protocolHealthSpecs(options),
    ...(options.fairnessSmoke ? fairnessSpecs(options) : []),
    ...(options.capabilitySmoke ? capabilitySpecs(options) : []),
  ];
  const reports: ServeRegressionRunReport[] = [];
  for (const spec of specs) {
    reports.push(await runServeBenchmark(spec, options, progress));
  }
  return reports;
}

export async function runServeRegression(
  options: CliOptions,
  progress: (text: string) => void = console.error,
): Promise<ServeRegressionResult> {
  await runFocusedUnitChecks(progress);
  const reports: ServeRegressionRunReport[] = [];
  if (options.realModels) {
    using _runtimeLock = acquireRuntimeCommandLock("regression:serve");
    reports.push(...(await runRealModelSmoke(options, progress)));
  }
  return {
    focusedChecks: "passed",
    realModels: options.realModels,
    reports,
  };
}

function toon(value: string | number | boolean | null): string {
  return typeof value === "string" ? JSON.stringify(value) : String(value);
}

function formatMultilineField(name: string, value: string): string[] {
  const lines = value.split(/\r?\n/);
  if (lines.length === 1) {
    return [`  ${name}: ${toon(value)}`];
  }
  return [`  ${name}: |`, ...lines.map((line) => `    ${line}`)];
}

export function formatServeRegressionSuccess(result: ServeRegressionResult): string {
  const reportState =
    result.realModels && result.reports.length > 0
      ? `passed:${result.reports.length}`
      : result.realModels
        ? "passed:0"
        : "skipped";
  return [
    "serve_regression:",
    "  status: passed",
    `  focused_checks: ${result.focusedChecks}`,
    `  real_model_smoke: ${reportState}`,
    `  reports: ${result.reports.length}`,
    `reports[${result.reports.length}]{label,model_id,rung,protocol,stream,path}:`,
    ...result.reports.map((report) =>
      [
        `  ${toon(report.label)}`,
        toon(report.modelId),
        toon(report.rung),
        toon(report.protocol),
        toon(report.stream),
        toon(report.reportPath),
      ].join(","),
    ),
  ].join("\n");
}

export function formatServeRegressionError(message: string, help: string): string {
  return ["error:", ...formatMultilineField("message", message), `help: ${toon(help)}`].join("\n");
}

export async function runServeRegressionCommand(
  argv: readonly string[],
  runtime: ServeRegressionRuntime = {},
): Promise<number> {
  const stdout = runtime.stdout ?? console.log;
  const stderr = runtime.stderr ?? console.error;
  const runRegression = runtime.runRegression ?? runServeRegression;
  let command: ServeRegressionCommand;

  try {
    command = parseServeRegressionArgs(argv);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    stdout(
      formatServeRegressionError(
        message,
        "bun run packages/serve/scripts/regression-serve-matrix.ts [--real-models]",
      ),
    );
    return error instanceof ServeRegressionUsageError ? 2 : 1;
  }

  if (command.kind === "help") {
    stdout(formatServeRegressionUsage());
    return 0;
  }

  try {
    const result = await runRegression(command.options, stderr);
    stdout(formatServeRegressionSuccess(result));
    return 0;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    stdout(
      formatServeRegressionError(
        message,
        "review stderr and rerun the serve regression command after fixing the failure",
      ),
    );
    return 1;
  }
}

if (import.meta.main) {
  const exitCode = await runServeRegressionCommand(Bun.argv.slice(2));
  process.exit(exitCode);
}
