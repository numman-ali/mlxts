#!/usr/bin/env bun

import { existsSync, mkdirSync, readFileSync } from "fs";
import { join } from "path";
import { acquireRuntimeCommandLock } from "../../../scripts/runtime-command-lock";
import type { BenchmarkReport, TrialMetrics } from "./benchmark-serve";
import {
  expectedCompletionTokensForRung,
  formatServeBenchmarkRung,
  rungConcurrency,
} from "./benchmark-serve-options";

type CliOptions = {
  realModels: boolean;
  capabilitySmoke: boolean;
  qwenModel: string;
  gemma4Model: string;
  reportDir: string;
  allowDownload: boolean;
  requestTimeoutMs: number;
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
};

type ServeRegressionSpec = {
  label: string;
  model: string;
  modelId: string;
  rungs?: string;
  mixedRungs?: string;
  stream: boolean;
  ignoreEos: boolean;
  greedy?: boolean;
  requestStaggerMs?: number;
  budget: ServeRegressionBudget;
};

const FOCUSED_TESTS = [
  "packages/serve/src/server.test.ts",
  "packages/serve/src/server-streaming.test.ts",
  "packages/serve/src/transformers-engine.test.ts",
  "packages/serve/src/model-server.test.ts",
  "packages/serve/src/batching-engine.test.ts",
  "packages/serve/src/protocols/openai-completions.test.ts",
  "packages/serve/src/protocols/openai-chat-completions.test.ts",
  "packages/serve/src/protocols/openai-responses.test.ts",
  "packages/serve/scripts/benchmark-serve-options.test.ts",
  "packages/serve/scripts/benchmark-serve-completions.test.ts",
  "packages/serve/scripts/benchmark-serve.test.ts",
  "packages/serve/scripts/regression-serve-matrix.test.ts",
];

function usage(): string {
  return [
    "Usage: bun run packages/serve/scripts/regression-serve-matrix.ts [options]",
    "",
    "Options:",
    "  --real-models             Run cached Qwen/Gemma endpoint smoke benchmarks.",
    "  --capability-smoke        Add longer output/context rungs; implies --real-models.",
    "  --qwen-model <id>         Qwen model id/path.",
    "  --gemma4-model <id>       Gemma 4 model id/path.",
    "  --report-dir <path>       Directory for benchmark JSON evidence.",
    "  --request-timeout-ms <n>  Client timeout per benchmark request.",
    "  --allow-download          Allow Hub downloads; default is cached/local only.",
  ].join("\n");
}

function defaultOptions(): CliOptions {
  return {
    realModels: false,
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
  if (value === undefined || value.startsWith("--")) {
    throw new Error(`${flag} requires a value.\n\n${usage()}`);
  }
  return value;
}

function readPositiveIntegerFlag(args: readonly string[], index: number, flag: string): number {
  const value = Number.parseInt(readStringFlag(args, index, flag), 10);
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${flag} must be a positive integer.`);
  }
  return value;
}

export function parseServeRegressionArgs(argv: readonly string[]): CliOptions {
  const options = defaultOptions();

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
      case "--help":
      case "-h":
        console.log(usage());
        process.exit(0);
        return options;
      case "--real-models":
        options.realModels = true;
        break;
      case "--capability-smoke":
        options.capabilitySmoke = true;
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
        throw new Error(`Unknown option: ${arg}\n\n${usage()}`);
    }
  }

  return options;
}

function inheritedStringEnv(): Record<string, string> {
  return Object.fromEntries(
    Object.entries(process.env).filter((entry): entry is [string, string] => {
      const value = entry[1];
      return typeof value === "string";
    }),
  );
}

async function runCommand(label: string, args: readonly string[]): Promise<void> {
  console.log(`[serve-regression] ${label}: ${args.join(" ")}`);
  const child = Bun.spawn([...args], {
    cwd: new URL("../../..", import.meta.url).pathname,
    env: inheritedStringEnv(),
    stdout: "inherit",
    stderr: "inherit",
  });
  const exitCode = await child.exited;
  if (exitCode !== 0) {
    throw new Error(`[serve-regression] ${label} failed with exit code ${exitCode}.`);
  }
}

async function runFocusedUnitChecks(): Promise<void> {
  await runCommand("focused unit checks", ["bun", "test", ...FOCUSED_TESTS]);
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

function assertFinishReasons(label: string, metrics: TrialMetrics): void {
  const badReasons = metrics.finishReasons.filter(
    (reason) => reason !== "length" && reason !== "stop",
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

function isAllowedStreamFinishReason(reason: string | undefined): boolean {
  return reason === "length" || reason === "stop" || reason === "eos";
}

function perRequestStreamLifecycleFailures(metrics: TrialMetrics, concurrency: number): string[] {
  const failures: string[] = [];
  if (metrics.requests.length < concurrency) {
    failures.push(`requests ${metrics.requests.length} < concurrency ${concurrency}`);
  }
  const nonStreamingRequests = metrics.requests.filter(
    (request) =>
      request.streamBytes <= 0 ||
      request.completionTokens <= 0 ||
      !isAllowedStreamFinishReason(request.finishReason),
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
      !isAllowedStreamFinishReason(request.serverStreamFinishReason),
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
): string[] {
  const failures = [
    ...aggregateStreamFailures(metrics, budget),
    ...streamTimingFailures(metrics, budget),
  ];
  if (budget.expectEveryRequestStreamed) {
    failures.push(...perRequestStreamLifecycleFailures(metrics, concurrency));
  }
  if (budget.expectEveryRequestOutputStreamed) {
    failures.push(...perRequestOutputStreamFailures(metrics));
  }
  if (budget.expectEveryServerRequestStreamed) {
    failures.push(...serverRequestStreamLifecycleFailures(metrics, concurrency));
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
    const missing = metrics.serverRequests.filter(
      (request) =>
        request.route === "continuous" &&
        (request.schedulerScheduledPromptTokens === null ||
          request.schedulerMaxScheduledPromptTokens === null ||
          request.schedulerScheduledCompletionTokens === null ||
          request.schedulerMaxScheduledCompletionTokens === null ||
          request.schedulerScheduledTotalTokens === null ||
          request.schedulerMaxScheduledTotalTokens === null),
    );
    if (missing.length > 0) {
      failures.push(
        `server_requests missing scheduler token pressure: ${missing.map((request) => request.id).join(",")}`,
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
    budget.maxServerSilentEventGapMs !== undefined
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
  return [
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
  ].filter((failure): failure is string => failure !== null);
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
      ...streamFailures(averages, concurrency, budget),
      ...routeFailures(averages, budget),
      ...evidenceFailures(averages, budget),
      ...batchCounterFailures(averages, budget),
      ...modelLaneWaitFailures(averages, budget),
      ...requestBudgetFailures(averages, budget),
    ];

    assertFinishReasons(`${label} ${rungLabel}`, averages);
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
      modelId: "qwen-local",
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
      modelId: "qwen-local",
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
      modelId: "qwen-local",
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
      modelId: "qwen-local",
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
      modelId: "qwen-local",
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
      modelId: "qwen-local",
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
      modelId: "qwen-local",
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
      modelId: "qwen-local",
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

function capabilitySpecs(options: CliOptions): ServeRegressionSpec[] {
  return [
    {
      label: "qwen36-long-output-stream",
      model: options.qwenModel,
      modelId: "qwen-local",
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
      modelId: "qwen-local",
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
    {
      label: "qwen36-mixed-long-short-staggered-stream",
      model: options.qwenModel,
      modelId: "qwen-local",
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
        maxObservedStreamChunkGapMs: 6_000,
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
            label: "short 128x32",
            promptTokens: 128,
            completionTokens: 32,
            maxClientTtftMs: 10_000,
            maxClientStreamChunkGapMs: 6_000,
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

async function runServeBenchmark(spec: ServeRegressionSpec, options: CliOptions): Promise<void> {
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

  await runCommand(spec.label, args);
  if (!existsSync(reportPath)) {
    throw new Error(`[serve-regression] ${spec.label} did not write ${reportPath}.`);
  }
  assertServeReportBudget(spec.label, readBenchmarkReport(reportPath), spec.budget);
  console.log(`[serve-regression] ${spec.label} passed budgets report=${reportPath}`);
}

async function runRealModelSmoke(options: CliOptions): Promise<void> {
  const specs = options.capabilitySmoke
    ? [...baseSpecs(options), ...capabilitySpecs(options)]
    : baseSpecs(options);
  for (const spec of specs) {
    await runServeBenchmark(spec, options);
  }
}

export async function runServeRegression(options: CliOptions): Promise<void> {
  await runFocusedUnitChecks();
  if (options.realModels) {
    using _runtimeLock = acquireRuntimeCommandLock("regression:serve");
    await runRealModelSmoke(options);
  }
}

if (import.meta.main) {
  runServeRegression(parseServeRegressionArgs(Bun.argv.slice(2))).catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  });
}
