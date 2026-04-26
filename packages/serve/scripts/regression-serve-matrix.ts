#!/usr/bin/env bun

import { existsSync, mkdirSync, readFileSync } from "fs";
import { join } from "path";
import { acquireRuntimeCommandLock } from "../../../scripts/runtime-command-lock";
import type { BenchmarkReport, TrialMetrics } from "./benchmark-serve";

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
  maxMeanTtftMs?: number;
  maxObservedStreamChunkGapMs?: number;
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
  minModelLaneWaitEvents?: number;
  minModelLaneBusyWaitEvents?: number;
};

type ServeRegressionSpec = {
  label: string;
  model: string;
  modelId: string;
  rungs: string;
  stream: boolean;
  ignoreEos: boolean;
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
  generationTokens: number,
  concurrency: number,
  budget: ServeRegressionBudget,
): string[] {
  const expectedCompletionTokens = generationTokens * concurrency;
  const minCompletionTokens = expectedCompletionTokens * budget.minCompletionTokenRatio;
  return metrics.completionTokens < minCompletionTokens
    ? [
        `completion_tokens ${metrics.completionTokens.toFixed(0)} < ${minCompletionTokens.toFixed(
          0,
        )}`,
      ]
    : [];
}

function perRequestStreamFailures(metrics: TrialMetrics, concurrency: number): string[] {
  const failures: string[] = [];
  if (metrics.requests.length < concurrency) {
    failures.push(`requests ${metrics.requests.length} < concurrency ${concurrency}`);
  }
  const nonStreamingRequests = metrics.requests.filter(
    (request) =>
      request.streamChunks <= 0 ||
      request.streamBytes <= 0 ||
      request.ttftMs === null ||
      (request.finishReason !== "length" && request.finishReason !== "stop"),
  );
  if (nonStreamingRequests.length > 0) {
    const ids = nonStreamingRequests.map((request) => request.id ?? `index:${request.index}`);
    failures.push(`requests missing per-request SSE evidence: ${ids.join(",")}`);
  }
  return failures;
}

function streamFailures(
  metrics: TrialMetrics,
  concurrency: number,
  budget: ServeRegressionBudget,
): string[] {
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
  if (budget.expectEveryRequestStreamed) {
    failures.push(...perRequestStreamFailures(metrics, concurrency));
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
    const failures = [
      ...throughputFailures(averages, budget),
      ...memoryFailures(averages, budget),
      ...tokenFailures(averages, rung.generationTokens, rung.concurrency, budget),
      ...streamFailures(averages, rung.concurrency, budget),
      ...routeFailures(averages, budget),
      ...evidenceFailures(averages, budget),
      ...batchCounterFailures(averages, budget),
      ...modelLaneWaitFailures(averages, budget),
    ];

    assertFinishReasons(`${label} ${rung.promptTokens}x${rung.generationTokens}`, averages);
    if (failures.length > 0) {
      throw new Error(
        `[serve-regression] ${label} ${rung.promptTokens}x${rung.generationTokens}@${
          rung.concurrency
        } failed: ${failures.join("; ")}.`,
      );
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
        minPostTtftCompletionTps: 12,
        maxPeakMemoryGb: 32,
        maxActiveDeltaGb: 1,
        minCompletionTokenRatio: 0.98,
        minStreamChunks: 1,
        minStreamBytes: 1,
        expectEveryRequestStreamed: true,
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
        minModelLaneWaitEvents: 0,
      },
    },
  ];
}

async function runServeBenchmark(spec: ServeRegressionSpec, options: CliOptions): Promise<void> {
  mkdirSync(options.reportDir, { recursive: true });
  const reportPath = join(options.reportDir, `${sanitizeLabel(spec.label)}.json`);
  const args = [
    "bun",
    "run",
    "bench:serve",
    "--model",
    spec.model,
    "--model-id",
    spec.modelId,
    "--rungs",
    spec.rungs,
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
    "--greedy",
  ];
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
