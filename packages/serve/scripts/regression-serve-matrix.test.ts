import { describe, expect, test } from "bun:test";

import type { BenchmarkReport, TrialMetrics } from "./benchmark-serve";
import {
  assertServeReportBudget,
  parseServeRegressionArgs,
  type ServeRegressionBudget,
} from "./regression-serve-matrix";

function trial(overrides: Partial<TrialMetrics> = {}): TrialMetrics {
  return {
    wallMs: 1000,
    requestTps: 1,
    completionTps: 50,
    totalTps: 60,
    meanTtftMs: 100,
    meanPromptToFirstTokenTps: 10,
    meanPostTtftCompletionTps: 55,
    meanStreamChunkGapMs: 20,
    maxStreamChunkGapMs: 25,
    promptTokens: 1024,
    completionTokens: 128,
    totalTokens: 1152,
    meanRequestMs: 1000,
    p95RequestMs: 1000,
    maxRequestMs: 1000,
    peakMemoryGb: 10,
    activeMemoryGb: 9,
    cacheMemoryGb: 1,
    activeDeltaGb: 0.1,
    admissionBatches: 1,
    admissionRows: 1,
    maxAdmissionBatchSize: 1,
    staticBatches: 0,
    staticBatchRows: 0,
    continuousAdmissions: 0,
    continuousAdmissionRows: 0,
    maxGenerationBatchSize: 1,
    streamChunks: 4,
    streamBytes: 100,
    finishReasons: ["length"],
    routeDecisions: [
      {
        id: "request",
        model: "local",
        protocol: "openai.completions",
        route: "single",
        eligible: false,
        reason: "streaming",
        modelType: "qwen3_5_text",
        maxBatchSize: 8,
        stream: true,
      },
    ],
    routeSummary: [{ key: "single:streaming", route: "single", reason: "streaming", count: 1 }],
    requests: [
      {
        id: "request",
        index: 0,
        launchDelayMs: 0,
        durationMs: 1000,
        ttftMs: 100,
        promptToFirstTokenTps: 10,
        postTtftCompletionTps: 55,
        meanStreamChunkGapMs: 20,
        maxStreamChunkGapMs: 25,
        promptTokens: 1024,
        completionTokens: 128,
        totalTokens: 1152,
        finishReason: "length",
        streamChunks: 4,
        streamBytes: 100,
      },
    ],
    serverRequests: [
      {
        id: "request",
        model: "local",
        protocol: "openai.completions",
        inputKind: "text",
        route: "single",
        routeReason: "streaming",
        routeDecisionMs: 1,
        firstPrefillProgressMs: null,
        lastPrefillProgressMs: null,
        prefillObservedMs: null,
        firstCompletionProgressMs: 100,
        completeObservedMs: 1000,
        durationMs: 1000,
        maxSilentEventGapMs: 900,
        prefillEvents: 0,
        progressEvents: 2,
        maxCompletionTokens: 128,
        finishReason: "length",
      },
    ],
    ...overrides,
  };
}

function report(metrics: TrialMetrics = trial()): BenchmarkReport {
  return {
    createdAt: "2026-04-26T00:00:00.000Z",
    model: "model",
    modelId: "local",
    snapshotPath: "/tmp/model",
    samplingMode: "greedy",
    transportMode: "streaming",
    protocolMode: "completions",
    ignoreEos: true,
    maxBatchSize: 8,
    batchWindowMs: 2,
    requestStaggerMs: 0,
    maxConcurrentRequests: 1,
    gpuMemoryUtilization: 0.9,
    rungs: [
      {
        rung: { promptTokens: 1024, generationTokens: 128, concurrency: 1 },
        arrivalSpanMs: 0,
        trials: [metrics],
        averages: metrics,
      },
    ],
  };
}

const budget: ServeRegressionBudget = {
  minCompletionTps: 20,
  minPostTtftCompletionTps: 25,
  maxPeakMemoryGb: 12,
  maxActiveDeltaGb: 1,
  minCompletionTokenRatio: 0.98,
  minStreamChunks: 1,
  minStreamBytes: 1,
  expectedRoute: "single",
  expectedReason: "streaming",
};

describe("serve regression matrix", () => {
  test("parses cheap, real-model, and capability options", () => {
    expect(parseServeRegressionArgs([])).toMatchObject({
      realModels: false,
      capabilitySmoke: false,
      qwenModel: "mlx-community/Qwen3.6-27B-4bit",
      gemma4Model: "google/gemma-4-E2B-it",
      reportDir: ".tmp/serve-regression",
      allowDownload: false,
    });

    expect(
      parseServeRegressionArgs([
        "--capability-smoke",
        "--qwen-model",
        "qwen",
        "--gemma4-model",
        "gemma",
        "--report-dir",
        ".tmp/reports",
        "--request-timeout-ms",
        "120000",
        "--allow-download",
      ]),
    ).toMatchObject({
      realModels: true,
      capabilitySmoke: true,
      qwenModel: "qwen",
      gemma4Model: "gemma",
      reportDir: ".tmp/reports",
      requestTimeoutMs: 120000,
      allowDownload: true,
    });
  });

  test("accepts reports that clear serving budgets", () => {
    expect(() => assertServeReportBudget("qwen", report(), budget)).not.toThrow();
  });

  test("fails on throughput, memory, token, stream, and finish regressions", () => {
    expect(() =>
      assertServeReportBudget("qwen", report(trial({ completionTps: 10 })), budget),
    ).toThrow("completion_tps");
    expect(() =>
      assertServeReportBudget("qwen", report(trial({ meanPostTtftCompletionTps: 10 })), budget),
    ).toThrow("post_ttft_completion_tps");
    expect(() =>
      assertServeReportBudget("qwen", report(trial({ peakMemoryGb: 13 })), budget),
    ).toThrow("peak_memory");
    expect(() =>
      assertServeReportBudget("qwen", report(trial({ completionTokens: 50 })), budget),
    ).toThrow("completion_tokens");
    expect(() =>
      assertServeReportBudget("qwen", report(trial({ streamChunks: 0 })), budget),
    ).toThrow("stream_chunks");
    expect(() =>
      assertServeReportBudget("qwen", report(trial({ streamBytes: 0 })), budget),
    ).toThrow("stream_bytes");
    expect(() =>
      assertServeReportBudget(
        "qwen",
        report(trial({ routeDecisions: [], routeSummary: [] })),
        budget,
      ),
    ).toThrow("route_decisions");
    expect(() =>
      assertServeReportBudget("qwen", report(trial({ finishReasons: ["unknown"] })), budget),
    ).toThrow("unexpected finish reasons");
  });
});
