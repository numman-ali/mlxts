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
    continuousSchedulerPhases: 0,
    maxContinuousBatchSize: 0,
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
        modelLaneWaitMs: 1,
        modelLaneQueuedAhead: 0,
        modelLaneInFlightAtQueue: 1,
        schedulerQueuedMs: null,
        schedulerPrefillStartMs: null,
        schedulerAdmittedMs: null,
        schedulerFirstTokenMs: null,
        schedulerFinishedMs: null,
        schedulerPhaseEvents: 0,
        schedulerAdmittedBatchSize: null,
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
    streamDecodeInterval: 1,
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
  maxMeanTtftMs: 500,
  maxObservedStreamChunkGapMs: 100,
  expectedRoute: "single",
  expectedReason: "streaming",
  minRouteDecisions: 1,
  minServerRequests: 1,
  expectedAdmissionBatches: 1,
  expectedStaticBatches: 0,
  expectedContinuousAdmissions: 0,
  expectedContinuousAdmissionRows: 0,
  expectedContinuousSchedulerPhases: 0,
  expectedMaxGenerationBatchSize: 1,
  minModelLaneWaitEvents: 1,
  minModelLaneBusyWaitEvents: 1,
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

  test("accepts static-batch reports while guarding continuous counters", () => {
    const baseServerRequest = trial().serverRequests[0];
    if (baseServerRequest === undefined) {
      throw new Error("Expected trial fixture to include one server request.");
    }

    const staticBudget: ServeRegressionBudget = {
      minCompletionTps: 20,
      maxPeakMemoryGb: 12,
      maxActiveDeltaGb: 1,
      minCompletionTokenRatio: 0.98,
      expectedRoute: "static",
      expectedReason: "eligible",
      minRouteDecisions: 2,
      minServerRequests: 2,
      expectedAdmissionBatches: 0,
      expectedStaticBatches: 1,
      expectedStaticBatchRows: 2,
      expectedContinuousAdmissions: 0,
      expectedContinuousSchedulerPhases: 0,
      expectedMaxGenerationBatchSize: 2,
      minModelLaneWaitEvents: 2,
    };
    const staticMetrics = trial({
      admissionBatches: 0,
      admissionRows: 0,
      maxAdmissionBatchSize: 0,
      staticBatches: 1,
      staticBatchRows: 2,
      maxGenerationBatchSize: 2,
      routeDecisions: [
        {
          id: "request-a",
          model: "local",
          protocol: "openai.completions",
          route: "static",
          eligible: true,
          reason: "eligible",
          modelType: "gemma4_text",
          maxBatchSize: 8,
          stream: false,
        },
        {
          id: "request-b",
          model: "local",
          protocol: "openai.completions",
          route: "static",
          eligible: true,
          reason: "eligible",
          modelType: "gemma4_text",
          maxBatchSize: 8,
          stream: false,
        },
      ],
      routeSummary: [{ key: "static:eligible", route: "static", reason: "eligible", count: 2 }],
      serverRequests: [
        {
          ...baseServerRequest,
          id: "request-a",
          route: "static",
          routeReason: "eligible",
          modelLaneInFlightAtQueue: 0,
        },
        {
          ...baseServerRequest,
          id: "request-b",
          route: "static",
          routeReason: "eligible",
          modelLaneInFlightAtQueue: 0,
        },
      ],
    });

    expect(() =>
      assertServeReportBudget("gemma", report(staticMetrics), staticBudget),
    ).not.toThrow();
    expect(() =>
      assertServeReportBudget(
        "qwen",
        report(
          trial({
            ...staticMetrics,
            routeDecisions: staticMetrics.routeDecisions.map((decision) => ({
              ...decision,
              modelType: "qwen3_5_text",
            })),
          }),
        ),
        staticBudget,
      ),
    ).not.toThrow();
    expect(() =>
      assertServeReportBudget(
        "gemma",
        report(trial({ ...staticMetrics, staticBatchRows: 1 })),
        staticBudget,
      ),
    ).toThrow("static_batch_rows");
    expect(() =>
      assertServeReportBudget(
        "gemma",
        report(trial({ ...staticMetrics, continuousSchedulerPhases: 1 })),
        staticBudget,
      ),
    ).toThrow("continuous_scheduler_phases");
  });

  test("accepts continuous reports with minimum scheduler evidence", () => {
    const continuousBudget: ServeRegressionBudget = {
      minCompletionTps: 20,
      maxPeakMemoryGb: 12,
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
    };
    const baseServerRequest = trial().serverRequests[0];
    if (baseServerRequest === undefined) {
      throw new Error("Expected trial fixture to include one server request.");
    }
    const continuousMetrics = trial({
      admissionBatches: 0,
      admissionRows: 0,
      maxAdmissionBatchSize: 0,
      continuousAdmissions: 2,
      continuousAdmissionRows: 3,
      continuousSchedulerPhases: 9,
      maxContinuousBatchSize: 2,
      maxGenerationBatchSize: 2,
      routeDecisions: [
        {
          id: "request-a",
          model: "local",
          protocol: "openai.completions",
          route: "continuous",
          eligible: true,
          reason: "eligible",
          modelType: "qwen3_5_text",
          maxBatchSize: 8,
          stream: false,
        },
        {
          id: "request-b",
          model: "local",
          protocol: "openai.completions",
          route: "continuous",
          eligible: true,
          reason: "eligible",
          modelType: "qwen3_5_text",
          maxBatchSize: 8,
          stream: false,
        },
      ],
      routeSummary: [
        { key: "continuous:eligible", route: "continuous", reason: "eligible", count: 2 },
      ],
      serverRequests: [
        {
          ...baseServerRequest,
          id: "request-a",
          route: "continuous",
          routeReason: "eligible",
          modelLaneWaitMs: null,
          modelLaneQueuedAhead: null,
          modelLaneInFlightAtQueue: null,
          schedulerPhaseEvents: 4,
          schedulerAdmittedBatchSize: 1,
        },
        {
          ...baseServerRequest,
          id: "request-b",
          route: "continuous",
          routeReason: "eligible",
          modelLaneWaitMs: null,
          modelLaneQueuedAhead: null,
          modelLaneInFlightAtQueue: null,
          schedulerPhaseEvents: 5,
          schedulerAdmittedBatchSize: 2,
        },
      ],
    });

    expect(() =>
      assertServeReportBudget("qwen", report(continuousMetrics), continuousBudget),
    ).not.toThrow();
    expect(() =>
      assertServeReportBudget(
        "qwen",
        report(trial({ ...continuousMetrics, continuousAdmissionRows: 1 })),
        continuousBudget,
      ),
    ).toThrow("continuous_admission_rows");
  });

  test("fails on throughput, memory, token, stream, route, evidence, batch, and finish regressions", () => {
    const baseServerRequest = trial().serverRequests[0];
    if (baseServerRequest === undefined) {
      throw new Error("Expected trial fixture to include one server request.");
    }

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
      assertServeReportBudget("qwen", report(trial({ meanTtftMs: 1_000 })), budget),
    ).toThrow("mean_ttft_ms");
    expect(() =>
      assertServeReportBudget("qwen", report(trial({ maxStreamChunkGapMs: 1_000 })), budget),
    ).toThrow("max_stream_chunk_gap_ms");
    expect(() =>
      assertServeReportBudget(
        "qwen",
        report(trial({ routeDecisions: [], routeSummary: [] })),
        budget,
      ),
    ).toThrow("route_decisions");
    expect(() =>
      assertServeReportBudget("qwen", report(trial({ serverRequests: [] })), budget),
    ).toThrow("server_requests");
    expect(() =>
      assertServeReportBudget("qwen", report(trial({ admissionBatches: 0 })), budget),
    ).toThrow("admission_batches");
    expect(() =>
      assertServeReportBudget("qwen", report(trial({ staticBatches: 1 })), budget),
    ).toThrow("static_batches");
    expect(() =>
      assertServeReportBudget("qwen", report(trial({ continuousAdmissions: 1 })), budget),
    ).toThrow("continuous_admissions");
    expect(() =>
      assertServeReportBudget("qwen", report(trial({ continuousAdmissionRows: 1 })), budget),
    ).toThrow("continuous_admission_rows");
    expect(() =>
      assertServeReportBudget("qwen", report(trial({ continuousSchedulerPhases: 1 })), budget),
    ).toThrow("continuous_scheduler_phases");
    expect(() =>
      assertServeReportBudget("qwen", report(trial({ maxGenerationBatchSize: 2 })), budget),
    ).toThrow("max_generation_batch");
    expect(() =>
      assertServeReportBudget(
        "qwen",
        report(
          trial({
            serverRequests: [
              {
                ...baseServerRequest,
                modelLaneWaitMs: null,
                modelLaneQueuedAhead: null,
                modelLaneInFlightAtQueue: null,
              },
            ],
          }),
        ),
        budget,
      ),
    ).toThrow("model_lane_wait_events");
    expect(() =>
      assertServeReportBudget(
        "qwen",
        report(
          trial({
            serverRequests: [
              {
                ...baseServerRequest,
                modelLaneInFlightAtQueue: 0,
              },
            ],
          }),
        ),
        budget,
      ),
    ).toThrow("model_lane_busy_wait_events");
    expect(() =>
      assertServeReportBudget("qwen", report(trial({ finishReasons: ["unknown"] })), budget),
    ).toThrow("unexpected finish reasons");
  });
});
