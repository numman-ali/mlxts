import { describe, expect, test } from "bun:test";

import type { BenchmarkReport, TrialMetrics } from "./benchmark-serve";
import type { ServeBenchmarkRung } from "./benchmark-serve-options";
import {
  assertServeReportBudget,
  parseServeRegressionArgs,
  type ServeRegressionBudget,
} from "./regression-serve-matrix";

const ROUTE_STRATEGY = {
  schedulerMode: "auto",
  cacheBackend: "managed",
  attentionBackend: "auto",
  decodingBackend: "model",
};

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
        ...ROUTE_STRATEGY,
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
        schedulerWaitingTotalTokens: null,
        schedulerPrefillingTotalTokens: null,
        schedulerActiveTotalTokens: null,
        schedulerScheduledPromptTokens: null,
        schedulerMaxScheduledPromptTokens: null,
        schedulerScheduledCompletionTokens: null,
        schedulerMaxScheduledCompletionTokens: null,
        schedulerScheduledTotalTokens: null,
        schedulerMaxScheduledTotalTokens: null,
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
        serverStreamChunkEvents: 4,
        serverStreamEndEvents: 1,
        serverStreamFirstChunkMs: 100,
        serverStreamLastChunkMs: 180,
        serverStreamChunks: 4,
        serverStreamBytes: 100,
        serverStreamOutputChunks: 4,
        serverStreamOutputBytes: 100,
        serverStreamTtftMs: 100,
        serverStreamDurationMs: 1000,
        serverStreamResult: "completed",
        serverStreamFinishReason: "length",
        finishReason: "length",
      },
    ],
    ...overrides,
  };
}

function report(
  metrics: TrialMetrics = trial(),
  rung: ServeBenchmarkRung = {
    promptTokens: 1024,
    generationTokens: 128,
    concurrency: 1,
  },
): BenchmarkReport {
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
        rung,
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
  expectEveryRequestStreamed: true,
  expectEveryRequestOutputStreamed: true,
  expectEveryServerRequestStreamed: true,
  expectEveryServerRequestOutputStreamed: true,
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
      fairnessSmoke: false,
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
      fairnessSmoke: true,
      capabilitySmoke: true,
      qwenModel: "qwen",
      gemma4Model: "gemma",
      reportDir: ".tmp/reports",
      requestTimeoutMs: 120000,
      allowDownload: true,
    });

    expect(parseServeRegressionArgs(["--fairness-smoke"])).toMatchObject({
      realModels: true,
      fairnessSmoke: true,
      capabilitySmoke: false,
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
          ...ROUTE_STRATEGY,
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
          ...ROUTE_STRATEGY,
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
      expectSchedulerTokenPressure: true,
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
          ...ROUTE_STRATEGY,
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
          ...ROUTE_STRATEGY,
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
          schedulerScheduledPromptTokens: 12,
          schedulerMaxScheduledPromptTokens: 24,
          schedulerScheduledCompletionTokens: 4,
          schedulerMaxScheduledCompletionTokens: 8,
          schedulerScheduledTotalTokens: 16,
          schedulerMaxScheduledTotalTokens: 32,
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
          schedulerScheduledPromptTokens: 12,
          schedulerMaxScheduledPromptTokens: 24,
          schedulerScheduledCompletionTokens: 4,
          schedulerMaxScheduledCompletionTokens: 8,
          schedulerScheduledTotalTokens: 16,
          schedulerMaxScheduledTotalTokens: 32,
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
    expect(() =>
      assertServeReportBudget(
        "qwen",
        report(
          trial({
            ...continuousMetrics,
            serverRequests: continuousMetrics.serverRequests.map((request) => ({
              ...request,
              schedulerScheduledTotalTokens: null,
            })),
          }),
        ),
        continuousBudget,
      ),
    ).toThrow("scheduler token pressure");
  });

  test("budgets mixed rungs against per-request generation targets", () => {
    const mixedRung: ServeBenchmarkRung = {
      promptTokens: 32768,
      generationTokens: 128,
      concurrency: 2,
      requestShapes: [
        { promptTokens: 32768, generationTokens: 128 },
        { promptTokens: 128, generationTokens: 32 },
      ],
    };
    const mixedBudget: ServeRegressionBudget = {
      minCompletionTps: 20,
      maxPeakMemoryGb: 12,
      maxActiveDeltaGb: 1,
      minCompletionTokenRatio: 0.98,
    };

    expect(() =>
      assertServeReportBudget(
        "mixed",
        report(trial({ completionTokens: 160, finishReasons: ["length", "length"] }), mixedRung),
        mixedBudget,
      ),
    ).not.toThrow();
    expect(() =>
      assertServeReportBudget(
        "mixed",
        report(trial({ completionTokens: 128, finishReasons: ["length", "length"] }), mixedRung),
        mixedBudget,
      ),
    ).toThrow("completion_tokens");
  });

  test("budgets mixed rung short-request fairness by client/server request id", () => {
    const baseRequest = trial().requests[0];
    const baseServerRequest = trial().serverRequests[0];
    if (baseRequest === undefined || baseServerRequest === undefined) {
      throw new Error("Expected trial fixture to include one request and one server request.");
    }

    const mixedRung: ServeBenchmarkRung = {
      promptTokens: 32768,
      generationTokens: 128,
      concurrency: 2,
      requestShapes: [
        { promptTokens: 32768, generationTokens: 128 },
        { promptTokens: 128, generationTokens: 32 },
      ],
    };
    const fairnessBudget: ServeRegressionBudget = {
      minCompletionTps: 20,
      maxPeakMemoryGb: 12,
      maxActiveDeltaGb: 1,
      minCompletionTokenRatio: 0.98,
      requestBudgets: [
        {
          label: "short 128x32",
          promptTokens: 128,
          completionTokens: 32,
          maxClientTtftMs: 1_000,
          maxClientStreamChunkGapMs: 500,
          maxServerSchedulerQueuedMs: 500,
          maxServerStreamTtftMs: 600,
          maxServerSilentEventGapMs: 700,
        },
        {
          label: "long 32768x128",
          promptTokens: 32768,
          completionTokens: 128,
          maxServerFirstPrefillProgressMs: 2_000,
          maxServerSilentEventGapMs: 2_000,
          minServerPrefillEvents: 8,
        },
      ],
    };
    const fairMetrics = trial({
      completionTokens: 160,
      finishReasons: ["length", "length"],
      requests: [
        {
          ...baseRequest,
          id: "long",
          index: 0,
          promptTokens: 32768,
          completionTokens: 128,
          totalTokens: 32896,
          ttftMs: 150_000,
          maxStreamChunkGapMs: 100,
        },
        {
          ...baseRequest,
          id: "short",
          index: 1,
          promptTokens: 128,
          completionTokens: 32,
          totalTokens: 160,
          ttftMs: 700,
          maxStreamChunkGapMs: 400,
        },
      ],
      serverRequests: [
        {
          ...baseServerRequest,
          id: "short",
          route: "continuous",
          routeReason: "eligible",
          schedulerQueuedMs: 300,
          serverStreamTtftMs: 500,
          maxSilentEventGapMs: 450,
        },
        {
          ...baseServerRequest,
          id: "long",
          route: "continuous",
          routeReason: "eligible",
          schedulerQueuedMs: 150_000,
          serverStreamTtftMs: 150_000,
          maxSilentEventGapMs: 600,
          firstPrefillProgressMs: 1_200,
          prefillEvents: 64,
        },
      ],
    });

    expect(() =>
      assertServeReportBudget("mixed", report(fairMetrics, mixedRung), fairnessBudget),
    ).not.toThrow();
    expect(() =>
      assertServeReportBudget(
        "mixed",
        report(
          trial({
            ...fairMetrics,
            requests: fairMetrics.requests.map((request) =>
              request.id === "short" ? { ...request, ttftMs: 1_500 } : request,
            ),
          }),
          mixedRung,
        ),
        fairnessBudget,
      ),
    ).toThrow("request_budget short 128x32 client ttft");
    expect(() =>
      assertServeReportBudget(
        "mixed",
        report(
          trial({
            ...fairMetrics,
            serverRequests: fairMetrics.serverRequests.map((request) =>
              request.id === "short" ? { ...request, schedulerQueuedMs: 900 } : request,
            ),
          }),
          mixedRung,
        ),
        fairnessBudget,
      ),
    ).toThrow("request_budget short 128x32 server scheduler queued");
    expect(() =>
      assertServeReportBudget(
        "mixed",
        report(
          trial({
            ...fairMetrics,
            serverRequests: fairMetrics.serverRequests.filter((request) => request.id !== "short"),
          }),
          mixedRung,
        ),
        fairnessBudget,
      ),
    ).toThrow("request_budget short 128x32 found no matching server request");
    expect(() =>
      assertServeReportBudget(
        "mixed",
        report(
          trial({
            ...fairMetrics,
            serverRequests: fairMetrics.serverRequests.map((request) =>
              request.id === "long" ? { ...request, firstPrefillProgressMs: 3_000 } : request,
            ),
          }),
          mixedRung,
        ),
        fairnessBudget,
      ),
    ).toThrow("request_budget long 32768x128 server first prefill progress");
    expect(() =>
      assertServeReportBudget(
        "mixed",
        report(
          trial({
            ...fairMetrics,
            serverRequests: fairMetrics.serverRequests.map((request) =>
              request.id === "long" ? { ...request, prefillEvents: 2 } : request,
            ),
          }),
          mixedRung,
        ),
        fairnessBudget,
      ),
    ).toThrow("request_budget long 32768x128 server prefill events");
  });

  test("accepts concurrent streaming continuous reports with per-request SSE evidence", () => {
    const baseRequest = trial().requests[0];
    const baseServerRequest = trial().serverRequests[0];
    if (baseRequest === undefined || baseServerRequest === undefined) {
      throw new Error("Expected trial fixture to include one request and one server request.");
    }

    const streamingBudget: ServeRegressionBudget = {
      minCompletionTps: 20,
      minPostTtftCompletionTps: 20,
      maxPeakMemoryGb: 12,
      maxActiveDeltaGb: 1,
      minCompletionTokenRatio: 0.98,
      minStreamChunks: 8,
      minStreamBytes: 1,
      expectEveryRequestStreamed: true,
      expectEveryRequestOutputStreamed: true,
      expectEveryServerRequestStreamed: true,
      expectEveryServerRequestOutputStreamed: true,
      maxClientRequestTtftMs: 500,
      maxServerSchedulerQueuedMs: 500,
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
    };
    const streamingMetrics = trial({
      promptTokens: 256,
      completionTokens: 64,
      totalTokens: 320,
      admissionBatches: 0,
      admissionRows: 0,
      maxAdmissionBatchSize: 0,
      continuousAdmissions: 1,
      continuousAdmissionRows: 2,
      continuousSchedulerPhases: 7,
      maxContinuousBatchSize: 2,
      maxGenerationBatchSize: 2,
      streamChunks: 16,
      streamBytes: 200,
      finishReasons: ["stop", "stop"],
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
          ...ROUTE_STRATEGY,
          stream: true,
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
          ...ROUTE_STRATEGY,
          stream: true,
        },
      ],
      routeSummary: [
        { key: "continuous:eligible", route: "continuous", reason: "eligible", count: 2 },
      ],
      requests: [
        {
          ...baseRequest,
          id: "request-a",
          index: 0,
          promptTokens: 128,
          completionTokens: 32,
          totalTokens: 160,
          finishReason: "stop",
          streamChunks: 8,
          streamBytes: 100,
        },
        {
          ...baseRequest,
          id: "request-b",
          index: 1,
          promptTokens: 128,
          completionTokens: 32,
          totalTokens: 160,
          finishReason: "stop",
          streamChunks: 8,
          streamBytes: 100,
        },
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
          schedulerQueuedMs: 100,
          schedulerPhaseEvents: 3,
          schedulerAdmittedBatchSize: 2,
        },
        {
          ...baseServerRequest,
          id: "request-b",
          route: "continuous",
          routeReason: "eligible",
          modelLaneWaitMs: null,
          modelLaneQueuedAhead: null,
          modelLaneInFlightAtQueue: null,
          schedulerQueuedMs: 120,
          schedulerPhaseEvents: 4,
          schedulerAdmittedBatchSize: 2,
        },
      ],
    });
    const streamingRung = { promptTokens: 128, generationTokens: 32, concurrency: 2 };

    expect(() =>
      assertServeReportBudget(
        "qwen-stream",
        report(streamingMetrics, streamingRung),
        streamingBudget,
      ),
    ).not.toThrow();
    expect(() =>
      assertServeReportBudget(
        "qwen-stream",
        report(
          trial({
            ...streamingMetrics,
            requests: streamingMetrics.requests.map((request) =>
              request.index === 1 ? { ...request, streamChunks: 0 } : request,
            ),
          }),
          streamingRung,
        ),
        streamingBudget,
      ),
    ).toThrow("requests missing per-request output SSE evidence");
    expect(() =>
      assertServeReportBudget(
        "qwen-stream",
        report(
          trial({
            ...streamingMetrics,
            requests: streamingMetrics.requests.map((request) =>
              request.index === 1 ? { ...request, ttftMs: 1_000 } : request,
            ),
          }),
          streamingRung,
        ),
        streamingBudget,
      ),
    ).toThrow("client_request_ttft_ms");
    expect(() =>
      assertServeReportBudget(
        "qwen-stream",
        report(
          trial({
            ...streamingMetrics,
            serverRequests: streamingMetrics.serverRequests.map((request) =>
              request.id === "request-b" ? { ...request, serverStreamChunkEvents: 0 } : request,
            ),
          }),
          streamingRung,
        ),
        streamingBudget,
      ),
    ).toThrow("server_requests missing server-side output stream evidence");
    expect(() =>
      assertServeReportBudget(
        "qwen-stream",
        report(
          trial({
            ...streamingMetrics,
            serverRequests: streamingMetrics.serverRequests.map((request) =>
              request.id === "request-b" ? { ...request, schedulerQueuedMs: 1_000 } : request,
            ),
          }),
          streamingRung,
        ),
        streamingBudget,
      ),
    ).toThrow("scheduler queued budget");
  });

  test("accepts terminal-only sampled streams when lifecycle evidence is present", () => {
    const baseRequest = trial().requests[0];
    const baseServerRequest = trial().serverRequests[0];
    if (baseRequest === undefined || baseServerRequest === undefined) {
      throw new Error("Expected trial fixture to include one request and one server request.");
    }

    const lifecycleBudget: ServeRegressionBudget = {
      minCompletionTps: 10,
      maxPeakMemoryGb: 12,
      maxActiveDeltaGb: 1,
      minCompletionTokenRatio: 0.98,
      minStreamChunks: 1,
      minStreamBytes: 1,
      expectEveryRequestStreamed: true,
      expectEveryServerRequestStreamed: true,
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
    };
    const terminalOnlyMetrics = trial({
      promptTokens: 256,
      completionTokens: 32,
      totalTokens: 288,
      admissionBatches: 0,
      admissionRows: 0,
      maxAdmissionBatchSize: 0,
      continuousAdmissions: 1,
      continuousAdmissionRows: 2,
      continuousSchedulerPhases: 7,
      maxContinuousBatchSize: 2,
      maxGenerationBatchSize: 2,
      streamChunks: 4,
      streamBytes: 240,
      meanTtftMs: 120,
      meanPromptToFirstTokenTps: 10,
      meanPostTtftCompletionTps: 25,
      finishReasons: ["stop", "stop"],
      routeDecisions: [
        {
          id: "request-a",
          model: "local",
          protocol: "openai.completions",
          route: "continuous",
          eligible: true,
          reason: "eligible",
          modelType: "gemma4_text",
          maxBatchSize: 8,
          ...ROUTE_STRATEGY,
          stream: true,
        },
        {
          id: "request-b",
          model: "local",
          protocol: "openai.completions",
          route: "continuous",
          eligible: true,
          reason: "eligible",
          modelType: "gemma4_text",
          maxBatchSize: 8,
          ...ROUTE_STRATEGY,
          stream: true,
        },
      ],
      routeSummary: [
        { key: "continuous:eligible", route: "continuous", reason: "eligible", count: 2 },
      ],
      requests: [
        {
          ...baseRequest,
          id: "request-a",
          index: 0,
          promptTokens: 128,
          completionTokens: 16,
          totalTokens: 144,
          finishReason: "stop",
          streamChunks: 4,
          streamBytes: 160,
          ttftMs: 120,
        },
        {
          ...baseRequest,
          id: "request-b",
          index: 1,
          promptTokens: 128,
          completionTokens: 16,
          totalTokens: 144,
          finishReason: "stop",
          streamChunks: 0,
          streamBytes: 80,
          ttftMs: null,
          promptToFirstTokenTps: null,
          postTtftCompletionTps: null,
          meanStreamChunkGapMs: null,
          maxStreamChunkGapMs: null,
        },
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
          schedulerPhaseEvents: 3,
          schedulerAdmittedBatchSize: 2,
          serverStreamFinishReason: "stop",
        },
        {
          ...baseServerRequest,
          id: "request-b",
          route: "continuous",
          routeReason: "eligible",
          modelLaneWaitMs: null,
          modelLaneQueuedAhead: null,
          modelLaneInFlightAtQueue: null,
          schedulerPhaseEvents: 4,
          schedulerAdmittedBatchSize: 2,
          serverStreamChunkEvents: 0,
          serverStreamChunks: 4,
          serverStreamBytes: 80,
          serverStreamOutputChunks: 0,
          serverStreamOutputBytes: 0,
          serverStreamTtftMs: null,
          serverStreamDurationMs: 500,
          serverStreamResult: "completed",
          serverStreamFinishReason: "eos",
        },
      ],
    });

    expect(() =>
      assertServeReportBudget(
        "gemma-sampled-stream",
        report(terminalOnlyMetrics, { promptTokens: 128, generationTokens: 16, concurrency: 2 }),
        lifecycleBudget,
      ),
    ).not.toThrow();
    expect(() =>
      assertServeReportBudget(
        "gemma-sampled-stream",
        report(terminalOnlyMetrics, { promptTokens: 128, generationTokens: 16, concurrency: 2 }),
        {
          ...lifecycleBudget,
          expectEveryRequestOutputStreamed: true,
          expectEveryServerRequestOutputStreamed: true,
        },
      ),
    ).toThrow("output");
  });

  test("fails on throughput, memory, token, stream, route, evidence, batch, and finish regressions", () => {
    const baseRequest = trial().requests[0];
    const baseServerRequest = trial().serverRequests[0];
    if (baseRequest === undefined || baseServerRequest === undefined) {
      throw new Error("Expected trial fixture to include one request and one server request.");
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
      assertServeReportBudget(
        "qwen",
        report(
          trial({
            requests: [
              {
                ...baseRequest,
                streamBytes: 0,
              },
            ],
          }),
        ),
        budget,
      ),
    ).toThrow("requests missing per-request SSE lifecycle evidence");
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
