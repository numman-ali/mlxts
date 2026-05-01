import { describe, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "fs";
import { join } from "path";

import {
  type BenchmarkReport,
  formatServeBenchmarkError,
  formatServeBenchmarkSuccess,
  runServeBenchmarkCommand,
  serverRequestTimingReports,
  type TrialMetrics,
  writeBenchmarkReport,
} from "./benchmark-serve";
import { formatServeBenchmarkUsage, requestLaunchDelayMs } from "./benchmark-serve-options";

const SCHEDULER_TOKENS = {
  waitingTotalTokens: 0,
  prefillingTotalTokens: 0,
  activeTotalTokens: 12,
  scheduledPromptTokens: 8,
  maxScheduledPromptTokens: 24,
  scheduledCompletionTokens: 4,
  maxScheduledCompletionTokens: 8,
  scheduledTotalTokens: 12,
  maxScheduledTotalTokens: 32,
  scheduledMemoryBytes: 1024,
  maxScheduledMemoryBytes: 4096,
};

function trialMetrics(overrides: Partial<TrialMetrics> = {}): TrialMetrics {
  return {
    wallMs: 1000,
    requestTps: 1,
    completionTps: 16,
    totalTps: 144,
    meanTtftMs: 100,
    meanPromptToFirstTokenTps: 1280,
    meanServerPrefillMs: 80,
    meanServerPrefillTps: 1600,
    meanPostTtftCompletionTps: 20,
    meanStreamChunkGapMs: null,
    maxStreamChunkGapMs: null,
    promptTokens: 128,
    completionTokens: 16,
    totalTokens: 144,
    cacheReadTokens: 64,
    cacheWriteTokens: 128,
    meanRequestMs: 1000,
    p95RequestMs: 1000,
    maxRequestMs: 1000,
    peakMemoryGb: 1.25,
    activeMemoryGb: 1,
    cacheMemoryGb: 0.25,
    activeDeltaGb: 0.5,
    admissionBatches: 1,
    admissionRows: 1,
    maxAdmissionBatchSize: 1,
    staticBatches: 1,
    staticBatchRows: 1,
    continuousAdmissions: 0,
    continuousAdmissionRows: 0,
    continuousSchedulerPhases: 0,
    maxContinuousBatchSize: 0,
    maxGenerationBatchSize: 1,
    promptCacheHits: 1,
    promptCacheMisses: 1,
    promptCacheWrites: 1,
    promptCacheReadTokens: 64,
    promptCacheWriteTokens: 128,
    promptCacheRetainedSnapshots: 2,
    promptCacheRetainedSnapshotBytes: 4096,
    promptCacheIndexedBlockHashes: 4,
    promptCacheTokenBlocks: 3,
    promptCacheTokenBlockReferences: 4,
    promptCacheUniqueTokenCount: 192,
    promptCacheReferencedTokenCount: 256,
    streamChunks: 0,
    streamBytes: 0,
    finishReasons: ["length"],
    routeDecisions: [],
    routeSummary: [{ key: "static:eligible", route: "static", reason: "eligible", count: 1 }],
    requests: [],
    serverRequests: [],
    ...overrides,
  };
}

function benchmarkReport(overrides: Partial<BenchmarkReport> = {}): BenchmarkReport {
  const metrics = trialMetrics();
  return {
    createdAt: "2026-04-30T00:00:00.000Z",
    model: "model",
    modelId: "local-model",
    snapshotPath: "/tmp/model",
    samplingMode: "greedy",
    transportMode: "non-streaming",
    protocolMode: "completions",
    ignoreEos: true,
    maxBatchSize: 8,
    batchWindowMs: 2,
    prefillStepSize: 512,
    activePrefillStepSize: 128,
    activeDecodeStepsPerPrefillChunk: 16,
    streamDecodeInterval: 1,
    requestStaggerMs: 25,
    maxConcurrentRequests: 1,
    gpuMemoryUtilization: 0.9,
    rungs: [
      {
        rung: { promptTokens: 128, generationTokens: 16, concurrency: 1 },
        arrivalSpanMs: 0,
        trials: [metrics],
        averages: metrics,
      },
    ],
    ...overrides,
  };
}

describe("serve benchmark reports", () => {
  test("computes staggered request launch offsets", () => {
    expect([0, 1, 2, 3].map((index) => requestLaunchDelayMs(index, 25))).toEqual([0, 25, 50, 75]);
  });

  test("creates parent directories before writing JSON evidence", async () => {
    const directory = mkdtempSync("/tmp/mlxts-serve-benchmark-");
    const reportPath = join(directory, "nested", "serve-report.json");

    try {
      await writeBenchmarkReport(reportPath, {
        createdAt: "2026-04-24T00:00:00.000Z",
        model: "model",
        modelId: "local",
        snapshotPath: "/tmp/model",
        samplingMode: "greedy",
        transportMode: "non-streaming",
        protocolMode: "completions",
        ignoreEos: true,
        maxBatchSize: 8,
        batchWindowMs: 2,
        prefillStepSize: 512,
        activePrefillStepSize: 128,
        activeDecodeStepsPerPrefillChunk: 16,
        streamDecodeInterval: 1,
        requestStaggerMs: 25,
        maxConcurrentRequests: 1,
        gpuMemoryUtilization: 0.9,
        rungs: [],
      });

      expect(existsSync(reportPath)).toBe(true);
      expect(JSON.parse(readFileSync(reportPath, "utf8"))).toMatchObject({
        model: "model",
        modelId: "local",
        requestStaggerMs: 25,
        rungs: [],
      });
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  test("formats compact AXI success and error output", () => {
    expect(formatServeBenchmarkSuccess(benchmarkReport(), ".tmp/serve-report.json")).toBe(
      [
        "serve_benchmark:",
        "  status: passed",
        '  model: "model"',
        '  model_id: "local-model"',
        '  snapshot_path: "/tmp/model"',
        '  transport: "non-streaming"',
        '  protocol: "completions"',
        '  sampling: "greedy"',
        "  rungs: 1",
        '  report_json: ".tmp/serve-report.json"',
        "rungs[1]{rung,trials,completion_tps,total_tps,mean_ttft_ms,p95_request_ms,prompt_cache_hits,prompt_cache_read_tokens,prompt_cache_retained_snapshots,prompt_cache_token_blocks,routes}:",
        '  "128x16@1",1,16.000,144.000,100.0,1000.0,1,64,2,3,"static:eligible=1"',
      ].join("\n"),
    );
    expect(formatServeBenchmarkError("bad flag", "rerun")).toBe(
      ["error:", '  message: "bad flag"', 'help: "rerun"'].join("\n"),
    );
    expect(formatServeBenchmarkError("bad\nflag", "rerun")).toContain("  message: |");
  });

  test("runs help, success, usage error, and runtime error paths with AXI stdout", async () => {
    const helpStdout: string[] = [];
    expect(
      await runServeBenchmarkCommand(["--help"], {
        stdout: (text) => helpStdout.push(text),
      }),
    ).toBe(0);
    expect(helpStdout.join("\n")).toBe(formatServeBenchmarkUsage());

    const stdout: string[] = [];
    const stderr: string[] = [];
    const writtenReports: string[] = [];
    let lockDepth = 0;
    expect(
      await runServeBenchmarkCommand(["--model", "model", "--report-json", ".tmp/report.json"], {
        stdout: (text) => stdout.push(text),
        stderr: (text) => stderr.push(text),
        acquireLock: () => {
          lockDepth += 1;
          return {
            [Symbol.dispose]: () => {
              lockDepth -= 1;
            },
          };
        },
        runBenchmark: async (options, progress) => {
          expect(options.model).toBe("model");
          expect(lockDepth).toBe(1);
          progress("benchmark-serve: probe");
          return benchmarkReport();
        },
        writeReport: async (path) => {
          writtenReports.push(path);
        },
      }),
    ).toBe(0);
    expect(lockDepth).toBe(0);
    expect(stderr).toEqual(["benchmark-serve: probe"]);
    expect(writtenReports).toEqual([".tmp/report.json"]);
    expect(stdout.join("\n")).toContain("serve_benchmark:");

    const usageStdout: string[] = [];
    let usageLockCalls = 0;
    expect(
      await runServeBenchmarkCommand(["--trials", "0"], {
        stdout: (text) => usageStdout.push(text),
        acquireLock: () => {
          usageLockCalls += 1;
          return { [Symbol.dispose]: () => undefined };
        },
      }),
    ).toBe(2);
    expect(usageLockCalls).toBe(0);
    expect(usageStdout.join("\n")).toContain("--trials expects a positive integer");

    const runtimeStdout: string[] = [];
    let runtimeLockDepth = 0;
    expect(
      await runServeBenchmarkCommand(["--model", "model"], {
        stdout: (text) => runtimeStdout.push(text),
        acquireLock: () => {
          runtimeLockDepth += 1;
          return {
            [Symbol.dispose]: () => {
              runtimeLockDepth -= 1;
            },
          };
        },
        runBenchmark: async () => {
          throw new Error("benchmark failed");
        },
      }),
    ).toBe(1);
    expect(runtimeLockDepth).toBe(0);
    expect(runtimeStdout.join("\n")).toContain("benchmark failed");
  });

  test("keeps model-lane wait timing in server request reports", () => {
    expect(
      serverRequestTimingReports([
        {
          type: "generation_start",
          id: "request",
          protocol: "openai.completions",
          model: "local",
          inputKind: "text",
          maxTokens: 8,
          stream: false,
          observedAtMs: 100,
        },
        {
          type: "generation_model_lane_wait",
          id: "request",
          protocol: "openai.completions",
          model: "local",
          lane: "model",
          waitMs: 12,
          inFlightAtQueue: 1,
          queuedAhead: 2,
          inFlightAtDispatch: 1,
          queuedAtDispatch: 0,
          maxConcurrentJobs: 1,
          observedAtMs: 112,
        },
        {
          type: "generation_complete",
          id: "request",
          protocol: "openai.completions",
          model: "local",
          finishReason: "length",
          completionTokens: 8,
          durationMs: 40,
          observedAtMs: 140,
        },
      ]),
    ).toMatchObject([
      {
        id: "request",
        modelLaneWaitMs: 12,
        modelLaneQueuedAhead: 2,
        modelLaneInFlightAtQueue: 1,
      },
    ]);
  });

  test("keeps prompt-prefix cache evidence in server request reports", () => {
    expect(
      serverRequestTimingReports([
        {
          type: "generation_start",
          id: "request",
          protocol: "openai.chat_completions",
          model: "local",
          inputKind: "messages",
          maxTokens: 8,
          stream: true,
          observedAtMs: 100,
        },
        {
          type: "generation_prompt_cache",
          id: "request",
          protocol: "openai.chat_completions",
          model: "local",
          result: "write",
          promptTokens: 12,
          cacheReadTokens: 0,
          cacheWriteTokens: 12,
          retainedSnapshots: 1,
          retainedSnapshotBytes: 1024,
          indexedBlockHashes: 2,
          tokenBlockSize: 64,
          tokenBlockCount: 1,
          tokenBlockReferences: 1,
          uniqueTokenCount: 64,
          referencedTokenCount: 64,
          observedAtMs: 105,
        },
        {
          type: "generation_prompt_cache",
          id: "request",
          protocol: "openai.chat_completions",
          model: "local",
          result: "hit",
          promptTokens: 12,
          cacheReadTokens: 12,
          cacheWriteTokens: 0,
          cacheMatchType: "exact",
          cacheSourceTokenLength: 12,
          cacheSourceSnapshotOffset: 12,
          cacheSourceEstimatedBytes: 1024,
          cacheSourceLayerKinds: ["full"],
          cacheSourceTrimmable: true,
          cacheSourceTokenBlockSize: 64,
          cacheSourceTokenBlockCount: 1,
          cacheSourceBlockAlignedTokenLength: 0,
          retainedSnapshots: 1,
          retainedSnapshotBytes: 1024,
          indexedBlockHashes: 2,
          tokenBlockSize: 64,
          tokenBlockCount: 1,
          tokenBlockReferences: 1,
          uniqueTokenCount: 64,
          referencedTokenCount: 64,
          observedAtMs: 110,
        },
        {
          type: "generation_prompt_cache",
          id: "request",
          protocol: "openai.chat_completions",
          model: "local",
          result: "miss",
          promptTokens: 4,
          cacheReadTokens: 0,
          cacheWriteTokens: 0,
          retainedSnapshots: 1,
          retainedSnapshotBytes: 1024,
          indexedBlockHashes: 2,
          tokenBlockSize: 64,
          tokenBlockCount: 1,
          tokenBlockReferences: 1,
          uniqueTokenCount: 64,
          referencedTokenCount: 64,
          observedAtMs: 115,
        },
      ]),
    ).toMatchObject([
      {
        id: "request",
        promptCacheEvents: 3,
        promptCacheHits: 1,
        promptCacheMisses: 1,
        promptCacheWrites: 1,
        promptCacheReadTokens: 12,
        promptCacheWriteTokens: 12,
        promptCacheRetainedSnapshots: 1,
        promptCacheRetainedSnapshotBytes: 1024,
        promptCacheIndexedBlockHashes: 2,
        promptCacheTokenBlocks: 1,
        promptCacheTokenBlockReferences: 1,
        promptCacheUniqueTokenCount: 64,
        promptCacheReferencedTokenCount: 64,
      },
    ]);
  });

  test("keeps server-side stream timing in server request reports", () => {
    expect(
      serverRequestTimingReports([
        {
          type: "generation_start",
          id: "request",
          protocol: "openai.completions",
          model: "local",
          inputKind: "text",
          maxTokens: 8,
          stream: true,
          observedAtMs: 100,
        },
        {
          type: "generation_stream_chunk",
          id: "request",
          protocol: "openai.completions",
          model: "local",
          chunkIndex: 1,
          elapsedMs: 25,
          bytes: 64,
          observedAtMs: 125,
        },
        {
          type: "generation_stream_chunk",
          id: "request",
          protocol: "openai.completions",
          model: "local",
          chunkIndex: 2,
          elapsedMs: 40,
          bytes: 32,
          observedAtMs: 140,
        },
        {
          type: "generation_stream_end",
          id: "request",
          protocol: "openai.completions",
          model: "local",
          result: "completed",
          finishReason: "length",
          chunks: 4,
          bytes: 160,
          outputChunks: 2,
          outputBytes: 96,
          ttftMs: 25,
          durationMs: 80,
          observedAtMs: 180,
        },
        {
          type: "generation_complete",
          id: "request",
          protocol: "openai.completions",
          model: "local",
          finishReason: "length",
          completionTokens: 8,
          durationMs: 82,
          observedAtMs: 182,
        },
      ]),
    ).toMatchObject([
      {
        id: "request",
        serverStreamChunkEvents: 2,
        serverStreamEndEvents: 1,
        serverStreamFirstChunkMs: 25,
        serverStreamLastChunkMs: 40,
        serverStreamChunks: 4,
        serverStreamBytes: 160,
        serverStreamOutputChunks: 2,
        serverStreamOutputBytes: 96,
        serverStreamTtftMs: 25,
        serverStreamDurationMs: 80,
        serverStreamResult: "completed",
        serverStreamFinishReason: "length",
      },
    ]);
  });

  test("keeps server-side prefill timing separate from client TTFT", () => {
    expect(
      serverRequestTimingReports([
        {
          type: "generation_start",
          id: "request",
          protocol: "openai.completions",
          model: "local",
          inputKind: "tokens",
          maxTokens: 8,
          stream: true,
          observedAtMs: 100,
        },
        {
          type: "generation_model_lane_wait",
          id: "request",
          protocol: "openai.completions",
          model: "local",
          lane: "model",
          waitMs: 0,
          inFlightAtQueue: 0,
          queuedAhead: 0,
          inFlightAtDispatch: 1,
          queuedAtDispatch: 0,
          maxConcurrentJobs: 1,
          observedAtMs: 110,
        },
        {
          type: "generation_progress",
          id: "request",
          protocol: "openai.completions",
          model: "local",
          promptTokens: 129,
          completionTokens: 0,
          maxTokens: 8,
          observedAtMs: 110,
        },
        {
          type: "generation_prefill_progress",
          id: "request",
          protocol: "openai.completions",
          model: "local",
          promptTokens: 129,
          processedPrefillTokens: 128,
          totalPrefillTokens: 128,
          chunkTokens: 128,
          maxTokens: 8,
          observedAtMs: 610,
        },
        {
          type: "generation_stream_chunk",
          id: "request",
          protocol: "openai.completions",
          model: "local",
          chunkIndex: 1,
          elapsedMs: 700,
          bytes: 64,
          observedAtMs: 800,
        },
      ]),
    ).toMatchObject([
      {
        id: "request",
        serverPrefillStartMs: 10,
        serverPrefillEndMs: 510,
        serverPrefillMs: 500,
        serverPrefillTokens: 128,
        serverPrefillTps: 256,
        serverStreamFirstChunkMs: 700,
      },
    ]);
  });

  test("keeps continuous scheduler phase timing in server request reports", () => {
    expect(
      serverRequestTimingReports([
        {
          type: "generation_start",
          id: "request",
          protocol: "openai.completions",
          model: "local",
          inputKind: "tokens",
          maxTokens: 8,
          stream: false,
          observedAtMs: 100,
        },
        {
          type: "generation_scheduler_phase",
          mode: "continuous",
          phase: "queued",
          model: "local",
          id: "request",
          ids: ["request"],
          queuedAhead: 0,
          promptTokens: 4,
          maxTokens: 8,
          schedulerMs: 0,
          waiting: 1,
          prefilling: 0,
          active: 0,
          maxBatchSize: 2,
          ...SCHEDULER_TOKENS,
          observedAtMs: 101,
        },
        {
          type: "generation_scheduler_phase",
          mode: "continuous",
          phase: "admitted",
          model: "local",
          ids: ["request"],
          batchSize: 1,
          maxTokens: 8,
          maxTokensByRequest: [8],
          queuedMsByRequest: [12],
          schedulerMs: 12,
          waiting: 0,
          prefilling: 0,
          active: 1,
          maxBatchSize: 2,
          ...SCHEDULER_TOKENS,
          observedAtMs: 112,
        },
        {
          type: "generation_scheduler_phase",
          mode: "continuous",
          phase: "first_token",
          model: "local",
          id: "request",
          ids: ["request"],
          completionTokens: 1,
          queuedMs: 15,
          schedulerMs: 15,
          waiting: 0,
          prefilling: 0,
          active: 1,
          maxBatchSize: 2,
          ...SCHEDULER_TOKENS,
          observedAtMs: 115,
        },
        {
          type: "generation_scheduler_phase",
          mode: "continuous",
          phase: "finished",
          model: "local",
          id: "request",
          ids: ["request"],
          completionTokens: 8,
          finishReason: "length",
          queuedMs: 40,
          schedulerMs: 40,
          waiting: 0,
          prefilling: 0,
          active: 1,
          maxBatchSize: 2,
          ...SCHEDULER_TOKENS,
          observedAtMs: 140,
        },
      ]),
    ).toMatchObject([
      {
        id: "request",
        schedulerQueuedMs: 12,
        schedulerAdmittedMs: 12,
        schedulerFirstTokenMs: 15,
        schedulerFinishedMs: 40,
        schedulerPhaseEvents: 4,
        schedulerAdmittedBatchSize: 1,
        schedulerScheduledPromptTokens: 8,
        schedulerMaxScheduledPromptTokens: 24,
        schedulerScheduledCompletionTokens: 4,
        schedulerMaxScheduledCompletionTokens: 8,
        schedulerScheduledTotalTokens: 12,
        schedulerMaxScheduledTotalTokens: 32,
        schedulerScheduledMemoryBytes: 1024,
        schedulerMaxScheduledMemoryBytes: 4096,
      },
    ]);
  });
});
