import { describe, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "fs";
import { join } from "path";

import { serverRequestTimingReports, writeBenchmarkReport } from "./benchmark-serve";
import { requestLaunchDelayMs } from "./benchmark-serve-options";

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
};

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
      },
    ]);
  });
});
