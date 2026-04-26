import { describe, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "fs";
import { join } from "path";

import { serverRequestTimingReports, writeBenchmarkReport } from "./benchmark-serve";
import { requestLaunchDelayMs } from "./benchmark-serve-options";

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
      },
    ]);
  });
});
