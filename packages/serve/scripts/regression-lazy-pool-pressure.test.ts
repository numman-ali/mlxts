import { describe, expect, test } from "bun:test";

import type { GenerationMemoryUsage } from "../src/types";
import {
  assertLazyPoolPressureReport,
  type LazyPoolPressureReport,
  parseLazyPoolPressureArgs,
  pressureGpuMemoryUtilization,
} from "./regression-lazy-pool-pressure";

const memory: GenerationMemoryUsage = {
  activeBytes: 0,
  cacheBytes: 0,
  peakBytes: 0,
  limitBytes: 64 * 1024 * 1024 * 1024,
};

function report(overrides: Partial<LazyPoolPressureReport> = {}): LazyPoolPressureReport {
  return {
    createdAt: "2026-04-30T00:00:00.000Z",
    activeModel: "gemma",
    blockedModel: "qwen",
    activeModelId: "gemma-pressure-active",
    blockedModelId: "qwen-pressure-blocked",
    gpuMemoryUtilization: 0.4,
    memory,
    memorySnapshots: [
      { stage: "start", observedAtMs: 0, ...memory },
      { stage: "after_active_first_chunk", observedAtMs: 10, ...memory },
      { stage: "after_pressure_event", observedAtMs: 20, ...memory },
      { stage: "after_blocked_completion", observedAtMs: 30, ...memory },
      { stage: "after_server_stop", observedAtMs: 40, ...memory },
    ],
    estimates: {
      active: {
        source: "gemma",
        snapshotPath: "/tmp/gemma",
        safetensorBytes: 4,
        estimatedBytes: 5,
      },
      blocked: {
        source: "qwen",
        snapshotPath: "/tmp/qwen",
        safetensorBytes: 20,
        estimatedBytes: 25,
      },
    },
    requests: {
      active: {
        id: "chatcmpl-active",
        status: 200,
        chunks: 1,
        bytes: 100,
        firstChunkMs: 10,
        done: true,
      },
      blocked: { status: 200, outputChars: 12, finishReason: "length", durationMs: 1000 },
    },
    pressure: {
      events: 1,
      abortActiveEvents: 1,
      abortedRequestIds: ["chatcmpl-active"],
      actions: [
        {
          targetModel: "qwen-pressure-blocked",
          action: "abort_active",
          reason: "model_load_memory_exceeded",
          evictedModels: [],
          abortedRequestIds: ["chatcmpl-active"],
        },
      ],
    },
    metrics: {
      pressureLines: ['mlxts_serve_model_pool_pressure_events_total{model="qwen"} 1'],
    },
    ...overrides,
  };
}

describe("lazy pool pressure regression", () => {
  test("parses defaults and explicit smoke options", () => {
    expect(parseLazyPoolPressureArgs([])).toMatchObject({
      qwenModel: "mlx-community/Qwen3.6-27B-4bit",
      gemma4Model: "google/gemma-4-E2B-it",
      reportDir: ".tmp/lazy-pool-pressure",
      requestTimeoutMs: 3_600_000,
      activeMaxTokens: 2048,
      blockedMaxTokens: 8,
      pressureReleaseTimeoutMs: 120_000,
      budgetMultiplier: 1.05,
      allowDownload: false,
    });

    expect(
      parseLazyPoolPressureArgs([
        "--qwen-model",
        "qwen",
        "--gemma4-model",
        "gemma",
        "--report-dir",
        ".tmp/reports",
        "--request-timeout-ms",
        "120000",
        "--active-max-tokens",
        "256",
        "--blocked-max-tokens",
        "4",
        "--pressure-release-timeout-ms",
        "5000",
        "--budget-multiplier",
        "1.1",
        "--allow-download",
      ]),
    ).toMatchObject({
      qwenModel: "qwen",
      gemma4Model: "gemma",
      reportDir: ".tmp/reports",
      requestTimeoutMs: 120000,
      activeMaxTokens: 256,
      blockedMaxTokens: 4,
      pressureReleaseTimeoutMs: 5000,
      budgetMultiplier: 1.1,
      allowDownload: true,
    });
  });

  test("rejects invalid option values", () => {
    expect(() => parseLazyPoolPressureArgs(["--request-timeout-ms", "0"])).toThrow(
      "--request-timeout-ms must be a positive integer",
    );
    expect(() => parseLazyPoolPressureArgs(["--request-timeout-ms", "10abc"])).toThrow(
      "--request-timeout-ms must be a positive integer",
    );
    expect(() => parseLazyPoolPressureArgs(["--budget-multiplier", "0"])).toThrow(
      "--budget-multiplier must be a positive number",
    );
    expect(() => parseLazyPoolPressureArgs(["--budget-multiplier", "1.1x"])).toThrow(
      "--budget-multiplier must be a positive number",
    );
  });

  test("computes a constrained memory fraction from model estimates", () => {
    const qwenEstimate = 20 * 1024 * 1024 * 1024;
    const gemmaEstimate = 4 * 1024 * 1024 * 1024;
    const utilization = pressureGpuMemoryUtilization({
      memory,
      activeEstimatedBytes: gemmaEstimate,
      blockedEstimatedBytes: qwenEstimate,
      multiplier: 1.05,
    });

    expect(utilization).toBeGreaterThan(0.3);
    expect(utilization).toBeLessThan(0.4);
  });

  test("rejects memory budgets that cannot constrain the blocked load", () => {
    expect(() =>
      pressureGpuMemoryUtilization({
        memory: { ...memory, limitBytes: 8 * 1024 * 1024 * 1024 },
        activeEstimatedBytes: 4 * 1024 * 1024 * 1024,
        blockedEstimatedBytes: 20 * 1024 * 1024 * 1024,
        multiplier: 1.05,
      }),
    ).toThrow("model estimates exceed the available MLX memory limit");
  });

  test("validates required pressure evidence in reports", () => {
    expect(() => assertLazyPoolPressureReport(report())).not.toThrow();
    expect(() =>
      assertLazyPoolPressureReport(
        report({ pressure: { ...report().pressure, abortActiveEvents: 0 } }),
      ),
    ).toThrow("expected at least one active-request pressure abort");
    expect(() =>
      assertLazyPoolPressureReport(
        report({
          requests: {
            ...report().requests,
            active: { ...report().requests.active, id: "other" },
          },
        }),
      ),
    ).toThrow("pressure abort did not target the active stream id");
    expect(() =>
      assertLazyPoolPressureReport(
        report({
          pressure: {
            ...report().pressure,
            actions: [
              {
                targetModel: "other",
                action: "abort_active",
                reason: "model_load_memory_exceeded",
                evictedModels: [],
                abortedRequestIds: ["chatcmpl-active"],
              },
            ],
          },
        }),
      ),
    ).toThrow("pressure abort action did not target the blocked load");
    expect(() =>
      assertLazyPoolPressureReport(
        report({
          requests: {
            ...report().requests,
            blocked: { ...report().requests.blocked, outputChars: 0 },
          },
        }),
      ),
    ).toThrow("blocked request did not complete");
    expect(() =>
      assertLazyPoolPressureReport(
        report({
          memorySnapshots: report().memorySnapshots.filter(
            (snapshot) => snapshot.stage !== "after_pressure_event",
          ),
        }),
      ),
    ).toThrow("report missing memory snapshot after_pressure_event");
  });
});
