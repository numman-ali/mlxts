import { describe, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "fs";
import { join } from "path";

import { writeBenchmarkReport } from "./benchmark-serve";
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
});
