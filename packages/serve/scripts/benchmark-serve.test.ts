import { describe, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "fs";
import { join } from "path";

import { writeBenchmarkReport } from "./benchmark-serve";

describe("serve benchmark reports", () => {
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
        ignoreEos: true,
        maxBatchSize: 8,
        batchWindowMs: 2,
        maxConcurrentRequests: 1,
        gpuMemoryUtilization: 0.9,
        rungs: [],
      });

      expect(existsSync(reportPath)).toBe(true);
      expect(JSON.parse(readFileSync(reportPath, "utf8"))).toMatchObject({
        model: "model",
        modelId: "local",
        rungs: [],
      });
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });
});
