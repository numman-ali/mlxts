import { describe, expect, test } from "bun:test";

import { parseQwenGemmaRegressionArgs } from "./regression-qwen-gemma";

describe("Qwen/Gemma regression options", () => {
  test("defaults to the cheap profile", () => {
    expect(parseQwenGemmaRegressionArgs([])).toMatchObject({
      profile: "quick",
      qwenModel: "mlx-community/Qwen3.6-27B-4bit",
      gemma4Model: "google/gemma-4-E2B-it",
      reportDir: ".tmp/qwen-gemma-regression",
      requestTimeoutMs: 3_600_000,
      allowDownload: false,
    });
  });

  test("parses substantial profile options", () => {
    expect(
      parseQwenGemmaRegressionArgs([
        "--profile",
        "substantial",
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
    ).toEqual({
      profile: "substantial",
      qwenModel: "qwen",
      gemma4Model: "gemma",
      reportDir: ".tmp/reports",
      requestTimeoutMs: 120000,
      allowDownload: true,
    });
  });

  test("rejects invalid profile and timeout values", () => {
    expect(() => parseQwenGemmaRegressionArgs(["--profile", "overnight"])).toThrow(
      '--profile must be "quick", "real", or "substantial"',
    );
    expect(() => parseQwenGemmaRegressionArgs(["--request-timeout-ms", "0"])).toThrow(
      "--request-timeout-ms must be a positive integer",
    );
  });
});
