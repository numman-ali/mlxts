import { describe, expect, test } from "bun:test";

import {
  buildServeBenchmarkRungs,
  parsePositiveIntegerList,
  parseServeBenchmarkArgs,
} from "./benchmark-serve-options";

describe("serve benchmark options", () => {
  test("parses positional model defaults for cached local endpoint runs", () => {
    const parsed = parseServeBenchmarkArgs(["mlx-community/Qwen3.6-27B-4bit"]);

    expect(parsed).toEqual({
      model: "mlx-community/Qwen3.6-27B-4bit",
      modelId: "Qwen3.6-27B-4bit",
      promptTokens: [128],
      generationTokens: [128],
      concurrency: [1],
      trials: 1,
      warmup: true,
      matrix: "cartesian",
      samplingMode: "model-defaults",
      transportMode: "non-streaming",
      ignoreEos: false,
      localFilesOnly: true,
      port: 0,
      maxBatchSize: 32,
      batchWindowMs: 1,
      maxConcurrentRequests: 1,
      requestTimeoutMs: 3_600_000,
      gpuMemoryUtilization: 0.9,
    });
  });

  test("parses benchmark, sampling, admission, and download flags", () => {
    const parsed = parseServeBenchmarkArgs([
      "--model",
      "google/gemma-4-E2B-it",
      "--model-id",
      "gemma-local",
      "--prompt-tokens",
      "128,1024",
      "--generation-tokens",
      "64,256",
      "--concurrency",
      "1,4",
      "--trials",
      "3",
      "--matrix",
      "zip",
      "--port",
      "8081",
      "--max-batch-size",
      "8",
      "--batch-window-ms",
      "2",
      "--max-concurrent-requests",
      "2",
      "--gpu-memory-utilization",
      "0.75",
      "--request-timeout-ms",
      "7200000",
      "--max-prompt-tokens",
      "2048",
      "--max-total-tokens",
      "4096",
      "--greedy",
      "--stream",
      "--ignore-eos",
      "--no-warmup",
      "--allow-download",
    ]);

    expect(parsed).toEqual({
      model: "google/gemma-4-E2B-it",
      modelId: "gemma-local",
      promptTokens: [128, 1024],
      generationTokens: [64, 256],
      concurrency: [1, 4],
      trials: 3,
      warmup: false,
      matrix: "zip",
      samplingMode: "greedy",
      transportMode: "streaming",
      ignoreEos: true,
      localFilesOnly: false,
      port: 8081,
      maxBatchSize: 8,
      batchWindowMs: 2,
      maxConcurrentRequests: 2,
      requestTimeoutMs: 7_200_000,
      gpuMemoryUtilization: 0.75,
      maxPromptTokens: 2048,
      maxTotalTokens: 4096,
    });
  });

  test("deduplicates comma-separated integer lists", () => {
    expect(parsePositiveIntegerList("--prompt-tokens", "128,1024,128")).toEqual([128, 1024]);
  });

  test("builds cartesian and zip benchmark rungs", () => {
    const base = parseServeBenchmarkArgs([
      "model",
      "--prompt-tokens",
      "128,1024",
      "--generation-tokens",
      "32,64",
      "--concurrency",
      "1,2",
    ]);

    expect(buildServeBenchmarkRungs(base)).toEqual([
      { promptTokens: 128, generationTokens: 32, concurrency: 1 },
      { promptTokens: 128, generationTokens: 32, concurrency: 2 },
      { promptTokens: 128, generationTokens: 64, concurrency: 1 },
      { promptTokens: 128, generationTokens: 64, concurrency: 2 },
      { promptTokens: 1024, generationTokens: 32, concurrency: 1 },
      { promptTokens: 1024, generationTokens: 32, concurrency: 2 },
      { promptTokens: 1024, generationTokens: 64, concurrency: 1 },
      { promptTokens: 1024, generationTokens: 64, concurrency: 2 },
    ]);

    const zipped = parseServeBenchmarkArgs([
      "model",
      "--prompt-tokens",
      "128,1024",
      "--generation-tokens",
      "32,64",
      "--concurrency",
      "4",
      "--matrix",
      "zip",
    ]);

    expect(buildServeBenchmarkRungs(zipped)).toEqual([
      { promptTokens: 128, generationTokens: 32, concurrency: 4 },
      { promptTokens: 1024, generationTokens: 64, concurrency: 4 },
    ]);
  });

  test("rejects malformed values before a benchmark starts", () => {
    expect(() => parseServeBenchmarkArgs(["model", "--model-id"])).toThrow(
      "missing value for --model-id",
    );
    expect(() => parseServeBenchmarkArgs(["model", "--matrix", "diagonal"])).toThrow(
      '--matrix must be "cartesian" or "zip"',
    );
    expect(() => parsePositiveIntegerList("--concurrency", "1,0")).toThrow(
      "--concurrency expects a positive integer",
    );
    expect(() =>
      buildServeBenchmarkRungs(
        parseServeBenchmarkArgs([
          "model",
          "--prompt-tokens",
          "128,1024",
          "--generation-tokens",
          "64",
          "--matrix",
          "zip",
        ]),
      ),
    ).toThrow("zip matrix requires equal prompt and generation counts");
  });
});
