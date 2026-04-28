import { describe, expect, test } from "bun:test";

import { DEFAULT_SERVE_PREFILL_STEP_SIZE } from "../src/runtime/strategy";
import {
  buildServeBenchmarkRungs,
  expectedCompletionTokensForRung,
  formatServeBenchmarkRung,
  maxGenerationTokensForRung,
  maxPromptTokensForRung,
  maxTotalTokensForRung,
  parsePositiveIntegerList,
  parseServeBenchmarkArgs,
  parseServeBenchmarkMixedRungs,
  parseServeBenchmarkRungs,
  requestShapesForRung,
  rungConcurrency,
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
      requestStaggerMs: 0,
      trials: 1,
      warmup: true,
      matrix: "cartesian",
      samplingMode: "model-defaults",
      transportMode: "non-streaming",
      protocolMode: "completions",
      ignoreEos: false,
      localFilesOnly: true,
      port: 0,
      maxBatchSize: 32,
      batchWindowMs: 1,
      prefillStepSize: DEFAULT_SERVE_PREFILL_STEP_SIZE,
      activePrefillStepSize: 128,
      activeDecodeStepsPerPrefillChunk: 16,
      streamDecodeInterval: 1,
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
      "--rungs",
      "128x128,1024x512@2",
      "--request-stagger-ms",
      "25",
      "--trials",
      "3",
      "--report-json",
      ".tmp/serve-report.json",
      "--protocol",
      "chat",
      "--matrix",
      "zip",
      "--port",
      "8081",
      "--max-batch-size",
      "8",
      "--batch-window-ms",
      "2",
      "--prefill-step-size",
      "1024",
      "--active-prefill-step-size",
      "256",
      "--active-decode-steps-per-prefill-chunk",
      "24",
      "--stream-decode-interval",
      "4",
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
      rungs: [
        { promptTokens: 128, generationTokens: 128, concurrency: 1 },
        { promptTokens: 1024, generationTokens: 512, concurrency: 2 },
      ],
      reportJson: ".tmp/serve-report.json",
      trials: 3,
      warmup: false,
      matrix: "zip",
      samplingMode: "greedy",
      transportMode: "streaming",
      protocolMode: "chat",
      ignoreEos: true,
      localFilesOnly: false,
      port: 8081,
      maxBatchSize: 8,
      batchWindowMs: 2,
      prefillStepSize: 1024,
      activePrefillStepSize: 256,
      activeDecodeStepsPerPrefillChunk: 24,
      streamDecodeInterval: 4,
      maxConcurrentRequests: 2,
      requestTimeoutMs: 7_200_000,
      requestStaggerMs: 25,
      gpuMemoryUtilization: 0.75,
      maxPromptTokens: 2048,
      maxTotalTokens: 4096,
    });
  });

  test("deduplicates comma-separated integer lists", () => {
    expect(parsePositiveIntegerList("--prompt-tokens", "128,1024,128")).toEqual([128, 1024]);
  });

  test("parses explicit staggered benchmark rungs", () => {
    expect(parseServeBenchmarkRungs("--rungs", "128x128,1024x512@2,5000x128@4")).toEqual([
      { promptTokens: 128, generationTokens: 128, concurrency: 1 },
      { promptTokens: 1024, generationTokens: 512, concurrency: 2 },
      { promptTokens: 5000, generationTokens: 128, concurrency: 4 },
    ]);
    expect(() => parseServeBenchmarkRungs("--rungs", "128/128/1")).toThrow(
      "entries must look like 128x128@1",
    );
  });

  test("parses mixed per-request benchmark rungs", () => {
    const mixed = parseServeBenchmarkMixedRungs(
      "--mixed-rungs",
      "32768x128+128x32,5000x128+128x32",
    );

    expect(mixed).toEqual([
      {
        promptTokens: 32768,
        generationTokens: 128,
        concurrency: 2,
        requestShapes: [
          { promptTokens: 32768, generationTokens: 128 },
          { promptTokens: 128, generationTokens: 32 },
        ],
      },
      {
        promptTokens: 5000,
        generationTokens: 128,
        concurrency: 2,
        requestShapes: [
          { promptTokens: 5000, generationTokens: 128 },
          { promptTokens: 128, generationTokens: 32 },
        ],
      },
    ]);

    const first = mixed[0];
    if (first === undefined) {
      throw new Error("Expected mixed rung.");
    }
    expect(requestShapesForRung(first)).toEqual([
      { promptTokens: 32768, generationTokens: 128 },
      { promptTokens: 128, generationTokens: 32 },
    ]);
    expect(rungConcurrency(first)).toBe(2);
    expect(expectedCompletionTokensForRung(first)).toBe(160);
    expect(maxPromptTokensForRung(first)).toBe(32768);
    expect(maxGenerationTokensForRung(first)).toBe(128);
    expect(maxTotalTokensForRung(first)).toBe(32896);
    expect(formatServeBenchmarkRung(first)).toBe("32768x128+128x32@2");
    expect(() => parseServeBenchmarkMixedRungs("--mixed-rungs", "128x32")).toThrow(
      "need at least two request shapes",
    );
    expect(() => parseServeBenchmarkMixedRungs("--mixed-rungs", "128/32+64x16")).toThrow(
      "entries must look like 32768x128+128x32",
    );
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

    const explicit = parseServeBenchmarkArgs(["model", "--rungs", "128x128,1024x512@2"]);
    expect(buildServeBenchmarkRungs(explicit)).toEqual([
      { promptTokens: 128, generationTokens: 128, concurrency: 1 },
      { promptTokens: 1024, generationTokens: 512, concurrency: 2 },
    ]);

    const mixed = parseServeBenchmarkArgs(["model", "--mixed-rungs", "32768x128+128x32"]);
    expect(buildServeBenchmarkRungs(mixed)).toEqual([
      {
        promptTokens: 32768,
        generationTokens: 128,
        concurrency: 2,
        requestShapes: [
          { promptTokens: 32768, generationTokens: 128 },
          { promptTokens: 128, generationTokens: 32 },
        ],
      },
    ]);
  });

  test("rejects malformed values before a benchmark starts", () => {
    expect(() => parseServeBenchmarkArgs(["model", "--model-id"])).toThrow(
      "missing value for --model-id",
    );
    expect(() => parseServeBenchmarkArgs(["model", "--matrix", "diagonal"])).toThrow(
      '--matrix must be "cartesian" or "zip"',
    );
    expect(parseServeBenchmarkArgs(["model", "--protocol", "anthropic"]).protocolMode).toBe(
      "anthropic",
    );
    expect(() => parseServeBenchmarkArgs(["model", "--protocol", "invalid"])).toThrow(
      '--protocol must be "completions", "chat", "responses", or "anthropic"',
    );
    expect(() =>
      parseServeBenchmarkArgs(["model", "--protocol", "responses", "--ignore-eos"]),
    ).toThrow("--ignore-eos is not supported with --protocol responses");
    expect(() =>
      parseServeBenchmarkArgs(["model", "--protocol", "anthropic", "--ignore-eos"]),
    ).toThrow("--ignore-eos is not supported with --protocol anthropic");
    expect(() => parsePositiveIntegerList("--concurrency", "1,0")).toThrow(
      "--concurrency expects a positive integer",
    );
    expect(() => parseServeBenchmarkArgs(["model", "--request-stagger-ms", "-1"])).toThrow(
      "--request-stagger-ms expects a non-negative integer",
    );
    expect(() => parseServeBenchmarkArgs(["model", "--stream-decode-interval", "0"])).toThrow(
      "--stream-decode-interval expects a positive integer",
    );
    expect(() => parseServeBenchmarkArgs(["model", "--active-prefill-step-size", "0"])).toThrow(
      "--active-prefill-step-size expects a positive integer",
    );
    expect(() => parseServeBenchmarkArgs(["model", "--prefill-step-size", "0"])).toThrow(
      "--prefill-step-size expects a positive integer",
    );
    expect(() =>
      parseServeBenchmarkArgs(["model", "--active-decode-steps-per-prefill-chunk", "0"]),
    ).toThrow("--active-decode-steps-per-prefill-chunk expects a positive integer");
    expect(() =>
      parseServeBenchmarkArgs(["model", "--rungs", "128x32", "--mixed-rungs", "128x32+64x16"]),
    ).toThrow("use either --rungs or --mixed-rungs");
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
