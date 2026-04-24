import { describe, expect, test } from "bun:test";
import type { Tokenizer } from "@mlxts/tokenizers";
import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import {
  type BenchmarkBaselines,
  compareAgainstBaseline,
  compareAgainstMlxLmReference,
  formatMlxLmReference,
  parseBaselineData,
  parseBenchmarkArgs,
  readBenchmarkVocabSize,
  resolveCachedSnapshotPath,
  safeDecodedTokenLength,
  selectTargets,
} from "./benchmark-common";

function tempDirectory(prefix: string): string {
  return mkdtempSync(join(tmpdir(), prefix));
}

describe("benchmark-common", () => {
  test("parseBenchmarkArgs reads model and integer flags", () => {
    const parsed = parseBenchmarkArgs([
      "--model",
      "google/gemma-3-1b-it",
      "--prompt-tokens",
      "2048",
      "--generation-tokens",
      "64",
      "--trials",
      "5",
      "--prefill-step-size",
      "1024",
      "--memory-sample-interval",
      "16",
      "--sync-decode",
      "--metal-trace",
    ]);

    expect(parsed.model).toBe("google/gemma-3-1b-it");
    expect(parsed.options).toEqual({
      promptTokens: 2048,
      generationTokens: 64,
      trials: 5,
      prefillStepSize: 1024,
      metalTrace: true,
      memorySampleInterval: 16,
      decodeSchedule: "sync",
      materializeCacheEachToken: false,
    });
    expect(parsed.reference).toEqual({
      captureMlxLmReference: true,
      enforceMlxLmDecodeBar: false,
      requireMlxLmReference: false,
      mlxLmPython: undefined,
    });
  });

  test("parseBenchmarkArgs reads mlx-lm reference flags", () => {
    const parsed = parseBenchmarkArgs([
      "--model",
      "google/gemma-4-E2B-it",
      "--capture-mlx-lm-reference",
      "--enforce-mlx-lm-decode-bar",
      "--require-mlx-lm-reference",
      "--mlx-lm-python",
      "/tmp/venv/bin/python",
    ]);

    expect(parsed.reference).toEqual({
      captureMlxLmReference: true,
      enforceMlxLmDecodeBar: true,
      requireMlxLmReference: true,
      mlxLmPython: "/tmp/venv/bin/python",
    });
  });

  test("parseBenchmarkArgs can skip live mlx-lm capture for local profiling", () => {
    const parsed = parseBenchmarkArgs([
      "--model",
      "mlx-community/Qwen3.6-27B-4bit",
      "--skip-mlx-lm-reference",
    ]);

    expect(parsed.reference.captureMlxLmReference).toBe(false);
  });

  test("parseBenchmarkArgs rejects requiring and skipping mlx-lm reference together", () => {
    expect(() =>
      parseBenchmarkArgs([
        "--model",
        "mlx-community/Qwen3.6-27B-4bit",
        "--skip-mlx-lm-reference",
        "--require-mlx-lm-reference",
      ]),
    ).toThrow("cannot be combined");
  });

  test("parseBaselineData requires synthetic and parity sections", () => {
    expect(() => parseBaselineData({ targets: [] })).toThrow("synthetic");
  });

  test("selectTargets uses the requested benchmark section and CLI overrides", () => {
    const baselines: BenchmarkBaselines = {
      synthetic: {
        targets: [
          {
            name: "gemma",
            model: "google/gemma-3-1b-it",
            promptTokens: 1024,
            generationTokens: 128,
            prefillStepSize: 2048,
            promptTps: 4000,
            explicitEvalCountPerToken: 1,
          },
        ],
      },
      parity: {
        targets: [
          {
            name: "gemma-parity",
            model: "google/gemma-3-1b-it",
            promptTokens: 1024,
            generationTokens: 128,
            prefillStepSize: 2048,
            generationTps: 50,
            mlxLmReference: {
              promptTps: 45,
              generationTps: 49,
              peakMemoryGb: 2.5,
              capturedAt: "2026-04-05",
            },
          },
        ],
      },
    };

    const selected = selectTargets("parity", baselines, {
      model: "google/gemma-3-1b-it",
      options: {
        promptTokens: 512,
        generationTokens: 32,
        trials: 1,
        prefillStepSize: 256,
        metalTrace: false,
        memorySampleInterval: 64,
        decodeSchedule: "async",
        materializeCacheEachToken: false,
      },
      reference: {
        captureMlxLmReference: true,
        enforceMlxLmDecodeBar: false,
        requireMlxLmReference: false,
        mlxLmPython: undefined,
      },
    });

    expect(selected).toEqual([
      {
        name: "gemma-parity",
        model: "google/gemma-3-1b-it",
        promptTokens: 512,
        generationTokens: 32,
        prefillStepSize: 256,
        generationTps: 50,
        mlxLmReference: {
          promptTps: 45,
          generationTps: 49,
          peakMemoryGb: 2.5,
          capturedAt: "2026-04-05",
        },
      },
    ]);
  });

  test("compareAgainstBaseline warns on >2x throughput regression", () => {
    const warnings = compareAgainstBaseline(
      {
        name: "gemma",
        model: "google/gemma-3-1b-it",
        promptTokens: 1024,
        generationTokens: 128,
        promptTps: 4000,
        generationTps: 80,
        peakMemoryGb: 4,
        explicitEvalCountPerToken: 1,
      },
      {
        promptTps: 1500,
        generationTps: 30,
        peakMemoryGb: 9,
        activeMemoryStartGb: 1,
        activeMemoryEndGb: 2,
        activeMemoryDeltaGb: 1,
        activeMemoryMaxGb: 2,
        activeMemorySlopeMbPerToken: 8,
        explicitEvalCountPerToken: 1.5,
        totalTimeSeconds: 1,
      },
    );

    expect(warnings).toEqual([
      "prompt_tps regressed >2x: baseline=4000.0, current=1500.0",
      "generation_tps regressed >2x: baseline=80.0, current=30.0",
      "peak_memory grew >2x: baseline=4.000, current=9.000",
      "evals_per_token regressed: baseline=1.00, current=1.50",
    ]);
  });

  test("formatMlxLmReference prints the paired reference block", () => {
    expect(
      formatMlxLmReference({
        name: "gemma-parity",
        model: "google/gemma-3-1b-it",
        promptTokens: 1024,
        generationTokens: 128,
        mlxLmReference: {
          promptTps: 45,
          generationTps: 49,
          peakMemoryGb: 2.5,
          capturedAt: "2026-04-05",
        },
      }),
    ).toBe(
      "MLX-LM reference: prompt_tps=45.000 generation_tps=49.000 peak_memory=2.500 captured_at=2026-04-05",
    );
  });

  test("formatMlxLmReference includes live reference trial counts", () => {
    expect(
      formatMlxLmReference({
        name: "qwen-parity",
        model: "mlx-community/Qwen3.6-27B-4bit",
        promptTokens: 128,
        generationTokens: 128,
        mlxLmReference: {
          promptTps: 45,
          generationTps: 49,
          peakMemoryGb: 2.5,
          trialCount: 3,
          capturedAt: "2026-04-23",
        },
      }),
    ).toBe(
      "MLX-LM reference: prompt_tps=45.000 generation_tps=49.000 peak_memory=2.500 trials=3 captured_at=2026-04-23",
    );
  });

  test("readBenchmarkVocabSize reads direct and nested text config vocab sizes", async () => {
    const directDirectory = tempDirectory("mlxts-bench-vocab-direct-");
    const nestedDirectory = tempDirectory("mlxts-bench-vocab-nested-");
    try {
      writeFileSync(join(directDirectory, "config.json"), JSON.stringify({ vocab_size: 32000 }));
      writeFileSync(
        join(nestedDirectory, "config.json"),
        JSON.stringify({ text_config: { vocab_size: 248064 } }),
      );

      await expect(readBenchmarkVocabSize(directDirectory)).resolves.toBe(32000);
      await expect(readBenchmarkVocabSize(nestedDirectory)).resolves.toBe(248064);
    } finally {
      rmSync(directDirectory, { recursive: true, force: true });
      rmSync(nestedDirectory, { recursive: true, force: true });
    }
  });

  test("compareAgainstMlxLmReference warns when current metrics trail mlx-lm", () => {
    expect(
      compareAgainstMlxLmReference(
        {
          promptTps: 100,
          generationTps: 20,
          peakMemoryGb: 5,
          activeMemoryStartGb: 1,
          activeMemoryEndGb: 2,
          activeMemoryDeltaGb: 1,
          activeMemoryMaxGb: 2,
          activeMemorySlopeMbPerToken: 8,
          explicitEvalCountPerToken: 1,
          totalTimeSeconds: 1,
        },
        {
          promptTps: 120,
          generationTps: 25,
          peakMemoryGb: 4.5,
          capturedAt: "2026-04-05",
        },
      ),
    ).toEqual(["generation_tps below mlx-lm: mlx_lm=25.0, current=20.0"]);
  });

  test("compareAgainstMlxLmReference warns when current memory is above mlx-lm", () => {
    expect(
      compareAgainstMlxLmReference(
        {
          promptTps: 100,
          generationTps: 25,
          peakMemoryGb: 6,
          activeMemoryStartGb: 1,
          activeMemoryEndGb: 2,
          activeMemoryDeltaGb: 1,
          activeMemoryMaxGb: 2,
          activeMemorySlopeMbPerToken: 8,
          explicitEvalCountPerToken: 1,
          totalTimeSeconds: 1,
        },
        {
          promptTps: 120,
          generationTps: 25,
          peakMemoryGb: 4.5,
          capturedAt: "2026-04-05",
        },
      ),
    ).toEqual(["peak_memory above mlx-lm: mlx_lm=4.500, current=6.000"]);
  });

  test("compareAgainstMlxLmReference tolerates tiny decode variance", () => {
    expect(
      compareAgainstMlxLmReference(
        {
          promptTps: 100,
          generationTps: 24.7,
          peakMemoryGb: 5,
          activeMemoryStartGb: 1,
          activeMemoryEndGb: 2,
          activeMemoryDeltaGb: 1,
          activeMemoryMaxGb: 2,
          activeMemorySlopeMbPerToken: 8,
          explicitEvalCountPerToken: 1,
          totalTimeSeconds: 1,
        },
        {
          promptTps: 120,
          generationTps: 25,
          peakMemoryGb: 4.5,
          capturedAt: "2026-04-05",
        },
      ),
    ).toEqual([]);
  });

  test("resolveCachedSnapshotPath rejects uncached repo ids", async () => {
    await expect(
      resolveCachedSnapshotPath("__this-cache-miss-should-not-exist__/__benchmark__"),
    ).rejects.toThrow("no cached snapshot");
  });

  test("safeDecodedTokenLength tolerates out-of-range tokenizer ids", () => {
    const tokenizer: Tokenizer = {
      vocabSize: 2,
      bosTokenId: undefined,
      eosTokenIds: [],
      padTokenId: undefined,
      encode() {
        throw new Error("not needed");
      },
      encodeWithOffsets() {
        throw new Error("not needed");
      },
      encodeBatch() {
        throw new Error("not needed");
      },
      decode(ids: readonly number[]) {
        const tokenId = ids[0];
        if (tokenId === 128000) {
          throw new Error("BPETokenizer.decode: token ID 128000 is out of range");
        }
        return "ok";
      },
      decodeBatch() {
        throw new Error("not needed");
      },
    };

    expect(safeDecodedTokenLength(tokenizer, 7)).toBe(2);
    expect(safeDecodedTokenLength(tokenizer, 128000)).toBe(0);
  });
});
