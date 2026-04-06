import { describe, expect, test } from "bun:test";
import type { Tokenizer } from "@mlxts/tokenizers";

import {
  type BenchmarkBaselines,
  compareAgainstBaseline,
  formatMlxLmReference,
  parseBaselineData,
  parseBenchmarkArgs,
  resolveCachedSnapshotPath,
  safeDecodedTokenLength,
  selectTargets,
} from "./benchmark-common";

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
      "--metal-trace",
    ]);

    expect(parsed.model).toBe("google/gemma-3-1b-it");
    expect(parsed.options).toEqual({
      promptTokens: 2048,
      generationTokens: 64,
      trials: 5,
      prefillStepSize: 1024,
      metalTrace: true,
    });
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
