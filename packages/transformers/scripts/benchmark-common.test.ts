import { describe, expect, test } from "bun:test";

import {
  type BenchmarkBaselines,
  parseBaselineData,
  parseBenchmarkArgs,
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
});
