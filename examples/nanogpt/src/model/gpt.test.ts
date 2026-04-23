import { describe, expect, test } from "bun:test";
import {
  array,
  clearMemoryCache,
  getActiveMemoryBytes,
  mxEval,
  random,
  reshape,
  synchronize,
  treeFlatten,
} from "@mlxts/core";
import { crossEntropy } from "@mlxts/nn";

import { estimateParameterCount, GPT_SMALL, GPT_TINY, resolveConfig } from "../config";
import { GPT } from "./gpt";
import { initializeGPT } from "./init";

// Use a small config for fast tests
const TINY_CONFIG = resolveConfig(
  { ...GPT_TINY, nLayer: 2, nHead: 2, nEmbd: 32, blockSize: 16, dropout: 0.0 },
  26,
);

function createCheckpointedConfig() {
  return resolveConfig(
    {
      ...GPT_TINY,
      nLayer: 2,
      nHead: 2,
      nEmbd: 32,
      blockSize: 16,
      dropout: 0.0,
      gradientCheckpointing: true,
    },
    26,
  );
}

function activeMemoryRange(samples: readonly number[]): number {
  return Math.max(...samples) - Math.min(...samples);
}

describe("GPT", () => {
  test("forward produces correct logits shape", () => {
    random.seed(42);
    const model = new GPT(TINY_CONFIG);
    initializeGPT(model, TINY_CONFIG);
    model.eval();

    const flat = array([0, 1, 2, 3, 4, 5, 6, 7], "int32");
    using input = reshape(flat, [1, 8]);
    flat.free();

    using logits = model.forward(input);
    mxEval(logits);
    expect(logits.shape).toEqual([1, 8, 26]);

    model[Symbol.dispose]();
  });

  test("weight tying: tokenEmbedding.weight appears once in params", () => {
    random.seed(42);
    const model = new GPT(TINY_CONFIG);
    const flat = treeFlatten(model.parameters());
    const weightPaths = flat.map(([p]) => p.join("."));

    // tokenEmbedding.weight should appear exactly once
    const tokenWeightCount = weightPaths.filter((p) => p === "tokenEmbedding.weight").length;
    expect(tokenWeightCount).toBe(1);

    // No lm_head or output_projection at root level
    expect(weightPaths.some((p) => p.includes("lm_head"))).toBe(false);

    model[Symbol.dispose]();
  });

  test("parameter count is reasonable for tiny config", () => {
    random.seed(42);
    const model = new GPT(TINY_CONFIG);
    const flat = treeFlatten(model.parameters());
    let totalParams = 0;
    for (const [, arr] of flat) {
      totalParams += arr.size;
    }
    expect(totalParams).toBe(estimateParameterCount(TINY_CONFIG));
    model[Symbol.dispose]();
  });

  test("GPT_SMALL preset resolves to a larger parameter count than GPT_TINY", () => {
    const tiny = resolveConfig(GPT_TINY, 26);
    const small = resolveConfig(GPT_SMALL, 26);
    expect(estimateParameterCount(small)).toBeGreaterThan(estimateParameterCount(tiny));
  });

  test("forward rejects float dtype input", () => {
    random.seed(42);
    const model = new GPT(TINY_CONFIG);
    model.eval();
    const input = array([[0, 1, 2, 3]]);
    expect(() => model.forward(input)).toThrow("integer token dtype");
    input.free();
    model[Symbol.dispose]();
  });

  test("forward rejects sequence longer than blockSize", () => {
    random.seed(42);
    const model = new GPT(TINY_CONFIG);
    model.eval();
    const tokens = Array.from({ length: 20 }, (_, i) => i % 26);
    const flat = array(tokens, "int32");
    using input = reshape(flat, [1, 20]);
    flat.free();
    expect(() => model.forward(input)).toThrow("exceeds blockSize");
    model[Symbol.dispose]();
  });

  test("forward rejects rank-1 input", () => {
    random.seed(42);
    const model = new GPT(TINY_CONFIG);
    model.eval();
    const input = array([0, 1, 2], "int32");
    expect(() => model.forward(input)).toThrow("rank-2");
    input.free();
    model[Symbol.dispose]();
  });

  test("forward still works when gradient checkpointing is enabled", () => {
    random.seed(42);
    const checkpointedConfig = createCheckpointedConfig();
    const model = new GPT(checkpointedConfig);
    initializeGPT(model, checkpointedConfig);
    model.eval();

    const flat = array([0, 1, 2, 3], "int32");
    using input = reshape(flat, [1, 4]);
    flat.free();

    using logits = model.forward(input);
    mxEval(logits);
    expect(logits.shape).toEqual([1, 4, 26]);

    model[Symbol.dispose]();
  });

  test("repeated forward + loss evaluation stays within a bounded active-memory range", () => {
    random.seed(42);
    const model = new GPT(TINY_CONFIG);
    initializeGPT(model, TINY_CONFIG);
    model.eval();

    const flatTokens = array([0, 1, 2, 3, 4, 5, 6, 7], "int32");
    const flatTargets = array([1, 2, 3, 4, 5, 6, 7, 8], "int32");
    using input = reshape(flatTokens, [1, 8]);
    using targets = reshape(flatTargets, [1, 8]);
    flatTokens.free();
    flatTargets.free();

    clearMemoryCache();
    synchronize();
    const activeSamples: number[] = [];

    try {
      for (let iteration = 0; iteration < 12; iteration++) {
        using logits = model.forward(input);
        using flatLogits = reshape(logits, [8, 26]);
        using flatLabelTargets = reshape(targets, [8]);
        using loss = crossEntropy(flatLogits, flatLabelTargets);
        mxEval(loss);
        synchronize();
        loss.item();

        if (iteration >= 3) {
          activeSamples.push(getActiveMemoryBytes());
        }
      }

      expect(activeSamples.length).toBeGreaterThan(0);
      expect(activeMemoryRange(activeSamples)).toBeLessThan(16 * 1024 * 1024);
    } finally {
      model[Symbol.dispose]();
    }
  });
});
