import { describe, expect, test } from "bun:test";

import { array, mxEval } from "@mlxts/core";

import { generateBatchTokens, generateTokens } from "../../generation";
import { BatchKVCache } from "../../infrastructure/cache";
import { LlamaLikeCausalLM } from "./model";
import type { LlamaLikeConfig } from "./types";

function testConfig(overrides: Partial<LlamaLikeConfig> = {}): LlamaLikeConfig {
  return {
    family: "llama",
    modelType: "llama",
    rawConfig: {},
    vocabSize: 16,
    hiddenSize: 8,
    intermediateSize: 16,
    numHiddenLayers: 1,
    numAttentionHeads: 2,
    numKeyValueHeads: 1,
    headDim: 4,
    maxPositionEmbeddings: 32,
    ropeTheta: 10000,
    rmsNormEps: 1e-5,
    tieWordEmbeddings: false,
    attentionBias: false,
    mlpActivation: "swiglu",
    ...overrides,
  };
}

describe("LlamaLikeCausalLM batch cache", () => {
  test("accepts BatchKVCache for full-cache batched forwards", () => {
    using model = new LlamaLikeCausalLM(testConfig());
    using cache = new BatchKVCache(model.layerCount, [1, 0]);
    using inputIds = array(
      [
        [0, 1],
        [2, 3],
      ],
      "int32",
    );
    using logits = model.forward(inputIds, { cache });
    mxEval(logits);

    const stateArrays = cache.arrays();
    try {
      expect(logits.shape).toEqual([2, 2, 16]);
      expect(cache.length).toBe(2);
      expect(cache.offsets).toEqual([1, 2]);
      expect(stateArrays[0]?.shape).toEqual([2, 1, 2, 4]);
      expect(stateArrays[1]?.shape).toEqual([2, 1, 2, 4]);
    } finally {
      for (const stateArray of stateArrays) {
        stateArray.free();
      }
    }
  });

  test("rejects BatchKVCache for sliding-window LLaMA-like models", () => {
    using model = new LlamaLikeCausalLM(testConfig({ slidingWindow: 4 }));
    using cache = new BatchKVCache(model.layerCount, [0, 0]);
    using inputIds = array([[1], [2]], "int32");

    expect(() => model.forward(inputIds, { cache })).toThrow(
      "BatchKVCache is only supported for full-cache models",
    );
  });

  test("runs per-head query/key RMSNorm when the family config requires it", () => {
    using model = new LlamaLikeCausalLM(testConfig({ queryKeyNorm: true }));
    using inputIds = array([[1, 2]], "int32");
    using logits = model.forward(inputIds);

    expect(model.model.layers[0]?.selfAttention.queryNorm).not.toBeNull();
    expect(model.model.layers[0]?.selfAttention.keyNorm).not.toBeNull();
    expect(logits.shape).toEqual([1, 2, 16]);
  });

  test("static greedy batch generation matches separate single-prompt generation", () => {
    using model = new LlamaLikeCausalLM(testConfig());
    const prompts = [[1, 2], [3]];
    const options = { maxTokens: 2, temperature: 0, eosTokenIds: [] };

    const separate = prompts.map((prompt) => generateTokens(model, prompt, options));
    const batched = generateBatchTokens(model, prompts, options);

    expect(batched).toEqual(separate);
  });

  test("static greedy batch generation supports per-prompt lengths", () => {
    using model = new LlamaLikeCausalLM(testConfig());
    const prompts = [[1, 2], [3], [4, 5]];
    const maxTokensPerPrompt = [1, 3, 0];

    const separate = prompts.map((prompt, index) =>
      generateTokens(model, prompt, {
        maxTokens: maxTokensPerPrompt[index] ?? 0,
        temperature: 0,
        eosTokenIds: [],
      }),
    );
    const batched = generateBatchTokens(model, prompts, {
      maxTokens: maxTokensPerPrompt,
      temperature: 0,
      eosTokenIds: [],
    });

    expect(batched).toEqual(separate);
  });

  test("exposes retained hidden states for encoder-style conditioning", () => {
    using model = new LlamaLikeCausalLM(testConfig({ numHiddenLayers: 2 }));
    using inputIds = array([[1, 2, 3]], "int32");

    const output = model.model.runWithHiddenStates(inputIds, { outputHiddenStates: true });
    try {
      mxEval(output.lastHiddenState, ...(output.hiddenStates ?? []));
      expect(output.lastHiddenState.shape).toEqual([1, 3, 8]);
      expect(output.hiddenStates?.map((hidden) => hidden.shape)).toEqual([
        [1, 3, 8],
        [1, 3, 8],
        [1, 3, 8],
      ]);
    } finally {
      output.lastHiddenState.free();
      for (const hiddenState of output.hiddenStates ?? []) {
        hiddenState.free();
      }
    }
  });
});
