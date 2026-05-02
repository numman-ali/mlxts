import { describe, expect, test } from "bun:test";
import { array, mxEval } from "@mlxts/core";

import {
  createContinuousBatchTokenScheduler,
  generateBatchTokens,
  generateTokens,
} from "../../generation";
import { disposeGemma3TextModelOutput, Gemma3TextCausalLM } from "./model";
import type { Gemma3TextConfig } from "./types";

function gemma3Config(overrides: Partial<Gemma3TextConfig> = {}): Gemma3TextConfig {
  return {
    family: "gemma",
    modelType: "gemma3_text",
    rawConfig: {},
    vocabSize: 16,
    hiddenSize: 8,
    intermediateSize: 16,
    numHiddenLayers: 2,
    numAttentionHeads: 2,
    numKeyValueHeads: 1,
    headDim: 4,
    maxPositionEmbeddings: 32,
    ropeTheta: 10_000,
    ropeLocalBaseFreq: 10_000,
    rmsNormEps: 1e-5,
    tieWordEmbeddings: false,
    attentionBias: false,
    queryPreAttentionScalar: 4,
    slidingWindow: 2,
    layerTypes: ["sliding_attention", "full_attention"],
    embeddingScale: 1,
    ...overrides,
  };
}

describe("Gemma3TextCausalLM batch cache", () => {
  test("retains embedding and layer hidden states for prompt conditioning", () => {
    using model = new Gemma3TextCausalLM(gemma3Config());
    using inputIds = array([[1, 2, 3]], "int32");
    using attentionMask = array([[1, 1, 1]], "int32");
    const output = model.model.runWithHiddenStates(inputIds, {
      attentionMask,
      outputHiddenStates: true,
    });

    try {
      mxEval(output.lastHiddenState, ...(output.hiddenStates ?? []));
      expect(output.lastHiddenState.shape).toEqual([1, 3, 8]);
      expect(output.hiddenStates?.map((hiddenState) => hiddenState.shape)).toEqual([
        [1, 3, 8],
        [1, 3, 8],
        [1, 3, 8],
      ]);
    } finally {
      disposeGemma3TextModelOutput(output);
    }
  });

  test("rejects prompt attention masks with cache-backed generation", () => {
    using model = new Gemma3TextCausalLM(gemma3Config());
    using inputIds = array([[1, 2, 3]], "int32");
    using attentionMask = array([[1, 1, 1]], "int32");
    using cache = model.createCache();

    expect(() =>
      model.model.runWithHiddenStates(inputIds, {
        cache,
        attentionMask,
        outputHiddenStates: true,
      }),
    ).toThrow("attentionMask is only supported without cache");
  });

  test("static greedy batch generation matches separate single-prompt generation", () => {
    using model = new Gemma3TextCausalLM(gemma3Config());
    const prompts = [
      [1, 2, 3, 4],
      [5, 6],
    ];
    const options = { maxTokens: 2, temperature: 0, eosTokenIds: [] };

    const separate = prompts.map((prompt) => generateTokens(model, prompt, options));
    const batched = generateBatchTokens(model, prompts, options);

    expect(batched).toEqual(separate);
  });

  test("continuous greedy batch generation matches separate single-prompt generation", async () => {
    using model = new Gemma3TextCausalLM(gemma3Config());
    const prompts = [
      [1, 2, 3, 4],
      [5, 6],
    ];
    const options = { maxTokens: 2, temperature: 0, eosTokenIds: [] };
    const separate = prompts.map((prompt) => generateTokens(model, prompt, options));
    const scheduler = createContinuousBatchTokenScheduler(model, {
      maxBatchSize: 2,
      temperature: 0,
      eosTokenIds: [],
    });

    const batched = await Promise.all(
      prompts.map((prompt) =>
        scheduler.enqueue({ promptTokenIds: prompt, maxTokens: options.maxTokens }),
      ),
    );

    expect(batched).toEqual(separate);
  });
});
