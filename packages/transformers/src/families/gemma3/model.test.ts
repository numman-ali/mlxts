import { describe, expect, test } from "bun:test";

import {
  createContinuousBatchTokenScheduler,
  generateBatchTokens,
  generateTokens,
} from "../../generation";
import { Gemma3TextCausalLM } from "./model";
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
