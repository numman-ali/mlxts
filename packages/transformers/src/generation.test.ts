import { describe, expect, test } from "bun:test";

import { array, type MxArray, type ParameterTree } from "@mlxts/core";
import { generateStep, makePromptCache } from "./generation";
import { KVCache } from "./infrastructure/cache";
import type { BaseModelConfig, CausalLM, ForwardOptions, TransformerCache } from "./types";

class DeterministicGenerationModel implements CausalLM {
  readonly family = "gemma";
  readonly layerCount = 1;
  readonly config: BaseModelConfig = {
    family: "gemma",
    modelType: "deterministic-test",
    rawConfig: {},
    vocabSize: 3,
    hiddenSize: 1,
    numHiddenLayers: 1,
  };
  lastForwardCache: TransformerCache | undefined;
  cacheCreates = 0;

  forward(_inputIds: MxArray, options?: ForwardOptions): MxArray {
    this.lastForwardCache = options?.cache;
    return array([[[0.1, 0.2, 0.9]]], "float32");
  }

  createCache(): TransformerCache {
    this.cacheCreates += 1;
    return new KVCache(1);
  }

  parameters(): ParameterTree {
    return {};
  }

  trainableParameters(): ParameterTree {
    return {};
  }

  update(_params: ParameterTree): void {}

  freeze(): this {
    return this;
  }

  unfreeze(): this {
    return this;
  }

  eval(): this {
    return this;
  }

  train(): this {
    return this;
  }

  [Symbol.dispose](): void {}
}

describe("generation", () => {
  test("makePromptCache delegates to model.createCache", () => {
    using model = new DeterministicGenerationModel();
    using cache = makePromptCache(model);

    expect(model.cacheCreates).toBe(1);
    expect(cache).toBeInstanceOf(KVCache);
  });

  test("generateStep samples the next token and forwards the provided cache", () => {
    using model = new DeterministicGenerationModel();
    using cache = new KVCache(1);

    const tokenId = generateStep(model, [0, 1], cache, [0, 1], { temperature: 0 });

    expect(tokenId).toBe(2);
    expect(model.lastForwardCache).toBe(cache);
  });
});
