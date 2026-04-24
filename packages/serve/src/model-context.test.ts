import { describe, expect, test } from "bun:test";
import type { MxArray, ParameterTree } from "@mlxts/core";
import type {
  BaseModelConfig,
  CausalLM,
  ForwardOptions,
  TransformerCache,
} from "@mlxts/transformers";
import {
  effectiveTotalTokenLimit,
  modelAdmissionMetadata,
  modelContextWindow,
} from "./model-context";

class ConfigOnlyModel implements CausalLM {
  readonly family = "llama";
  readonly layerCount = 0;
  readonly config: BaseModelConfig;

  constructor(rawConfig: Record<string, unknown>) {
    this.config = {
      family: "llama",
      modelType: "config-only",
      rawConfig,
      vocabSize: 1,
      hiddenSize: 1,
      numHiddenLayers: 0,
    };
  }

  forward(_inputIds: MxArray, _options?: ForwardOptions): MxArray {
    throw new Error("ConfigOnlyModel.forward should not be called.");
  }

  createCache(): TransformerCache {
    throw new Error("ConfigOnlyModel.createCache should not be called.");
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

describe("model admission metadata", () => {
  test("reads top-level and nested context windows", () => {
    expect(modelContextWindow(new ConfigOnlyModel({ max_position_embeddings: 8192 }))).toBe(8192);
    expect(
      modelContextWindow(
        new ConfigOnlyModel({ text_config: { max_position_embeddings: 262_144 } }),
      ),
    ).toBe(262_144);
    expect(modelContextWindow(new ConfigOnlyModel({ max_position_embeddings: "large" }))).toBe(
      undefined,
    );
  });

  test("combines server and model context limits into operator metadata", () => {
    const model = new ConfigOnlyModel({ text_config: { max_position_embeddings: 131_072 } });

    expect(effectiveTotalTokenLimit({ maxTotalTokens: 4096, contextWindow: 131_072 })).toBe(4096);
    expect(modelAdmissionMetadata(model, { maxPromptTokens: 2048, maxTotalTokens: 4096 })).toEqual({
      contextWindow: 131_072,
      maxPromptTokens: 2048,
      maxTotalTokens: 4096,
      effectiveTotalTokens: 4096,
    });
  });
});
