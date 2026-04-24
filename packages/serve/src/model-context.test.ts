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
  estimateGenerationMemory,
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

  test("estimates Qwen hybrid KV, recurrent, and prefill memory from text config", () => {
    const layerTypes = Array.from({ length: 64 }, (_, layerIndex) =>
      (layerIndex + 1) % 4 === 0 ? "full_attention" : "linear_attention",
    );
    const estimate = estimateGenerationMemory(
      new ConfigOnlyModel({
        text_config: {
          model_type: "qwen3_5_text",
          num_hidden_layers: 64,
          num_attention_heads: 24,
          num_key_value_heads: 4,
          hidden_size: 5120,
          head_dim: 256,
          linear_conv_kernel_dim: 4,
          linear_key_head_dim: 128,
          linear_value_head_dim: 128,
          linear_num_key_heads: 16,
          linear_num_value_heads: 48,
          layer_types: layerTypes,
        },
      }),
      { promptTokens: 131_072, totalTokens: 131_136, prefillStepSize: 2048 },
    );

    expect(estimate).toMatchObject({
      bytesPerToken: 65_536,
      kvCacheLayers: 16,
      keyValueHeads: 4,
      attentionHeads: 24,
      headDim: 256,
      dtypeSizeBytes: 2,
      batchSize: 1,
    });
    expect(estimate?.kvCacheBytes).toBe(131_136 * 65_536);
    expect(estimate?.fixedStateBytes).toBe(48 * (3 * 10_240 * 2 + 48 * 128 * 128 * 4));
    expect(estimate?.prefillTemporaryBytes).toBe(2048 * 131_072 * 24 * 4);
  });

  test("estimates Gemma sliding and global full-attention cache geometry", () => {
    const estimate = estimateGenerationMemory(
      new ConfigOnlyModel({
        model_type: "gemma4_text",
        num_hidden_layers: 4,
        num_attention_heads: 8,
        num_key_value_heads: 2,
        num_global_key_value_heads: 1,
        attention_k_eq_v: true,
        hidden_size: 128,
        head_dim: 16,
        global_head_dim: 32,
        sliding_window: 3,
        layer_types: ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
      }),
      { promptTokens: 5, totalTokens: 10, prefillStepSize: 4, batchSize: 2 },
    );

    expect(estimate).toMatchObject({
      kvCacheBytes: 6656,
      fixedStateBytes: 0,
      bytesPerToken: 1024,
      kvCacheLayers: 4,
      keyValueHeads: 2,
      attentionHeads: 8,
      headDim: 32,
      batchSize: 2,
    });
  });
});
