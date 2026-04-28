import { describe, expect, test } from "bun:test";

import { parseGemma4Config, parseGemma4TextConfig } from "./config";

function gemma4Raw(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    model_type: "gemma4_text",
    vocab_size: 32,
    hidden_size: 16,
    intermediate_size: 32,
    num_hidden_layers: 2,
    num_attention_heads: 4,
    num_key_value_heads: 2,
    global_head_dim: 8,
    max_position_embeddings: 128,
    sliding_window: 64,
    layer_types: ["sliding_attention", "full_attention"],
    rope_parameters: {
      sliding_attention: { rope_type: "default", rope_theta: 10000 },
      full_attention: {
        rope_type: "proportional",
        rope_theta: 1_000_000,
        partial_rotary_factor: 0.25,
      },
    },
    hidden_activation: "gelu_pytorch_tanh",
    ...overrides,
  };
}

describe("Gemma 4 config parsing", () => {
  test("parses Gemma 4 MoE text fields without changing the CausalLM contract", () => {
    const textConfig = parseGemma4TextConfig(
      gemma4Raw({
        enable_moe_block: true,
        moe_intermediate_size: 8,
        num_experts: 4,
        top_k_experts: 2,
      }),
    );
    const wrapperConfig = parseGemma4Config({
      model_type: "gemma4",
      text_config: gemma4Raw({
        enable_moe_block: true,
        moe_intermediate_size: 12,
        num_experts: 6,
        top_k_experts: 3,
      }),
    });

    expect(textConfig.enableMoeBlock).toBe(true);
    expect(textConfig.moeIntermediateSize).toBe(8);
    expect(textConfig.numExperts).toBe(4);
    expect(textConfig.topKExperts).toBe(2);
    expect(wrapperConfig.modelType).toBe("gemma4");
    expect(wrapperConfig.enableMoeBlock).toBe(true);
    expect(wrapperConfig.moeIntermediateSize).toBe(12);
    expect(wrapperConfig.numExperts).toBe(6);
    expect(wrapperConfig.topKExperts).toBe(3);
  });

  test("validates Gemma 4 MoE routing cardinality", () => {
    expect(() =>
      parseGemma4TextConfig(
        gemma4Raw({
          enable_moe_block: true,
          moe_intermediate_size: 8,
          num_experts: 2,
          top_k_experts: 3,
        }),
      ),
    ).toThrow("top_k_experts must be <= num_experts");
  });
});
