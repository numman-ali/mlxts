import { describe, expect, test } from "bun:test";

import { FAMILY_REGISTRY } from "../../registry";
import { parseQwen3Config } from "./config";
import { isIgnoredQwen3Weight, sanitizeQwen3Weight } from "./weights";

function zImageTextEncoderConfig(): Record<string, unknown> {
  return {
    architectures: ["Qwen3ForCausalLM"],
    attention_bias: false,
    attention_dropout: 0.0,
    bos_token_id: 151643,
    eos_token_id: 151645,
    head_dim: 128,
    hidden_act: "silu",
    hidden_size: 2560,
    initializer_range: 0.02,
    intermediate_size: 9728,
    max_position_embeddings: 40960,
    max_window_layers: 36,
    model_type: "qwen3",
    num_attention_heads: 32,
    num_hidden_layers: 36,
    num_key_value_heads: 8,
    rms_norm_eps: 1e-6,
    rope_scaling: null,
    rope_theta: 1_000_000,
    sliding_window: null,
    tie_word_embeddings: true,
    torch_dtype: "bfloat16",
    use_cache: true,
    use_sliding_window: false,
    vocab_size: 151936,
  };
}

describe("Qwen3 config", () => {
  test("parses the Z-Image Turbo Qwen3 text encoder shape", () => {
    const config = parseQwen3Config(zImageTextEncoderConfig());

    expect(config.family).toBe("qwen");
    expect(config.modelType).toBe("qwen3");
    expect(config.vocabSize).toBe(151936);
    expect(config.hiddenSize).toBe(2560);
    expect(config.intermediateSize).toBe(9728);
    expect(config.numHiddenLayers).toBe(36);
    expect(config.numAttentionHeads).toBe(32);
    expect(config.numKeyValueHeads).toBe(8);
    expect(config.headDim).toBe(128);
    expect(config.ropeTheta).toBe(1_000_000);
    expect(config.tieWordEmbeddings).toBe(true);
    expect(config.queryKeyNorm).toBe(true);
    expect(config.mlpActivation).toBe("swiglu");
    expect(config.slidingWindow).toBeUndefined();
  });

  test("registers qwen3 as a lean llama-like family", () => {
    const registration = FAMILY_REGISTRY.get("qwen3");

    expect(registration).toBeDefined();
    expect(registration?.family).toBe("qwen");
  });

  test("uses the shared llama-like checkpoint weight contract", () => {
    const config = parseQwen3Config(zImageTextEncoderConfig());

    expect(sanitizeQwen3Weight(config, "model.embed_tokens.weight")).toBe(
      "model.embedTokens.weight",
    );
    expect(sanitizeQwen3Weight(config, "model.layers.0.self_attn.q_proj.weight")).toBe(
      "model.layers.0.selfAttention.qProjection.weight",
    );
    expect(sanitizeQwen3Weight(config, "model.layers.0.self_attn.q_norm.weight")).toBe(
      "model.layers.0.selfAttention.queryNorm.weight",
    );
    expect(sanitizeQwen3Weight(config, "model.layers.0.self_attn.k_norm.weight")).toBe(
      "model.layers.0.selfAttention.keyNorm.weight",
    );
    expect(sanitizeQwen3Weight(config, "model.layers.0.mlp.down_proj.weight")).toBe(
      "model.layers.0.mlp.downProjection.weight",
    );
    expect(isIgnoredQwen3Weight(config, "model.layers.0.self_attn.rotary_emb.inv_freq")).toBe(true);
    expect(isIgnoredQwen3Weight(config, "lm_head.weight")).toBe(true);
  });

  test("rejects non-Qwen3 configs", () => {
    expect(() => parseQwen3Config({ ...zImageTextEncoderConfig(), model_type: "qwen3_5" })).toThrow(
      'model_type must be "qwen3"',
    );
    expect(() => parseQwen3Config({ ...zImageTextEncoderConfig(), hidden_act: "gelu" })).toThrow(
      'hidden_act must be "silu"',
    );
  });
});
