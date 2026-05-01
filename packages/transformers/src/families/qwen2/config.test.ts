import { describe, expect, test } from "bun:test";

import { FAMILY_REGISTRY } from "../../registry";
import { parseQwen2Config } from "./config";
import { isIgnoredQwen2Weight, sanitizeQwen2Weight } from "./weights";

function qwenImageTextEncoderConfig(): Record<string, unknown> {
  return {
    architectures: ["Qwen2_5_VLForConditionalGeneration"],
    attention_dropout: 0.0,
    bos_token_id: 151643,
    dtype: "bfloat16",
    eos_token_id: 151645,
    hidden_act: "silu",
    hidden_size: 3584,
    image_token_id: 151655,
    initializer_range: 0.02,
    intermediate_size: 18944,
    max_position_embeddings: 128000,
    max_window_layers: 28,
    model_type: "qwen2_5_vl",
    num_attention_heads: 28,
    num_hidden_layers: 28,
    num_key_value_heads: 4,
    rms_norm_eps: 1e-6,
    rope_scaling: {
      mrope_section: [16, 24, 24],
      rope_type: "default",
      type: "default",
    },
    rope_theta: 1_000_000,
    sliding_window: 32768,
    text_config: {
      architectures: ["Qwen2_5_VLForConditionalGeneration"],
      attention_dropout: 0.0,
      bos_token_id: 151643,
      eos_token_id: 151645,
      hidden_act: "silu",
      hidden_size: 3584,
      initializer_range: 0.02,
      intermediate_size: 18944,
      layer_types: Array.from({ length: 28 }, () => "full_attention"),
      max_position_embeddings: 128000,
      max_window_layers: 28,
      model_type: "qwen2_5_vl_text",
      num_attention_heads: 28,
      num_hidden_layers: 28,
      num_key_value_heads: 4,
      rms_norm_eps: 1e-6,
      rope_scaling: {
        mrope_section: [16, 24, 24],
        rope_type: "default",
        type: "default",
      },
      rope_theta: 1_000_000,
      sliding_window: null,
      use_cache: true,
      use_sliding_window: false,
      vocab_size: 152064,
    },
    tie_word_embeddings: false,
    use_cache: true,
    use_sliding_window: false,
    video_token_id: 151656,
    vision_end_token_id: 151653,
    vision_start_token_id: 151652,
    vocab_size: 152064,
  };
}

function qwen2Config(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    attention_dropout: 0.0,
    bos_token_id: 151643,
    eos_token_id: 151645,
    hidden_act: "silu",
    hidden_size: 1024,
    intermediate_size: 2816,
    max_position_embeddings: 32768,
    model_type: "qwen2",
    num_attention_heads: 16,
    num_hidden_layers: 2,
    num_key_value_heads: 4,
    rms_norm_eps: 1e-6,
    rope_theta: 1_000_000,
    sliding_window: 32768,
    tie_word_embeddings: false,
    use_sliding_window: false,
    vocab_size: 151936,
    ...overrides,
  };
}

describe("Qwen2 config", () => {
  test("parses Qwen-Image Qwen2.5-VL text encoder config", () => {
    const config = parseQwen2Config(qwenImageTextEncoderConfig());

    expect(config.family).toBe("qwen");
    expect(config.modelType).toBe("qwen2_5_vl");
    expect(config.vocabSize).toBe(152064);
    expect(config.hiddenSize).toBe(3584);
    expect(config.intermediateSize).toBe(18944);
    expect(config.numHiddenLayers).toBe(28);
    expect(config.numAttentionHeads).toBe(28);
    expect(config.numKeyValueHeads).toBe(4);
    expect(config.headDim).toBe(128);
    expect(config.ropeTheta).toBe(1_000_000);
    expect(config.rmsNormEps).toBe(1e-6);
    expect(config.tieWordEmbeddings).toBe(false);
    expect(config.attentionBias).toBe(true);
    expect(config.attentionOutputBias).toBe(false);
    expect(config.slidingWindow).toBeUndefined();
  });

  test("parses dense Qwen2 configs with disabled sliding-window metadata", () => {
    const config = parseQwen2Config(qwen2Config());

    expect(config.modelType).toBe("qwen2");
    expect(config.hiddenSize).toBe(1024);
    expect(config.slidingWindow).toBeUndefined();
  });

  test("registers qwen2 families as lean llama-like families", () => {
    expect(FAMILY_REGISTRY.get("qwen2")?.family).toBe("qwen");
    expect(FAMILY_REGISTRY.get("qwen2_5_vl")?.family).toBe("qwen");
  });

  test("maps Qwen2.5-VL wrapper weights to the text parameter tree", () => {
    const config = parseQwen2Config(qwenImageTextEncoderConfig());

    expect(sanitizeQwen2Weight(config, "model.language_model.embed_tokens.weight")).toBe(
      "model.embedTokens.weight",
    );
    expect(sanitizeQwen2Weight(config, "model.language_model.layers.0.self_attn.q_proj.bias")).toBe(
      "model.layers.0.selfAttention.qProjection.bias",
    );
    expect(
      sanitizeQwen2Weight(config, "model.language_model.layers.0.self_attn.o_proj.bias"),
    ).toBeNull();
    expect(sanitizeQwen2Weight(config, "lm_head.weight")).toBe("lmHead.weight");
    expect(sanitizeQwen2Weight(config, "model.visual.patch_embed.proj.weight")).toBeNull();
    expect(isIgnoredQwen2Weight(config, "model.visual.patch_embed.proj.weight")).toBe(true);
    expect(
      isIgnoredQwen2Weight(config, "model.language_model.layers.0.self_attn.rotary_emb.inv_freq"),
    ).toBe(true);
  });

  test("rejects unsupported Qwen2 text shapes deliberately", () => {
    expect(() => parseQwen2Config(qwen2Config({ use_sliding_window: true }))).toThrow(
      "sliding-window",
    );
    expect(() => parseQwen2Config(qwen2Config({ layer_types: ["sliding_attention"] }))).toThrow(
      "sliding-window",
    );
    expect(() => parseQwen2Config(qwen2Config({ model_type: "qwen2_5_vl_text" }))).toThrow(
      'model_type must be "qwen2" or "qwen2_5_vl"',
    );
    expect(() => parseQwen2Config(qwen2Config({ rope_scaling: { rope_type: "yarn" } }))).toThrow(
      "rope_scaling",
    );
    expect(() => parseQwen2Config(qwen2Config({ rope_parameters: { rope_type: "yarn" } }))).toThrow(
      "rope_parameters",
    );
  });

  test("uses default rope_parameters theta when present", () => {
    const config = parseQwen2Config(
      qwen2Config({
        rope_parameters: {
          rope_type: "default",
          rope_theta: 500000,
        },
      }),
    );

    expect(config.ropeTheta).toBe(500000);
  });
});
