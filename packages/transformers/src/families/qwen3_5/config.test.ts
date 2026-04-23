import { describe, expect, test } from "bun:test";

import { parseQwen3_5Config, parseQwen3_5TextConfig, parseQwen3_5VisionConfig } from "./config";

function qwen3_5LayerTypes(
  numHiddenLayers: number,
  fullAttentionInterval: number,
): Array<"linear_attention" | "full_attention"> {
  return Array.from({ length: numHiddenLayers }, (_, layerIndex) =>
    (layerIndex + 1) % fullAttentionInterval === 0 ? "full_attention" : "linear_attention",
  );
}

function qwen3_5TextRaw(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    model_type: "qwen3_5_text",
    attention_bias: false,
    attention_dropout: 0,
    attn_output_gate: true,
    bos_token_id: 248044,
    eos_token_id: 248044,
    full_attention_interval: 4,
    head_dim: 256,
    hidden_act: "silu",
    hidden_size: 5120,
    initializer_range: 0.02,
    intermediate_size: 17408,
    layer_types: qwen3_5LayerTypes(64, 4),
    linear_conv_kernel_dim: 4,
    linear_key_head_dim: 128,
    linear_num_key_heads: 16,
    linear_num_value_heads: 48,
    linear_value_head_dim: 128,
    mamba_ssm_dtype: "float32",
    max_position_embeddings: 262144,
    mtp_num_hidden_layers: 1,
    mtp_use_dedicated_embeddings: false,
    num_attention_heads: 24,
    num_hidden_layers: 64,
    num_key_value_heads: 4,
    output_gate_type: "swish",
    pad_token_id: null,
    partial_rotary_factor: 0.25,
    rms_norm_eps: 1e-6,
    rope_parameters: {
      mrope_interleaved: true,
      mrope_section: [11, 11, 10],
      partial_rotary_factor: 0.25,
      rope_theta: 10_000_000,
      rope_type: "default",
    },
    tie_word_embeddings: false,
    use_cache: true,
    vocab_size: 248320,
    ...overrides,
  };
}

function qwen3_5VisionRaw(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    model_type: "qwen3_5",
    deepstack_visual_indexes: [],
    depth: 27,
    hidden_act: "gelu_pytorch_tanh",
    hidden_size: 1152,
    in_channels: 3,
    initializer_range: 0.02,
    intermediate_size: 4304,
    num_heads: 16,
    num_position_embeddings: 2304,
    out_hidden_size: 5120,
    patch_size: 16,
    spatial_merge_size: 2,
    temporal_patch_size: 2,
    ...overrides,
  };
}

function qwen3_5Raw(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    architectures: ["Qwen3_5ForConditionalGeneration"],
    image_token_id: 248056,
    language_model_only: false,
    model_type: "qwen3_5",
    text_config: qwen3_5TextRaw(),
    tie_word_embeddings: false,
    video_token_id: 248057,
    vision_config: qwen3_5VisionRaw(),
    vision_end_token_id: 248054,
    vision_start_token_id: 248053,
    ...overrides,
  };
}

describe("Qwen 3.5 config parsing", () => {
  test("parses the live Qwen3.6 wrapper shape and keeps nested configs intact", () => {
    const config = parseQwen3_5Config(qwen3_5Raw());

    expect(config.family).toBe("qwen");
    expect(config.modelType).toBe("qwen3_5");
    expect(config.languageModelOnly).toBe(false);
    expect(config.textConfig.modelType).toBe("qwen3_5_text");
    expect(config.textConfig.hiddenSize).toBe(5120);
    expect(config.textConfig.numHiddenLayers).toBe(64);
    expect(config.textConfig.fullAttentionInterval).toBe(4);
    expect(config.textConfig.layerTypes[3]).toBe("full_attention");
    expect(config.textConfig.ropeParameters.mropeInterleaved).toBe(true);
    expect(config.textConfig.ropeParameters.mropeSection).toEqual([11, 11, 10]);
    expect(config.textConfig.outputGateType).toBe("swish");
    expect(config.visionConfig.modelType).toBe("qwen3_5");
    expect(config.visionConfig.hiddenSize).toBe(1152);
    expect(config.visionConfig.outHiddenSize).toBe(5120);
    expect(config.visionConfig.deepstackVisualIndexes).toEqual([]);
  });

  test("derives text layer types when the raw config omits them", () => {
    const config = parseQwen3_5TextConfig(
      qwen3_5TextRaw({
        full_attention_interval: 3,
        layer_types: undefined,
        num_hidden_layers: 6,
      }),
    );

    expect(config.fullAttentionInterval).toBe(3);
    expect(config.layerTypes).toEqual([
      "linear_attention",
      "linear_attention",
      "full_attention",
      "linear_attention",
      "linear_attention",
      "full_attention",
    ]);
  });

  test("accepts both current and canonical nested vision model_type values", () => {
    const currentConfig = parseQwen3_5VisionConfig(qwen3_5VisionRaw({ model_type: "qwen3_5" }));
    const canonicalConfig = parseQwen3_5VisionConfig(
      qwen3_5VisionRaw({ model_type: "qwen3_5_vision" }),
    );

    expect(currentConfig.modelType).toBe("qwen3_5");
    expect(canonicalConfig.modelType).toBe("qwen3_5_vision");
  });

  test("rejects mismatched wrapper and nested model types", () => {
    expect(() => parseQwen3_5Config(qwen3_5Raw({ model_type: "qwen3_vl" }))).toThrow(
      'must be "qwen3_5"',
    );
    expect(() =>
      parseQwen3_5Config(qwen3_5Raw({ text_config: qwen3_5TextRaw({ model_type: "qwen3_next" }) })),
    ).toThrow('must be "qwen3_5_text"');
    expect(() => parseQwen3_5VisionConfig(qwen3_5VisionRaw({ model_type: "qwen3_vl" }))).toThrow(
      'must be "qwen3_5" or "qwen3_5_vision"',
    );
  });

  test("validates layer patterns, rope parameters, and vision shape fields", () => {
    expect(() =>
      parseQwen3_5TextConfig(
        qwen3_5TextRaw({
          num_hidden_layers: 4,
          layer_types: ["linear_attention", "full_attention"],
        }),
      ),
    ).toThrow("layer_types must be an array with 4 entries");

    expect(() =>
      parseQwen3_5TextConfig(
        qwen3_5TextRaw({
          num_hidden_layers: 4,
          full_attention_interval: 2,
          layer_types: ["linear_attention", "linear_attention", "full_attention", "full_attention"],
        }),
      ),
    ).toThrow("must match full_attention_interval=2");

    expect(() =>
      parseQwen3_5TextConfig(
        qwen3_5TextRaw({
          rope_parameters: {
            rope_type: "default",
            rope_theta: 10_000_000,
            partial_rotary_factor: 0.25,
            mrope_section: [11, "bad", 10],
            mrope_interleaved: true,
          },
        }),
      ),
    ).toThrow("mrope_section[1] must be an integer");

    expect(() => parseQwen3_5VisionConfig(qwen3_5VisionRaw({ patch_size: [] }))).toThrow(
      "patch_size must be a non-empty array of integers",
    );

    expect(() => parseQwen3_5VisionConfig(qwen3_5VisionRaw({ hidden_act: "silu" }))).toThrow(
      'hidden_act must be "gelu_pytorch_tanh"',
    );
  });
});
