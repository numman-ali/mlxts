import { describe, expect, test } from "bun:test";

import { parseGemmaConfig } from "./gemma/config";
import { isIgnoredGemmaWeight, sanitizeGemmaWeight } from "./gemma/weights";
import { parseGemma3TextConfig } from "./gemma3/config";
import type { Gemma3TextConfig } from "./gemma3/types";
import { isIgnoredGemma3Weight, sanitizeGemma3Weight } from "./gemma3/weights";
import { parseGemma4Config, parseGemma4TextConfig } from "./gemma4/config";
import type { Gemma4TextConfig } from "./gemma4/types";
import {
  isIgnoredGemma4TextWeight,
  isIgnoredGemma4Weight,
  sanitizeGemma4TextWeight,
  sanitizeGemma4Weight,
} from "./gemma4/weights";
import { parseLlamaConfig } from "./llama/config";
import { isIgnoredLlamaWeight, sanitizeLlamaWeight } from "./llama/weights";
import type { LlamaLikeConfig } from "./llama-like/types";
import { isIgnoredLlamaLikeWeight, sanitizeLlamaLikeWeight } from "./llama-like/types";
import { parseMistralConfig } from "./mistral/config";
import { isIgnoredMistralWeight, sanitizeMistralWeight } from "./mistral/weights";
import { parseMistral3Config } from "./mistral3/config";
import { isIgnoredMistral3Weight, sanitizeMistral3Weight } from "./mistral3/weights";
import { parsePhiConfig } from "./phi/config";
import { isIgnoredPhiWeight, sanitizePhiWeight } from "./phi/weights";

function baseRawConfig(modelType: "llama" | "mistral" | "gemma" | "phi3"): Record<string, unknown> {
  return {
    model_type: modelType,
    vocab_size: 32,
    hidden_size: 16,
    intermediate_size: 32,
    num_hidden_layers: 2,
    num_attention_heads: 4,
    num_key_value_heads: 2,
    max_position_embeddings: 128,
    rope_theta: 5000,
    rms_norm_eps: 1e-5,
    attention_bias: false,
  };
}

function mappedConfig(overrides: Partial<LlamaLikeConfig> = {}): LlamaLikeConfig {
  return {
    family: "llama",
    modelType: "llama",
    rawConfig: {},
    vocabSize: 32,
    hiddenSize: 16,
    intermediateSize: 32,
    numHiddenLayers: 2,
    numAttentionHeads: 4,
    numKeyValueHeads: 2,
    headDim: 4,
    maxPositionEmbeddings: 128,
    ropeTheta: 5000,
    rmsNormEps: 1e-5,
    tieWordEmbeddings: false,
    attentionBias: false,
    mlpActivation: "swiglu",
    ...overrides,
  };
}

function gemma3MappedConfig(overrides: Partial<Gemma3TextConfig> = {}): Gemma3TextConfig {
  return {
    family: "gemma",
    modelType: "gemma3_text",
    rawConfig: {},
    vocabSize: 32,
    hiddenSize: 16,
    intermediateSize: 32,
    numHiddenLayers: 2,
    numAttentionHeads: 4,
    numKeyValueHeads: 2,
    headDim: 4,
    maxPositionEmbeddings: 128,
    ropeTheta: 1_000_000,
    ropeLocalBaseFreq: 10_000,
    rmsNormEps: 1e-5,
    tieWordEmbeddings: true,
    attentionBias: false,
    queryPreAttentionScalar: 4,
    slidingWindow: 64,
    layerTypes: ["sliding_attention", "full_attention"],
    embeddingScale: Math.sqrt(16),
    ...overrides,
  };
}

function gemma4MappedConfig(overrides: Partial<Gemma4TextConfig> = {}): Gemma4TextConfig {
  return {
    family: "gemma",
    modelType: "gemma4_text",
    rawConfig: {},
    vocabSize: 32,
    vocabSizePerLayerInput: 32,
    hiddenSize: 16,
    intermediateSize: 32,
    enableMoeBlock: false,
    moeIntermediateSize: null,
    numExperts: null,
    topKExperts: null,
    numHiddenLayers: 3,
    numAttentionHeads: 4,
    numKeyValueHeads: 2,
    numGlobalKeyValueHeads: null,
    headDim: 4,
    globalHeadDim: 8,
    maxPositionEmbeddings: 128,
    slidingWindow: 64,
    layerTypes: ["sliding_attention", "full_attention", "full_attention"],
    rmsNormEps: 1e-5,
    attentionBias: false,
    tieWordEmbeddings: true,
    hiddenSizePerLayerInput: 4,
    useDoubleWideMLP: true,
    attentionKEqV: false,
    numKvSharedLayers: 1,
    slidingRopeTheta: 10_000,
    fullRopeTheta: 1_000_000,
    fullRotaryDimensions: 2,
    finalLogitSoftcapping: 30,
    embeddingScale: Math.sqrt(16),
    ...overrides,
  };
}

describe("family config parsing", () => {
  test("dense family config parsers keep their defaults honest", () => {
    const llama = parseLlamaConfig({
      ...baseRawConfig("llama"),
      tie_word_embeddings: false,
    });
    const mistral = parseMistralConfig({
      ...baseRawConfig("mistral"),
      tie_word_embeddings: false,
      sliding_window: 64,
    });
    const gemma = parseGemmaConfig({
      ...baseRawConfig("gemma"),
      hidden_act: "gelu_pytorch_tanh",
      head_dim: 8,
    });
    const phi = parsePhiConfig({
      ...baseRawConfig("phi3"),
      hidden_act: "silu",
      sliding_window: 64,
      partial_rotary_factor: 0.5,
    });
    const mistral3 = parseMistral3Config({
      model_type: "mistral3",
      text_config: {
        ...baseRawConfig("mistral"),
        tie_word_embeddings: false,
      },
    });
    const gemma3 = parseGemma3TextConfig({
      ...baseRawConfig("gemma"),
      model_type: "gemma3_text",
      hidden_activation: "gelu_pytorch_tanh",
      head_dim: 8,
      query_pre_attn_scalar: 8,
      sliding_window: 64,
      sliding_window_pattern: 2,
      tie_word_embeddings: true,
    });
    const gemma4Text = parseGemma4TextConfig({
      model_type: "gemma4_text",
      vocab_size: 32,
      vocab_size_per_layer_input: 32,
      hidden_size: 16,
      intermediate_size: 32,
      num_hidden_layers: 3,
      num_attention_heads: 4,
      num_key_value_heads: 2,
      num_global_key_value_heads: 1,
      head_dim: 4,
      global_head_dim: 8,
      max_position_embeddings: 128,
      sliding_window: 64,
      layer_types: ["sliding_attention", "full_attention", "full_attention"],
      rope_parameters: {
        sliding_attention: { rope_type: "default", rope_theta: 10000 },
        full_attention: {
          rope_type: "proportional",
          rope_theta: 1_000_000,
          partial_rotary_factor: 0.25,
        },
      },
      hidden_activation: "gelu_pytorch_tanh",
      hidden_size_per_layer_input: 4,
      use_double_wide_mlp: true,
      num_kv_shared_layers: 1,
      final_logit_softcapping: 30,
      attention_k_eq_v: false,
      tie_word_embeddings: true,
    });
    const gemma4 = parseGemma4Config({
      model_type: "gemma4",
      text_config: {
        model_type: "gemma4_text",
        vocab_size: 32,
        vocab_size_per_layer_input: 32,
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 4,
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
        hidden_size_per_layer_input: 0,
        use_double_wide_mlp: false,
        num_kv_shared_layers: 0,
        final_logit_softcapping: 30,
        attention_k_eq_v: true,
        num_global_key_value_heads: 1,
        tie_word_embeddings: true,
        use_bidirectional_attention: "vision",
      },
    });

    expect(llama.family).toBe("llama");
    expect(llama.headDim).toBe(4);
    expect(mistral.family).toBe("mistral");
    expect(mistral.slidingWindow).toBe(64);
    expect(gemma.family).toBe("gemma");
    expect(gemma.tieWordEmbeddings).toBe(true);
    expect(gemma.normWeightOffset).toBe(true);
    expect(gemma.embeddingScale).toBe(Math.sqrt(16));
    expect(gemma.mlpActivation).toBe("gelu_pytorch_tanh");
    expect(phi.family).toBe("phi");
    expect(phi.attentionProjectionLayout).toBe("packed_qkv");
    expect(phi.mlpProjectionLayout).toBe("packed_gate_up");
    expect(phi.rotaryDimensions).toBe(2);
    expect(mistral3.family).toBe("mistral");
    expect(mistral3.modelType).toBe("mistral3");
    expect(gemma3.family).toBe("gemma");
    expect(gemma3.layerTypes).toEqual(["sliding_attention", "full_attention"]);
    expect(gemma3.queryPreAttentionScalar).toBe(8);
    expect(gemma3.tieWordEmbeddings).toBe(true);
    expect(gemma4Text.family).toBe("gemma");
    expect(gemma4Text.hiddenSizePerLayerInput).toBe(4);
    expect(gemma4Text.numKvSharedLayers).toBe(1);
    expect(gemma4Text.fullRotaryDimensions).toBe(2);
    expect(gemma4.modelType).toBe("gemma4");
    expect(gemma4.attentionKEqV).toBe(true);
    expect(gemma4.hiddenSizePerLayerInput).toBe(0);

    const gemma4WithNullGlobalHeads = parseGemma4Config({
      model_type: "gemma4",
      text_config: {
        model_type: "gemma4_text",
        vocab_size: 32,
        vocab_size_per_layer_input: 32,
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        num_global_key_value_heads: null,
        head_dim: 4,
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
        hidden_size_per_layer_input: 0,
        use_double_wide_mlp: false,
        num_kv_shared_layers: 0,
        final_logit_softcapping: 30,
        attention_k_eq_v: false,
        tie_word_embeddings: true,
      },
    });
    expect(gemma4WithNullGlobalHeads.numGlobalKeyValueHeads).toBeNull();
  });

  test("Gemma 4 top-level text loading accepts the upstream vision attention marker only", () => {
    const validTextConfig = {
      model_type: "gemma4_text",
      vocab_size: 32,
      hidden_size: 16,
      intermediate_size: 32,
      num_hidden_layers: 2,
      num_attention_heads: 4,
      num_key_value_heads: 2,
      head_dim: 4,
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
      attention_k_eq_v: true,
      use_bidirectional_attention: "vision",
    };

    expect(
      parseGemma4Config({
        model_type: "gemma4",
        text_config: validTextConfig,
      }).modelType,
    ).toBe("gemma4");

    expect(() =>
      parseGemma4Config({
        model_type: "gemma4",
        text_config: {
          ...validTextConfig,
          use_bidirectional_attention: true,
        },
      }),
    ).toThrow("use_bidirectional_attention is multimodal Gemma 4 behavior");
  });

  test("family config parsers reject mismatched model_type values", () => {
    expect(() => parseLlamaConfig(baseRawConfig("gemma"))).toThrow('must be "llama"');
    expect(() => parseMistralConfig(baseRawConfig("llama"))).toThrow('must be "mistral"');
    expect(() => parseGemmaConfig(baseRawConfig("mistral"))).toThrow('must be "gemma"');
    expect(() => parsePhiConfig(baseRawConfig("llama"))).toThrow('must be "phi3"');
    expect(() => parseMistral3Config(baseRawConfig("mistral"))).toThrow('must be "mistral3"');
    expect(() => parseGemma3TextConfig(baseRawConfig("llama"))).toThrow('must be "gemma3_text"');
    expect(() => parseGemma4TextConfig(baseRawConfig("llama"))).toThrow('must be "gemma4_text"');
    expect(() => parseGemma4Config(baseRawConfig("gemma"))).toThrow('must be "gemma4"');
  });

  test("gemma3 config derives sliding patterns and rope defaults when optional fields are absent", () => {
    const config = parseGemma3TextConfig({
      model_type: "gemma3_text",
      vocab_size: 32,
      hidden_size: 16,
      intermediate_size: 32,
      num_hidden_layers: 4,
      num_attention_heads: 4,
      max_position_embeddings: 128,
      hidden_act: "gelu_pytorch_tanh",
      sliding_window_pattern: 2,
    });

    expect(config.numKeyValueHeads).toBe(4);
    expect(config.headDim).toBe(4);
    expect(config.layerTypes).toEqual([
      "sliding_attention",
      "full_attention",
      "sliding_attention",
      "full_attention",
    ]);
    expect(config.ropeTheta).toBe(1_000_000);
    expect(config.ropeLocalBaseFreq).toBe(10_000);
    expect(config.queryPreAttentionScalar).toBe(4);
  });

  test("gemma3 config validates layer_types, rope_parameters, activation, and sliding windows", () => {
    expect(() =>
      parseGemma3TextConfig({
        ...baseRawConfig("gemma"),
        model_type: "gemma3_text",
        hidden_act: "gelu_pytorch_tanh",
        layer_types: ["full_attention"],
      }),
    ).toThrow("layer_types must be an array with 2 entries");

    expect(() =>
      parseGemma3TextConfig({
        ...baseRawConfig("gemma"),
        model_type: "gemma3_text",
        hidden_act: "gelu_pytorch_tanh",
        layer_types: ["invalid", "full_attention"],
      }),
    ).toThrow('layer_types[0] must be "full_attention" or "sliding_attention"');

    const ropeConfig = parseGemma3TextConfig({
      ...baseRawConfig("gemma"),
      model_type: "gemma3_text",
      hidden_act: "gelu_pytorch_tanh",
      layer_types: ["full_attention", "sliding_attention"],
      rope_parameters: {
        full_attention: { rope_theta: 123456 },
        sliding_attention: { rope_theta: 7890 },
      },
    });
    expect(ropeConfig.ropeTheta).toBe(123456);
    expect(ropeConfig.ropeLocalBaseFreq).toBe(7890);

    expect(() =>
      parseGemma3TextConfig({
        ...baseRawConfig("gemma"),
        model_type: "gemma3_text",
        hidden_activation: "silu",
      }),
    ).toThrow('hidden activation must be "gelu_pytorch_tanh"');

    expect(() =>
      parseGemma3TextConfig({
        ...baseRawConfig("gemma"),
        model_type: "gemma3_text",
        hidden_act: "gelu_pytorch_tanh",
        layer_types: ["sliding_attention", "full_attention"],
        sliding_window: 0,
      }),
    ).toThrow("sliding_window must be positive");
  });
});

describe("llama-like weight mapping", () => {
  test("sanitizeLlamaLikeWeight maps checkpoint names onto model parameter paths", () => {
    const plainConfig = mappedConfig();
    const attentionBiasConfig = mappedConfig({ attentionBias: true });
    const tiedConfig = mappedConfig({ tieWordEmbeddings: true });

    expect(sanitizeLlamaLikeWeight(plainConfig, "model.embed_tokens.weight")).toBe(
      "model.embedTokens.weight",
    );
    expect(sanitizeLlamaLikeWeight(plainConfig, "model.norm.weight")).toBe("model.norm.weight");
    expect(sanitizeLlamaLikeWeight(plainConfig, "lm_head.weight")).toBe("lmHead.weight");
    expect(sanitizeLlamaLikeWeight(tiedConfig, "lm_head.weight")).toBeNull();
    expect(sanitizeLlamaLikeWeight(plainConfig, "model.layers.1.self_attn.q_proj.weight")).toBe(
      "model.layers.1.selfAttention.qProjection.weight",
    );
    expect(
      sanitizeLlamaLikeWeight(attentionBiasConfig, "model.layers.1.self_attn.q_proj.bias"),
    ).toBe("model.layers.1.selfAttention.qProjection.bias");
    expect(sanitizeLlamaLikeWeight(plainConfig, "model.layers.1.self_attn.q_proj.bias")).toBeNull();
    expect(sanitizeLlamaLikeWeight(plainConfig, "model.layers.1.mlp.down_proj.weight")).toBe(
      "model.layers.1.mlp.downProjection.weight",
    );
    expect(sanitizeLlamaLikeWeight(plainConfig, "unknown.weight")).toBeNull();
  });

  test("family wrappers delegate to the shared llama-like mapping and ignore rules", () => {
    const config = mappedConfig({ attentionBias: true, tieWordEmbeddings: true });
    const packedConfig = mappedConfig({
      family: "phi",
      modelType: "phi3",
      attentionProjectionLayout: "packed_qkv",
      mlpProjectionLayout: "packed_gate_up",
    });
    const gemma3Config = gemma3MappedConfig();
    const gemma4Config = gemma4MappedConfig();
    const gemma4FullAttentionConfig = gemma4MappedConfig({
      hiddenSizePerLayerInput: 0,
      numKvSharedLayers: 0,
      attentionKEqV: true,
      numGlobalKeyValueHeads: 1,
      layerTypes: ["full_attention", "full_attention", "full_attention"],
      modelType: "gemma4",
    });

    expect(sanitizeLlamaWeight(config, "model.layers.0.self_attn.k_proj.bias")).toBe(
      "model.layers.0.selfAttention.kProjection.bias",
    );
    expect(sanitizeGemmaWeight(config, "model.layers.0.self_attn.v_proj.bias")).toBe(
      "model.layers.0.selfAttention.vProjection.bias",
    );
    expect(sanitizeMistralWeight(config, "model.layers.0.self_attn.o_proj.bias")).toBe(
      "model.layers.0.selfAttention.outputProjection.bias",
    );
    expect(sanitizePhiWeight(packedConfig, "model.layers.0.self_attn.qkv_proj.weight")).toBe(
      "model.layers.0.selfAttention.qkvProjection.weight",
    );
    expect(sanitizePhiWeight(packedConfig, "model.layers.0.mlp.gate_up_proj.weight")).toBe(
      "model.layers.0.mlp.gateUpProjection.weight",
    );
    expect(
      sanitizeMistral3Weight(config, "language_model.model.layers.0.self_attn.k_proj.weight"),
    ).toBe("model.layers.0.selfAttention.kProjection.weight");
    expect(sanitizeMistral3Weight(config, "model.layers.0.self_attn.k_proj.weight")).toBeNull();
    expect(sanitizeGemma3Weight(gemma3Config, "model.layers.0.self_attn.q_norm.weight")).toBe(
      "model.layers.0.selfAttention.qNorm.weight",
    );
    expect(
      sanitizeGemma3Weight(gemma3Config, "model.layers.0.pre_feedforward_layernorm.weight"),
    ).toBe("model.layers.0.preFeedforwardLayerNorm.weight");
    expect(
      sanitizeGemma4TextWeight(gemma4Config, "model.layers.0.per_layer_input_gate.weight"),
    ).toBe("model.layers.0.perLayerInputGate.weight");
    expect(sanitizeGemma4TextWeight(gemma4Config, "model.per_layer_projection_norm.weight")).toBe(
      "model.perLayerProjectionNorm.weight",
    );
    expect(
      sanitizeGemma4Weight(
        gemma4FullAttentionConfig,
        "model.language_model.layers.0.self_attn.k_proj.weight",
      ),
    ).toBe("model.layers.0.selfAttention.kProjection.weight");
    expect(
      sanitizeGemma4Weight(
        gemma4FullAttentionConfig,
        "language_model.model.layers.0.self_attn.k_proj.weight",
      ),
    ).toBe("model.layers.0.selfAttention.kProjection.weight");
    expect(
      sanitizeGemma4Weight(
        gemma4FullAttentionConfig,
        "model.language_model.layers.0.self_attn.v_proj.weight",
      ),
    ).toBeNull();
    expect(sanitizeGemma4Weight(gemma4Config, "language_model.model.embed_tokens.weight")).toBe(
      "model.embedTokens.weight",
    );
    expect(
      sanitizeGemma4Weight(
        gemma4FullAttentionConfig,
        "model.embed_vision.embedding_projection.weight",
      ),
    ).toBeNull();

    expect(isIgnoredLlamaLikeWeight(config, "model.layers.0.self_attn.rotary_emb.inv_freq")).toBe(
      true,
    );
    expect(isIgnoredLlamaLikeWeight(config, "lm_head.weight")).toBe(true);
    expect(isIgnoredLlamaWeight(config, "lm_head.weight")).toBe(true);
    expect(isIgnoredGemmaWeight(config, "lm_head.weight")).toBe(true);
    expect(isIgnoredMistralWeight(config, "lm_head.weight")).toBe(true);
    expect(isIgnoredMistral3Weight(config, "vision_tower.patch_embed.weight")).toBe(true);
    expect(
      isIgnoredMistral3Weight(config, "language_model.model.layers.0.self_attn.k_proj.weight"),
    ).toBe(false);
    expect(isIgnoredPhiWeight(config, "lm_head.weight")).toBe(true);
    expect(isIgnoredGemma3Weight(gemma3Config, "lm_head.weight")).toBe(true);
    expect(
      isIgnoredGemma4TextWeight(gemma4Config, "model.layers.0.self_attn.rotary_emb.inv_freq"),
    ).toBe(true);
    expect(
      isIgnoredGemma4Weight(
        gemma4FullAttentionConfig,
        "model.language_model.layers.0.self_attn.k_proj.weight",
      ),
    ).toBe(false);
    expect(
      isIgnoredGemma4Weight(gemma4Config, "model.embed_vision.embedding_projection.weight"),
    ).toBe(true);
  });
});
