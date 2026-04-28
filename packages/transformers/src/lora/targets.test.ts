import { describe, expect, test } from "bun:test";

import { createModel, rawConfigForFamily } from "./__test__/fixtures.test";
import { resolveLoRATargets } from "./targets";

function rawConfigForGemma3Text(): Record<string, unknown> {
  return {
    model_type: "gemma3_text",
    vocab_size: 5,
    hidden_size: 8,
    intermediate_size: 16,
    num_hidden_layers: 2,
    num_attention_heads: 2,
    num_key_value_heads: 1,
    head_dim: 4,
    max_position_embeddings: 32,
    rms_norm_eps: 1e-6,
    attention_bias: false,
    hidden_activation: "gelu_pytorch_tanh",
    query_pre_attn_scalar: 4,
    sliding_window: 2,
    sliding_window_pattern: 2,
    rope_theta: 1_000_000,
    rope_local_base_freq: 10_000,
    tie_word_embeddings: false,
  };
}

function rawConfigForGemma4Text(): Record<string, unknown> {
  return {
    model_type: "gemma4_text",
    vocab_size: 5,
    vocab_size_per_layer_input: 5,
    hidden_size: 8,
    intermediate_size: 16,
    num_hidden_layers: 3,
    num_attention_heads: 2,
    num_key_value_heads: 1,
    num_global_key_value_heads: 1,
    head_dim: 4,
    global_head_dim: 4,
    max_position_embeddings: 32,
    rms_norm_eps: 1e-6,
    attention_bias: false,
    hidden_activation: "gelu_pytorch_tanh",
    sliding_window: 2,
    layer_types: ["sliding_attention", "full_attention", "full_attention"],
    rope_parameters: {
      sliding_attention: {
        rope_type: "default",
        rope_theta: 10_000,
      },
      full_attention: {
        rope_type: "proportional",
        rope_theta: 1_000_000,
        partial_rotary_factor: 0.5,
      },
    },
    tie_word_embeddings: false,
    hidden_size_per_layer_input: 2,
    use_double_wide_mlp: true,
    num_kv_shared_layers: 1,
    final_logit_softcapping: 30,
    attention_k_eq_v: false,
    enable_moe_block: false,
  };
}

function rawConfigForGemma4Wrapper(): Record<string, unknown> {
  return {
    model_type: "gemma4",
    text_config: rawConfigForGemma4Text(),
    vision_config: {
      model_type: "gemma4_vision",
    },
    audio_config: {
      model_type: "gemma4_audio",
    },
  };
}

function rawConfigForMistral3TextWrapper(): Record<string, unknown> {
  return {
    model_type: "mistral3",
    text_config: {
      ...rawConfigForFamily("mistral"),
      tie_word_embeddings: false,
    },
    vision_config: {
      model_type: "pixtral",
    },
  };
}

describe("resolveLoRATargets", () => {
  test("resolves standard decoder presets across the supported families", () => {
    const cases = [
      {
        name: "llama",
        rawConfig: rawConfigForFamily("llama"),
        expectedAttentionLeafs: ["qProjection", "kProjection", "vProjection", "outputProjection"],
        expectedMlpLeafs: ["gateProjection", "upProjection", "downProjection"],
        allLinearAddsExtra: false,
      },
      {
        name: "mistral",
        rawConfig: rawConfigForFamily("mistral"),
        expectedAttentionLeafs: ["qProjection", "kProjection", "vProjection", "outputProjection"],
        expectedMlpLeafs: ["gateProjection", "upProjection", "downProjection"],
        allLinearAddsExtra: false,
      },
      {
        name: "gemma",
        rawConfig: rawConfigForFamily("gemma"),
        expectedAttentionLeafs: ["qProjection", "kProjection", "vProjection", "outputProjection"],
        expectedMlpLeafs: ["gateProjection", "upProjection", "downProjection"],
        allLinearAddsExtra: false,
      },
      {
        name: "phi3",
        rawConfig: rawConfigForFamily("phi3"),
        expectedAttentionLeafs: ["qkvProjection", "outputProjection"],
        expectedMlpLeafs: ["gateUpProjection", "downProjection"],
        allLinearAddsExtra: false,
      },
      {
        name: "mistral3",
        rawConfig: rawConfigForMistral3TextWrapper(),
        expectedAttentionLeafs: ["qProjection", "kProjection", "vProjection", "outputProjection"],
        expectedMlpLeafs: ["gateProjection", "upProjection", "downProjection"],
        allLinearAddsExtra: false,
      },
      {
        name: "gemma3_text",
        rawConfig: rawConfigForGemma3Text(),
        expectedAttentionLeafs: ["qProjection", "kProjection", "vProjection", "outputProjection"],
        expectedMlpLeafs: ["gateProjection", "upProjection", "downProjection"],
        allLinearAddsExtra: false,
      },
      {
        name: "gemma4_text",
        rawConfig: rawConfigForGemma4Text(),
        expectedAttentionLeafs: ["qProjection", "kProjection", "outputProjection"],
        expectedMlpLeafs: ["gateProjection", "upProjection", "downProjection"],
        allLinearAddsExtra: true,
      },
      {
        name: "gemma4",
        rawConfig: rawConfigForGemma4Wrapper(),
        expectedAttentionLeafs: ["qProjection", "kProjection", "outputProjection"],
        expectedMlpLeafs: ["gateProjection", "upProjection", "downProjection"],
        allLinearAddsExtra: true,
      },
    ] as const;

    for (const testCase of cases) {
      using model = createModel(testCase.rawConfig);

      const attention = resolveLoRATargets(model, { preset: "attention" });
      const attentionMlp = resolveLoRATargets(model, { preset: "attention+mlp" });
      const allLinear = resolveLoRATargets(model, { preset: "all-linear" });

      for (const leaf of testCase.expectedAttentionLeafs) {
        expect(attention.paths.some((path) => path.endsWith(`.${leaf}`))).toBe(true);
      }
      for (const leaf of testCase.expectedMlpLeafs) {
        expect(attentionMlp.paths.some((path) => path.endsWith(`.${leaf}`))).toBe(true);
      }

      expect(attentionMlp.paths.length).toBeGreaterThanOrEqual(attention.paths.length);
      expect(allLinear.paths.every((path) => path !== "lmHead")).toBe(true);
      expect(allLinear.paths.every((path) => !path.includes("embedTokens"))).toBe(true);

      if (testCase.allLinearAddsExtra) {
        expect(allLinear.paths.length).toBeGreaterThan(attentionMlp.paths.length);
        expect(allLinear.paths).toContain("model.perLayerModelProjection");
      } else {
        expect(allLinear.paths).toEqual(attentionMlp.paths);
      }
    }
  });

  test("handles layer slicing and explicit lmHead opt-in", () => {
    using gemma4 = createModel(rawConfigForGemma4Text());

    const lastLayerTargets = resolveLoRATargets(gemma4, {
      preset: "all-linear",
      lastLayers: 1,
    });

    expect(lastLayerTargets.paths).toContain("model.perLayerModelProjection");
    expect(lastLayerTargets.paths.every((path) => !path.startsWith("model.layers.0."))).toBe(true);
    expect(lastLayerTargets.paths.some((path) => path.startsWith("model.layers.2."))).toBe(true);

    using llama = createModel(rawConfigForFamily("llama"));

    const defaultTargets = resolveLoRATargets(llama, { preset: "all-linear" });
    const withLmHead = resolveLoRATargets(llama, {
      preset: "all-linear",
      includeLmHead: true,
    });

    expect(defaultTargets.paths).not.toContain("lmHead");
    expect(withLmHead.paths).toContain("lmHead");
  });
});
