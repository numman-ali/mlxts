import { afterEach, describe, expect, test } from "bun:test";
import { loadSafetensors, saveSafetensors, zeros } from "@mlxts/core";
import { applyLoRAToModule } from "@mlxts/lora";
import { LoRALinear } from "@mlxts/nn";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { loadCausalLMAdapters, saveCausalLMAdapters } from "./lora-adapters";
import { collectLoRAWrapperStates, expectCausalLMModule } from "./lora-module-traversal";
import { resolveLoRATargets } from "./lora-targets";
import { resolveFamily } from "./registry";
import type { CausalLM } from "./types";

const tempRoots: string[] = [];

function createTempDir(prefix: string): string {
  const directory = mkdtempSync(join(tmpdir(), prefix));
  tempRoots.push(directory);
  return directory;
}

afterEach(() => {
  while (tempRoots.length > 0) {
    const directory = tempRoots.pop();
    if (directory !== undefined) {
      rmSync(directory, { recursive: true, force: true });
    }
  }
});

function rawConfigForFamily(
  modelType: "llama" | "mistral" | "gemma" | "phi3",
): Record<string, unknown> {
  const base = {
    vocab_size: 5,
    hidden_size: 8,
    intermediate_size: 16,
    num_hidden_layers: 2,
    num_attention_heads: 2,
    num_key_value_heads: 2,
    max_position_embeddings: 32,
    rope_theta: 10_000,
    rms_norm_eps: 1e-6,
    attention_bias: false,
  };

  if (modelType === "phi3") {
    return {
      ...base,
      model_type: "phi3",
      tie_word_embeddings: false,
      hidden_act: "silu",
      partial_rotary_factor: 1.0,
      sliding_window: 8,
    };
  }

  if (modelType === "gemma") {
    return {
      ...base,
      model_type: "gemma",
      tie_word_embeddings: false,
      head_dim: 4,
      hidden_act: "gelu_pytorch_tanh",
    };
  }

  if (modelType === "mistral") {
    return {
      ...base,
      model_type: "mistral",
      tie_word_embeddings: false,
      sliding_window: 8,
    };
  }

  return {
    ...base,
    model_type: "llama",
    tie_word_embeddings: false,
  };
}

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

function createModel(rawConfig: Record<string, unknown>): CausalLM {
  const modelType = rawConfig.model_type;
  if (typeof modelType !== "string") {
    throw new Error("expected rawConfig.model_type to be a string");
  }
  const registration = resolveFamily(modelType);
  return registration.createModel(registration.parseConfig(rawConfig));
}

function basePeftConfig(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    peft_type: "LORA",
    task_type: "CAUSAL_LM",
    r: 4,
    lora_alpha: 8,
    lora_dropout: 0,
    bias: "none",
    fan_in_fan_out: false,
    use_rslora: false,
    target_modules: ["q_proj"],
    ...overrides,
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

  test("lastLayers keeps decoder-root auxiliary paths while slicing layer-local targets", () => {
    using model = createModel(rawConfigForGemma4Text());

    const resolved = resolveLoRATargets(model, {
      preset: "all-linear",
      lastLayers: 1,
    });

    expect(resolved.paths).toContain("model.perLayerModelProjection");
    expect(resolved.paths.every((path) => !path.startsWith("model.layers.0."))).toBe(true);
    expect(resolved.paths.some((path) => path.startsWith("model.layers.2."))).toBe(true);
  });

  test("can opt lmHead into all-linear selection", () => {
    using model = createModel(rawConfigForFamily("llama"));

    const defaultTargets = resolveLoRATargets(model, { preset: "all-linear" });
    const withLmHead = resolveLoRATargets(model, {
      preset: "all-linear",
      includeLmHead: true,
    });

    expect(defaultTargets.paths).not.toContain("lmHead");
    expect(withLmHead.paths).toContain("lmHead");
  });
});

describe("causal LM adapter interop", () => {
  test("saves and reloads a first-pass PEFT LoRA adapter directory", async () => {
    const directory = createTempDir("mlxts-transformers-peft-");
    using sourceModel = createModel(rawConfigForFamily("llama"));
    using targetModel = createModel(rawConfigForFamily("llama"));

    const targets = resolveLoRATargets(sourceModel, {
      preset: "attention",
      lastLayers: 1,
    });
    applyLoRAToModule(expectCausalLMModule(sourceModel), {
      paths: targets.paths,
      rank: 4,
      alpha: 8,
      dropout: 0,
    });

    await saveCausalLMAdapters(sourceModel, directory, {
      format: "peft",
      baseModelNameOrPath: "meta-llama/Llama-3.2-1B-Instruct",
    });

    const config = JSON.parse(readFileSync(join(directory, "adapter_config.json"), "utf8")) as {
      peft_type: string;
      task_type: string;
      target_modules: string[];
      base_model_name_or_path: string | null;
    };
    expect(config.peft_type).toBe("LORA");
    expect(config.task_type).toBe("CAUSAL_LM");
    expect(config.base_model_name_or_path).toBe("meta-llama/Llama-3.2-1B-Instruct");
    expect(config.target_modules).toEqual(["q_proj", "k_proj", "v_proj", "o_proj"]);

    const saved = await loadSafetensors(join(directory, "adapter_model.safetensors"));
    try {
      expect(Object.keys(saved.tensors).sort()).toEqual([
        "base_model.model.model.layers.1.self_attn.k_proj.lora_A.weight",
        "base_model.model.model.layers.1.self_attn.k_proj.lora_B.weight",
        "base_model.model.model.layers.1.self_attn.o_proj.lora_A.weight",
        "base_model.model.model.layers.1.self_attn.o_proj.lora_B.weight",
        "base_model.model.model.layers.1.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.layers.1.self_attn.q_proj.lora_B.weight",
        "base_model.model.model.layers.1.self_attn.v_proj.lora_A.weight",
        "base_model.model.model.layers.1.self_attn.v_proj.lora_B.weight",
      ]);
    } finally {
      for (const tensor of Object.values(saved.tensors)) {
        tensor.free();
      }
    }

    await loadCausalLMAdapters(targetModel, directory);
    const wrappers = collectLoRAWrapperStates(targetModel);
    expect(wrappers).toHaveLength(4);
    expect(wrappers.every((slot) => slot.child instanceof LoRALinear)).toBe(true);
  });

  test("auto-detects and reloads native mlxts adapter directories", async () => {
    const directory = createTempDir("mlxts-transformers-native-");
    using sourceModel = createModel(rawConfigForFamily("llama"));
    using targetModel = createModel(rawConfigForFamily("llama"));

    const targets = resolveLoRATargets(sourceModel, {
      preset: "attention",
      lastLayers: 1,
    });
    applyLoRAToModule(expectCausalLMModule(sourceModel), {
      paths: targets.paths,
      rank: 4,
      alpha: 8,
      dropout: 0,
    });

    await saveCausalLMAdapters(sourceModel, directory);
    await loadCausalLMAdapters(targetModel, directory);

    const wrappers = collectLoRAWrapperStates(targetModel);
    expect(wrappers).toHaveLength(4);
  });

  test("rejects PEFT export when active wrappers do not share one config", async () => {
    const directory = createTempDir("mlxts-transformers-peft-mismatch-");
    using model = createModel(rawConfigForFamily("llama"));

    applyLoRAToModule(expectCausalLMModule(model), {
      paths: ["model.layers.0.selfAttention.qProjection"],
      rank: 4,
      alpha: 8,
      dropout: 0,
    });
    applyLoRAToModule(expectCausalLMModule(model), {
      paths: ["model.layers.0.selfAttention.kProjection"],
      rank: 8,
      alpha: 16,
      dropout: 0,
    });

    await expect(saveCausalLMAdapters(model, directory, { format: "peft" })).rejects.toThrow(
      "one shared rank, alpha, and dropout",
    );
  });

  test("rejects unsupported PEFT modules_to_save configs", async () => {
    const directory = createTempDir("mlxts-transformers-peft-unsupported-");
    writeFileSync(
      join(directory, "adapter_config.json"),
      JSON.stringify(basePeftConfig({ modules_to_save: ["lm_head"] })),
    );

    using model = createModel(rawConfigForFamily("llama"));
    await expect(loadCausalLMAdapters(model, directory, { format: "peft" })).rejects.toThrow(
      "modules_to_save",
    );
  });

  test("rejects unsupported or malformed PEFT configs before loading tensors", async () => {
    const cases = [
      {
        name: "wrong peft_type",
        config: basePeftConfig({ peft_type: "ADALORA" }),
        expected: 'peft_type to "LORA"',
      },
      {
        name: "wrong task_type",
        config: basePeftConfig({ task_type: "SEQ_CLS" }),
        expected: 'task_type to "CAUSAL_LM"',
      },
      {
        name: "trainable token indices",
        config: basePeftConfig({ trainable_token_indices: [1] }),
        expected: "trainable_token_indices",
      },
      {
        name: "rank pattern",
        config: basePeftConfig({ rank_pattern: { q_proj: 8 } }),
        expected: "rank_pattern",
      },
      {
        name: "dora",
        config: basePeftConfig({ use_dora: true }),
        expected: "standard LoRA only",
      },
      {
        name: "fan in fan out",
        config: basePeftConfig({ fan_in_fan_out: true }),
        expected: "fan_in_fan_out=true",
      },
      {
        name: "rslora",
        config: basePeftConfig({ use_rslora: true }),
        expected: "use_rslora=true",
      },
      {
        name: "bias",
        config: basePeftConfig({ bias: "all" }),
        expected: 'bias="none"',
      },
      {
        name: "invalid rank",
        config: basePeftConfig({ r: 0 }),
        expected: "PEFT r must be a positive integer",
      },
      {
        name: "invalid alpha",
        config: basePeftConfig({ lora_alpha: Number.NaN }),
        expected: "lora_alpha must be a finite number",
      },
      {
        name: "invalid dropout",
        config: basePeftConfig({ lora_dropout: 1 }),
        expected: "lora_dropout must satisfy 0 <= p < 1",
      },
      {
        name: "invalid target modules",
        config: basePeftConfig({ target_modules: [1, 2, 3] }),
        expected: "target_modules must be a string or an array of strings",
      },
      {
        name: "unsupported key",
        config: {
          ...basePeftConfig(),
          mysterious_setting: true,
        },
        expected: 'unsupported PEFT config key "mysterious_setting"',
      },
    ] as const;

    for (const testCase of cases) {
      const directory = createTempDir(
        `mlxts-transformers-peft-${testCase.name.replace(/\s+/g, "-")}-`,
      );
      writeFileSync(join(directory, "adapter_config.json"), JSON.stringify(testCase.config));

      using model = createModel(rawConfigForFamily("llama"));
      await expect(loadCausalLMAdapters(model, directory, { format: "peft" })).rejects.toThrow(
        testCase.expected,
      );
    }
  });

  test("rejects adapter configs that do not identify a supported format", async () => {
    const directory = createTempDir("mlxts-transformers-adapter-unknown-");
    writeFileSync(join(directory, "adapter_config.json"), JSON.stringify({ mystery: true }));

    using model = createModel(rawConfigForFamily("llama"));
    await expect(loadCausalLMAdapters(model, directory)).rejects.toThrow(
      "could not detect adapter format",
    );
  });

  test("rejects unexpected PEFT tensor names", async () => {
    const directory = createTempDir("mlxts-transformers-peft-unexpected-");
    writeFileSync(join(directory, "adapter_config.json"), JSON.stringify(basePeftConfig()));

    using tensor = zeros([4, 8]);
    await saveSafetensors(
      {
        "base_model.model.not_a_real_module.lora_A.weight": tensor,
      },
      join(directory, "adapter_model.safetensors"),
    );

    using model = createModel(rawConfigForFamily("llama"));
    await expect(loadCausalLMAdapters(model, directory, { format: "peft" })).rejects.toThrow(
      "unexpected PEFT adapter tensor",
    );
  });

  test("rejects incomplete PEFT adapter tensor pairs", async () => {
    const directory = createTempDir("mlxts-transformers-peft-missing-");
    writeFileSync(join(directory, "adapter_config.json"), JSON.stringify(basePeftConfig()));

    using tensor = zeros([4, 8]);
    await saveSafetensors(
      {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": tensor,
      },
      join(directory, "adapter_model.safetensors"),
    );

    using model = createModel(rawConfigForFamily("llama"));
    await expect(loadCausalLMAdapters(model, directory, { format: "peft" })).rejects.toThrow(
      "missing PEFT adapter tensor",
    );
  });
});
