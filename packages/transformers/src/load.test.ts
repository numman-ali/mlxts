import { afterEach, describe, expect, test } from "bun:test";
import { array, type MxArray, mxEval, retainArray, saveSafetensors } from "@mlxts/core";
import { Linear } from "@mlxts/nn";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { AutoModel, AutoTokenizer } from "./auto";
import { Gemma3TextCausalLM } from "./families/gemma3/model";
import { Gemma4TextMLP } from "./families/gemma4/mlp";
import { Gemma4TextCausalLM } from "./families/gemma4/model";
import { LlamaLikeMLP } from "./families/llama-like/mlp";
import { LlamaLikeCausalLM } from "./families/llama-like/model";
import { generateText, generateTokens } from "./generation";
import { KVCache, LayerPatternKVCache, SlidingWindowKVCache } from "./infrastructure/cache";
import { loadCausalLM, loadPretrainedTokenizer } from "./load";
import { resolveFamily } from "./registry";
import { ConfigParseError, MissingWeightsError, type SupportedModelFamily } from "./types";

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

function flattenNumbers(value: unknown): number[] {
  if (typeof value === "number") {
    return [value];
  }
  if (!Array.isArray(value)) {
    throw new Error("Expected nested numeric arrays");
  }
  return value.flatMap((entry) => flattenNumbers(entry));
}

function expectCloseLists(actual: unknown, expected: unknown): void {
  const actualValues = flattenNumbers(actual);
  const expectedValues = flattenNumbers(expected);
  expect(actualValues.length).toBe(expectedValues.length);
  for (let index = 0; index < actualValues.length; index += 1) {
    expect(actualValues[index]).toBeCloseTo(expectedValues[index] ?? 0, 6);
  }
}

function expectTensor(value: MxArray | null, message: string): MxArray {
  if (value === null) {
    throw new Error(message);
  }
  return value;
}

function tokenizerFixture() {
  return {
    tokenizerJson: {
      model: {
        type: "BPE",
        vocab: {
          H: 0,
          i: 1,
          "!": 2,
          "<eos>": 3,
          "<bos>": 4,
        },
        merges: [],
        unk_token: "<eos>",
        byte_fallback: false,
      },
      added_tokens: [
        { id: 3, content: "<eos>", special: true },
        { id: 4, content: "<bos>", special: true },
      ],
      pre_tokenizer: {
        type: "ByteLevel",
        add_prefix_space: false,
        trim_offsets: true,
        use_regex: true,
      },
      decoder: {
        type: "ByteLevel",
      },
    },
    tokenizerConfig: {
      eos_token: "<eos>",
      bos_token: "<bos>",
      add_bos_token: true,
      add_eos_token: false,
    },
    specialTokensMap: {
      eos_token: { content: "<eos>" },
      bos_token: { content: "<bos>" },
    },
  };
}

function base64Token(...bytes: number[]): string {
  return btoa(String.fromCharCode(...bytes));
}

function tekkenTokenizerFixture() {
  return {
    tekkenJson: {
      config: {
        pattern: "\\p{L}+| ?[^\\s\\p{L}\\p{N}]+|\\s+",
        default_num_special_tokens: 4,
      },
      special_tokens: [
        { rank: 0, token_str: "<unk>", is_control: true },
        { rank: 1, token_str: "<s>", is_control: true },
        { rank: 2, token_str: "</s>", is_control: true },
        { rank: 3, token_str: "<pad>", is_control: true },
      ],
      vocab: [
        { rank: 0, token_bytes: base64Token(72), token_str: "H" },
        { rank: 1, token_bytes: base64Token(105), token_str: "i" },
        { rank: 2, token_bytes: base64Token(33), token_str: "!" },
      ],
    },
    generationConfig: {
      bos_token_id: 1,
      eos_token_id: 2,
    },
  };
}

function rawConfigForFamily(family: SupportedModelFamily): Record<string, unknown> {
  const base = {
    vocab_size: 5,
    hidden_size: 8,
    intermediate_size: 16,
    num_hidden_layers: 2,
    num_attention_heads: 2,
    num_key_value_heads: 2,
    max_position_embeddings: 32,
    rope_theta: 10000,
    rms_norm_eps: 1e-6,
    attention_bias: false,
  };

  if (family === "llama") {
    return {
      ...base,
      model_type: "llama",
      tie_word_embeddings: false,
    };
  }

  if (family === "mistral") {
    return {
      ...base,
      model_type: "mistral",
      tie_word_embeddings: false,
      sliding_window: 2,
    };
  }

  if (family === "phi") {
    return {
      ...base,
      model_type: "phi3",
      tie_word_embeddings: false,
      hidden_act: "silu",
      partial_rotary_factor: 1.0,
      sliding_window: 8,
    };
  }

  return {
    ...base,
    model_type: "gemma",
    tie_word_embeddings: true,
    head_dim: 4,
    hidden_act: "gelu_pytorch_tanh",
  };
}

function addAttentionCheckpointTensors(
  tensors: Record<string, MxArray>,
  prefix: string,
  layer: LlamaLikeCausalLM["model"]["layers"][number],
): void {
  if (layer === undefined) {
    return;
  }

  tensors[`${prefix}.self_attn.o_proj.weight`] = retainArray(
    layer.selfAttention.outputProjection.weight,
  );
  if (layer.selfAttention.qkvProjection !== null) {
    tensors[`${prefix}.self_attn.qkv_proj.weight`] = retainArray(
      layer.selfAttention.qkvProjection.weight,
    );
    return;
  }

  if (
    layer.selfAttention.qProjection === null ||
    layer.selfAttention.kProjection === null ||
    layer.selfAttention.vProjection === null
  ) {
    throw new Error(
      "Expected split query/key/value projections for this llama-like attention layout.",
    );
  }

  tensors[`${prefix}.self_attn.q_proj.weight`] = retainArray(
    layer.selfAttention.qProjection.weight,
  );
  tensors[`${prefix}.self_attn.k_proj.weight`] = retainArray(
    layer.selfAttention.kProjection.weight,
  );
  tensors[`${prefix}.self_attn.v_proj.weight`] = retainArray(
    layer.selfAttention.vProjection.weight,
  );
}

function addMlpCheckpointTensors(
  tensors: Record<string, MxArray>,
  prefix: string,
  mlp: LlamaLikeMLP,
): void {
  if (mlp.gateUpProjection !== null) {
    tensors[`${prefix}.mlp.gate_up_proj.weight`] = retainArray(mlp.gateUpProjection.weight);
  } else {
    if (mlp.gateProjection === null || mlp.upProjection === null) {
      throw new Error("Expected split gate and up projections for this llama-like MLP layout.");
    }
    tensors[`${prefix}.mlp.gate_proj.weight`] = retainArray(mlp.gateProjection.weight);
    tensors[`${prefix}.mlp.up_proj.weight`] = retainArray(mlp.upProjection.weight);
  }
  tensors[`${prefix}.mlp.down_proj.weight`] = retainArray(mlp.downProjection.weight);
}

function checkpointTensors(model: LlamaLikeCausalLM): Record<string, MxArray> {
  const tensors: Record<string, MxArray> = {
    "model.embed_tokens.weight": retainArray(model.model.embedTokens.weight),
    "model.norm.weight": retainArray(model.model.norm.weight),
  };

  if (model.lmHead !== null) {
    tensors["lm_head.weight"] = retainArray(model.lmHead.weight);
  }

  for (let layerIndex = 0; layerIndex < model.model.layers.length; layerIndex += 1) {
    const layer = model.model.layers[layerIndex];
    if (layer === undefined) {
      continue;
    }

    const prefix = `model.layers.${layerIndex}`;
    tensors[`${prefix}.input_layernorm.weight`] = retainArray(layer.inputLayerNorm.weight);
    tensors[`${prefix}.post_attention_layernorm.weight`] = retainArray(
      layer.postAttentionLayerNorm.weight,
    );
    addAttentionCheckpointTensors(tensors, prefix, layer);
    if (!(layer.mlp instanceof LlamaLikeMLP)) {
      throw new Error("Expected the Phase 7 llama-like decoder block to own a LlamaLikeMLP.");
    }
    addMlpCheckpointTensors(tensors, prefix, layer.mlp);
  }

  return tensors;
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

function checkpointTensorsForGemma3Text(model: Gemma3TextCausalLM): Record<string, MxArray> {
  const tensors: Record<string, MxArray> = {
    "model.embed_tokens.weight": retainArray(model.model.embedTokens.weight),
    "model.norm.weight": retainArray(model.model.norm.weight),
  };

  if (model.lmHead !== null) {
    tensors["lm_head.weight"] = retainArray(model.lmHead.weight);
  }

  for (let layerIndex = 0; layerIndex < model.model.layers.length; layerIndex += 1) {
    const layer = model.model.layers[layerIndex];
    if (layer === undefined) {
      continue;
    }

    const prefix = `model.layers.${layerIndex}`;
    tensors[`${prefix}.input_layernorm.weight`] = retainArray(layer.inputLayerNorm.weight);
    tensors[`${prefix}.post_attention_layernorm.weight`] = retainArray(
      layer.postAttentionLayerNorm.weight,
    );
    tensors[`${prefix}.pre_feedforward_layernorm.weight`] = retainArray(
      layer.preFeedforwardLayerNorm.weight,
    );
    tensors[`${prefix}.post_feedforward_layernorm.weight`] = retainArray(
      layer.postFeedforwardLayerNorm.weight,
    );
    tensors[`${prefix}.self_attn.q_proj.weight`] = retainArray(
      layer.selfAttention.qProjection.weight,
    );
    tensors[`${prefix}.self_attn.k_proj.weight`] = retainArray(
      layer.selfAttention.kProjection.weight,
    );
    tensors[`${prefix}.self_attn.v_proj.weight`] = retainArray(
      layer.selfAttention.vProjection.weight,
    );
    tensors[`${prefix}.self_attn.o_proj.weight`] = retainArray(
      layer.selfAttention.outputProjection.weight,
    );
    tensors[`${prefix}.self_attn.q_norm.weight`] = retainArray(layer.selfAttention.qNorm.weight);
    tensors[`${prefix}.self_attn.k_norm.weight`] = retainArray(layer.selfAttention.kNorm.weight);
    tensors[`${prefix}.mlp.gate_proj.weight`] = retainArray(layer.mlp.gateProjection.weight);
    tensors[`${prefix}.mlp.up_proj.weight`] = retainArray(layer.mlp.upProjection.weight);
    tensors[`${prefix}.mlp.down_proj.weight`] = retainArray(layer.mlp.downProjection.weight);
  }

  return tensors;
}

function checkpointTensorsForGemma4Text(model: Gemma4TextCausalLM): Record<string, MxArray> {
  const tensors: Record<string, MxArray> = {
    "model.embed_tokens.weight": retainArray(model.model.embedTokens.weight),
    "model.norm.weight": retainArray(
      expectTensor(model.model.norm.weight, "Expected a scaled Gemma 4 model norm weight."),
    ),
  };

  addGemma4RootCheckpointTensors(tensors, model);

  for (let layerIndex = 0; layerIndex < model.model.layers.length; layerIndex += 1) {
    const layer = model.model.layers[layerIndex];
    if (layer === undefined) {
      continue;
    }
    addGemma4LayerCheckpointTensors(tensors, layerIndex, layer);
  }

  return tensors;
}

function addGemma4RootCheckpointTensors(
  tensors: Record<string, MxArray>,
  model: Gemma4TextCausalLM,
): void {
  if (model.model.embedTokensPerLayer !== null) {
    tensors["model.embed_tokens_per_layer.weight"] = retainArray(
      model.model.embedTokensPerLayer.weight,
    );
  }
  if (model.model.perLayerModelProjection !== null) {
    tensors["model.per_layer_model_projection.weight"] = retainArray(
      model.model.perLayerModelProjection.weight,
    );
  }
  if (
    model.model.perLayerProjectionNorm !== null &&
    model.model.perLayerProjectionNorm.weight !== null
  ) {
    tensors["model.per_layer_projection_norm.weight"] = retainArray(
      model.model.perLayerProjectionNorm.weight,
    );
  }
  if (model.lmHead !== null) {
    tensors["lm_head.weight"] = retainArray(model.lmHead.weight);
  }
}

function addGemma4LayerCheckpointTensors(
  tensors: Record<string, MxArray>,
  layerIndex: number,
  layer: Gemma4TextCausalLM["model"]["layers"][number],
): void {
  const prefix = `model.layers.${layerIndex}`;
  tensors[`${prefix}.input_layernorm.weight`] = retainArray(
    expectTensor(layer.inputLayerNorm.weight, "Expected Gemma 4 input layer norm weight."),
  );
  tensors[`${prefix}.post_attention_layernorm.weight`] = retainArray(
    expectTensor(
      layer.postAttentionLayerNorm.weight,
      "Expected Gemma 4 post-attention layer norm weight.",
    ),
  );
  tensors[`${prefix}.pre_feedforward_layernorm.weight`] = retainArray(
    expectTensor(
      layer.preFeedforwardLayerNorm.weight,
      "Expected Gemma 4 pre-feedforward layer norm weight.",
    ),
  );
  tensors[`${prefix}.post_feedforward_layernorm.weight`] = retainArray(
    expectTensor(
      layer.postFeedforwardLayerNorm.weight,
      "Expected Gemma 4 post-feedforward layer norm weight.",
    ),
  );
  tensors[`${prefix}.layer_scalar`] = retainArray(layer.layerScalar);
  tensors[`${prefix}.self_attn.q_proj.weight`] = retainArray(
    layer.selfAttention.qProjection.weight,
  );
  tensors[`${prefix}.self_attn.k_proj.weight`] = retainArray(
    layer.selfAttention.kProjection.weight,
  );
  if (layer.selfAttention.vProjection !== null) {
    tensors[`${prefix}.self_attn.v_proj.weight`] = retainArray(
      layer.selfAttention.vProjection.weight,
    );
  }
  tensors[`${prefix}.self_attn.o_proj.weight`] = retainArray(
    layer.selfAttention.outputProjection.weight,
  );
  if (layer.selfAttention.qNorm.weight !== null) {
    tensors[`${prefix}.self_attn.q_norm.weight`] = retainArray(layer.selfAttention.qNorm.weight);
  }
  if (layer.selfAttention.kNorm.weight !== null) {
    tensors[`${prefix}.self_attn.k_norm.weight`] = retainArray(layer.selfAttention.kNorm.weight);
  }
  if (!(layer.mlp instanceof Gemma4TextMLP)) {
    throw new Error("Expected the Gemma 4 decoder block to own a Gemma4TextMLP.");
  }
  tensors[`${prefix}.mlp.gate_proj.weight`] = retainArray(layer.mlp.gateProjection.weight);
  tensors[`${prefix}.mlp.up_proj.weight`] = retainArray(layer.mlp.upProjection.weight);
  tensors[`${prefix}.mlp.down_proj.weight`] = retainArray(layer.mlp.downProjection.weight);
  if (layer.perLayerInputGate instanceof Linear) {
    tensors[`${prefix}.per_layer_input_gate.weight`] = retainArray(layer.perLayerInputGate.weight);
  }
  if (layer.perLayerProjection instanceof Linear) {
    tensors[`${prefix}.per_layer_projection.weight`] = retainArray(layer.perLayerProjection.weight);
  }
  if (layer.postPerLayerInputNorm !== null && layer.postPerLayerInputNorm.weight !== null) {
    tensors[`${prefix}.post_per_layer_input_norm.weight`] = retainArray(
      layer.postPerLayerInputNorm.weight,
    );
  }
}

async function createTinySnapshot(family: SupportedModelFamily): Promise<{
  directory: string;
  model: LlamaLikeCausalLM;
}> {
  const directory = createTempDir(`mlxts-transformers-${family}-`);
  const rawConfig = rawConfigForFamily(family);
  const registration = resolveFamily(rawConfig.model_type as string);
  const model = registration.createModel(registration.parseConfig(rawConfig));
  if (!(model instanceof LlamaLikeCausalLM)) {
    throw new Error("Expected a LlamaLikeCausalLM for the supported Phase 7 families.");
  }

  const tensors = checkpointTensors(model);
  const tokenizer = tokenizerFixture();

  try {
    await Bun.write(join(directory, "config.json"), `${JSON.stringify(rawConfig, null, 2)}\n`);
    await Bun.write(
      join(directory, "tokenizer.json"),
      `${JSON.stringify(tokenizer.tokenizerJson, null, 2)}\n`,
    );
    await Bun.write(
      join(directory, "tokenizer_config.json"),
      `${JSON.stringify(tokenizer.tokenizerConfig, null, 2)}\n`,
    );
    await Bun.write(
      join(directory, "special_tokens_map.json"),
      `${JSON.stringify(tokenizer.specialTokensMap, null, 2)}\n`,
    );
    await saveSafetensors(tensors, join(directory, "model.safetensors"));
  } finally {
    for (const tensor of Object.values(tensors)) {
      tensor.free();
    }
  }

  return { directory, model };
}

async function createTinyGemma3TextSnapshot(): Promise<{
  directory: string;
  model: Gemma3TextCausalLM;
}> {
  const directory = createTempDir("mlxts-transformers-gemma3-");
  const rawConfig = rawConfigForGemma3Text();
  const registration = resolveFamily("gemma3_text");
  const model = registration.createModel(registration.parseConfig(rawConfig));
  if (!(model instanceof Gemma3TextCausalLM)) {
    throw new Error("Expected a Gemma3TextCausalLM for the supported Gemma 3 text snapshot.");
  }

  const tensors = checkpointTensorsForGemma3Text(model);
  const tokenizer = tokenizerFixture();

  try {
    await Bun.write(join(directory, "config.json"), `${JSON.stringify(rawConfig, null, 2)}\n`);
    await Bun.write(
      join(directory, "tokenizer.json"),
      `${JSON.stringify(tokenizer.tokenizerJson, null, 2)}\n`,
    );
    await Bun.write(
      join(directory, "tokenizer_config.json"),
      `${JSON.stringify(tokenizer.tokenizerConfig, null, 2)}\n`,
    );
    await Bun.write(
      join(directory, "special_tokens_map.json"),
      `${JSON.stringify(tokenizer.specialTokensMap, null, 2)}\n`,
    );
    await saveSafetensors(tensors, join(directory, "model.safetensors"));
  } finally {
    for (const tensor of Object.values(tensors)) {
      tensor.free();
    }
  }

  return { directory, model };
}

async function createTinyMistral3TextSnapshot(): Promise<{
  directory: string;
  model: LlamaLikeCausalLM;
}> {
  const directory = createTempDir("mlxts-transformers-mistral3-");
  const rawConfig = rawConfigForMistral3TextWrapper();
  const registration = resolveFamily("mistral3");
  const model = registration.createModel(registration.parseConfig(rawConfig));
  if (!(model instanceof LlamaLikeCausalLM)) {
    throw new Error("Expected a LlamaLikeCausalLM for the supported Mistral 3 text snapshot.");
  }

  const baseTensors = checkpointTensors(model);
  const tokenizer = tekkenTokenizerFixture();
  const tensors = Object.fromEntries(
    Object.entries(baseTensors).map(([name, tensor]) => [`language_model.${name}`, tensor]),
  );

  try {
    await Bun.write(join(directory, "config.json"), `${JSON.stringify(rawConfig, null, 2)}\n`);
    await Bun.write(
      join(directory, "tekken.json"),
      `${JSON.stringify(tokenizer.tekkenJson, null, 2)}\n`,
    );
    await Bun.write(
      join(directory, "generation_config.json"),
      `${JSON.stringify(tokenizer.generationConfig, null, 2)}\n`,
    );
    await saveSafetensors(tensors, join(directory, "model.safetensors"));
  } finally {
    for (const tensor of Object.values(baseTensors)) {
      tensor.free();
    }
  }

  return { directory, model };
}

async function createTinyGemma4TextSnapshot(): Promise<{
  directory: string;
  model: Gemma4TextCausalLM;
}> {
  const directory = createTempDir("mlxts-transformers-gemma4-text-");
  const rawConfig = rawConfigForGemma4Text();
  const registration = resolveFamily("gemma4_text");
  const model = registration.createModel(registration.parseConfig(rawConfig));
  if (!(model instanceof Gemma4TextCausalLM)) {
    throw new Error("Expected a Gemma4TextCausalLM for the supported Gemma 4 text snapshot.");
  }

  const tensors = checkpointTensorsForGemma4Text(model);
  const tokenizer = tokenizerFixture();

  try {
    await Bun.write(join(directory, "config.json"), `${JSON.stringify(rawConfig, null, 2)}\n`);
    await Bun.write(
      join(directory, "tokenizer.json"),
      `${JSON.stringify(tokenizer.tokenizerJson, null, 2)}\n`,
    );
    await Bun.write(
      join(directory, "tokenizer_config.json"),
      `${JSON.stringify(tokenizer.tokenizerConfig, null, 2)}\n`,
    );
    await Bun.write(
      join(directory, "special_tokens_map.json"),
      `${JSON.stringify(tokenizer.specialTokensMap, null, 2)}\n`,
    );
    await saveSafetensors(tensors, join(directory, "model.safetensors"));
  } finally {
    for (const tensor of Object.values(tensors)) {
      tensor.free();
    }
  }

  return { directory, model };
}

async function createTinyGemma4Snapshot(): Promise<{
  directory: string;
  model: Gemma4TextCausalLM;
}> {
  const directory = createTempDir("mlxts-transformers-gemma4-");
  const rawConfig = rawConfigForGemma4Wrapper();
  const registration = resolveFamily("gemma4");
  const model = registration.createModel(registration.parseConfig(rawConfig));
  if (!(model instanceof Gemma4TextCausalLM)) {
    throw new Error("Expected a Gemma4TextCausalLM for the supported Gemma 4 snapshot.");
  }

  const baseTensors = checkpointTensorsForGemma4Text(model);
  const tokenizer = tokenizerFixture();
  const tensors: Record<string, MxArray> = {
    "model.embed_vision.embedding_projection.weight": retainArray(
      expectTensor(model.model.norm.weight, "Expected a scaled Gemma 4 model norm weight."),
    ),
  };
  for (const [name, tensor] of Object.entries(baseTensors)) {
    tensors[
      name.startsWith("model.")
        ? `model.language_model.${name.slice("model.".length)}`
        : `model.language_model.${name}`
    ] = tensor;
  }

  try {
    await Bun.write(join(directory, "config.json"), `${JSON.stringify(rawConfig, null, 2)}\n`);
    await Bun.write(
      join(directory, "tokenizer.json"),
      `${JSON.stringify(tokenizer.tokenizerJson, null, 2)}\n`,
    );
    await Bun.write(
      join(directory, "tokenizer_config.json"),
      `${JSON.stringify(tokenizer.tokenizerConfig, null, 2)}\n`,
    );
    await Bun.write(
      join(directory, "special_tokens_map.json"),
      `${JSON.stringify(tokenizer.specialTokensMap, null, 2)}\n`,
    );
    await saveSafetensors(tensors, join(directory, "model.safetensors"));
  } finally {
    const uniqueTensors = new Set(Object.values(tensors));
    for (const tensor of uniqueTensors) {
      tensor.free();
    }
  }

  return { directory, model };
}

async function rewriteCheckpoint(
  directory: string,
  model: LlamaLikeCausalLM,
  transform: (tensors: Record<string, MxArray>) => Record<string, MxArray>,
): Promise<void> {
  const original = checkpointTensors(model);
  const rewritten = transform(original);

  try {
    await saveSafetensors(rewritten, join(directory, "model.safetensors"));
  } finally {
    const tensors = new Set<MxArray>([...Object.values(original), ...Object.values(rewritten)]);
    for (const tensor of tensors) {
      tensor.free();
    }
  }
}

describe("pretrained loading", () => {
  test.each([
    "llama",
    "mistral",
    "gemma",
    "phi",
  ] as const)("loadCausalLM round-trips a tiny %s snapshot", async (family) => {
    const { directory, model: originalModel } = await createTinySnapshot(family);
    using loadedModel = await loadCausalLM(directory);
    using inputIds = array([[0, 1, 2]], "int32");
    using expectedLogits = originalModel.forward(inputIds);
    using actualLogits = loadedModel.forward(inputIds);

    mxEval(expectedLogits, actualLogits);

    expect(loadedModel.family).toBe(family);
    expect(loadedModel.layerCount).toBe(originalModel.layerCount);
    expectCloseLists(actualLogits.toList(), expectedLogits.toList());

    using cache = loadedModel.createCache();
    if (family === "mistral" || family === "phi") {
      expect(cache).toBeInstanceOf(SlidingWindowKVCache);
    } else {
      expect(cache).toBeInstanceOf(KVCache);
    }

    originalModel[Symbol.dispose]();
  });

  test("loadCausalLM round-trips a tiny gemma3_text snapshot", async () => {
    const { directory, model: originalModel } = await createTinyGemma3TextSnapshot();
    using loadedModel = await loadCausalLM(directory);
    using inputIds = array([[0, 1, 2]], "int32");
    using expectedLogits = originalModel.forward(inputIds);
    using actualLogits = loadedModel.forward(inputIds);

    mxEval(expectedLogits, actualLogits);

    expect(loadedModel.family).toBe("gemma");
    expect(loadedModel.layerCount).toBe(originalModel.layerCount);
    expectCloseLists(actualLogits.toList(), expectedLogits.toList());

    using cache = loadedModel.createCache();
    expect(cache).toBeInstanceOf(LayerPatternKVCache);

    originalModel[Symbol.dispose]();
  });

  test("loadCausalLM round-trips a tiny mistral3 text-decoder snapshot", async () => {
    const { directory, model: originalModel } = await createTinyMistral3TextSnapshot();
    using loadedModel = await loadCausalLM(directory);
    using inputIds = array([[0, 1, 2]], "int32");
    using expectedLogits = originalModel.forward(inputIds);
    using actualLogits = loadedModel.forward(inputIds);

    mxEval(expectedLogits, actualLogits);

    expect(loadedModel.family).toBe("mistral");
    expect(loadedModel.layerCount).toBe(originalModel.layerCount);
    expectCloseLists(actualLogits.toList(), expectedLogits.toList());

    using cache = loadedModel.createCache();
    expect(cache).toBeInstanceOf(SlidingWindowKVCache);

    originalModel[Symbol.dispose]();
  });

  test("loadCausalLM round-trips a tiny gemma4_text snapshot", async () => {
    const { directory, model: originalModel } = await createTinyGemma4TextSnapshot();
    using loadedModel = await loadCausalLM(directory);
    using inputIds = array([[0, 1, 2]], "int32");
    using expectedLogits = originalModel.forward(inputIds);
    using actualLogits = loadedModel.forward(inputIds);

    mxEval(expectedLogits, actualLogits);

    expect(loadedModel.family).toBe("gemma");
    expect(loadedModel.layerCount).toBe(originalModel.layerCount);
    expectCloseLists(actualLogits.toList(), expectedLogits.toList());

    using cache = loadedModel.createCache();
    expect(cache).toBeInstanceOf(LayerPatternKVCache);

    originalModel[Symbol.dispose]();
  });

  test("loadCausalLM round-trips a tiny top-level gemma4 snapshot", async () => {
    const { directory, model: originalModel } = await createTinyGemma4Snapshot();
    using loadedModel = await loadCausalLM(directory);
    using inputIds = array([[0, 1, 2]], "int32");
    using expectedLogits = originalModel.forward(inputIds);
    using actualLogits = loadedModel.forward(inputIds);

    mxEval(expectedLogits, actualLogits);

    expect(loadedModel.family).toBe("gemma");
    expect(loadedModel.layerCount).toBe(originalModel.layerCount);
    expectCloseLists(actualLogits.toList(), expectedLogits.toList());

    using cache = loadedModel.createCache();
    expect(cache).toBeInstanceOf(LayerPatternKVCache);

    originalModel[Symbol.dispose]();
  });

  test("loadPretrainedTokenizer and AutoTokenizer read the snapshot tokenizer", async () => {
    const { directory, model } = await createTinySnapshot("llama");
    const tokenizer = await loadPretrainedTokenizer(directory);
    const autoTokenizer = await AutoTokenizer.fromPretrained(directory);

    expect(tokenizer.encode("Hi!")).toEqual([4, 0, 1, 2]);
    expect(autoTokenizer.decode([4, 0, 1, 2], { skipSpecialTokens: true })).toBe("Hi!");

    model[Symbol.dispose]();
  });

  test("loadPretrainedTokenizer reads Tekken snapshots", async () => {
    const { directory, model } = await createTinyMistral3TextSnapshot();
    const tokenizer = await loadPretrainedTokenizer(directory);

    expect(tokenizer.encode("Hi!")).toEqual([1, 4, 5, 6]);
    expect(tokenizer.decode([1, 4, 5, 6, 2], { skipSpecialTokens: true })).toBe("Hi!");

    model[Symbol.dispose]();
  });

  test("AutoModel loads the same snapshot contract as loadCausalLM", async () => {
    const { directory, model } = await createTinySnapshot("llama");
    using loadedModel = await AutoModel.fromPretrained(directory);
    using inputIds = array([[0, 1, 2]], "int32");
    using expectedLogits = model.forward(inputIds);
    using actualLogits = loadedModel.forward(inputIds);

    mxEval(expectedLogits, actualLogits);
    expectCloseLists(actualLogits.toList(), expectedLogits.toList());

    model[Symbol.dispose]();
  });

  test("loadCausalLM validates the config shape before loading weights", async () => {
    const { directory, model } = await createTinySnapshot("llama");
    await Bun.write(
      join(directory, "config.json"),
      `${JSON.stringify({ hidden_size: 8 }, null, 2)}\n`,
    );

    await expect(loadCausalLM(directory)).rejects.toThrow(ConfigParseError);
    model[Symbol.dispose]();
  });

  test("loadCausalLM reports missing required weights", async () => {
    const { directory, model } = await createTinySnapshot("llama");
    await rewriteCheckpoint(directory, model, (tensors) => {
      const { "model.norm.weight": _, ...rest } = tensors;
      if (_ !== undefined) {
        _.free();
      }
      return rest;
    });

    await expect(loadCausalLM(directory)).rejects.toThrow(MissingWeightsError);
    model[Symbol.dispose]();
  });

  test("loadCausalLM warns about unexpected unmapped weights by default and can fail strictly", async () => {
    const { directory, model } = await createTinySnapshot("llama");
    await rewriteCheckpoint(directory, model, (tensors) => ({
      ...tensors,
      "model.extra.weight": retainArray(model.model.norm.weight),
    }));

    const warn = console.warn;
    const warnings: string[] = [];
    console.warn = (message?: unknown) => {
      warnings.push(String(message));
    };

    try {
      using loadedModel = await loadCausalLM(directory);
      expect(loadedModel.family).toBe("llama");
      expect(warnings.some((message) => message.includes("unexpected unmapped weights"))).toBe(
        true,
      );
      await expect(loadCausalLM(directory, { strictUnexpectedWeights: true })).rejects.toThrow(
        "unexpected unmapped weights",
      );
    } finally {
      console.warn = warn;
      model[Symbol.dispose]();
    }
  });

  test("loadPretrainedTokenizer rejects snapshots without tokenizer artifacts", async () => {
    const { directory, model } = await createTinySnapshot("llama");
    rmSync(join(directory, "tokenizer.json"));
    rmSync(join(directory, "tokenizer_config.json"));
    rmSync(join(directory, "special_tokens_map.json"));

    await expect(loadPretrainedTokenizer(directory)).rejects.toThrow(
      "snapshot does not include tokenizer.json, tekken.json, or tokenizer.model",
    );
    model[Symbol.dispose]();
  });
});

describe("generation helpers", () => {
  test("generateTokens matches between cached and uncached decoding", async () => {
    const { directory, model: originalModel } = await createTinySnapshot("llama");
    using loadedModel = await loadCausalLM(directory);
    const tokenizer = await loadPretrainedTokenizer(directory);
    const promptTokenIds = tokenizer.encode("Hi", { addSpecialTokens: true });

    const cached = generateTokens(loadedModel, promptTokenIds, {
      maxTokens: 4,
      temperature: 0,
      useCache: true,
    });
    const uncached = generateTokens(loadedModel, promptTokenIds, {
      maxTokens: 4,
      temperature: 0,
      useCache: false,
    });

    expect(cached).toEqual(uncached);
    expect(
      typeof generateText(loadedModel, tokenizer, "Hi", { maxTokens: 2, temperature: 0 }),
    ).toBe("string");

    originalModel[Symbol.dispose]();
  });

  test("cached and uncached continuation logits match for the same prefix", async () => {
    const { directory, model } = await createTinySnapshot("llama");
    using loadedModel = await loadCausalLM(directory);
    using fullInput = array([[0, 1, 2]], "int32");
    using prefixInput = array([[0, 1]], "int32");
    using nextTokenInput = array([[2]], "int32");
    using fullLogits = loadedModel.forward(fullInput);
    using cache = loadedModel.createCache();
    using prefixLogits = loadedModel.forward(prefixInput, { cache });
    using cachedLogits = loadedModel.forward(nextTokenInput, { cache });

    mxEval(fullLogits, prefixLogits, cachedLogits);

    const fullContinuation = (fullLogits.toList() as number[][][])[0]?.[2];
    const cachedContinuation = (cachedLogits.toList() as number[][][])[0]?.[0];
    expectCloseLists(cachedContinuation, fullContinuation);

    model[Symbol.dispose]();
  });
});
