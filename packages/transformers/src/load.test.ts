import { afterEach, describe, expect, test } from "bun:test";
import {
  array,
  inspectSafetensors,
  MxArray,
  mxEval,
  type QuantizationMode,
  quantize,
  retainArray,
  saveSafetensors,
  slice,
  transpose,
} from "@mlxts/core";
import { Linear, QuantizedEmbedding, QuantizedLinear } from "@mlxts/nn";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { AutoModel, AutoTokenizer } from "./auto";
import { Gemma3TextCausalLM } from "./families/gemma3/model";
import { Gemma4TextMLP } from "./families/gemma4/mlp";
import { Gemma4TextCausalLM } from "./families/gemma4/model";
import { LlamaLikeMLP } from "./families/llama-like/mlp";
import { LlamaLikeCausalLM } from "./families/llama-like/model";
import { Qwen3_5TextCache } from "./families/qwen3_5/cache";
import { Qwen3_5ForConditionalGeneration } from "./families/qwen3_5/conditional";
import { qwen3_5ConditionalFamily } from "./families/qwen3_5/config";
import {
  loadQwen3_5ForConditionalGeneration,
  shouldLoadQwen3_5ForConditionalGeneration,
} from "./families/qwen3_5/load";
import { Qwen3_5TextMLP, Qwen3_5TextMoE } from "./families/qwen3_5/mlp";
import { Qwen3_5TextCausalLM, type Qwen3_5TextModel } from "./families/qwen3_5/model";
import { generateText, generateTextStream, generateTokens } from "./generation";
import { KVCache, LayerPatternKVCache, SlidingWindowKVCache } from "./infrastructure/cache";
import { PackedSwitchGLUExperts, SwitchGLUExperts } from "./infrastructure/moe";
import { loadCausalLM, loadPretrainedTokenizer } from "./load";
import { quantizePretrainedSnapshot } from "./quantize";
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

async function writeTokenizerFixture(directory: string): Promise<void> {
  const tokenizer = tokenizerFixture();
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

function castCheckpointTensors(
  tensors: Record<string, MxArray>,
  dtype: "float16" | "bfloat16" | "float32",
): Record<string, MxArray> {
  const converted: Record<string, MxArray> = {};
  for (const [name, tensor] of Object.entries(tensors)) {
    converted[name] = tensor.asType(dtype);
  }
  return converted;
}

function quantizeCheckpointTensors(
  tensors: Record<string, MxArray>,
  targets: readonly string[],
  options: { groupSize?: number; bits?: number; mode?: QuantizationMode } = {},
): Record<string, MxArray> {
  const targetWeights = new Set(targets.map((target) => `${target}.weight`));
  const rewritten: Record<string, MxArray> = {};

  for (const [name, tensor] of Object.entries(tensors)) {
    if (!targetWeights.has(name)) {
      rewritten[name] = retainArray(tensor);
      continue;
    }

    const quantizedTensor = quantize(tensor, {
      groupSize: options.groupSize ?? 64,
      bits: options.bits ?? 4,
      mode: options.mode ?? "affine",
    });
    rewritten[name] = quantizedTensor.weight;
    rewritten[`${name.slice(0, -".weight".length)}.scales`] = quantizedTensor.scales;
    if (quantizedTensor.biases !== undefined) {
      rewritten[`${name.slice(0, -".weight".length)}.biases`] = quantizedTensor.biases;
    }
  }

  return rewritten;
}

function splitExpertGateUpTensor(tensor: MxArray, context: string): { gate: MxArray; up: MxArray } {
  const [expertCount, gateUpSize, hiddenSize] = tensor.shape;
  if (
    tensor.shape.length !== 3 ||
    expertCount === undefined ||
    gateUpSize === undefined ||
    hiddenSize === undefined ||
    gateUpSize % 2 !== 0
  ) {
    throw new Error(`${context}: expected [experts, 2 * intermediate, hidden] gate-up tensor.`);
  }

  const intermediateSize = gateUpSize / 2;
  return {
    gate: slice(tensor, [0, 0, 0], [expertCount, intermediateSize, hiddenSize]),
    up: slice(tensor, [0, intermediateSize, 0], [expertCount, gateUpSize, hiddenSize]),
  };
}

function qwenSplitMoeCheckpointTensors(tensors: Record<string, MxArray>): Record<string, MxArray> {
  const rewritten: Record<string, MxArray> = {};
  for (const [name, tensor] of Object.entries(tensors)) {
    if (name.endsWith(".mlp.experts.gate_up_proj")) {
      const base = name.slice(0, -".experts.gate_up_proj".length);
      const splitWeights = splitExpertGateUpTensor(tensor, name);
      rewritten[`${base}.switch_mlp.gate_proj.weight`] = splitWeights.gate;
      rewritten[`${base}.switch_mlp.up_proj.weight`] = splitWeights.up;
      continue;
    }
    if (name.endsWith(".mlp.experts.down_proj")) {
      const base = name.slice(0, -".experts.down_proj".length);
      rewritten[`${base}.switch_mlp.down_proj.weight`] = retainArray(tensor);
      continue;
    }
    rewritten[name] = retainArray(tensor);
  }
  return rewritten;
}

function gemma4SplitMoeCheckpointTensors(
  tensors: Record<string, MxArray>,
): Record<string, MxArray> {
  const rewritten: Record<string, MxArray> = {};
  for (const [name, tensor] of Object.entries(tensors)) {
    if (name.endsWith(".experts.gate_up_proj")) {
      const base = name.slice(0, -".gate_up_proj".length);
      const splitWeights = splitExpertGateUpTensor(tensor, name);
      rewritten[`${base}.switch_glu.gate_proj.weight`] = splitWeights.gate;
      rewritten[`${base}.switch_glu.up_proj.weight`] = splitWeights.up;
      continue;
    }
    if (name.endsWith(".experts.down_proj")) {
      const base = name.slice(0, -".down_proj".length);
      rewritten[`${base}.switch_glu.down_proj.weight`] = retainArray(tensor);
      continue;
    }
    rewritten[name] = retainArray(tensor);
  }
  return rewritten;
}

function freeTensorRecords(...records: Array<Record<string, MxArray>>): void {
  for (const record of records) {
    for (const tensor of Object.values(record)) {
      tensor.free();
    }
  }
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

function rawConfigForGemma4MoeText(): Record<string, unknown> {
  return {
    ...rawConfigForGemma4Text(),
    vocab_size: 7,
    vocab_size_per_layer_input: 7,
    hidden_size: 64,
    intermediate_size: 128,
    num_hidden_layers: 1,
    num_attention_heads: 4,
    num_key_value_heads: 2,
    num_global_key_value_heads: 2,
    head_dim: 16,
    global_head_dim: 16,
    layer_types: ["full_attention"],
    hidden_size_per_layer_input: 0,
    use_double_wide_mlp: false,
    num_kv_shared_layers: 0,
    enable_moe_block: true,
    moe_intermediate_size: 64,
    num_experts: 2,
    top_k_experts: 1,
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

function rawConfigForQwen3_5Text(): Record<string, unknown> {
  return {
    model_type: "qwen3_5_text",
    vocab_size: 5,
    hidden_size: 8,
    intermediate_size: 16,
    num_hidden_layers: 2,
    num_attention_heads: 2,
    num_key_value_heads: 1,
    head_dim: 4,
    hidden_act: "silu",
    max_position_embeddings: 64,
    rms_norm_eps: 1e-6,
    attention_bias: false,
    tie_word_embeddings: false,
    full_attention_interval: 2,
    layer_types: ["linear_attention", "full_attention"],
    linear_num_value_heads: 2,
    linear_num_key_heads: 1,
    linear_key_head_dim: 4,
    linear_value_head_dim: 4,
    linear_conv_kernel_dim: 2,
    rope_parameters: {
      rope_type: "default",
      rope_theta: 10000,
      partial_rotary_factor: 1,
      mrope_section: [1, 1, 0],
      mrope_interleaved: true,
    },
  };
}

function rawConfigForQwen3_5MoeText(): Record<string, unknown> {
  return {
    ...rawConfigForQwen3_5Text(),
    model_type: "qwen3_5_moe_text",
    vocab_size: 7,
    hidden_size: 64,
    intermediate_size: 64,
    shared_expert_intermediate_size: 64,
    moe_intermediate_size: 64,
    num_experts: 2,
    num_experts_per_tok: 1,
    num_hidden_layers: 1,
    num_attention_heads: 4,
    num_key_value_heads: 2,
    head_dim: 16,
    full_attention_interval: 1,
    layer_types: ["full_attention"],
  };
}

function rawConfigForQwen3_5Wrapper(): Record<string, unknown> {
  return {
    model_type: "qwen3_5",
    architectures: ["Qwen3_5ForConditionalGeneration"],
    text_config: rawConfigForQwen3_5Text(),
    vision_config: {
      model_type: "qwen3_5",
      depth: 1,
      hidden_size: 8,
      hidden_act: "gelu_pytorch_tanh",
      intermediate_size: 16,
      num_heads: 2,
      in_channels: 3,
      patch_size: 2,
      spatial_merge_size: 1,
      temporal_patch_size: 1,
      out_hidden_size: 8,
      num_position_embeddings: 16,
      deepstack_visual_indexes: [],
    },
    image_token_id: 4,
    video_token_id: 3,
    vision_start_token_id: 2,
    vision_end_token_id: 1,
    tie_word_embeddings: false,
    language_model_only: false,
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
  if (layer.router !== null && layer.experts !== null) {
    tensors[`${prefix}.router.proj.weight`] = retainArray(layer.router.proj.weight);
    tensors[`${prefix}.router.scale`] = retainArray(layer.router.scale);
    tensors[`${prefix}.router.per_expert_scale`] = retainArray(layer.router.perExpertScale);
    if (!(layer.experts instanceof PackedSwitchGLUExperts)) {
      throw new Error(`Expected Gemma 4 layer ${layerIndex} to own packed experts.`);
    }
    tensors[`${prefix}.experts.gate_up_proj`] = retainArray(layer.experts.gateUpProjection);
    tensors[`${prefix}.experts.down_proj`] = retainArray(layer.experts.downProjection);
    if (
      layer.preFeedforwardLayerNorm2 === null ||
      layer.postFeedforwardLayerNorm1 === null ||
      layer.postFeedforwardLayerNorm2 === null
    ) {
      throw new Error(`Expected Gemma 4 MoE layer ${layerIndex} to own MoE norms.`);
    }
    tensors[`${prefix}.pre_feedforward_layernorm_2.weight`] = retainArray(
      expectTensor(
        layer.preFeedforwardLayerNorm2.weight,
        "Expected Gemma 4 second pre-feedforward norm weight.",
      ),
    );
    tensors[`${prefix}.post_feedforward_layernorm_1.weight`] = retainArray(
      expectTensor(
        layer.postFeedforwardLayerNorm1.weight,
        "Expected Gemma 4 first post-feedforward norm weight.",
      ),
    );
    tensors[`${prefix}.post_feedforward_layernorm_2.weight`] = retainArray(
      expectTensor(
        layer.postFeedforwardLayerNorm2.weight,
        "Expected Gemma 4 second post-feedforward norm weight.",
      ),
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

function checkpointTensorsForQwen3_5TextParts(
  textModel: Qwen3_5TextModel,
  lmHead: Linear | null,
  rootPrefix: string,
): Record<string, MxArray> {
  const tensors: Record<string, MxArray> = {
    [`${rootPrefix}.embed_tokens.weight`]: retainArray(textModel.embedTokens.weight),
    [`${rootPrefix}.norm.weight`]: retainArray(textModel.norm.weight),
  };
  if (lmHead !== null) {
    tensors["lm_head.weight"] = retainArray(lmHead.weight);
  }

  for (let layerIndex = 0; layerIndex < textModel.layers.length; layerIndex += 1) {
    const layer = textModel.layers[layerIndex];
    if (layer === undefined) {
      throw new Error(`Expected Qwen 3.5 layer ${layerIndex} to exist.`);
    }
    const prefix = `${rootPrefix}.layers.${layerIndex}`;
    tensors[`${prefix}.input_layernorm.weight`] = retainArray(layer.inputLayerNorm.weight);
    tensors[`${prefix}.post_attention_layernorm.weight`] = retainArray(
      layer.postAttentionLayerNorm.weight,
    );
    if (layer.mlp instanceof Qwen3_5TextMLP) {
      tensors[`${prefix}.mlp.gate_proj.weight`] = retainArray(layer.mlp.gateProjection.weight);
      tensors[`${prefix}.mlp.up_proj.weight`] = retainArray(layer.mlp.upProjection.weight);
      tensors[`${prefix}.mlp.down_proj.weight`] = retainArray(layer.mlp.downProjection.weight);
    } else if (layer.mlp instanceof Qwen3_5TextMoE) {
      tensors[`${prefix}.mlp.gate.weight`] = retainArray(layer.mlp.gate.weight);
      if (!(layer.mlp.experts instanceof PackedSwitchGLUExperts)) {
        throw new Error(`Expected Qwen 3.5 MoE layer ${layerIndex} to own packed experts.`);
      }
      tensors[`${prefix}.mlp.experts.gate_up_proj`] = retainArray(
        layer.mlp.experts.gateUpProjection,
      );
      tensors[`${prefix}.mlp.experts.down_proj`] = retainArray(layer.mlp.experts.downProjection);
      tensors[`${prefix}.mlp.shared_expert.gate_proj.weight`] = retainArray(
        layer.mlp.sharedExpert.gateProjection.weight,
      );
      tensors[`${prefix}.mlp.shared_expert.up_proj.weight`] = retainArray(
        layer.mlp.sharedExpert.upProjection.weight,
      );
      tensors[`${prefix}.mlp.shared_expert.down_proj.weight`] = retainArray(
        layer.mlp.sharedExpert.downProjection.weight,
      );
      tensors[`${prefix}.mlp.shared_expert_gate.weight`] = retainArray(
        layer.mlp.sharedExpertGate.weight,
      );
    }

    if (layer.linearAttention !== null) {
      tensors[`${prefix}.linear_attn.in_proj_qkv.weight`] = retainArray(
        layer.linearAttention.inProjectionQkv.weight,
      );
      tensors[`${prefix}.linear_attn.in_proj_z.weight`] = retainArray(
        layer.linearAttention.inProjectionZ.weight,
      );
      tensors[`${prefix}.linear_attn.in_proj_b.weight`] = retainArray(
        layer.linearAttention.inProjectionB.weight,
      );
      tensors[`${prefix}.linear_attn.in_proj_a.weight`] = retainArray(
        layer.linearAttention.inProjectionA.weight,
      );
      tensors[`${prefix}.linear_attn.conv1d.weight`] = retainArray(
        layer.linearAttention.conv1d.weight,
      );
      tensors[`${prefix}.linear_attn.dt_bias`] = retainArray(layer.linearAttention.dtBias);
      tensors[`${prefix}.linear_attn.A_log`] = retainArray(layer.linearAttention.aLog);
      tensors[`${prefix}.linear_attn.norm.weight`] = retainArray(layer.linearAttention.norm.weight);
      tensors[`${prefix}.linear_attn.out_proj.weight`] = retainArray(
        layer.linearAttention.outProjection.weight,
      );
    }

    if (layer.selfAttention !== null) {
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
    }
  }

  return tensors;
}

function checkpointTensorsForQwen3_5Text(model: Qwen3_5TextCausalLM): Record<string, MxArray> {
  return checkpointTensorsForQwen3_5TextParts(model.model, model.lmHead, "model");
}

function rawQwen3_5TextTensors(tensors: Record<string, MxArray>): Record<string, MxArray> {
  const raw: Record<string, MxArray> = {};

  for (const [name, tensor] of Object.entries(tensors)) {
    if (name.endsWith(".linear_attn.conv1d.weight")) {
      raw[name] = transpose(tensor, [0, 2, 1]);
      continue;
    }
    if (
      name === "model.norm.weight" ||
      name.endsWith(".input_layernorm.weight") ||
      name.endsWith(".post_attention_layernorm.weight") ||
      name.endsWith(".self_attn.q_norm.weight") ||
      name.endsWith(".self_attn.k_norm.weight")
    ) {
      raw[name] = MxArray.fromData(
        flattenNumbers(tensor.toList()).map((value) => value - 1),
        [...tensor.shape],
        tensor.dtype,
      );
      continue;
    }
    raw[name] = retainArray(tensor);
  }

  return raw;
}

function convertedQwen3_5TextTensors(tensors: Record<string, MxArray>): Record<string, MxArray> {
  const converted: Record<string, MxArray> = {};

  for (const [name, tensor] of Object.entries(tensors)) {
    converted[name] = retainArray(tensor);
  }

  return converted;
}

function wrapperQwen3_5TextTensors(tensors: Record<string, MxArray>): Record<string, MxArray> {
  const converted: Record<string, MxArray> = {};
  for (const [name, tensor] of Object.entries(tensors)) {
    converted[`language_model.${name}`] = retainArray(tensor);
  }
  converted["model.visual.patch_embed.proj.weight"] = MxArray.fromData([1, 2, 3, 4], [1, 4]);
  return converted;
}

function wrapperLanguageModelTensors(tensors: Record<string, MxArray>): Record<string, MxArray> {
  const converted: Record<string, MxArray> = {};
  for (const [name, tensor] of Object.entries(tensors)) {
    converted[`language_model.${name}`] = retainArray(tensor);
  }
  return converted;
}

function checkpointTensorsForQwen3_5Conditional(
  model: Qwen3_5ForConditionalGeneration,
): Record<string, MxArray> {
  const tensors = checkpointTensorsForQwen3_5TextParts(
    model.model.languageModel,
    model.lmHead,
    "language_model.model",
  );
  const visual = model.model.visual;
  tensors["model.visual.patch_embed.proj.weight"] = retainArray(visual.patchEmbed.weight);
  tensors["model.visual.patch_embed.proj.bias"] = retainArray(visual.patchEmbed.bias);
  tensors["model.visual.pos_embed.weight"] = retainArray(visual.posEmbed.weight);
  tensors["model.visual.merger.norm.weight"] = retainArray(visual.merger.norm.weight);
  tensors["model.visual.merger.norm.bias"] = retainArray(visual.merger.norm.bias);
  tensors["model.visual.merger.linear_fc1.weight"] = retainArray(visual.merger.linearFc1.weight);
  tensors["model.visual.merger.linear_fc1.bias"] = retainArray(
    expectTensor(visual.merger.linearFc1.bias, "Expected Qwen merger linear_fc1 bias."),
  );
  tensors["model.visual.merger.linear_fc2.weight"] = retainArray(visual.merger.linearFc2.weight);
  tensors["model.visual.merger.linear_fc2.bias"] = retainArray(
    expectTensor(visual.merger.linearFc2.bias, "Expected Qwen merger linear_fc2 bias."),
  );

  for (let layerIndex = 0; layerIndex < visual.blocks.length; layerIndex += 1) {
    const block = visual.blocks[layerIndex];
    if (block === undefined) {
      throw new Error(`Expected Qwen 3.5 vision block ${layerIndex} to exist.`);
    }
    const prefix = `model.visual.blocks.${layerIndex}`;
    tensors[`${prefix}.norm1.weight`] = retainArray(block.norm1.weight);
    tensors[`${prefix}.norm1.bias`] = retainArray(block.norm1.bias);
    tensors[`${prefix}.norm2.weight`] = retainArray(block.norm2.weight);
    tensors[`${prefix}.norm2.bias`] = retainArray(block.norm2.bias);
    tensors[`${prefix}.attn.qkv.weight`] = retainArray(block.attention.qkv.weight);
    tensors[`${prefix}.attn.qkv.bias`] = retainArray(
      expectTensor(block.attention.qkv.bias, "Expected Qwen vision qkv bias."),
    );
    tensors[`${prefix}.attn.proj.weight`] = retainArray(block.attention.proj.weight);
    tensors[`${prefix}.attn.proj.bias`] = retainArray(
      expectTensor(block.attention.proj.bias, "Expected Qwen vision proj bias."),
    );
    tensors[`${prefix}.mlp.linear_fc1.weight`] = retainArray(block.mlp.linearFc1.weight);
    tensors[`${prefix}.mlp.linear_fc1.bias`] = retainArray(
      expectTensor(block.mlp.linearFc1.bias, "Expected Qwen vision linear_fc1 bias."),
    );
    tensors[`${prefix}.mlp.linear_fc2.weight`] = retainArray(block.mlp.linearFc2.weight);
    tensors[`${prefix}.mlp.linear_fc2.bias`] = retainArray(
      expectTensor(block.mlp.linearFc2.bias, "Expected Qwen vision linear_fc2 bias."),
    );
  }

  return tensors;
}

async function createTinyQwen3_5TextSnapshot(style: "raw" | "mlx-converted"): Promise<{
  directory: string;
  model: Qwen3_5TextCausalLM;
}> {
  const directory = createTempDir(`mlxts-transformers-qwen3_5-${style}-`);
  const rawConfig = rawConfigForQwen3_5Text();
  const registration = resolveFamily("qwen3_5_text");
  const model = registration.createModel(registration.parseConfig(rawConfig));
  if (!(model instanceof Qwen3_5TextCausalLM)) {
    throw new Error("Expected a Qwen3_5TextCausalLM for the supported Qwen 3.5 text snapshot.");
  }

  const baseTensors = checkpointTensorsForQwen3_5Text(model);
  const tensors =
    style === "raw" ? rawQwen3_5TextTensors(baseTensors) : convertedQwen3_5TextTensors(baseTensors);
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
    const uniqueTensors = new Set([...Object.values(baseTensors), ...Object.values(tensors)]);
    for (const tensor of uniqueTensors) {
      tensor.free();
    }
  }

  return { directory, model };
}

async function createTinyQwen3_5Snapshot(): Promise<{
  directory: string;
  model: Qwen3_5TextCausalLM;
}> {
  const directory = createTempDir("mlxts-transformers-qwen3_5-wrapper-");
  const rawConfig = rawConfigForQwen3_5Wrapper();
  const registration = resolveFamily("qwen3_5");
  const model = registration.createModel(registration.parseConfig(rawConfig));
  if (!(model instanceof Qwen3_5TextCausalLM)) {
    throw new Error("Expected loadCausalLM to use the text Qwen 3.5 model by default.");
  }

  const baseTensors = checkpointTensorsForQwen3_5Text(model);
  const tensors = wrapperQwen3_5TextTensors(baseTensors);
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
    const uniqueTensors = new Set([...Object.values(baseTensors), ...Object.values(tensors)]);
    for (const tensor of uniqueTensors) {
      tensor.free();
    }
  }

  return { directory, model };
}

async function createTinyQwen3_5ConditionalSnapshot(): Promise<{
  directory: string;
  model: Qwen3_5ForConditionalGeneration;
}> {
  const directory = createTempDir("mlxts-transformers-qwen3_5-conditional-");
  const rawConfig = rawConfigForQwen3_5Wrapper();
  const parsedConfig = qwen3_5ConditionalFamily.parseConfig(rawConfig);
  const model = new Qwen3_5ForConditionalGeneration(parsedConfig);
  const tensors = checkpointTensorsForQwen3_5Conditional(model);
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
    for (const tensor of new Set(Object.values(tensors))) {
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

function firstQwenMoeLayer(model: Qwen3_5TextCausalLM): Qwen3_5TextMoE {
  const layer = model.model.layers[0];
  if (layer === undefined || !(layer.mlp instanceof Qwen3_5TextMoE)) {
    throw new Error("Expected the loaded Qwen fixture to own a MoE layer.");
  }
  return layer.mlp;
}

function firstGemma4Experts(model: Gemma4TextCausalLM): PackedSwitchGLUExperts | SwitchGLUExperts {
  const layer = model.model.layers[0];
  if (layer === undefined || layer.experts === null) {
    throw new Error("Expected the loaded Gemma 4 fixture to own MoE experts.");
  }
  return layer.experts;
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

  test.each([
    "raw",
    "mlx-converted",
  ] as const)("loadCausalLM round-trips a tiny qwen3_5_text %s snapshot", async (style) => {
    const { directory, model: originalModel } = await createTinyQwen3_5TextSnapshot(style);
    using loadedModel = await loadCausalLM(directory);
    using inputIds = array([[0, 1, 2]], "int32");
    using expectedLogits = originalModel.forward(inputIds);
    using actualLogits = loadedModel.forward(inputIds);

    mxEval(expectedLogits, actualLogits);

    expect(loadedModel.family).toBe("qwen");
    expect(loadedModel.layerCount).toBe(originalModel.layerCount);
    expectCloseLists(actualLogits.toList(), expectedLogits.toList());

    using cache = loadedModel.createCache();
    expect(cache).toBeInstanceOf(Qwen3_5TextCache);

    originalModel[Symbol.dispose]();
  });

  test("loadCausalLM loads top-level qwen3_5 snapshots as text-only causal LMs", async () => {
    const { directory, model: originalModel } = await createTinyQwen3_5Snapshot();
    using loadedModel = await loadCausalLM(directory);
    using inputIds = array([[0, 1, 2]], "int32");
    using expectedLogits = originalModel.forward(inputIds);
    using actualLogits = loadedModel.forward(inputIds);

    mxEval(expectedLogits, actualLogits);

    expect(loadedModel).toBeInstanceOf(Qwen3_5TextCausalLM);
    expect(loadedModel.config.modelType).toBe("qwen3_5_text");
    expect(loadedModel.family).toBe("qwen");
    expectCloseLists(actualLogits.toList(), expectedLogits.toList());

    originalModel[Symbol.dispose]();
  });

  test("loadQwen3_5ForConditionalGeneration loads the explicit multimodal wrapper", async () => {
    const { directory, model: originalModel } = await createTinyQwen3_5ConditionalSnapshot();
    using loadedModel = await loadQwen3_5ForConditionalGeneration(directory);
    using inputIds = array([[0, 1, 2]], "int32");
    using expectedLogits = originalModel.forward(inputIds);
    using actualLogits = loadedModel.forward(inputIds);

    mxEval(expectedLogits, actualLogits);

    expect(loadedModel).toBeInstanceOf(Qwen3_5ForConditionalGeneration);
    expect(loadedModel.config.modelType).toBe("qwen3_5");
    expect(loadedModel.family).toBe("qwen");
    expectCloseLists(actualLogits.toList(), expectedLogits.toList());
    expectCloseLists(
      loadedModel.model.visual.patchEmbed.bias.toList(),
      originalModel.model.visual.patchEmbed.bias.toList(),
    );

    originalModel[Symbol.dispose]();
  });

  test("shouldLoadQwen3_5ForConditionalGeneration accepts dense and MoE conditional wrappers", async () => {
    const denseDirectory = createTempDir("mlxts-transformers-qwen3_5-detect-");
    const moeDirectory = createTempDir("mlxts-transformers-qwen3_5-moe-detect-");
    const textOnlyDirectory = createTempDir("mlxts-transformers-qwen3_5-text-detect-");

    await Bun.write(
      join(denseDirectory, "config.json"),
      `${JSON.stringify(rawConfigForQwen3_5Wrapper(), null, 2)}\n`,
    );
    await Bun.write(
      join(moeDirectory, "config.json"),
      `${JSON.stringify(
        {
          ...rawConfigForQwen3_5Wrapper(),
          model_type: "qwen3_5_moe",
          architectures: ["Qwen3_5MoeForConditionalGeneration"],
          text_config: {
            ...rawConfigForQwen3_5Text(),
            model_type: "qwen3_5_moe_text",
            moe_intermediate_size: 2,
            num_experts: 4,
            num_experts_per_tok: 2,
            shared_expert_intermediate_size: 2,
          },
        },
        null,
        2,
      )}\n`,
    );
    await Bun.write(
      join(textOnlyDirectory, "config.json"),
      `${JSON.stringify(rawConfigForQwen3_5Text(), null, 2)}\n`,
    );

    expect(await shouldLoadQwen3_5ForConditionalGeneration(denseDirectory)).toBe(true);
    expect(await shouldLoadQwen3_5ForConditionalGeneration(moeDirectory)).toBe(true);
    expect(await shouldLoadQwen3_5ForConditionalGeneration(textOnlyDirectory)).toBe(false);
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

  test("loadCausalLM attaches checkpoint generation defaults to the loaded model config", async () => {
    const { directory, model } = await createTinySnapshot("llama");
    await Bun.write(
      join(directory, "generation_config.json"),
      `${JSON.stringify({ do_sample: false, eos_token_id: [3, 4], top_p: 0.92 }, null, 2)}\n`,
    );

    using loadedModel = await loadCausalLM(directory);
    expect(loadedModel.config.generationDefaults).toEqual({
      temperature: 0,
      eosTokenIds: [3, 4],
      topP: 0.92,
    });
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

  test("loadCausalLM prepares quantized linear modules from explicit checkpoint metadata", async () => {
    const directory = createTempDir("mlxts-transformers-quantized-");
    const rawConfig: Record<string, unknown> = {
      ...rawConfigForFamily("llama"),
      hidden_size: 64,
      intermediate_size: 128,
      num_attention_heads: 4,
      num_key_value_heads: 4,
      max_position_embeddings: 64,
      vocab_size: 7,
    };
    rawConfig.quantization = {
      bits: 4,
      group_size: 64,
      "model.layers.0.self_attn.q_proj": true,
      "model.layers.1.self_attn.q_proj": true,
    };
    const registration = resolveFamily("llama");
    using model = registration.createModel(registration.parseConfig(rawConfig));
    if (!(model instanceof LlamaLikeCausalLM)) {
      throw new Error("Expected a LlamaLikeCausalLM for the quantized checkpoint fixture.");
    }

    const baseTensors = checkpointTensors(model);
    const quantizedTensors = quantizeCheckpointTensors(baseTensors, [
      "model.embed_tokens",
      "model.layers.0.self_attn.q_proj",
      "model.layers.1.self_attn.q_proj",
    ]);
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
      await saveSafetensors(quantizedTensors, join(directory, "model.safetensors"));
    } finally {
      for (const tensor of Object.values(baseTensors)) {
        tensor.free();
      }
      for (const tensor of Object.values(quantizedTensors)) {
        tensor.free();
      }
    }

    using loadedModel = await loadCausalLM(directory);
    if (!(loadedModel instanceof LlamaLikeCausalLM)) {
      throw new Error("Expected a LlamaLikeCausalLM after loading the quantized checkpoint.");
    }

    const firstLayer = loadedModel.model.layers[0];
    const secondLayer = loadedModel.model.layers[1];
    if (firstLayer === undefined || secondLayer === undefined) {
      throw new Error("Expected two decoder layers in the quantized checkpoint fixture.");
    }

    expect(firstLayer.selfAttention.qProjection).toBeInstanceOf(QuantizedLinear);
    expect(secondLayer.selfAttention.qProjection).toBeInstanceOf(QuantizedLinear);
    expect(firstLayer.selfAttention.kProjection).toBeInstanceOf(Linear);
    expect(secondLayer.selfAttention.kProjection).toBeInstanceOf(Linear);
    expect(loadedModel.model.embedTokens).toBeInstanceOf(QuantizedEmbedding);
    expect(loadedModel.model.embedTokens.weight.dtype).toBe("uint32");

    using inputIds = array([[0, 1, 2]], "int32");
    using logits = loadedModel.forward(inputIds);
    expect(logits.shape).toEqual([1, 3, rawConfig.vocab_size as number]);
  });

  test("loadCausalLM supports official combined Qwen MoE expert weights", async () => {
    const directory = createTempDir("mlxts-transformers-qwen-moe-combined-");
    const rawConfig = rawConfigForQwen3_5MoeText();
    const registration = resolveFamily("qwen3_5_moe_text");
    using model = registration.createModel(registration.parseConfig(rawConfig));
    if (!(model instanceof Qwen3_5TextCausalLM)) {
      throw new Error("Expected a Qwen3_5TextCausalLM for the Qwen MoE fixture.");
    }

    const tensors = checkpointTensorsForQwen3_5Text(model);
    try {
      await Bun.write(join(directory, "config.json"), `${JSON.stringify(rawConfig, null, 2)}\n`);
      await writeTokenizerFixture(directory);
      await saveSafetensors(tensors, join(directory, "model.safetensors"));
    } finally {
      freeTensorRecords(tensors);
    }

    using loadedModel = await loadCausalLM(directory);
    if (!(loadedModel instanceof Qwen3_5TextCausalLM)) {
      throw new Error("Expected loadCausalLM to return a Qwen3_5TextCausalLM.");
    }
    expect(firstQwenMoeLayer(loadedModel).experts).toBeInstanceOf(PackedSwitchGLUExperts);
  });

  test("loadCausalLM supports split quantized Qwen MoE switch experts", async () => {
    const directory = createTempDir("mlxts-transformers-qwen-moe-split-");
    const rawConfig = {
      ...rawConfigForQwen3_5MoeText(),
      quantization: { group_size: 64, bits: 4, mode: "affine" },
    };
    const registration = resolveFamily("qwen3_5_moe_text");
    using model = registration.createModel(registration.parseConfig(rawConfig));
    if (!(model instanceof Qwen3_5TextCausalLM)) {
      throw new Error("Expected a Qwen3_5TextCausalLM for the split Qwen MoE fixture.");
    }

    const baseTensors = checkpointTensorsForQwen3_5Text(model);
    const splitTensors = qwenSplitMoeCheckpointTensors(baseTensors);
    const quantizedTensors = quantizeCheckpointTensors(splitTensors, [
      "model.layers.0.mlp.switch_mlp.gate_proj",
      "model.layers.0.mlp.switch_mlp.up_proj",
      "model.layers.0.mlp.switch_mlp.down_proj",
    ]);
    try {
      await Bun.write(join(directory, "config.json"), `${JSON.stringify(rawConfig, null, 2)}\n`);
      await writeTokenizerFixture(directory);
      await saveSafetensors(quantizedTensors, join(directory, "model.safetensors"));
    } finally {
      freeTensorRecords(baseTensors, splitTensors, quantizedTensors);
    }

    using loadedModel = await loadCausalLM(directory);
    if (!(loadedModel instanceof Qwen3_5TextCausalLM)) {
      throw new Error("Expected loadCausalLM to return a Qwen3_5TextCausalLM.");
    }
    const experts = firstQwenMoeLayer(loadedModel).experts;
    expect(experts).toBeInstanceOf(SwitchGLUExperts);
    if (!(experts instanceof SwitchGLUExperts)) {
      throw new Error("Expected split Qwen MoE experts after quantized preparation.");
    }
    expect(experts.gateProjection.weight.dtype).toBe("uint32");

    using inputIds = array([[0, 1, 2]], "int32");
    using logits = loadedModel.forward(inputIds);
    expect(logits.shape).toEqual([1, 3, 7]);
  });

  test("loadCausalLM supports official combined Gemma 4 MoE expert weights", async () => {
    const directory = createTempDir("mlxts-transformers-gemma4-moe-combined-");
    const rawConfig = rawConfigForGemma4MoeText();
    const registration = resolveFamily("gemma4_text");
    using model = registration.createModel(registration.parseConfig(rawConfig));
    if (!(model instanceof Gemma4TextCausalLM)) {
      throw new Error("Expected a Gemma4TextCausalLM for the Gemma 4 MoE fixture.");
    }

    const tensors = checkpointTensorsForGemma4Text(model);
    try {
      await Bun.write(join(directory, "config.json"), `${JSON.stringify(rawConfig, null, 2)}\n`);
      await writeTokenizerFixture(directory);
      await saveSafetensors(tensors, join(directory, "model.safetensors"));
    } finally {
      freeTensorRecords(tensors);
    }

    using loadedModel = await loadCausalLM(directory);
    if (!(loadedModel instanceof Gemma4TextCausalLM)) {
      throw new Error("Expected loadCausalLM to return a Gemma4TextCausalLM.");
    }
    expect(firstGemma4Experts(loadedModel)).toBeInstanceOf(PackedSwitchGLUExperts);
  });

  test("loadCausalLM supports split quantized Gemma 4 MoE switch experts", async () => {
    const directory = createTempDir("mlxts-transformers-gemma4-moe-split-");
    const rawConfig = {
      ...rawConfigForGemma4MoeText(),
      quantization: { group_size: 64, bits: 4, mode: "affine" },
    };
    const registration = resolveFamily("gemma4_text");
    using model = registration.createModel(registration.parseConfig(rawConfig));
    if (!(model instanceof Gemma4TextCausalLM)) {
      throw new Error("Expected a Gemma4TextCausalLM for the split Gemma 4 MoE fixture.");
    }

    const baseTensors = checkpointTensorsForGemma4Text(model);
    const splitTensors = gemma4SplitMoeCheckpointTensors(baseTensors);
    const quantizedTensors = quantizeCheckpointTensors(splitTensors, [
      "model.layers.0.experts.switch_glu.gate_proj",
      "model.layers.0.experts.switch_glu.up_proj",
      "model.layers.0.experts.switch_glu.down_proj",
    ]);
    try {
      await Bun.write(join(directory, "config.json"), `${JSON.stringify(rawConfig, null, 2)}\n`);
      await writeTokenizerFixture(directory);
      await saveSafetensors(quantizedTensors, join(directory, "model.safetensors"));
    } finally {
      freeTensorRecords(baseTensors, splitTensors, quantizedTensors);
    }

    using loadedModel = await loadCausalLM(directory);
    if (!(loadedModel instanceof Gemma4TextCausalLM)) {
      throw new Error("Expected loadCausalLM to return a Gemma4TextCausalLM.");
    }
    const experts = firstGemma4Experts(loadedModel);
    expect(experts).toBeInstanceOf(SwitchGLUExperts);
    if (!(experts instanceof SwitchGLUExperts)) {
      throw new Error("Expected split Gemma 4 MoE experts after quantized preparation.");
    }
    expect(experts.gateProjection.weight.dtype).toBe("uint32");

    using inputIds = array([[0, 1, 2]], "int32");
    using logits = loadedModel.forward(inputIds);
    expect(logits.shape).toEqual([1, 3, 7]);
  });

  test("loadCausalLM preserves mixed per-path quantization for Gemma 4 wrapper checkpoints", async () => {
    const directory = createTempDir("mlxts-transformers-gemma4-moe-mixed-quant-");
    const eightBitTargets = [
      "language_model.model.layers.0.mlp.gate_proj",
      "language_model.model.layers.0.mlp.up_proj",
      "language_model.model.layers.0.mlp.down_proj",
      "language_model.model.layers.0.router.proj",
    ];
    const fourBitTargets = [
      "language_model.model.layers.0.experts.switch_glu.gate_proj",
      "language_model.model.layers.0.experts.switch_glu.up_proj",
      "language_model.model.layers.0.experts.switch_glu.down_proj",
    ];
    const rawConfig = {
      model_type: "gemma4",
      text_config: rawConfigForGemma4MoeText(),
      quantization_config: {
        group_size: 64,
        bits: 4,
        mode: "affine",
        ...Object.fromEntries(eightBitTargets.map((target) => [target, { bits: 8 }])),
      },
    };
    const registration = resolveFamily("gemma4");
    using model = registration.createModel(registration.parseConfig(rawConfig));
    if (!(model instanceof Gemma4TextCausalLM)) {
      throw new Error("Expected a Gemma4TextCausalLM for the Gemma 4 wrapper fixture.");
    }

    const baseTensors = checkpointTensorsForGemma4Text(model);
    const splitTensors = gemma4SplitMoeCheckpointTensors(baseTensors);
    const wrapperTensors = wrapperLanguageModelTensors(splitTensors);
    const fourBitTensors = quantizeCheckpointTensors(wrapperTensors, fourBitTargets);
    const quantizedTensors = quantizeCheckpointTensors(fourBitTensors, eightBitTargets, {
      bits: 8,
    });
    try {
      await Bun.write(join(directory, "config.json"), `${JSON.stringify(rawConfig, null, 2)}\n`);
      await writeTokenizerFixture(directory);
      await saveSafetensors(quantizedTensors, join(directory, "model.safetensors"));
    } finally {
      freeTensorRecords(
        baseTensors,
        splitTensors,
        wrapperTensors,
        fourBitTensors,
        quantizedTensors,
      );
    }

    using loadedModel = await loadCausalLM(directory);
    if (!(loadedModel instanceof Gemma4TextCausalLM)) {
      throw new Error("Expected loadCausalLM to return a Gemma4TextCausalLM.");
    }
    const layer = loadedModel.model.layers[0];
    if (layer === undefined) {
      throw new Error("Expected the loaded Gemma 4 fixture to own a decoder layer.");
    }

    expect(layer.mlp.gateProjection).toBeInstanceOf(QuantizedLinear);
    expect(layer.mlp.upProjection).toBeInstanceOf(QuantizedLinear);
    expect(layer.mlp.downProjection).toBeInstanceOf(QuantizedLinear);
    if (!(layer.mlp.downProjection instanceof QuantizedLinear)) {
      throw new Error("Expected the dense Gemma 4 MLP down projection to remain quantized.");
    }
    expect(layer.mlp.downProjection.bits).toBe(8);
    expect(layer.mlp.downProjection.weight.shape).toEqual([64, 32]);

    expect(layer.router?.proj).toBeInstanceOf(QuantizedLinear);
    if (!(layer.router?.proj instanceof QuantizedLinear)) {
      throw new Error("Expected the Gemma 4 router projection to remain quantized.");
    }
    expect(layer.router.proj.bits).toBe(8);

    const experts = firstGemma4Experts(loadedModel);
    expect(experts).toBeInstanceOf(SwitchGLUExperts);
    if (!(experts instanceof SwitchGLUExperts)) {
      throw new Error("Expected split Gemma 4 experts after quantized preparation.");
    }
    expect(experts.downProjection.weight.shape).toEqual([2, 64, 8]);

    using inputIds = array([[0, 1, 2]], "int32");
    using logits = loadedModel.forward(inputIds);
    expect(logits.shape).toEqual([1, 3, 7]);
  });

  test("quantizePretrainedSnapshot rewrites a dense snapshot into a loadable quantized snapshot", async () => {
    const directory = createTempDir("mlxts-transformers-quantized-export-source-");
    const rawConfig: Record<string, unknown> = {
      ...rawConfigForFamily("llama"),
      hidden_size: 64,
      intermediate_size: 128,
      num_attention_heads: 4,
      num_key_value_heads: 4,
      max_position_embeddings: 64,
      vocab_size: 7,
    };
    const registration = resolveFamily("llama");
    using model = registration.createModel(registration.parseConfig(rawConfig));
    if (!(model instanceof LlamaLikeCausalLM)) {
      throw new Error("Expected a LlamaLikeCausalLM for the quantized export fixture.");
    }

    const tensors = checkpointTensors(model);
    const tokenizer = tokenizerFixture();
    const outputDirectory = join(createTempDir("mlxts-transformers-quantized-export-"), "out");

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

    const result = await quantizePretrainedSnapshot(directory, {
      outputDir: outputDirectory,
      bits: 4,
      groupSize: 64,
      mode: "affine",
    });

    const writtenConfig = JSON.parse(
      await Bun.file(join(outputDirectory, "config.json")).text(),
    ) as {
      quantization?: { group_size?: number; bits?: number; mode?: string };
    };
    expect(writtenConfig.quantization).toEqual({
      group_size: 64,
      bits: 4,
      mode: "affine",
    });
    expect(result.quantizedTensorCount).toBeGreaterThan(0);

    using loadedModel = await loadCausalLM(outputDirectory);
    if (!(loadedModel instanceof LlamaLikeCausalLM)) {
      throw new Error("Expected a LlamaLikeCausalLM after loading the quantized export.");
    }

    const firstLayer = loadedModel.model.layers[0];
    if (firstLayer === undefined) {
      throw new Error("Expected a decoder layer in the quantized export.");
    }

    expect(firstLayer.selfAttention.qProjection).toBeInstanceOf(QuantizedLinear);
    expect(firstLayer.selfAttention.kProjection).toBeInstanceOf(QuantizedLinear);
    expect(firstLayer.selfAttention.vProjection).toBeInstanceOf(QuantizedLinear);
    expect(firstLayer.selfAttention.outputProjection).toBeInstanceOf(QuantizedLinear);
    expect(loadedModel.model.embedTokens).toBeInstanceOf(QuantizedEmbedding);
  });

  test("quantizePretrainedSnapshot preserves MLX quantized auxiliary dtypes", async () => {
    const directory = createTempDir("mlxts-transformers-quantized-export-bf16-source-");
    const rawConfig: Record<string, unknown> = {
      ...rawConfigForFamily("llama"),
      hidden_size: 64,
      intermediate_size: 128,
      num_attention_heads: 4,
      num_key_value_heads: 4,
      max_position_embeddings: 64,
      vocab_size: 7,
    };
    const registration = resolveFamily("llama");
    using model = registration.createModel(registration.parseConfig(rawConfig));
    if (!(model instanceof LlamaLikeCausalLM)) {
      throw new Error("Expected a LlamaLikeCausalLM for the quantized export dtype fixture.");
    }

    const baseTensors = checkpointTensors(model);
    const bf16Tensors = castCheckpointTensors(baseTensors, "bfloat16");
    const tokenizer = tokenizerFixture();
    const outputDirectory = join(createTempDir("mlxts-transformers-quantized-export-bf16-"), "out");

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
      await saveSafetensors(bf16Tensors, join(directory, "model.safetensors"));
    } finally {
      for (const tensor of Object.values(baseTensors)) {
        tensor.free();
      }
      for (const tensor of Object.values(bf16Tensors)) {
        tensor.free();
      }
    }

    await quantizePretrainedSnapshot(directory, {
      outputDir: outputDirectory,
      bits: 4,
      groupSize: 64,
      mode: "affine",
    });

    const inspection = await inspectSafetensors(join(outputDirectory, "model.safetensors"));
    const scales = inspection.tensors.find(
      (tensor) => tensor.name === "model.layers.0.self_attn.q_proj.scales",
    );
    const biases = inspection.tensors.find(
      (tensor) => tensor.name === "model.layers.0.self_attn.q_proj.biases",
    );
    if (scales === undefined || biases === undefined) {
      throw new Error("Expected quantized q_proj auxiliary tensors in the exported snapshot.");
    }

    expect(scales.dtype).toBe("bfloat16");
    expect(biases.dtype).toBe("bfloat16");
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

  test("generateTextStream emits the same continuation text that generateText returns", async () => {
    const { directory, model: originalModel } = await createTinySnapshot("llama");
    using loadedModel = await loadCausalLM(directory);
    const tokenizer = await loadPretrainedTokenizer(directory);
    const chunks: string[] = [];

    const streamed = generateTextStream(
      loadedModel,
      tokenizer,
      "Hi",
      {
        maxTokens: 4,
        temperature: 0,
        useCache: true,
      },
      (chunk) => {
        chunks.push(chunk);
      },
    );

    expect(streamed.text).toBe(
      generateText(loadedModel, tokenizer, "Hi", { maxTokens: 4, temperature: 0, useCache: true }),
    );
    expect(chunks.join("")).toBe(streamed.text);

    originalModel[Symbol.dispose]();
  });

  test("generateTextStream can reuse an external prompt cache across turns", async () => {
    const { directory, model: originalModel } = await createTinySnapshot("llama");
    using loadedModel = await loadCausalLM(directory);
    const tokenizer = await loadPretrainedTokenizer(directory);
    using cache = loadedModel.createCache();

    const firstChunks: string[] = [];
    const first = generateTextStream(
      loadedModel,
      tokenizer,
      "Hi",
      {
        maxTokens: 2,
        temperature: 0,
        cache,
      },
      (chunk) => {
        firstChunks.push(chunk);
      },
    );

    const secondChunks: string[] = [];
    const second = generateTextStream(
      loadedModel,
      tokenizer,
      "!",
      {
        maxTokens: 2,
        temperature: 0,
        cache,
      },
      (chunk) => {
        secondChunks.push(chunk);
      },
    );

    expect(firstChunks.join("")).toBe(first.text);
    expect(secondChunks.join("")).toBe(second.text);
    expect(cache.offset).toBeGreaterThan(0);

    originalModel[Symbol.dispose]();
  });

  test("generation rejects an external cache when cache usage is disabled", async () => {
    const { directory, model: originalModel } = await createTinySnapshot("llama");
    using loadedModel = await loadCausalLM(directory);
    const tokenizer = await loadPretrainedTokenizer(directory);
    using cache = loadedModel.createCache();

    expect(() =>
      generateText(loadedModel, tokenizer, "Hi", {
        maxTokens: 1,
        temperature: 0,
        useCache: false,
        cache,
      }),
    ).toThrow("cache cannot be provided when useCache is false");

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
