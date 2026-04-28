/**
 * Qwen 3.5 multimodal config parsing.
 * @module
 */

import {
  expectConfigRecord,
  expectInteger,
  expectString,
  optionalBoolean,
  optionalInteger,
  optionalNumber,
  optionalString,
} from "../../infrastructure/config-parsing";
import type { FamilyRegistration } from "../../types";
import { ConfigParseError } from "../../types";
import { Qwen3_5ForConditionalGeneration } from "./conditional";
import { parseQwen3_5FeedForward } from "./config-feedforward";
import {
  expectPositiveInteger,
  optionalIntegerOrNull,
  parseEosTokenId,
  parseIntegerArrayField,
  parseIntegerOrIntegerArray,
  parseLayerTypes,
  parseTextRopeParameters,
} from "./config-helpers";
import { Qwen3_5TextCausalLM } from "./model";
import type {
  Qwen3_5Config,
  Qwen3_5EosTokenId,
  Qwen3_5ModelType,
  Qwen3_5TextConfig,
  Qwen3_5TextModelType,
  Qwen3_5VisionConfig,
  Qwen3_5VisionModelType,
} from "./types";
import {
  detectQwen3_5CheckpointStyle,
  isIgnoredQwen3_5CausalLMWeight,
  isIgnoredQwen3_5TextWeight,
  isIgnoredQwen3_5Weight,
  sanitizeQwen3_5CausalLMWeight,
  sanitizeQwen3_5TextWeight,
  sanitizeQwen3_5Weight,
  transformQwen3_5CheckpointTensor,
} from "./weights";

function parseQwen3_5ModelType(config: Record<string, unknown>, context: string): Qwen3_5ModelType {
  const modelType = expectString(config, "model_type", context);
  if (modelType !== "qwen3_5" && modelType !== "qwen3_5_moe") {
    throw new ConfigParseError(
      `${context}.model_type must be "qwen3_5" or "qwen3_5_moe", got "${modelType}".`,
    );
  }
  return modelType;
}

function parseQwen3_5TextModelType(
  config: Record<string, unknown>,
  context: string,
): Qwen3_5TextModelType {
  const modelType = expectString(config, "model_type", context);
  if (modelType !== "qwen3_5_text" && modelType !== "qwen3_5_moe_text") {
    throw new ConfigParseError(
      `${context}.model_type must be "qwen3_5_text" or "qwen3_5_moe_text", got "${modelType}".`,
    );
  }
  return modelType;
}

function parseQwen3_5TextCore(
  config: Record<string, unknown>,
  context: string,
): {
  hiddenSize: number;
  numAttentionHeads: number;
  numHiddenLayers: number;
  numKeyValueHeads: number;
  headDim: number;
  hiddenAct: string;
} {
  const hiddenSize = expectPositiveInteger(
    expectInteger(config, "hidden_size", context),
    `${context}.hidden_size`,
  );
  const numAttentionHeads = expectPositiveInteger(
    expectInteger(config, "num_attention_heads", context),
    `${context}.num_attention_heads`,
  );
  const numHiddenLayers = expectPositiveInteger(
    expectInteger(config, "num_hidden_layers", context),
    `${context}.num_hidden_layers`,
  );
  const numKeyValueHeads =
    optionalInteger(config, "num_key_value_heads", context) ?? numAttentionHeads;
  expectPositiveInteger(numKeyValueHeads, `${context}.num_key_value_heads`);

  const headDim =
    optionalInteger(config, "head_dim", context) ?? Math.floor(hiddenSize / numAttentionHeads);
  expectPositiveInteger(headDim, `${context}.head_dim`);

  const hiddenAct = optionalString(config, "hidden_act", context) ?? "silu";
  if (hiddenAct !== "silu") {
    throw new ConfigParseError(`${context}.hidden_act must be "silu", got "${hiddenAct}".`);
  }

  return {
    hiddenSize,
    numAttentionHeads,
    numHiddenLayers,
    numKeyValueHeads,
    headDim,
    hiddenAct,
  };
}

function parseQwen3_5TextOptionalSettings(
  config: Record<string, unknown>,
  context: string,
): {
  initializerRange: number;
  rmsNormEps: number;
  useCache: boolean;
  tieWordEmbeddings: boolean;
  attentionBias: boolean;
  attentionDropout: number;
  attnOutputGate: boolean;
  outputGateType: string | null;
  linearConvKernelDim: number;
  linearKeyHeadDim: number;
  linearValueHeadDim: number;
  linearNumKeyHeads: number;
  linearNumValueHeads: number;
  mtpNumHiddenLayers: number;
  mtpUseDedicatedEmbeddings: boolean;
  mambaSsmDtype: string | null;
  bosTokenId: number | null;
  eosTokenId: Qwen3_5EosTokenId;
  padTokenId: number | null;
} {
  const attentionSettings = parseQwen3_5TextAttentionSettings(config, context);
  const tokenSettings = parseQwen3_5TextTokenSettings(config, context);

  return {
    initializerRange: optionalNumber(config, "initializer_range", context) ?? 0.02,
    rmsNormEps: optionalNumber(config, "rms_norm_eps", context) ?? 1e-6,
    useCache: optionalBoolean(config, "use_cache", context) ?? true,
    tieWordEmbeddings: optionalBoolean(config, "tie_word_embeddings", context) ?? false,
    attentionBias: attentionSettings.attentionBias,
    attentionDropout: attentionSettings.attentionDropout,
    attnOutputGate: attentionSettings.attnOutputGate,
    outputGateType: attentionSettings.outputGateType,
    linearConvKernelDim: attentionSettings.linearConvKernelDim,
    linearKeyHeadDim: attentionSettings.linearKeyHeadDim,
    linearValueHeadDim: attentionSettings.linearValueHeadDim,
    linearNumKeyHeads: attentionSettings.linearNumKeyHeads,
    linearNumValueHeads: attentionSettings.linearNumValueHeads,
    mtpNumHiddenLayers: optionalInteger(config, "mtp_num_hidden_layers", context) ?? 0,
    mtpUseDedicatedEmbeddings:
      optionalBoolean(config, "mtp_use_dedicated_embeddings", context) ?? false,
    mambaSsmDtype: optionalString(config, "mamba_ssm_dtype", context) ?? null,
    bosTokenId: tokenSettings.bosTokenId,
    eosTokenId: tokenSettings.eosTokenId,
    padTokenId: tokenSettings.padTokenId,
  };
}

function parseQwen3_5TextAttentionSettings(
  config: Record<string, unknown>,
  context: string,
): {
  attentionBias: boolean;
  attentionDropout: number;
  attnOutputGate: boolean;
  outputGateType: string | null;
  linearConvKernelDim: number;
  linearKeyHeadDim: number;
  linearValueHeadDim: number;
  linearNumKeyHeads: number;
  linearNumValueHeads: number;
} {
  return {
    attentionBias: optionalBoolean(config, "attention_bias", context) ?? false,
    attentionDropout: optionalNumber(config, "attention_dropout", context) ?? 0,
    attnOutputGate: optionalBoolean(config, "attn_output_gate", context) ?? true,
    outputGateType: optionalString(config, "output_gate_type", context) ?? null,
    linearConvKernelDim: expectPositiveInteger(
      optionalInteger(config, "linear_conv_kernel_dim", context) ?? 4,
      `${context}.linear_conv_kernel_dim`,
    ),
    linearKeyHeadDim: expectPositiveInteger(
      optionalInteger(config, "linear_key_head_dim", context) ?? 128,
      `${context}.linear_key_head_dim`,
    ),
    linearValueHeadDim: expectPositiveInteger(
      optionalInteger(config, "linear_value_head_dim", context) ?? 128,
      `${context}.linear_value_head_dim`,
    ),
    linearNumKeyHeads: expectPositiveInteger(
      optionalInteger(config, "linear_num_key_heads", context) ?? 16,
      `${context}.linear_num_key_heads`,
    ),
    linearNumValueHeads: expectPositiveInteger(
      optionalInteger(config, "linear_num_value_heads", context) ?? 32,
      `${context}.linear_num_value_heads`,
    ),
  };
}

function parseQwen3_5TextTokenSettings(
  config: Record<string, unknown>,
  context: string,
): {
  bosTokenId: number | null;
  eosTokenId: Qwen3_5EosTokenId;
  padTokenId: number | null;
} {
  return {
    bosTokenId: optionalIntegerOrNull(config, "bos_token_id", context),
    eosTokenId: parseEosTokenId(config, context),
    padTokenId: optionalIntegerOrNull(config, "pad_token_id", context),
  };
}

function parseQwen3_5TextConfigInternal(
  rawConfig: Record<string, unknown>,
  context: string,
  sourceConfig: Record<string, unknown>,
): Qwen3_5TextConfig {
  const config = expectConfigRecord(rawConfig, context);
  const modelType = parseQwen3_5TextModelType(config, context);
  const core = parseQwen3_5TextCore(config, context);
  const layerTypes = parseLayerTypes(config, core.numHiddenLayers, context);
  const ropeParameters = parseTextRopeParameters(config, context);
  const optionalSettings = parseQwen3_5TextOptionalSettings(config, context);
  const feedForward = parseQwen3_5FeedForward(config, context, modelType);

  return {
    family: "qwen",
    modelType,
    rawConfig: sourceConfig,
    vocabSize: expectPositiveInteger(
      expectInteger(config, "vocab_size", context),
      `${context}.vocab_size`,
    ),
    hiddenSize: core.hiddenSize,
    intermediateSize: feedForward.intermediateSize,
    feedForwardKind: feedForward.feedForwardKind,
    moeIntermediateSize: feedForward.moeIntermediateSize,
    sharedExpertIntermediateSize: feedForward.sharedExpertIntermediateSize,
    numExperts: feedForward.numExperts,
    numExpertsPerToken: feedForward.numExpertsPerToken,
    routerAuxLossCoef: feedForward.routerAuxLossCoef,
    numHiddenLayers: core.numHiddenLayers,
    numAttentionHeads: core.numAttentionHeads,
    numKeyValueHeads: core.numKeyValueHeads,
    headDim: core.headDim,
    hiddenAct: core.hiddenAct,
    maxPositionEmbeddings: expectPositiveInteger(
      expectInteger(config, "max_position_embeddings", context),
      `${context}.max_position_embeddings`,
    ),
    initializerRange: optionalSettings.initializerRange,
    rmsNormEps: optionalSettings.rmsNormEps,
    useCache: optionalSettings.useCache,
    tieWordEmbeddings: optionalSettings.tieWordEmbeddings,
    attentionBias: optionalSettings.attentionBias,
    attentionDropout: optionalSettings.attentionDropout,
    attnOutputGate: optionalSettings.attnOutputGate,
    outputGateType: optionalSettings.outputGateType,
    linearConvKernelDim: optionalSettings.linearConvKernelDim,
    linearKeyHeadDim: optionalSettings.linearKeyHeadDim,
    linearValueHeadDim: optionalSettings.linearValueHeadDim,
    linearNumKeyHeads: optionalSettings.linearNumKeyHeads,
    linearNumValueHeads: optionalSettings.linearNumValueHeads,
    layerTypes: layerTypes.layerTypes,
    fullAttentionInterval: layerTypes.fullAttentionInterval,
    ropeParameters,
    partialRotaryFactor: ropeParameters.partialRotaryFactor,
    mtpNumHiddenLayers: optionalSettings.mtpNumHiddenLayers,
    mtpUseDedicatedEmbeddings: optionalSettings.mtpUseDedicatedEmbeddings,
    mambaSsmDtype: optionalSettings.mambaSsmDtype,
    bosTokenId: optionalSettings.bosTokenId,
    eosTokenId: optionalSettings.eosTokenId,
    padTokenId: optionalSettings.padTokenId,
  };
}

function parseVisionModelType(
  config: Record<string, unknown>,
  context: string,
): Qwen3_5VisionModelType {
  const modelType = expectString(config, "model_type", context);
  if (modelType !== "qwen3_5" && modelType !== "qwen3_5_vision") {
    throw new ConfigParseError(
      `${context}.model_type must be "qwen3_5" or "qwen3_5_vision", got "${modelType}".`,
    );
  }
  return modelType;
}

function parseQwen3_5VisionConfigInternal(
  rawConfig: Record<string, unknown>,
  context: string,
  sourceConfig: Record<string, unknown>,
): Qwen3_5VisionConfig {
  const config = expectConfigRecord(rawConfig, context);
  const modelType = parseVisionModelType(config, context);
  const hiddenAct = optionalString(config, "hidden_act", context) ?? "gelu_pytorch_tanh";
  if (hiddenAct !== "gelu_pytorch_tanh") {
    throw new ConfigParseError(
      `${context}.hidden_act must be "gelu_pytorch_tanh", got "${hiddenAct}".`,
    );
  }

  return {
    family: "qwen",
    modelType,
    rawConfig: sourceConfig,
    depth: expectPositiveInteger(expectInteger(config, "depth", context), `${context}.depth`),
    hiddenSize: expectPositiveInteger(
      expectInteger(config, "hidden_size", context),
      `${context}.hidden_size`,
    ),
    hiddenAct,
    intermediateSize: expectPositiveInteger(
      expectInteger(config, "intermediate_size", context),
      `${context}.intermediate_size`,
    ),
    numHeads: expectPositiveInteger(
      expectInteger(config, "num_heads", context),
      `${context}.num_heads`,
    ),
    inChannels: expectPositiveInteger(
      optionalInteger(config, "in_channels", context) ?? 3,
      `${context}.in_channels`,
    ),
    patchSize: parseIntegerOrIntegerArray(config, "patch_size", context),
    spatialMergeSize: expectPositiveInteger(
      optionalInteger(config, "spatial_merge_size", context) ?? 2,
      `${context}.spatial_merge_size`,
    ),
    temporalPatchSize: parseIntegerOrIntegerArray(config, "temporal_patch_size", context),
    outHiddenSize: expectPositiveInteger(
      expectInteger(config, "out_hidden_size", context),
      `${context}.out_hidden_size`,
    ),
    numPositionEmbeddings: expectPositiveInteger(
      expectInteger(config, "num_position_embeddings", context),
      `${context}.num_position_embeddings`,
    ),
    deepstackVisualIndexes:
      parseIntegerArrayField(config, "deepstack_visual_indexes", context, true) ?? [],
    initializerRange: optionalNumber(config, "initializer_range", context) ?? 0.02,
  };
}

export function parseQwen3_5TextConfig(rawConfig: Record<string, unknown>): Qwen3_5TextConfig {
  return parseQwen3_5TextConfigInternal(rawConfig, "Qwen 3.5 text config", rawConfig);
}

export function parseQwen3_5VisionConfig(rawConfig: Record<string, unknown>): Qwen3_5VisionConfig {
  return parseQwen3_5VisionConfigInternal(rawConfig, "Qwen 3.5 vision config", rawConfig);
}

export function parseQwen3_5Config(rawConfig: Record<string, unknown>): Qwen3_5Config {
  const config = expectConfigRecord(rawConfig, "Qwen 3.5 config");
  const modelType = parseQwen3_5ModelType(config, "Qwen 3.5 config");
  if (modelType !== "qwen3_5") {
    throw new ConfigParseError(
      `Qwen 3.5 config.model_type must be "qwen3_5" for the conditional image wrapper, got "${modelType}".`,
    );
  }

  const textConfig = parseQwen3_5TextConfigInternal(
    expectConfigRecord(config.text_config, "Qwen 3.5 config.text_config"),
    "Qwen 3.5 config.text_config",
    expectConfigRecord(config.text_config, "Qwen 3.5 config.text_config"),
  );
  const visionConfig = parseQwen3_5VisionConfigInternal(
    expectConfigRecord(config.vision_config, "Qwen 3.5 config.vision_config"),
    "Qwen 3.5 config.vision_config",
    expectConfigRecord(config.vision_config, "Qwen 3.5 config.vision_config"),
  );

  return {
    family: "qwen",
    modelType: "qwen3_5",
    rawConfig: config,
    vocabSize: textConfig.vocabSize,
    hiddenSize: textConfig.hiddenSize,
    numHiddenLayers: textConfig.numHiddenLayers,
    textConfig,
    visionConfig,
    imageTokenId: expectInteger(config, "image_token_id", "Qwen 3.5 config"),
    videoTokenId: expectInteger(config, "video_token_id", "Qwen 3.5 config"),
    visionStartTokenId: expectInteger(config, "vision_start_token_id", "Qwen 3.5 config"),
    visionEndTokenId: expectInteger(config, "vision_end_token_id", "Qwen 3.5 config"),
    tieWordEmbeddings: optionalBoolean(config, "tie_word_embeddings", "Qwen 3.5 config") ?? false,
    languageModelOnly: optionalBoolean(config, "language_model_only", "Qwen 3.5 config") ?? false,
  };
}

export function parseQwen3_5CausalLMConfig(rawConfig: Record<string, unknown>): Qwen3_5TextConfig {
  const config = expectConfigRecord(rawConfig, "Qwen 3.5 config");
  parseQwen3_5ModelType(config, "Qwen 3.5 config");

  const textConfig = expectConfigRecord(config.text_config, "Qwen 3.5 config.text_config");
  return parseQwen3_5TextConfigInternal(textConfig, "Qwen 3.5 config.text_config", config);
}

export const qwen3_5TextFamily: FamilyRegistration<Qwen3_5TextConfig> = {
  family: "qwen",
  modelTypes: ["qwen3_5_text", "qwen3_5_moe_text"],
  parseConfig: parseQwen3_5TextConfig,
  createModel: (config) => new Qwen3_5TextCausalLM(config),
  sanitizeWeight: sanitizeQwen3_5TextWeight,
  isIgnoredWeight: isIgnoredQwen3_5TextWeight,
  createCheckpointTensorTransform: async ({ snapshot }) => {
    const style = await detectQwen3_5CheckpointStyle(snapshot);
    return (_checkpointName, weightPath, tensor) =>
      transformQwen3_5CheckpointTensor(style, weightPath, tensor);
  },
};

export const qwen3_5Family: FamilyRegistration<Qwen3_5TextConfig> = {
  family: "qwen",
  modelTypes: ["qwen3_5", "qwen3_5_moe"],
  parseConfig: parseQwen3_5CausalLMConfig,
  createModel: (config) => new Qwen3_5TextCausalLM(config),
  sanitizeWeight: sanitizeQwen3_5CausalLMWeight,
  isIgnoredWeight: isIgnoredQwen3_5CausalLMWeight,
  createCheckpointTensorTransform: async ({ snapshot }) => {
    const style = await detectQwen3_5CheckpointStyle(snapshot);
    return (_checkpointName, weightPath, tensor) =>
      transformQwen3_5CheckpointTensor(style, weightPath, tensor);
  },
};

export const qwen3_5ConditionalFamily: FamilyRegistration<Qwen3_5Config> = {
  family: "qwen",
  modelTypes: ["qwen3_5"],
  parseConfig: parseQwen3_5Config,
  createModel: (config) => new Qwen3_5ForConditionalGeneration(config),
  sanitizeWeight: sanitizeQwen3_5Weight,
  isIgnoredWeight: isIgnoredQwen3_5Weight,
  createCheckpointTensorTransform: async ({ snapshot }) => {
    const style = await detectQwen3_5CheckpointStyle(snapshot);
    return (_checkpointName, weightPath, tensor) =>
      transformQwen3_5CheckpointTensor(style, weightPath, tensor);
  },
};
