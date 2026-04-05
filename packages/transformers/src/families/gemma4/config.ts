/**
 * Gemma 4 dense text config parsing.
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
import { Gemma4TextCausalLM } from "./model";
import type { Gemma4LayerType, Gemma4TextConfig } from "./types";
import {
  exceptionalGemma4WeightNames,
  isIgnoredGemma4TextWeight,
  isIgnoredGemma4Weight,
  loadExceptionalGemma4Weights,
  sanitizeGemma4TextWeight,
  sanitizeGemma4Weight,
} from "./weights";

function parseLayerTypes(
  config: Record<string, unknown>,
  numHiddenLayers: number,
  context: string,
): Gemma4LayerType[] {
  const rawLayerTypes = config.layer_types;
  if (!Array.isArray(rawLayerTypes) || rawLayerTypes.length !== numHiddenLayers) {
    throw new ConfigParseError(
      `${context}.layer_types must be an array with ${numHiddenLayers} entries.`,
    );
  }

  return rawLayerTypes.map((entry, layerIndex) => {
    if (entry !== "full_attention" && entry !== "sliding_attention") {
      throw new ConfigParseError(
        `${context}.layer_types[${layerIndex}] must be "full_attention" or "sliding_attention".`,
      );
    }
    return entry;
  });
}

function parseRopeParameters(
  config: Record<string, unknown>,
  context: string,
): { slidingRopeTheta: number; fullRopeTheta: number; fullRotaryDimensions: number } {
  const ropeParameters = expectConfigRecord(config.rope_parameters, `${context}.rope_parameters`);
  const slidingAttention = expectConfigRecord(
    ropeParameters.sliding_attention,
    `${context}.rope_parameters.sliding_attention`,
  );
  const fullAttention = expectConfigRecord(
    ropeParameters.full_attention,
    `${context}.rope_parameters.full_attention`,
  );
  const ropeType =
    optionalString(fullAttention, "rope_type", `${context}.rope_parameters.full_attention`) ??
    "default";
  if (ropeType !== "proportional" && ropeType !== "default") {
    throw new ConfigParseError(
      `${context}.rope_parameters.full_attention.rope_type must be "default" or "proportional", got "${ropeType}".`,
    );
  }

  const globalHeadDim = expectInteger(config, "global_head_dim", context);
  const partialRotaryFactor =
    optionalNumber(
      fullAttention,
      "partial_rotary_factor",
      `${context}.rope_parameters.full_attention`,
    ) ?? 1.0;
  const fullRotaryDimensions = Math.floor(globalHeadDim * partialRotaryFactor);
  if (fullRotaryDimensions <= 0 || fullRotaryDimensions % 2 !== 0) {
    throw new ConfigParseError(
      `${context}.rope_parameters.full_attention.partial_rotary_factor yields ${fullRotaryDimensions} rotary dimensions, expected a positive even integer.`,
    );
  }

  return {
    slidingRopeTheta:
      optionalNumber(
        slidingAttention,
        "rope_theta",
        `${context}.rope_parameters.sliding_attention`,
      ) ?? 10_000,
    fullRopeTheta:
      optionalNumber(fullAttention, "rope_theta", `${context}.rope_parameters.full_attention`) ??
      1_000_000,
    fullRotaryDimensions,
  };
}

function parseHiddenActivation(config: Record<string, unknown>, context: string): void {
  const hiddenActivation =
    optionalString(config, "hidden_activation", context) ??
    optionalString(config, "hidden_act", context) ??
    "gelu_pytorch_tanh";
  if (hiddenActivation !== "gelu_pytorch_tanh") {
    throw new ConfigParseError(
      `${context}.hidden_activation must be "gelu_pytorch_tanh" for the supported dense Gemma 4 path, got "${hiddenActivation}".`,
    );
  }
}

function assertDensePhase7Scope(config: Record<string, unknown>, context: string): void {
  const enableMoeBlock = optionalBoolean(config, "enable_moe_block", context) ?? false;
  if (enableMoeBlock) {
    throw new ConfigParseError(`${context}.enable_moe_block=true is Phase 7e work, not Phase 7.`);
  }

  const useBidirectionalAttention = config.use_bidirectional_attention;
  if (useBidirectionalAttention !== undefined && useBidirectionalAttention !== null) {
    throw new ConfigParseError(
      `${context}.use_bidirectional_attention is multimodal Gemma 4 behavior and is out of Phase 7 scope.`,
    );
  }
}

function parseNumKvSharedLayers(
  config: Record<string, unknown>,
  numHiddenLayers: number,
  context: string,
): number {
  const numKvSharedLayers = optionalInteger(config, "num_kv_shared_layers", context) ?? 0;
  if (numKvSharedLayers < 0 || numKvSharedLayers >= numHiddenLayers) {
    throw new ConfigParseError(
      `${context}.num_kv_shared_layers must be in the range [0, ${numHiddenLayers - 1}], got ${numKvSharedLayers}.`,
    );
  }
  return numKvSharedLayers;
}

function optionalIntegerOrNull(
  config: Record<string, unknown>,
  key: string,
  context: string,
): number | null {
  const value = config[key];
  if (value === undefined || value === null) {
    return null;
  }
  if (typeof value !== "number" || !Number.isInteger(value)) {
    throw new ConfigParseError(
      `${context}.${key} must be an integer when present, got ${String(value)}.`,
    );
  }
  return value;
}

function parseGemma4CoreConfig(config: Record<string, unknown>, context: string) {
  parseHiddenActivation(config, context);
  assertDensePhase7Scope(config, context);

  const hiddenSize = expectInteger(config, "hidden_size", context);
  const numAttentionHeads = expectInteger(config, "num_attention_heads", context);
  const numHiddenLayers = expectInteger(config, "num_hidden_layers", context);
  const numKeyValueHeads =
    optionalInteger(config, "num_key_value_heads", context) ?? numAttentionHeads;
  const layerTypes = parseLayerTypes(config, numHiddenLayers, context);

  return {
    hiddenSize,
    numAttentionHeads,
    numHiddenLayers,
    numKeyValueHeads,
    layerTypes,
    ropeParameters: parseRopeParameters(config, context),
    numKvSharedLayers: parseNumKvSharedLayers(config, numHiddenLayers, context),
  };
}

function parseGemma4TextConfigInternal(
  rawConfig: Record<string, unknown>,
  context: string,
  modelType: Gemma4TextConfig["modelType"],
  sourceConfig: Record<string, unknown>,
): Gemma4TextConfig {
  const config = expectConfigRecord(rawConfig, context);
  const core = parseGemma4CoreConfig(config, context);

  return {
    family: "gemma",
    modelType,
    rawConfig: sourceConfig,
    vocabSize: expectInteger(config, "vocab_size", context),
    vocabSizePerLayerInput:
      optionalInteger(config, "vocab_size_per_layer_input", context) ??
      expectInteger(config, "vocab_size", context),
    hiddenSize: core.hiddenSize,
    intermediateSize: expectInteger(config, "intermediate_size", context),
    numHiddenLayers: core.numHiddenLayers,
    numAttentionHeads: core.numAttentionHeads,
    numKeyValueHeads: core.numKeyValueHeads,
    numGlobalKeyValueHeads: optionalIntegerOrNull(config, "num_global_key_value_heads", context),
    headDim:
      optionalInteger(config, "head_dim", context) ??
      Math.floor(core.hiddenSize / core.numAttentionHeads),
    globalHeadDim: expectInteger(config, "global_head_dim", context),
    maxPositionEmbeddings: expectInteger(config, "max_position_embeddings", context),
    slidingWindow: expectInteger(config, "sliding_window", context),
    layerTypes: core.layerTypes,
    rmsNormEps: optionalNumber(config, "rms_norm_eps", context) ?? 1e-6,
    attentionBias: optionalBoolean(config, "attention_bias", context) ?? false,
    tieWordEmbeddings: optionalBoolean(config, "tie_word_embeddings", context) ?? true,
    hiddenSizePerLayerInput: optionalInteger(config, "hidden_size_per_layer_input", context) ?? 0,
    useDoubleWideMLP: optionalBoolean(config, "use_double_wide_mlp", context) ?? false,
    attentionKEqV: optionalBoolean(config, "attention_k_eq_v", context) ?? false,
    numKvSharedLayers: core.numKvSharedLayers,
    slidingRopeTheta: core.ropeParameters.slidingRopeTheta,
    fullRopeTheta: core.ropeParameters.fullRopeTheta,
    fullRotaryDimensions: core.ropeParameters.fullRotaryDimensions,
    finalLogitSoftcapping: optionalNumber(config, "final_logit_softcapping", context) ?? null,
    embeddingScale: Math.sqrt(core.hiddenSize),
  };
}

export function parseGemma4TextConfig(rawConfig: Record<string, unknown>): Gemma4TextConfig {
  const config = expectConfigRecord(rawConfig, "Gemma 4 text config");
  const modelType = expectString(config, "model_type", "Gemma 4 text config");
  if (modelType !== "gemma4_text") {
    throw new Error(`Gemma 4 text config.model_type must be "gemma4_text", got "${modelType}".`);
  }
  return parseGemma4TextConfigInternal(config, "Gemma 4 text config", "gemma4_text", config);
}

export function parseGemma4Config(rawConfig: Record<string, unknown>): Gemma4TextConfig {
  const config = expectConfigRecord(rawConfig, "Gemma 4 config");
  const modelType = expectString(config, "model_type", "Gemma 4 config");
  if (modelType !== "gemma4") {
    throw new Error(`Gemma 4 config.model_type must be "gemma4", got "${modelType}".`);
  }

  const textConfig = expectConfigRecord(config.text_config, "Gemma 4 config.text_config");
  const textModelType = expectString(textConfig, "model_type", "Gemma 4 config.text_config");
  if (textModelType !== "gemma4_text") {
    throw new ConfigParseError(
      `Gemma 4 config.text_config.model_type must be "gemma4_text" for the Phase 7 dense text path, got "${textModelType}".`,
    );
  }

  return parseGemma4TextConfigInternal(textConfig, "Gemma 4 config.text_config", "gemma4", config);
}

export const gemma4TextFamily: FamilyRegistration<Gemma4TextConfig> = {
  family: "gemma",
  modelTypes: ["gemma4_text"],
  parseConfig: parseGemma4TextConfig,
  createModel: (config) => new Gemma4TextCausalLM(config),
  sanitizeWeight: sanitizeGemma4TextWeight,
  isIgnoredWeight: isIgnoredGemma4TextWeight,
  exceptionalWeightNames: exceptionalGemma4WeightNames,
  loadExceptionalWeights: loadExceptionalGemma4Weights,
};

export const gemma4Family: FamilyRegistration<Gemma4TextConfig> = {
  family: "gemma",
  modelTypes: ["gemma4"],
  parseConfig: parseGemma4Config,
  createModel: (config) => new Gemma4TextCausalLM(config),
  sanitizeWeight: sanitizeGemma4Weight,
  isIgnoredWeight: isIgnoredGemma4Weight,
  exceptionalWeightNames: exceptionalGemma4WeightNames,
  loadExceptionalWeights: loadExceptionalGemma4Weights,
};
