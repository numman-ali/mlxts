/**
 * Gemma 3 text config parsing.
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
import { Gemma3TextCausalLM } from "./model";
import type { Gemma3LayerType, Gemma3TextConfig } from "./types";
import { isIgnoredGemma3Weight, sanitizeGemma3Weight } from "./weights";

function parseLayerTypes(
  config: Record<string, unknown>,
  numHiddenLayers: number,
  context: string,
): Gemma3LayerType[] {
  const rawLayerTypes = config.layer_types;
  if (rawLayerTypes === undefined) {
    const slidingWindowPattern = optionalInteger(config, "sliding_window_pattern", context) ?? 6;
    return Array.from({ length: numHiddenLayers }, (_, layerIndex) =>
      (layerIndex + 1) % slidingWindowPattern === 0 ? "full_attention" : "sliding_attention",
    );
  }

  if (!Array.isArray(rawLayerTypes) || rawLayerTypes.length !== numHiddenLayers) {
    throw new ConfigParseError(
      `${context}.layer_types must be an array with ${numHiddenLayers} entries when present.`,
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

function parseRopeTheta(
  config: Record<string, unknown>,
  context: string,
): { ropeTheta: number; ropeLocalBaseFreq: number } {
  const ropeParameters = config.rope_parameters;
  if (ropeParameters === undefined) {
    return {
      ropeTheta: optionalNumber(config, "rope_theta", context) ?? 1_000_000,
      ropeLocalBaseFreq: optionalNumber(config, "rope_local_base_freq", context) ?? 10_000,
    };
  }

  const parameters = expectConfigRecord(ropeParameters, `${context}.rope_parameters`);
  const fullAttention = expectConfigRecord(
    parameters.full_attention,
    `${context}.rope_parameters.full_attention`,
  );
  const slidingAttention = expectConfigRecord(
    parameters.sliding_attention,
    `${context}.rope_parameters.sliding_attention`,
  );
  return {
    ropeTheta:
      optionalNumber(fullAttention, "rope_theta", `${context}.rope_parameters.full_attention`) ??
      1_000_000,
    ropeLocalBaseFreq:
      optionalNumber(
        slidingAttention,
        "rope_theta",
        `${context}.rope_parameters.sliding_attention`,
      ) ?? 10_000,
  };
}

export function parseGemma3TextConfig(rawConfig: Record<string, unknown>): Gemma3TextConfig {
  const config = expectConfigRecord(rawConfig, "Gemma 3 text config");
  const modelType = expectString(config, "model_type", "Gemma 3 text config");
  if (modelType === "gemma3") {
    return parseGemma3TextConfig(
      expectConfigRecord(config.text_config, "Gemma 3 text config.text_config"),
    );
  }
  if (modelType !== "gemma3_text") {
    throw new Error(`Gemma 3 text config.model_type must be "gemma3_text", got "${modelType}".`);
  }

  const hiddenSize = expectInteger(config, "hidden_size", "Gemma 3 text config");
  const numAttentionHeads = expectInteger(config, "num_attention_heads", "Gemma 3 text config");
  const numHiddenLayers = expectInteger(config, "num_hidden_layers", "Gemma 3 text config");
  const numKeyValueHeads =
    optionalInteger(config, "num_key_value_heads", "Gemma 3 text config") ?? numAttentionHeads;
  const headDim =
    optionalInteger(config, "head_dim", "Gemma 3 text config") ??
    Math.floor(hiddenSize / numAttentionHeads);
  const hiddenActivation =
    optionalString(config, "hidden_activation", "Gemma 3 text config") ??
    optionalString(config, "hidden_act", "Gemma 3 text config") ??
    "gelu_pytorch_tanh";
  if (hiddenActivation !== "gelu_pytorch_tanh") {
    throw new ConfigParseError(
      `Gemma 3 text config hidden activation must be "gelu_pytorch_tanh", got "${hiddenActivation}".`,
    );
  }

  const layerTypes = parseLayerTypes(config, numHiddenLayers, "Gemma 3 text config");
  const hasSlidingLayers = layerTypes.includes("sliding_attention");
  const slidingWindow = optionalInteger(config, "sliding_window", "Gemma 3 text config") ?? 4096;
  if (hasSlidingLayers && slidingWindow <= 0) {
    throw new ConfigParseError("Gemma 3 text config.sliding_window must be positive.");
  }
  const ropeTheta = parseRopeTheta(config, "Gemma 3 text config");

  return {
    family: "gemma",
    modelType: "gemma3_text",
    rawConfig: config,
    vocabSize: expectInteger(config, "vocab_size", "Gemma 3 text config"),
    hiddenSize,
    intermediateSize: expectInteger(config, "intermediate_size", "Gemma 3 text config"),
    numHiddenLayers,
    numAttentionHeads,
    numKeyValueHeads,
    headDim,
    maxPositionEmbeddings: expectInteger(config, "max_position_embeddings", "Gemma 3 text config"),
    ropeTheta: ropeTheta.ropeTheta,
    ropeLocalBaseFreq: ropeTheta.ropeLocalBaseFreq,
    rmsNormEps: optionalNumber(config, "rms_norm_eps", "Gemma 3 text config") ?? 1e-6,
    tieWordEmbeddings:
      optionalBoolean(config, "tie_word_embeddings", "Gemma 3 text config") ?? true,
    attentionBias: optionalBoolean(config, "attention_bias", "Gemma 3 text config") ?? false,
    queryPreAttentionScalar:
      optionalNumber(config, "query_pre_attn_scalar", "Gemma 3 text config") ?? headDim,
    slidingWindow,
    layerTypes,
    embeddingScale: Math.sqrt(hiddenSize),
  };
}

export const gemma3TextFamily: FamilyRegistration<Gemma3TextConfig> = {
  family: "gemma",
  modelTypes: ["gemma3", "gemma3_text"],
  parseConfig: parseGemma3TextConfig,
  createModel: (config) => new Gemma3TextCausalLM(config),
  sanitizeWeight: sanitizeGemma3Weight,
  isIgnoredWeight: isIgnoredGemma3Weight,
};
