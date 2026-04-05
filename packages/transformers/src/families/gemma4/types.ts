/**
 * Shared config and weight-name mapping for Gemma 4 dense text decoders.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import type { BaseModelConfig } from "../../types";

export type Gemma4LayerType = "full_attention" | "sliding_attention";

export type Gemma4TextModelType = "gemma4_text" | "gemma4";

export type Gemma4TextConfig = BaseModelConfig & {
  family: "gemma";
  modelType: Gemma4TextModelType;
  vocabSize: number;
  vocabSizePerLayerInput: number;
  hiddenSize: number;
  intermediateSize: number;
  numHiddenLayers: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  numGlobalKeyValueHeads: number | null;
  headDim: number;
  globalHeadDim: number;
  maxPositionEmbeddings: number;
  slidingWindow: number;
  layerTypes: Gemma4LayerType[];
  rmsNormEps: number;
  attentionBias: boolean;
  tieWordEmbeddings: boolean;
  hiddenSizePerLayerInput: number;
  useDoubleWideMLP: boolean;
  attentionKEqV: boolean;
  numKvSharedLayers: number;
  slidingRopeTheta: number;
  fullRopeTheta: number;
  fullRotaryDimensions: number;
  finalLogitSoftcapping: number | null;
  embeddingScale: number;
};

export type Gemma4SharedKeyValues = {
  keys: MxArray;
  values: MxArray;
};

function layerPath(layerIndex: string, suffix: readonly string[]): string {
  return ["model", "layers", layerIndex, ...suffix].join(".");
}

export function gemma4UsesAlternativeAttention(
  config: Gemma4TextConfig,
  layerIndex: number,
): boolean {
  return config.attentionKEqV && config.layerTypes[layerIndex] === "full_attention";
}

function hasPerLayerInputs(config: Gemma4TextConfig): boolean {
  return config.hiddenSizePerLayerInput > 0;
}

function rootWeightPath(
  config: Gemma4TextConfig,
  checkpointName: string,
): string | null | undefined {
  const perLayerInputs = hasPerLayerInputs(config);
  const mapping: Record<string, string | null> = {
    "model.embed_tokens.weight": "model.embedTokens.weight",
    "model.norm.weight": "model.norm.weight",
    "model.embed_tokens_per_layer.weight": perLayerInputs
      ? "model.embedTokensPerLayer.weight"
      : null,
    "model.per_layer_model_projection.weight": perLayerInputs
      ? "model.perLayerModelProjection.weight"
      : null,
    "model.per_layer_projection_norm.weight": perLayerInputs
      ? "model.perLayerProjectionNorm.weight"
      : null,
    "lm_head.weight": config.tieWordEmbeddings ? null : "lmHead.weight",
  };
  return mapping[checkpointName];
}

function layerMatchParts(
  checkpointName: string,
): { layerIndex: number; layerIndexText: string; suffix: string } | null {
  const layerMatch = checkpointName.match(/^model\.layers\.(\d+)\.(.+)$/);
  if (layerMatch === null) {
    return null;
  }

  const layerIndexText = layerMatch[1];
  const suffix = layerMatch[2];
  if (layerIndexText === undefined || suffix === undefined) {
    return null;
  }

  const layerIndex = Number(layerIndexText);
  if (!Number.isInteger(layerIndex) || layerIndex < 0) {
    return null;
  }

  return { layerIndex, layerIndexText, suffix };
}

function layerBiasPath(
  config: Gemma4TextConfig,
  layerIndexText: string,
  projection: string,
): string | null {
  return config.attentionBias
    ? layerPath(layerIndexText, ["selfAttention", projection, "bias"])
    : null;
}

function layerWeightPath(
  config: Gemma4TextConfig,
  layerIndex: number,
  layerIndexText: string,
  suffix: string,
): string | null {
  const perLayerInputs = hasPerLayerInputs(config);
  const usesAlternativeAttention = gemma4UsesAlternativeAttention(config, layerIndex);
  const mapping: Record<string, string | null> = {
    "input_layernorm.weight": layerPath(layerIndexText, ["inputLayerNorm", "weight"]),
    "post_attention_layernorm.weight": layerPath(layerIndexText, [
      "postAttentionLayerNorm",
      "weight",
    ]),
    "pre_feedforward_layernorm.weight": layerPath(layerIndexText, [
      "preFeedforwardLayerNorm",
      "weight",
    ]),
    "post_feedforward_layernorm.weight": layerPath(layerIndexText, [
      "postFeedforwardLayerNorm",
      "weight",
    ]),
    layer_scalar: layerPath(layerIndexText, ["layerScalar"]),
    "self_attn.q_proj.weight": layerPath(layerIndexText, [
      "selfAttention",
      "qProjection",
      "weight",
    ]),
    "self_attn.q_proj.bias": layerBiasPath(config, layerIndexText, "qProjection"),
    "self_attn.k_proj.weight": layerPath(layerIndexText, [
      "selfAttention",
      "kProjection",
      "weight",
    ]),
    "self_attn.k_proj.bias": layerBiasPath(config, layerIndexText, "kProjection"),
    "self_attn.v_proj.weight": usesAlternativeAttention
      ? null
      : layerPath(layerIndexText, ["selfAttention", "vProjection", "weight"]),
    "self_attn.v_proj.bias": usesAlternativeAttention
      ? null
      : layerBiasPath(config, layerIndexText, "vProjection"),
    "self_attn.o_proj.weight": layerPath(layerIndexText, [
      "selfAttention",
      "outputProjection",
      "weight",
    ]),
    "self_attn.o_proj.bias": layerBiasPath(config, layerIndexText, "outputProjection"),
    "self_attn.q_norm.weight": layerPath(layerIndexText, ["selfAttention", "qNorm", "weight"]),
    "self_attn.k_norm.weight": layerPath(layerIndexText, ["selfAttention", "kNorm", "weight"]),
    "mlp.gate_proj.weight": layerPath(layerIndexText, ["mlp", "gateProjection", "weight"]),
    "mlp.up_proj.weight": layerPath(layerIndexText, ["mlp", "upProjection", "weight"]),
    "mlp.down_proj.weight": layerPath(layerIndexText, ["mlp", "downProjection", "weight"]),
    "per_layer_input_gate.weight": perLayerInputs
      ? layerPath(layerIndexText, ["perLayerInputGate", "weight"])
      : null,
    "per_layer_projection.weight": perLayerInputs
      ? layerPath(layerIndexText, ["perLayerProjection", "weight"])
      : null,
    "post_per_layer_input_norm.weight": perLayerInputs
      ? layerPath(layerIndexText, ["postPerLayerInputNorm", "weight"])
      : null,
  };

  return mapping[suffix] ?? null;
}

export function sanitizeGemma4TextWeightName(
  config: Gemma4TextConfig,
  checkpointName: string,
): string | null {
  const rootPath = rootWeightPath(config, checkpointName);
  if (rootPath !== undefined) {
    return rootPath;
  }

  const parts = layerMatchParts(checkpointName);
  if (parts === null || parts.layerIndex >= config.numHiddenLayers) {
    return null;
  }

  return layerWeightPath(config, parts.layerIndex, parts.layerIndexText, parts.suffix);
}

export function isIgnoredGemma4TextWeightName(
  config: Gemma4TextConfig,
  checkpointName: string,
): boolean {
  return (
    checkpointName.endsWith("rotary_emb.inv_freq") ||
    (config.tieWordEmbeddings && checkpointName === "lm_head.weight") ||
    checkpointName.endsWith(".self_attn.v_norm.weight")
  );
}
