/**
 * Shared config and weight-name mapping for Gemma 3 text decoders.
 * @module
 */

import type { BaseModelConfig } from "../../types";

export type Gemma3LayerType = "full_attention" | "sliding_attention";

export type Gemma3TextConfig = BaseModelConfig & {
  family: "gemma";
  modelType: "gemma3_text";
  vocabSize: number;
  hiddenSize: number;
  intermediateSize: number;
  numHiddenLayers: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  maxPositionEmbeddings: number;
  ropeTheta: number;
  ropeLocalBaseFreq: number;
  rmsNormEps: number;
  tieWordEmbeddings: boolean;
  attentionBias: boolean;
  queryPreAttentionScalar: number;
  slidingWindow: number;
  layerTypes: Gemma3LayerType[];
  embeddingScale: number;
};

function layerPath(layerIndex: string, suffix: readonly string[]): string {
  return ["model", "layers", layerIndex, ...suffix].join(".");
}

export function sanitizeGemma3TextWeight(
  config: Gemma3TextConfig,
  checkpointName: string,
): string | null {
  if (checkpointName === "model.embed_tokens.weight") {
    return "model.embedTokens.weight";
  }
  if (checkpointName === "model.norm.weight") {
    return "model.norm.weight";
  }
  if (checkpointName === "lm_head.weight") {
    return config.tieWordEmbeddings ? null : "lmHead.weight";
  }

  const layerMatch = checkpointName.match(/^model\.layers\.(\d+)\.(.+)$/);
  if (layerMatch === null) {
    return null;
  }

  const layerIndex = layerMatch[1];
  const suffix = layerMatch[2];
  if (layerIndex === undefined || suffix === undefined) {
    return null;
  }

  const biasPath = (projection: string) =>
    config.attentionBias ? layerPath(layerIndex, ["selfAttention", projection, "bias"]) : null;

  const mapping: Record<string, string | null> = {
    "input_layernorm.weight": layerPath(layerIndex, ["inputLayerNorm", "weight"]),
    "post_attention_layernorm.weight": layerPath(layerIndex, ["postAttentionLayerNorm", "weight"]),
    "pre_feedforward_layernorm.weight": layerPath(layerIndex, [
      "preFeedforwardLayerNorm",
      "weight",
    ]),
    "post_feedforward_layernorm.weight": layerPath(layerIndex, [
      "postFeedforwardLayerNorm",
      "weight",
    ]),
    "self_attn.q_proj.weight": layerPath(layerIndex, ["selfAttention", "qProjection", "weight"]),
    "self_attn.q_proj.bias": biasPath("qProjection"),
    "self_attn.k_proj.weight": layerPath(layerIndex, ["selfAttention", "kProjection", "weight"]),
    "self_attn.k_proj.bias": biasPath("kProjection"),
    "self_attn.v_proj.weight": layerPath(layerIndex, ["selfAttention", "vProjection", "weight"]),
    "self_attn.v_proj.bias": biasPath("vProjection"),
    "self_attn.o_proj.weight": layerPath(layerIndex, [
      "selfAttention",
      "outputProjection",
      "weight",
    ]),
    "self_attn.o_proj.bias": biasPath("outputProjection"),
    "self_attn.q_norm.weight": layerPath(layerIndex, ["selfAttention", "qNorm", "weight"]),
    "self_attn.k_norm.weight": layerPath(layerIndex, ["selfAttention", "kNorm", "weight"]),
    "mlp.gate_proj.weight": layerPath(layerIndex, ["mlp", "gateProjection", "weight"]),
    "mlp.up_proj.weight": layerPath(layerIndex, ["mlp", "upProjection", "weight"]),
    "mlp.down_proj.weight": layerPath(layerIndex, ["mlp", "downProjection", "weight"]),
  };

  return mapping[suffix] ?? null;
}

export function isIgnoredGemma3TextWeight(
  config: Gemma3TextConfig,
  checkpointName: string,
): boolean {
  return (
    checkpointName.endsWith("rotary_emb.inv_freq") ||
    (config.tieWordEmbeddings && checkpointName === "lm_head.weight")
  );
}
