/**
 * Shared config and weight-name mapping for LLaMA-like decoder families.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import type { Module } from "@mlxts/nn";

import type { BaseModelConfig, SupportedModelFamily } from "../../types";

export type LlamaLikeActivation = "swiglu" | "gelu_pytorch_tanh";
export type AttentionProjectionLayout = "split" | "packed_qkv";
export type MlpProjectionLayout = "split" | "packed_gate_up";

export interface ForwardModule extends Module {
  forward(x: MxArray): MxArray;
}

export type LlamaLikeConfig = BaseModelConfig & {
  family: SupportedModelFamily;
  vocabSize: number;
  hiddenSize: number;
  intermediateSize: number;
  numHiddenLayers: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  headDim: number;
  maxPositionEmbeddings: number;
  ropeTheta: number;
  rmsNormEps: number;
  tieWordEmbeddings: boolean;
  attentionBias: boolean;
  slidingWindow?: number;
  embeddingScale?: number;
  normWeightOffset?: boolean;
  rotaryDimensions?: number;
  attentionProjectionLayout?: AttentionProjectionLayout;
  mlpProjectionLayout?: MlpProjectionLayout;
  mlpActivation: LlamaLikeActivation;
};

function layerPath(layerIndex: string, suffix: readonly string[]): string {
  return ["model", "layers", layerIndex, ...suffix].join(".");
}

export function sanitizeLlamaLikeWeight(
  config: LlamaLikeConfig,
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
    "self_attn.o_proj.weight": layerPath(layerIndex, [
      "selfAttention",
      "outputProjection",
      "weight",
    ]),
    "self_attn.o_proj.bias": biasPath("outputProjection"),
    "mlp.down_proj.weight": layerPath(layerIndex, ["mlp", "downProjection", "weight"]),
  };

  if (config.attentionProjectionLayout === "packed_qkv") {
    mapping["self_attn.qkv_proj.weight"] = layerPath(layerIndex, [
      "selfAttention",
      "qkvProjection",
      "weight",
    ]);
    mapping["self_attn.qkv_proj.bias"] = biasPath("qkvProjection");
  } else {
    mapping["self_attn.q_proj.weight"] = layerPath(layerIndex, [
      "selfAttention",
      "qProjection",
      "weight",
    ]);
    mapping["self_attn.q_proj.bias"] = biasPath("qProjection");
    mapping["self_attn.k_proj.weight"] = layerPath(layerIndex, [
      "selfAttention",
      "kProjection",
      "weight",
    ]);
    mapping["self_attn.k_proj.bias"] = biasPath("kProjection");
    mapping["self_attn.v_proj.weight"] = layerPath(layerIndex, [
      "selfAttention",
      "vProjection",
      "weight",
    ]);
    mapping["self_attn.v_proj.bias"] = biasPath("vProjection");
  }

  if (config.mlpProjectionLayout === "packed_gate_up") {
    mapping["mlp.gate_up_proj.weight"] = layerPath(layerIndex, [
      "mlp",
      "gateUpProjection",
      "weight",
    ]);
  } else {
    mapping["mlp.gate_proj.weight"] = layerPath(layerIndex, ["mlp", "gateProjection", "weight"]);
    mapping["mlp.up_proj.weight"] = layerPath(layerIndex, ["mlp", "upProjection", "weight"]);
  }

  return mapping[suffix] ?? null;
}

export function isIgnoredLlamaLikeWeight(config: LlamaLikeConfig, checkpointName: string): boolean {
  return (
    checkpointName.endsWith("rotary_emb.inv_freq") ||
    (config.tieWordEmbeddings && checkpointName === "lm_head.weight")
  );
}
