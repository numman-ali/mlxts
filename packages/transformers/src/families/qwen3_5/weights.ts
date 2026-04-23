/**
 * Weight-name mapping for Qwen 3.5 text and multimodal checkpoints.
 * @module
 */

import { add, inspectSafetensors, type MxArray, transpose } from "@mlxts/core";

import type { ResolvedSnapshot } from "../../pretrained/types";
import type { Qwen3_5Config, Qwen3_5TextConfig } from "./types";

const LANGUAGE_MODEL_PREFIXES = ["language_model.", "model.language_model."] as const;
const VISION_PREFIXES = ["vision_tower.", "model.visual."] as const;

export type Qwen3_5CheckpointStyle = "raw-hf" | "mlx-converted";

function joinPath(parts: readonly string[]): string {
  return parts.join(".");
}

function textLayerPath(
  treePrefix: readonly string[],
  layerIndex: string,
  suffix: readonly string[],
): string {
  return joinPath([...treePrefix, "layers", layerIndex, ...suffix]);
}

function standaloneTextRootWeightPath(
  config: Qwen3_5TextConfig,
  checkpointName: string,
): string | null | undefined {
  const mapping: Record<string, string | null> = {
    "model.embed_tokens.weight": "model.embedTokens.weight",
    "model.norm.weight": "model.norm.weight",
    "lm_head.weight": config.tieWordEmbeddings ? null : "lmHead.weight",
  };
  return mapping[checkpointName];
}

function wrapperTextRootWeightPath(
  config: Qwen3_5Config,
  checkpointName: string,
): string | null | undefined {
  const mapping: Record<string, string | null> = {
    "embed_tokens.weight": "model.languageModel.embedTokens.weight",
    "norm.weight": "model.languageModel.norm.weight",
    "lm_head.weight": config.tieWordEmbeddings ? null : "lmHead.weight",
  };
  return mapping[checkpointName];
}

function normalizeWrapperTextCheckpointName(checkpointName: string): string {
  return checkpointName.startsWith("model.")
    ? checkpointName.slice("model.".length)
    : checkpointName;
}

function stripPrefix(checkpointName: string, prefixes: readonly string[]): string | null {
  for (const prefix of prefixes) {
    if (checkpointName.startsWith(prefix)) {
      return checkpointName.slice(prefix.length);
    }
  }
  return null;
}

function isZeroCenteredQwen3_5NormPath(weightPath: string): boolean {
  return (
    weightPath === "model.norm.weight" ||
    weightPath === "model.languageModel.norm.weight" ||
    weightPath.endsWith(".inputLayerNorm.weight") ||
    weightPath.endsWith(".postAttentionLayerNorm.weight") ||
    weightPath.endsWith(".qNorm.weight") ||
    weightPath.endsWith(".kNorm.weight")
  );
}

function isQwen3_5Conv1dPath(weightPath: string): boolean {
  return weightPath.endsWith(".linearAttention.conv1d.weight");
}

function conv1dWeightShapeStyle(shape: readonly number[]): Qwen3_5CheckpointStyle | null {
  const [outputChannels, secondDimension, thirdDimension] = shape;
  if (
    outputChannels === undefined ||
    secondDimension === undefined ||
    thirdDimension === undefined ||
    shape.length !== 3
  ) {
    return null;
  }
  if (thirdDimension === 1 && secondDimension > 1) {
    return "mlx-converted";
  }
  if (secondDimension === 1 && thirdDimension > 1) {
    return "raw-hf";
  }
  return null;
}

export async function detectQwen3_5CheckpointStyle(
  snapshot: ResolvedSnapshot,
): Promise<Qwen3_5CheckpointStyle | null> {
  for (const file of snapshot.files) {
    if (!file.localPath.endsWith(".safetensors")) {
      continue;
    }
    const inspection = await inspectSafetensors(file.localPath);
    for (const tensor of inspection.tensors) {
      if (!tensor.name.endsWith("conv1d.weight")) {
        continue;
      }
      const style = conv1dWeightShapeStyle(tensor.shape);
      if (style !== null) {
        return style;
      }
    }
  }
  return null;
}

export function transformQwen3_5CheckpointTensor(
  style: Qwen3_5CheckpointStyle | null,
  weightPath: string,
  tensor: MxArray,
): MxArray {
  if (style === "raw-hf" && isQwen3_5Conv1dPath(weightPath)) {
    return transpose(tensor, [0, 2, 1]);
  }
  if (style === "raw-hf" && isZeroCenteredQwen3_5NormPath(weightPath)) {
    return add(tensor, 1);
  }
  return tensor;
}

function textLayerWeightPath(
  config: Qwen3_5TextConfig,
  treePrefix: readonly string[],
  layerIndexText: string,
  suffix: string,
): string | null {
  const attentionBiasPath = (projection: string) =>
    config.attentionBias
      ? textLayerPath(treePrefix, layerIndexText, ["selfAttention", projection, "bias"])
      : null;

  const mapping: Record<string, string | null> = {
    "input_layernorm.weight": textLayerPath(treePrefix, layerIndexText, [
      "inputLayerNorm",
      "weight",
    ]),
    "post_attention_layernorm.weight": textLayerPath(treePrefix, layerIndexText, [
      "postAttentionLayerNorm",
      "weight",
    ]),
    "self_attn.q_proj.weight": textLayerPath(treePrefix, layerIndexText, [
      "selfAttention",
      "qProjection",
      "weight",
    ]),
    "self_attn.q_proj.bias": attentionBiasPath("qProjection"),
    "self_attn.k_proj.weight": textLayerPath(treePrefix, layerIndexText, [
      "selfAttention",
      "kProjection",
      "weight",
    ]),
    "self_attn.k_proj.bias": attentionBiasPath("kProjection"),
    "self_attn.v_proj.weight": textLayerPath(treePrefix, layerIndexText, [
      "selfAttention",
      "vProjection",
      "weight",
    ]),
    "self_attn.v_proj.bias": attentionBiasPath("vProjection"),
    "self_attn.o_proj.weight": textLayerPath(treePrefix, layerIndexText, [
      "selfAttention",
      "outputProjection",
      "weight",
    ]),
    "self_attn.o_proj.bias": attentionBiasPath("outputProjection"),
    "self_attn.q_norm.weight": textLayerPath(treePrefix, layerIndexText, [
      "selfAttention",
      "qNorm",
      "weight",
    ]),
    "self_attn.k_norm.weight": textLayerPath(treePrefix, layerIndexText, [
      "selfAttention",
      "kNorm",
      "weight",
    ]),
    "linear_attn.in_proj_qkv.weight": textLayerPath(treePrefix, layerIndexText, [
      "linearAttention",
      "inProjectionQkv",
      "weight",
    ]),
    "linear_attn.in_proj_z.weight": textLayerPath(treePrefix, layerIndexText, [
      "linearAttention",
      "inProjectionZ",
      "weight",
    ]),
    "linear_attn.in_proj_b.weight": textLayerPath(treePrefix, layerIndexText, [
      "linearAttention",
      "inProjectionB",
      "weight",
    ]),
    "linear_attn.in_proj_a.weight": textLayerPath(treePrefix, layerIndexText, [
      "linearAttention",
      "inProjectionA",
      "weight",
    ]),
    "linear_attn.conv1d.weight": textLayerPath(treePrefix, layerIndexText, [
      "linearAttention",
      "conv1d",
      "weight",
    ]),
    "linear_attn.dt_bias": textLayerPath(treePrefix, layerIndexText, ["linearAttention", "dtBias"]),
    "linear_attn.A_log": textLayerPath(treePrefix, layerIndexText, ["linearAttention", "aLog"]),
    "linear_attn.norm.weight": textLayerPath(treePrefix, layerIndexText, [
      "linearAttention",
      "norm",
      "weight",
    ]),
    "linear_attn.out_proj.weight": textLayerPath(treePrefix, layerIndexText, [
      "linearAttention",
      "outProjection",
      "weight",
    ]),
    "mlp.gate_proj.weight": textLayerPath(treePrefix, layerIndexText, [
      "mlp",
      "gateProjection",
      "weight",
    ]),
    "mlp.up_proj.weight": textLayerPath(treePrefix, layerIndexText, [
      "mlp",
      "upProjection",
      "weight",
    ]),
    "mlp.down_proj.weight": textLayerPath(treePrefix, layerIndexText, [
      "mlp",
      "downProjection",
      "weight",
    ]),
  };
  return mapping[suffix] ?? null;
}

function sanitizeTextWeight(
  config: Qwen3_5TextConfig,
  checkpointName: string,
  rootPath: (checkpointName: string) => string | null | undefined,
  layerPrefix: RegExp,
  treePrefix: readonly string[],
): string | null {
  const directPath = rootPath(checkpointName);
  if (directPath !== undefined) {
    return directPath;
  }

  const layerMatch = checkpointName.match(layerPrefix);
  if (layerMatch === null) {
    return null;
  }

  const layerIndexText = layerMatch[1];
  const suffix = layerMatch[2];
  if (layerIndexText === undefined || suffix === undefined) {
    return null;
  }

  const layerIndex = Number(layerIndexText);
  if (!Number.isInteger(layerIndex) || layerIndex < 0 || layerIndex >= config.numHiddenLayers) {
    return null;
  }
  return textLayerWeightPath(config, treePrefix, layerIndexText, suffix);
}

function visionBlockPath(layerIndex: string, suffix: readonly string[]): string {
  return joinPath(["model", "visual", "blocks", layerIndex, ...suffix]);
}

function visionMergerPath(suffix: readonly string[]): string {
  return joinPath(["model", "visual", "merger", ...suffix]);
}

function sanitizeVisionWeight(config: Qwen3_5Config, checkpointName: string): string | null {
  const directMapping: Record<string, string> = {
    "patch_embed.proj.weight": "model.visual.patchEmbed.weight",
    "patch_embed.proj.bias": "model.visual.patchEmbed.bias",
    "pos_embed.weight": "model.visual.posEmbed.weight",
    "merger.norm.weight": visionMergerPath(["norm", "weight"]),
    "merger.norm.bias": visionMergerPath(["norm", "bias"]),
    "merger.linear_fc1.weight": visionMergerPath(["linearFc1", "weight"]),
    "merger.linear_fc1.bias": visionMergerPath(["linearFc1", "bias"]),
    "merger.linear_fc2.weight": visionMergerPath(["linearFc2", "weight"]),
    "merger.linear_fc2.bias": visionMergerPath(["linearFc2", "bias"]),
  };
  const directPath = directMapping[checkpointName];
  if (directPath !== undefined) {
    return directPath;
  }

  const blockMatch = checkpointName.match(/^blocks\.(\d+)\.(.+)$/);
  if (blockMatch === null) {
    return null;
  }

  const layerIndexText = blockMatch[1];
  const suffix = blockMatch[2];
  if (layerIndexText === undefined || suffix === undefined) {
    return null;
  }

  const layerIndex = Number(layerIndexText);
  if (!Number.isInteger(layerIndex) || layerIndex < 0 || layerIndex >= config.visionConfig.depth) {
    return null;
  }

  const blockMapping: Record<string, string> = {
    "norm1.weight": visionBlockPath(layerIndexText, ["norm1", "weight"]),
    "norm1.bias": visionBlockPath(layerIndexText, ["norm1", "bias"]),
    "norm2.weight": visionBlockPath(layerIndexText, ["norm2", "weight"]),
    "norm2.bias": visionBlockPath(layerIndexText, ["norm2", "bias"]),
    "attn.qkv.weight": visionBlockPath(layerIndexText, ["attention", "qkv", "weight"]),
    "attn.qkv.bias": visionBlockPath(layerIndexText, ["attention", "qkv", "bias"]),
    "attn.proj.weight": visionBlockPath(layerIndexText, ["attention", "proj", "weight"]),
    "attn.proj.bias": visionBlockPath(layerIndexText, ["attention", "proj", "bias"]),
    "mlp.linear_fc1.weight": visionBlockPath(layerIndexText, ["mlp", "linearFc1", "weight"]),
    "mlp.linear_fc1.bias": visionBlockPath(layerIndexText, ["mlp", "linearFc1", "bias"]),
    "mlp.linear_fc2.weight": visionBlockPath(layerIndexText, ["mlp", "linearFc2", "weight"]),
    "mlp.linear_fc2.bias": visionBlockPath(layerIndexText, ["mlp", "linearFc2", "bias"]),
  };
  return blockMapping[suffix] ?? null;
}

export function sanitizeQwen3_5TextWeight(
  config: Qwen3_5TextConfig,
  checkpointName: string,
): string | null {
  return sanitizeTextWeight(
    config,
    checkpointName,
    (name) => standaloneTextRootWeightPath(config, name),
    /^model\.layers\.(\d+)\.(.+)$/,
    ["model"],
  );
}

export function sanitizeQwen3_5Weight(
  config: Qwen3_5Config,
  checkpointName: string,
): string | null {
  if (checkpointName === "lm_head.weight") {
    return config.tieWordEmbeddings ? null : "lmHead.weight";
  }

  const languageModelName = stripPrefix(checkpointName, LANGUAGE_MODEL_PREFIXES);
  if (languageModelName !== null) {
    return sanitizeTextWeight(
      config.textConfig,
      normalizeWrapperTextCheckpointName(languageModelName),
      (name) => wrapperTextRootWeightPath(config, name),
      /^layers\.(\d+)\.(.+)$/,
      ["model", "languageModel"],
    );
  }

  const visionName = stripPrefix(checkpointName, VISION_PREFIXES);
  if (visionName !== null) {
    return sanitizeVisionWeight(config, visionName);
  }

  return null;
}

export function isIgnoredQwen3_5TextWeight(
  config: Qwen3_5TextConfig,
  checkpointName: string,
): boolean {
  return (
    checkpointName.endsWith("rotary_emb.inv_freq") ||
    checkpointName.startsWith("mtp.") ||
    (config.tieWordEmbeddings && checkpointName === "lm_head.weight")
  );
}

export function isIgnoredQwen3_5Weight(config: Qwen3_5Config, checkpointName: string): boolean {
  return (
    checkpointName.endsWith("rotary_emb.inv_freq") ||
    checkpointName.endsWith("rotary_pos_emb.inv_freq") ||
    checkpointName.startsWith("mtp.") ||
    (config.tieWordEmbeddings && checkpointName === "lm_head.weight")
  );
}
