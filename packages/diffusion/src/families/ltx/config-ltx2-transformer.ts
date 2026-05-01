import { DiffusionConfigError } from "../../errors";
import {
  expectRecord,
  fieldName,
  optionalBoolean,
  optionalClassName,
  optionalExactString,
  optionalPositiveInteger,
  optionalPositiveNumber,
  rejectUnknownFields,
} from "../flux2/config-parsing";
import {
  optionalNullablePositiveInteger,
  optionalPositiveIntegerArray,
  optionalStringValue,
} from "./config-common";

export type Ltx2RopeType = "interleaved" | "split";
export type Ltx2QkNorm = "rms_norm_across_heads";

/** Package-native config for the LTX-2 `LTX2VideoTransformer3DModel`. */
export type Ltx2VideoTransformerConfig = {
  inChannels: number;
  outChannels: number;
  patchSize: number;
  patchSizeT: number;
  numAttentionHeads: number;
  attentionHeadDim: number;
  hiddenSize: number;
  crossAttentionDim: number;
  vaeScaleFactors: readonly [number, number, number];
  posEmbedMaxPos: number;
  baseHeight: number;
  baseWidth: number;
  audioInChannels: number;
  audioOutChannels: number;
  audioPatchSize: number;
  audioPatchSizeT: number;
  audioNumAttentionHeads: number;
  audioAttentionHeadDim: number;
  audioHiddenSize: number;
  audioCrossAttentionDim: number;
  audioScaleFactor: number;
  audioPosEmbedMaxPos: number;
  audioSamplingRate: number;
  audioHopLength: number;
  numLayers: number;
  activationFn: "gelu-approximate";
  qkNorm: Ltx2QkNorm;
  normElementwiseAffine: boolean;
  normEps: number;
  captionChannels: number;
  attentionBias: boolean;
  attentionOutBias: boolean;
  ropeTheta: number;
  ropeDoublePrecision: boolean;
  causalOffset: number;
  timestepScaleMultiplier: number;
  crossAttnTimestepScaleMultiplier: number;
  ropeType: Ltx2RopeType;
  gatedAttn: boolean;
  crossAttnMod: boolean;
  audioGatedAttn: boolean;
  audioCrossAttnMod: boolean;
  usePromptEmbeddings: boolean;
  perturbedAttn: boolean;
  rawConfig: Record<string, unknown>;
};

const ROPE_TYPES = new Set<Ltx2RopeType>(["interleaved", "split"]);
const QK_NORMS = new Set<Ltx2QkNorm>(["rms_norm_across_heads"]);

const LTX2_TRANSFORMER_KEYS = new Set([
  "_class_name",
  "_diffusers_version",
  "_name_or_path",
  "activation_fn",
  "attention_bias",
  "attention_head_dim",
  "attention_out_bias",
  "audio_attention_head_dim",
  "audio_cross_attention_dim",
  "audio_cross_attn_mod",
  "audio_gated_attn",
  "audio_hop_length",
  "audio_in_channels",
  "audio_num_attention_heads",
  "audio_out_channels",
  "audio_patch_size",
  "audio_patch_size_t",
  "audio_pos_embed_max_pos",
  "audio_sampling_rate",
  "audio_scale_factor",
  "base_height",
  "base_width",
  "caption_channels",
  "causal_offset",
  "cross_attention_dim",
  "cross_attn_mod",
  "cross_attn_timestep_scale_multiplier",
  "gated_attn",
  "in_channels",
  "norm_elementwise_affine",
  "norm_eps",
  "num_attention_heads",
  "num_layers",
  "out_channels",
  "patch_size",
  "patch_size_t",
  "perturbed_attn",
  "pos_embed_max_pos",
  "qk_norm",
  "rope_double_precision",
  "rope_theta",
  "rope_type",
  "timestep_scale_multiplier",
  "use_prompt_embeddings",
  "vae_scale_factors",
]);

function tuple3(values: readonly number[], context: string, key: string): [number, number, number] {
  const [first, second, third] = values;
  if (first === undefined || second === undefined || third === undefined) {
    throw new DiffusionConfigError(`${fieldName(context, key)} must contain 3 integers.`);
  }
  return [first, second, third];
}

/** Parse a Diffusers LTX-2 `LTX2VideoTransformer3DModel` config. */
export function parseLtx2VideoTransformerConfig(rawConfig: unknown): Ltx2VideoTransformerConfig {
  const context = "transformer/config.json";
  const record = expectRecord(rawConfig, context);
  rejectUnknownFields(record, LTX2_TRANSFORMER_KEYS, context);
  optionalClassName(record, "LTX2VideoTransformer3DModel", context);
  optionalExactString(record, "activation_fn", context, "gelu-approximate");

  const inChannels = optionalPositiveInteger(record, "in_channels", context, 128);
  const audioInChannels = optionalPositiveInteger(record, "audio_in_channels", context, 128);
  const attentionHeadDim = optionalPositiveInteger(record, "attention_head_dim", context, 128);
  const numAttentionHeads = optionalPositiveInteger(record, "num_attention_heads", context, 32);
  const audioAttentionHeadDim = optionalPositiveInteger(
    record,
    "audio_attention_head_dim",
    context,
    64,
  );
  const audioNumAttentionHeads = optionalPositiveInteger(
    record,
    "audio_num_attention_heads",
    context,
    32,
  );
  return {
    inChannels,
    outChannels:
      optionalNullablePositiveInteger(record, "out_channels", context, 128) ?? inChannels,
    patchSize: optionalPositiveInteger(record, "patch_size", context, 1),
    patchSizeT: optionalPositiveInteger(record, "patch_size_t", context, 1),
    numAttentionHeads,
    attentionHeadDim,
    hiddenSize: attentionHeadDim * numAttentionHeads,
    crossAttentionDim: optionalPositiveInteger(record, "cross_attention_dim", context, 4096),
    vaeScaleFactors: tuple3(
      optionalPositiveIntegerArray(record, "vae_scale_factors", context, [8, 32, 32], 3),
      context,
      "vae_scale_factors",
    ),
    posEmbedMaxPos: optionalPositiveInteger(record, "pos_embed_max_pos", context, 20),
    baseHeight: optionalPositiveInteger(record, "base_height", context, 2048),
    baseWidth: optionalPositiveInteger(record, "base_width", context, 2048),
    audioInChannels,
    audioOutChannels:
      optionalNullablePositiveInteger(record, "audio_out_channels", context, 128) ??
      audioInChannels,
    audioPatchSize: optionalPositiveInteger(record, "audio_patch_size", context, 1),
    audioPatchSizeT: optionalPositiveInteger(record, "audio_patch_size_t", context, 1),
    audioNumAttentionHeads,
    audioAttentionHeadDim,
    audioHiddenSize: audioAttentionHeadDim * audioNumAttentionHeads,
    audioCrossAttentionDim: optionalPositiveInteger(
      record,
      "audio_cross_attention_dim",
      context,
      2048,
    ),
    audioScaleFactor: optionalPositiveInteger(record, "audio_scale_factor", context, 4),
    audioPosEmbedMaxPos: optionalPositiveInteger(record, "audio_pos_embed_max_pos", context, 20),
    audioSamplingRate: optionalPositiveInteger(record, "audio_sampling_rate", context, 16000),
    audioHopLength: optionalPositiveInteger(record, "audio_hop_length", context, 160),
    numLayers: optionalPositiveInteger(record, "num_layers", context, 48),
    activationFn: "gelu-approximate",
    qkNorm: optionalStringValue(record, "qk_norm", context, "rms_norm_across_heads", QK_NORMS),
    normElementwiseAffine: optionalBoolean(record, "norm_elementwise_affine", context, false),
    normEps: optionalPositiveNumber(record, "norm_eps", context, 1e-6),
    captionChannels: optionalPositiveInteger(record, "caption_channels", context, 3840),
    attentionBias: optionalBoolean(record, "attention_bias", context, true),
    attentionOutBias: optionalBoolean(record, "attention_out_bias", context, true),
    ropeTheta: optionalPositiveNumber(record, "rope_theta", context, 10000),
    ropeDoublePrecision: optionalBoolean(record, "rope_double_precision", context, true),
    causalOffset: optionalPositiveInteger(record, "causal_offset", context, 1),
    timestepScaleMultiplier: optionalPositiveInteger(
      record,
      "timestep_scale_multiplier",
      context,
      1000,
    ),
    crossAttnTimestepScaleMultiplier: optionalPositiveInteger(
      record,
      "cross_attn_timestep_scale_multiplier",
      context,
      1000,
    ),
    ropeType: optionalStringValue(record, "rope_type", context, "interleaved", ROPE_TYPES),
    gatedAttn: optionalBoolean(record, "gated_attn", context, false),
    crossAttnMod: optionalBoolean(record, "cross_attn_mod", context, false),
    audioGatedAttn: optionalBoolean(record, "audio_gated_attn", context, false),
    audioCrossAttnMod: optionalBoolean(record, "audio_cross_attn_mod", context, false),
    usePromptEmbeddings: optionalBoolean(record, "use_prompt_embeddings", context, true),
    perturbedAttn: optionalBoolean(record, "perturbed_attn", context, false),
    rawConfig: record,
  };
}
