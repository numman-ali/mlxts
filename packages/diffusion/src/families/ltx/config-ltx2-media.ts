import {
  expectRecord,
  optionalBoolean,
  optionalClassName,
  optionalPositiveInteger,
  optionalPositiveNumber,
  rejectUnknownFields,
} from "../flux2/config-parsing";
import {
  optionalNullablePositiveInteger,
  optionalNullableStringValue,
  optionalPositiveIntegerArray,
  optionalPositiveIntegerMatrix,
  optionalStringValue,
} from "./config-common";
import type { Ltx2RopeType } from "./config-ltx2-transformer";

export type Ltx2VocoderActivation = "snakebeta" | "snake" | "leaky_relu";
export type Ltx2VocoderFinalActivation = "tanh" | "clamp" | null;

/** Package-native config for the LTX-2 `LTX2TextConnectors` component. */
export type Ltx2TextConnectorsConfig = {
  captionChannels: number;
  textProjInFactor: number;
  textEncoderDim: number;
  videoConnectorNumAttentionHeads: number;
  videoConnectorAttentionHeadDim: number;
  videoConnectorHiddenSize: number;
  videoConnectorNumLayers: number;
  videoConnectorNumLearnableRegisters: number | null;
  videoGatedAttn: boolean;
  audioConnectorNumAttentionHeads: number;
  audioConnectorAttentionHeadDim: number;
  audioConnectorHiddenSize: number;
  audioConnectorNumLayers: number;
  audioConnectorNumLearnableRegisters: number | null;
  audioGatedAttn: boolean;
  connectorRopeBaseSeqLen: number;
  ropeTheta: number;
  ropeDoublePrecision: boolean;
  causalTemporalPositioning: boolean;
  ropeType: Ltx2RopeType;
  perModalityProjections: boolean;
  videoHiddenDim: number;
  audioHiddenDim: number;
  projBias: boolean;
  rawConfig: Record<string, unknown>;
};

/** Package-native config for the LTX-2 `LTX2Vocoder` component. */
export type Ltx2VocoderConfig = {
  inChannels: number;
  hiddenChannels: number;
  outChannels: number;
  upsampleKernelSizes: readonly number[];
  upsampleFactors: readonly number[];
  totalUpsampleFactor: number;
  resnetKernelSizes: readonly number[];
  resnetDilations: readonly (readonly number[])[];
  actFn: Ltx2VocoderActivation;
  leakyReluNegativeSlope: number;
  antialias: boolean;
  antialiasRatio: number;
  antialiasKernelSize: number;
  finalActFn: Ltx2VocoderFinalActivation;
  finalBias: boolean;
  outputSamplingRate: number;
  rawConfig: Record<string, unknown>;
};

const ROPE_TYPES = new Set<Ltx2RopeType>(["interleaved", "split"]);
const VOCODER_ACTIVATIONS = new Set<Ltx2VocoderActivation>(["snakebeta", "snake", "leaky_relu"]);
const VOCODER_FINAL_ACTIVATIONS = new Set<Exclude<Ltx2VocoderFinalActivation, null>>([
  "tanh",
  "clamp",
]);

const LTX2_CONNECTORS_KEYS = new Set([
  "_class_name",
  "_diffusers_version",
  "_name_or_path",
  "audio_connector_attention_head_dim",
  "audio_connector_num_attention_heads",
  "audio_connector_num_layers",
  "audio_connector_num_learnable_registers",
  "audio_gated_attn",
  "audio_hidden_dim",
  "caption_channels",
  "causal_temporal_positioning",
  "connector_rope_base_seq_len",
  "per_modality_projections",
  "proj_bias",
  "rope_double_precision",
  "rope_theta",
  "rope_type",
  "text_proj_in_factor",
  "video_connector_attention_head_dim",
  "video_connector_num_attention_heads",
  "video_connector_num_layers",
  "video_connector_num_learnable_registers",
  "video_gated_attn",
  "video_hidden_dim",
]);

const LTX2_VOCODER_KEYS = new Set([
  "_class_name",
  "_diffusers_version",
  "_name_or_path",
  "act_fn",
  "antialias",
  "antialias_kernel_size",
  "antialias_ratio",
  "final_act_fn",
  "final_bias",
  "hidden_channels",
  "in_channels",
  "leaky_relu_negative_slope",
  "out_channels",
  "output_sampling_rate",
  "resnet_dilations",
  "resnet_kernel_sizes",
  "upsample_factors",
  "upsample_kernel_sizes",
]);

/** Parse a Diffusers LTX-2 `LTX2TextConnectors` config. */
export function parseLtx2TextConnectorsConfig(rawConfig: unknown): Ltx2TextConnectorsConfig {
  const context = "connectors/config.json";
  const record = expectRecord(rawConfig, context);
  rejectUnknownFields(record, LTX2_CONNECTORS_KEYS, context);
  optionalClassName(record, "LTX2TextConnectors", context);
  const captionChannels = optionalPositiveInteger(record, "caption_channels", context, 3840);
  const textProjInFactor = optionalPositiveInteger(record, "text_proj_in_factor", context, 49);
  const videoConnectorNumAttentionHeads = optionalPositiveInteger(
    record,
    "video_connector_num_attention_heads",
    context,
    30,
  );
  const videoConnectorAttentionHeadDim = optionalPositiveInteger(
    record,
    "video_connector_attention_head_dim",
    context,
    128,
  );
  const audioConnectorNumAttentionHeads = optionalPositiveInteger(
    record,
    "audio_connector_num_attention_heads",
    context,
    30,
  );
  const audioConnectorAttentionHeadDim = optionalPositiveInteger(
    record,
    "audio_connector_attention_head_dim",
    context,
    128,
  );
  return {
    captionChannels,
    textProjInFactor,
    textEncoderDim: captionChannels * textProjInFactor,
    videoConnectorNumAttentionHeads,
    videoConnectorAttentionHeadDim,
    videoConnectorHiddenSize: videoConnectorNumAttentionHeads * videoConnectorAttentionHeadDim,
    videoConnectorNumLayers: optionalPositiveInteger(
      record,
      "video_connector_num_layers",
      context,
      2,
    ),
    videoConnectorNumLearnableRegisters: optionalNullablePositiveInteger(
      record,
      "video_connector_num_learnable_registers",
      context,
      128,
    ),
    videoGatedAttn: optionalBoolean(record, "video_gated_attn", context, false),
    audioConnectorNumAttentionHeads,
    audioConnectorAttentionHeadDim,
    audioConnectorHiddenSize: audioConnectorNumAttentionHeads * audioConnectorAttentionHeadDim,
    audioConnectorNumLayers: optionalPositiveInteger(
      record,
      "audio_connector_num_layers",
      context,
      2,
    ),
    audioConnectorNumLearnableRegisters: optionalNullablePositiveInteger(
      record,
      "audio_connector_num_learnable_registers",
      context,
      128,
    ),
    audioGatedAttn: optionalBoolean(record, "audio_gated_attn", context, false),
    connectorRopeBaseSeqLen: optionalPositiveInteger(
      record,
      "connector_rope_base_seq_len",
      context,
      4096,
    ),
    ropeTheta: optionalPositiveNumber(record, "rope_theta", context, 10000),
    ropeDoublePrecision: optionalBoolean(record, "rope_double_precision", context, true),
    causalTemporalPositioning: optionalBoolean(
      record,
      "causal_temporal_positioning",
      context,
      false,
    ),
    ropeType: optionalStringValue(record, "rope_type", context, "interleaved", ROPE_TYPES),
    perModalityProjections: optionalBoolean(record, "per_modality_projections", context, false),
    videoHiddenDim: optionalPositiveInteger(record, "video_hidden_dim", context, 4096),
    audioHiddenDim: optionalPositiveInteger(record, "audio_hidden_dim", context, 2048),
    projBias: optionalBoolean(record, "proj_bias", context, false),
    rawConfig: record,
  };
}

/** Parse a Diffusers LTX-2 `LTX2Vocoder` config. */
export function parseLtx2VocoderConfig(rawConfig: unknown): Ltx2VocoderConfig {
  const context = "vocoder/config.json";
  const record = expectRecord(rawConfig, context);
  rejectUnknownFields(record, LTX2_VOCODER_KEYS, context);
  optionalClassName(record, "LTX2Vocoder", context);
  const upsampleKernelSizes = optionalPositiveIntegerArray(
    record,
    "upsample_kernel_sizes",
    context,
    [16, 15, 8, 4, 4],
  );
  const upsampleFactors = optionalPositiveIntegerArray(
    record,
    "upsample_factors",
    context,
    [6, 5, 2, 2, 2],
    upsampleKernelSizes.length,
  );
  const resnetKernelSizes = optionalPositiveIntegerArray(
    record,
    "resnet_kernel_sizes",
    context,
    [3, 7, 11],
  );
  const resnetDilations = optionalPositiveIntegerMatrix(
    record,
    "resnet_dilations",
    context,
    [
      [1, 3, 5],
      [1, 3, 5],
      [1, 3, 5],
    ],
    resnetKernelSizes.length,
    3,
  );
  return {
    inChannels: optionalPositiveInteger(record, "in_channels", context, 128),
    hiddenChannels: optionalPositiveInteger(record, "hidden_channels", context, 1024),
    outChannels: optionalPositiveInteger(record, "out_channels", context, 2),
    upsampleKernelSizes,
    upsampleFactors,
    totalUpsampleFactor: upsampleFactors.reduce((product, factor) => product * factor, 1),
    resnetKernelSizes,
    resnetDilations,
    actFn: optionalStringValue(record, "act_fn", context, "leaky_relu", VOCODER_ACTIVATIONS),
    leakyReluNegativeSlope: optionalPositiveNumber(
      record,
      "leaky_relu_negative_slope",
      context,
      0.1,
    ),
    antialias: optionalBoolean(record, "antialias", context, false),
    antialiasRatio: optionalPositiveInteger(record, "antialias_ratio", context, 2),
    antialiasKernelSize: optionalPositiveInteger(record, "antialias_kernel_size", context, 12),
    finalActFn: optionalNullableStringValue(
      record,
      "final_act_fn",
      context,
      "tanh",
      VOCODER_FINAL_ACTIVATIONS,
    ),
    finalBias: optionalBoolean(record, "final_bias", context, true),
    outputSamplingRate: optionalPositiveInteger(record, "output_sampling_rate", context, 24000),
    rawConfig: record,
  };
}
