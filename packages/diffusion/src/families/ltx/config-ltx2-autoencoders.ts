import { DiffusionConfigError } from "../../errors";
import {
  expectRecord,
  fieldName,
  optionalBoolean,
  optionalClassName,
  optionalPositiveInteger,
  optionalPositiveNumber,
  rejectUnknownFields,
} from "../flux2/config-parsing";
import {
  optionalBooleanArray,
  optionalNonNegativeNumber,
  optionalNullablePositiveInteger,
  optionalNullableStringValue,
  optionalPositiveIntegerArray,
  optionalStringArray,
  optionalStringValue,
} from "./config-common";

export type Ltx2VideoDownBlockType = "LTX2VideoDownBlock3D";
export type Ltx2VideoDownsampleType = "spatial" | "temporal" | "spatiotemporal";
export type Ltx2VideoUpsampleType = "spatiotemporal";
export type Ltx2SpatialPaddingMode = "zeros" | "reflect";
export type Ltx2AudioNormType = "pixel";
export type Ltx2AudioCausalityAxis = "none" | "width" | "height" | "width-compatibility" | null;

/** Package-native config for the LTX-2 `AutoencoderKLLTX2Video` component. */
export type Ltx2VideoAutoencoderConfig = {
  inChannels: number;
  outChannels: number;
  latentChannels: number;
  latentChannelsOut: number;
  blockOutChannels: readonly number[];
  downBlockTypes: readonly Ltx2VideoDownBlockType[];
  decoderBlockOutChannels: readonly number[];
  layersPerBlock: readonly number[];
  decoderLayersPerBlock: readonly number[];
  spatioTemporalScaling: readonly boolean[];
  decoderSpatioTemporalScaling: readonly boolean[];
  decoderInjectNoise: readonly boolean[];
  downsampleTypes: readonly Ltx2VideoDownsampleType[];
  upsampleTypes: readonly Ltx2VideoUpsampleType[];
  upsampleResidual: readonly boolean[];
  upsampleFactors: readonly number[];
  timestepConditioning: boolean;
  patchSize: number;
  patchSizeT: number;
  resnetNormEps: number;
  scalingFactor: number;
  encoderCausal: boolean;
  decoderCausal: boolean;
  encoderSpatialPaddingMode: Ltx2SpatialPaddingMode;
  decoderSpatialPaddingMode: Ltx2SpatialPaddingMode;
  spatialCompressionRatio: number | null;
  temporalCompressionRatio: number | null;
  rawConfig: Record<string, unknown>;
};

/** Package-native config for the LTX-2 `AutoencoderKLLTX2Audio` component. */
export type Ltx2AudioAutoencoderConfig = {
  baseChannels: number;
  outputChannels: number;
  chMult: readonly number[];
  numResBlocks: number;
  attnResolutions: readonly number[] | null;
  inChannels: number;
  resolution: number;
  latentChannels: number;
  normType: Ltx2AudioNormType;
  causalityAxis: Ltx2AudioCausalityAxis;
  dropout: number;
  midBlockAddAttention: boolean;
  sampleRate: number;
  melHopLength: number;
  isCausal: boolean;
  melBins: number | null;
  melCompressionRatio: 4;
  temporalCompressionRatio: 4;
  packedFeatureSize: number | null;
  doubleZ: boolean;
  rawConfig: Record<string, unknown>;
};

const DOWN_BLOCK_TYPES = new Set<Ltx2VideoDownBlockType>(["LTX2VideoDownBlock3D"]);
const DOWNSAMPLE_TYPES = new Set<Ltx2VideoDownsampleType>([
  "spatial",
  "temporal",
  "spatiotemporal",
]);
const UPSAMPLE_TYPES = new Set<Ltx2VideoUpsampleType>(["spatiotemporal"]);
const PADDING_MODES = new Set<Ltx2SpatialPaddingMode>(["zeros", "reflect"]);
const AUDIO_NORM_TYPES = new Set<Ltx2AudioNormType>(["pixel"]);
const AUDIO_CAUSALITY_AXES = new Set<Exclude<Ltx2AudioCausalityAxis, null>>([
  "none",
  "width",
  "height",
  "width-compatibility",
]);
const LTX2_AUDIO_COMPRESSION_RATIO = 4;

const LTX2_VIDEO_VAE_KEYS = new Set([
  "_class_name",
  "_diffusers_version",
  "_name_or_path",
  "block_out_channels",
  "decoder_block_out_channels",
  "decoder_causal",
  "decoder_inject_noise",
  "decoder_layers_per_block",
  "decoder_spatial_padding_mode",
  "decoder_spatio_temporal_scaling",
  "down_block_types",
  "downsample_type",
  "encoder_causal",
  "encoder_spatial_padding_mode",
  "in_channels",
  "latent_channels",
  "layers_per_block",
  "out_channels",
  "patch_size",
  "patch_size_t",
  "resnet_norm_eps",
  "scaling_factor",
  "spatial_compression_ratio",
  "spatio_temporal_scaling",
  "temporal_compression_ratio",
  "timestep_conditioning",
  "upsample_factor",
  "upsample_residual",
  "upsample_type",
]);

const LTX2_AUDIO_VAE_KEYS = new Set([
  "_class_name",
  "_diffusers_version",
  "_name_or_path",
  "attn_resolutions",
  "base_channels",
  "causality_axis",
  "ch_mult",
  "double_z",
  "dropout",
  "in_channels",
  "is_causal",
  "latent_channels",
  "mel_bins",
  "mel_hop_length",
  "mid_block_add_attention",
  "norm_type",
  "num_res_blocks",
  "output_channels",
  "resolution",
  "sample_rate",
]);

/** Parse a Diffusers LTX-2 `AutoencoderKLLTX2Video` config. */
export function parseLtx2VideoAutoencoderConfig(rawConfig: unknown): Ltx2VideoAutoencoderConfig {
  const context = "vae/config.json";
  const record = expectRecord(rawConfig, context);
  rejectUnknownFields(record, LTX2_VIDEO_VAE_KEYS, context);
  optionalClassName(record, "AutoencoderKLLTX2Video", context);

  const blockOutChannels = optionalPositiveIntegerArray(
    record,
    "block_out_channels",
    context,
    [256, 512, 1024, 2048],
  );
  const encoderBlocks = blockOutChannels.length;
  const decoderBlockOutChannels = optionalPositiveIntegerArray(
    record,
    "decoder_block_out_channels",
    context,
    [256, 512, 1024],
  );
  const decoderBlocks = decoderBlockOutChannels.length + 1;
  const latentChannels = optionalPositiveInteger(record, "latent_channels", context, 128);
  return {
    inChannels: optionalPositiveInteger(record, "in_channels", context, 3),
    outChannels: optionalPositiveInteger(record, "out_channels", context, 3),
    latentChannels,
    latentChannelsOut: 2 * latentChannels,
    blockOutChannels,
    downBlockTypes: optionalStringArray(
      record,
      "down_block_types",
      context,
      Array.from({ length: encoderBlocks }, () => "LTX2VideoDownBlock3D"),
      DOWN_BLOCK_TYPES,
      encoderBlocks,
    ),
    decoderBlockOutChannels,
    layersPerBlock: optionalPositiveIntegerArray(
      record,
      "layers_per_block",
      context,
      [4, 6, 6, 2, 2],
      encoderBlocks + 1,
    ),
    decoderLayersPerBlock: optionalPositiveIntegerArray(
      record,
      "decoder_layers_per_block",
      context,
      [5, 5, 5, 5],
      decoderBlocks,
    ),
    spatioTemporalScaling: optionalBooleanArray(
      record,
      "spatio_temporal_scaling",
      context,
      [true, true, true, true],
      encoderBlocks,
    ),
    decoderSpatioTemporalScaling: optionalBooleanArray(
      record,
      "decoder_spatio_temporal_scaling",
      context,
      [true, true, true],
      decoderBlocks - 1,
    ),
    decoderInjectNoise: optionalBooleanArray(
      record,
      "decoder_inject_noise",
      context,
      [false, false, false, false],
      decoderBlocks,
    ),
    downsampleTypes: optionalStringArray(
      record,
      "downsample_type",
      context,
      ["spatial", "temporal", "spatiotemporal", "spatiotemporal"],
      DOWNSAMPLE_TYPES,
      encoderBlocks,
    ),
    upsampleTypes: optionalStringArray(
      record,
      "upsample_type",
      context,
      ["spatiotemporal", "spatiotemporal", "spatiotemporal"],
      UPSAMPLE_TYPES,
      decoderBlocks - 1,
    ),
    upsampleResidual: optionalBooleanArray(
      record,
      "upsample_residual",
      context,
      [true, true, true],
      decoderBlocks - 1,
    ),
    upsampleFactors: optionalPositiveIntegerArray(
      record,
      "upsample_factor",
      context,
      [2, 2, 2],
      decoderBlocks - 1,
    ),
    timestepConditioning: optionalBoolean(record, "timestep_conditioning", context, false),
    patchSize: optionalPositiveInteger(record, "patch_size", context, 4),
    patchSizeT: optionalPositiveInteger(record, "patch_size_t", context, 1),
    resnetNormEps: optionalPositiveNumber(record, "resnet_norm_eps", context, 1e-6),
    scalingFactor: optionalPositiveNumber(record, "scaling_factor", context, 1),
    encoderCausal: optionalBoolean(record, "encoder_causal", context, true),
    decoderCausal: optionalBoolean(record, "decoder_causal", context, true),
    encoderSpatialPaddingMode: optionalStringValue(
      record,
      "encoder_spatial_padding_mode",
      context,
      "zeros",
      PADDING_MODES,
    ),
    decoderSpatialPaddingMode: optionalStringValue(
      record,
      "decoder_spatial_padding_mode",
      context,
      "reflect",
      PADDING_MODES,
    ),
    spatialCompressionRatio: optionalNullablePositiveInteger(
      record,
      "spatial_compression_ratio",
      context,
      null,
    ),
    temporalCompressionRatio: optionalNullablePositiveInteger(
      record,
      "temporal_compression_ratio",
      context,
      null,
    ),
    rawConfig: record,
  };
}

/** Parse a Diffusers LTX-2 `AutoencoderKLLTX2Audio` config. */
export function parseLtx2AudioAutoencoderConfig(rawConfig: unknown): Ltx2AudioAutoencoderConfig {
  const context = "audio_vae/config.json";
  const record = expectRecord(rawConfig, context);
  rejectUnknownFields(record, LTX2_AUDIO_VAE_KEYS, context);
  optionalClassName(record, "AutoencoderKLLTX2Audio", context);
  const latentChannels = optionalPositiveInteger(record, "latent_channels", context, 8);
  const melBins = optionalNullablePositiveInteger(record, "mel_bins", context, 64);
  if (melBins !== null && melBins % LTX2_AUDIO_COMPRESSION_RATIO !== 0) {
    throw new DiffusionConfigError(
      `${fieldName(context, "mel_bins")} must be divisible by ${LTX2_AUDIO_COMPRESSION_RATIO}.`,
    );
  }
  return {
    baseChannels: optionalPositiveInteger(record, "base_channels", context, 128),
    outputChannels: optionalPositiveInteger(record, "output_channels", context, 2),
    chMult: optionalPositiveIntegerArray(record, "ch_mult", context, [1, 2, 4]),
    numResBlocks: optionalPositiveInteger(record, "num_res_blocks", context, 2),
    attnResolutions:
      record.attn_resolutions === null || record.attn_resolutions === undefined
        ? null
        : optionalPositiveIntegerArray(record, "attn_resolutions", context, []),
    inChannels: optionalPositiveInteger(record, "in_channels", context, 2),
    resolution: optionalPositiveInteger(record, "resolution", context, 256),
    latentChannels,
    normType: optionalStringValue(record, "norm_type", context, "pixel", AUDIO_NORM_TYPES),
    causalityAxis: optionalNullableStringValue(
      record,
      "causality_axis",
      context,
      "height",
      AUDIO_CAUSALITY_AXES,
    ),
    dropout: optionalNonNegativeNumber(record, "dropout", context, 0),
    midBlockAddAttention: optionalBoolean(record, "mid_block_add_attention", context, false),
    sampleRate: optionalPositiveInteger(record, "sample_rate", context, 16000),
    melHopLength: optionalPositiveInteger(record, "mel_hop_length", context, 160),
    isCausal: optionalBoolean(record, "is_causal", context, true),
    melBins,
    melCompressionRatio: LTX2_AUDIO_COMPRESSION_RATIO,
    temporalCompressionRatio: LTX2_AUDIO_COMPRESSION_RATIO,
    packedFeatureSize:
      melBins === null ? null : latentChannels * (melBins / LTX2_AUDIO_COMPRESSION_RATIO),
    doubleZ: optionalBoolean(record, "double_z", context, true),
    rawConfig: record,
  };
}
