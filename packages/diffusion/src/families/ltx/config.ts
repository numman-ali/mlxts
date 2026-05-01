import { DiffusionConfigError } from "../../errors";
import type { DiffusionSnapshotManifest } from "../../pretrained/snapshot-manifest";
import {
  componentConfigPath,
  expectRecord,
  fieldName,
  optionalBoolean,
  optionalClassName,
  optionalExactString,
  optionalPositiveInteger,
  optionalPositiveNumber,
  readComponentJson,
  rejectUnknownFields,
} from "../flux2/config-parsing";
import {
  optionalBooleanArray,
  optionalNullablePositiveInteger,
  optionalPositiveIntegerArray,
  optionalStringArray,
  optionalStringValue,
} from "./config-common";
import {
  type Ltx2AudioAutoencoderConfig,
  type Ltx2ComponentConfigs,
  type Ltx2VideoTransformerConfig,
  type Ltx2VocoderConfig,
  parseLtx2AudioAutoencoderConfig,
  parseLtx2TextConnectorsConfig,
  parseLtx2VideoAutoencoderConfig,
  parseLtx2VideoTransformerConfig,
  parseLtx2VocoderConfig,
} from "./config-ltx2";

export {
  type Ltx2AudioAutoencoderConfig,
  type Ltx2AudioCausalityAxis,
  type Ltx2AudioNormType,
  type Ltx2ComponentConfigs,
  type Ltx2QkNorm,
  type Ltx2RopeType,
  type Ltx2SpatialPaddingMode,
  type Ltx2TextConnectorsConfig,
  type Ltx2VideoAutoencoderConfig,
  type Ltx2VideoDownBlockType,
  type Ltx2VideoDownsampleType,
  type Ltx2VideoTransformerConfig,
  type Ltx2VideoUpsampleType,
  type Ltx2VocoderActivation,
  type Ltx2VocoderConfig,
  type Ltx2VocoderFinalActivation,
  parseLtx2AudioAutoencoderConfig,
  parseLtx2TextConnectorsConfig,
  parseLtx2VideoAutoencoderConfig,
  parseLtx2VideoTransformerConfig,
  parseLtx2VocoderConfig,
} from "./config-ltx2";

export type LtxQkNorm = "rms_norm_across_heads";
export type LtxVideoDownBlockType = "LTXVideoDownBlock3D";
export type LtxVideoDownsampleType = "conv";

/** Package-native config for the LTX-Video `LTXVideoTransformer3DModel` component. */
export type LtxVideoTransformerConfig = {
  inChannels: number;
  outChannels: number;
  patchSize: number;
  patchSizeT: number;
  numAttentionHeads: number;
  attentionHeadDim: number;
  hiddenSize: number;
  crossAttentionDim: number;
  numLayers: number;
  activationFn: "gelu-approximate";
  qkNorm: LtxQkNorm;
  normElementwiseAffine: boolean;
  normEps: number;
  captionChannels: number;
  attentionBias: boolean;
  attentionOutBias: boolean;
  rawConfig: Record<string, unknown>;
};

/** Package-native config for the LTX-Video `AutoencoderKLLTXVideo` component. */
export type LtxVideoAutoencoderConfig = {
  inChannels: number;
  outChannels: number;
  latentChannels: number;
  latentChannelsOut: number;
  blockOutChannels: readonly number[];
  downBlockTypes: readonly LtxVideoDownBlockType[];
  decoderBlockOutChannels: readonly number[];
  layersPerBlock: readonly number[];
  decoderLayersPerBlock: readonly number[];
  spatioTemporalScaling: readonly boolean[];
  decoderSpatioTemporalScaling: readonly boolean[];
  decoderInjectNoise: readonly boolean[];
  downsampleTypes: readonly LtxVideoDownsampleType[];
  upsampleResidual: readonly boolean[];
  upsampleFactors: readonly number[];
  timestepConditioning: boolean;
  patchSize: number;
  patchSizeT: number;
  resnetNormEps: number;
  scalingFactor: number;
  encoderCausal: boolean;
  decoderCausal: boolean;
  spatialCompressionRatio: number | null;
  temporalCompressionRatio: number | null;
  rawConfig: Record<string, unknown>;
};

/** Configs required before LTX-Video model construction can begin. */
export type LtxVideoComponentConfigs = {
  pipelineKind: "ltx-video";
  transformer: LtxVideoTransformerConfig;
  vae: LtxVideoAutoencoderConfig;
};

export type LtxComponentConfigs = LtxVideoComponentConfigs | Ltx2ComponentConfigs;

const QK_NORMS = new Set<LtxQkNorm>(["rms_norm_across_heads"]);
const DOWN_BLOCK_TYPES = new Set<LtxVideoDownBlockType>(["LTXVideoDownBlock3D"]);
const DOWNSAMPLE_TYPES = new Set<LtxVideoDownsampleType>(["conv"]);

const LTX_TRANSFORMER_KEYS = new Set([
  "_class_name",
  "_diffusers_version",
  "_name_or_path",
  "activation_fn",
  "attention_bias",
  "attention_head_dim",
  "attention_out_bias",
  "caption_channels",
  "cross_attention_dim",
  "in_channels",
  "norm_elementwise_affine",
  "norm_eps",
  "num_attention_heads",
  "num_layers",
  "out_channels",
  "patch_size",
  "patch_size_t",
  "qk_norm",
]);

const LTX_VIDEO_VAE_KEYS = new Set([
  "_class_name",
  "_diffusers_version",
  "_name_or_path",
  "block_out_channels",
  "decoder_block_out_channels",
  "decoder_causal",
  "decoder_inject_noise",
  "decoder_layers_per_block",
  "decoder_spatio_temporal_scaling",
  "down_block_types",
  "downsample_type",
  "encoder_causal",
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
]);

/** Parse a Diffusers LTX-Video `LTXVideoTransformer3DModel` config. */
export function parseLtxVideoTransformerConfig(rawConfig: unknown): LtxVideoTransformerConfig {
  const context = "transformer/config.json";
  const record = expectRecord(rawConfig, context);
  rejectUnknownFields(record, LTX_TRANSFORMER_KEYS, context);
  optionalClassName(record, "LTXVideoTransformer3DModel", context);
  optionalExactString(record, "activation_fn", context, "gelu-approximate");

  const inChannels = optionalPositiveInteger(record, "in_channels", context, 128);
  const attentionHeadDim = optionalPositiveInteger(record, "attention_head_dim", context, 64);
  const numAttentionHeads = optionalPositiveInteger(record, "num_attention_heads", context, 32);
  return {
    inChannels,
    outChannels: optionalPositiveInteger(record, "out_channels", context, inChannels),
    patchSize: optionalPositiveInteger(record, "patch_size", context, 1),
    patchSizeT: optionalPositiveInteger(record, "patch_size_t", context, 1),
    numAttentionHeads,
    attentionHeadDim,
    hiddenSize: attentionHeadDim * numAttentionHeads,
    crossAttentionDim: optionalPositiveInteger(record, "cross_attention_dim", context, 2048),
    numLayers: optionalPositiveInteger(record, "num_layers", context, 28),
    activationFn: "gelu-approximate",
    qkNorm: optionalStringValue(record, "qk_norm", context, "rms_norm_across_heads", QK_NORMS),
    normElementwiseAffine: optionalBoolean(record, "norm_elementwise_affine", context, false),
    normEps: optionalPositiveNumber(record, "norm_eps", context, 1e-6),
    captionChannels: optionalPositiveInteger(record, "caption_channels", context, 4096),
    attentionBias: optionalBoolean(record, "attention_bias", context, true),
    attentionOutBias: optionalBoolean(record, "attention_out_bias", context, true),
    rawConfig: record,
  };
}

/** Parse a Diffusers LTX-Video `AutoencoderKLLTXVideo` config. */
export function parseLtxVideoAutoencoderConfig(rawConfig: unknown): LtxVideoAutoencoderConfig {
  const context = "vae/config.json";
  const record = expectRecord(rawConfig, context);
  rejectUnknownFields(record, LTX_VIDEO_VAE_KEYS, context);
  optionalClassName(record, "AutoencoderKLLTXVideo", context);

  const blockOutChannels = optionalPositiveIntegerArray(
    record,
    "block_out_channels",
    context,
    [128, 256, 512, 512],
  );
  const encoderBlocks = blockOutChannels.length;
  const decoderBlockOutChannels = optionalPositiveIntegerArray(
    record,
    "decoder_block_out_channels",
    context,
    [128, 256, 512, 512],
  );
  const decoderBlocks = decoderBlockOutChannels.length;
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
      Array.from({ length: encoderBlocks }, () => "LTXVideoDownBlock3D"),
      DOWN_BLOCK_TYPES,
      encoderBlocks,
    ),
    decoderBlockOutChannels,
    layersPerBlock: optionalPositiveIntegerArray(
      record,
      "layers_per_block",
      context,
      [4, 3, 3, 3, 4],
      encoderBlocks + 1,
    ),
    decoderLayersPerBlock: optionalPositiveIntegerArray(
      record,
      "decoder_layers_per_block",
      context,
      [4, 3, 3, 3, 4],
      decoderBlocks + 1,
    ),
    spatioTemporalScaling: optionalBooleanArray(
      record,
      "spatio_temporal_scaling",
      context,
      [true, true, true, false],
      encoderBlocks,
    ),
    decoderSpatioTemporalScaling: optionalBooleanArray(
      record,
      "decoder_spatio_temporal_scaling",
      context,
      [true, true, true, false],
      decoderBlocks,
    ),
    decoderInjectNoise: optionalBooleanArray(
      record,
      "decoder_inject_noise",
      context,
      [false, false, false, false, false],
      decoderBlocks + 1,
    ),
    downsampleTypes: optionalStringArray(
      record,
      "downsample_type",
      context,
      ["conv", "conv", "conv", "conv"],
      DOWNSAMPLE_TYPES,
      encoderBlocks,
    ),
    upsampleResidual: optionalBooleanArray(
      record,
      "upsample_residual",
      context,
      [false, false, false, false],
      decoderBlocks,
    ),
    upsampleFactors: optionalPositiveIntegerArray(
      record,
      "upsample_factor",
      context,
      [1, 1, 1, 1],
      decoderBlocks,
    ),
    timestepConditioning: optionalBoolean(record, "timestep_conditioning", context, false),
    patchSize: optionalPositiveInteger(record, "patch_size", context, 4),
    patchSizeT: optionalPositiveInteger(record, "patch_size_t", context, 1),
    resnetNormEps: optionalPositiveNumber(record, "resnet_norm_eps", context, 1e-6),
    scalingFactor: optionalPositiveNumber(record, "scaling_factor", context, 1),
    encoderCausal: optionalBoolean(record, "encoder_causal", context, true),
    decoderCausal: optionalBoolean(record, "decoder_causal", context, false),
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

async function loadLtxVideoComponentConfigs(
  manifest: DiffusionSnapshotManifest,
): Promise<LtxVideoComponentConfigs> {
  const transformer = parseLtxVideoTransformerConfig(
    await readComponentJson(
      componentConfigPath(manifest, "transformer"),
      "transformer/config.json",
    ),
  );
  const vae = parseLtxVideoAutoencoderConfig(
    await readComponentJson(componentConfigPath(manifest, "vae"), "vae/config.json"),
  );
  if (vae.latentChannels !== transformer.inChannels) {
    throw new DiffusionConfigError(
      `LTX-Video VAE latent channels ${vae.latentChannels} do not match transformer input channels ${transformer.inChannels}.`,
    );
  }
  return { pipelineKind: "ltx-video", transformer, vae };
}

function validateLtx2AudioComponentAgreement(
  transformer: Ltx2VideoTransformerConfig,
  audioVae: Ltx2AudioAutoencoderConfig,
  vocoder: Ltx2VocoderConfig,
): void {
  if (audioVae.sampleRate !== transformer.audioSamplingRate) {
    throw new DiffusionConfigError(
      `LTX-2 audio VAE sample rate ${audioVae.sampleRate} does not match transformer audio sampling rate ${transformer.audioSamplingRate}.`,
    );
  }
  if (audioVae.melHopLength !== transformer.audioHopLength) {
    throw new DiffusionConfigError(
      `LTX-2 audio VAE hop length ${audioVae.melHopLength} does not match transformer audio hop length ${transformer.audioHopLength}.`,
    );
  }
  if (audioVae.packedFeatureSize !== null && audioVae.packedFeatureSize !== audioVae.baseChannels) {
    throw new DiffusionConfigError(
      `LTX-2 audio VAE packed feature size ${audioVae.packedFeatureSize} must match base channels ${audioVae.baseChannels}.`,
    );
  }
  if (
    audioVae.packedFeatureSize !== null &&
    audioVae.packedFeatureSize !== transformer.audioInChannels
  ) {
    throw new DiffusionConfigError(
      `LTX-2 audio VAE packed feature size ${audioVae.packedFeatureSize} does not match transformer audio input channels ${transformer.audioInChannels}.`,
    );
  }
  if (audioVae.baseChannels !== transformer.audioInChannels) {
    throw new DiffusionConfigError(
      `LTX-2 audio VAE base channels ${audioVae.baseChannels} do not match transformer audio input channels ${transformer.audioInChannels}.`,
    );
  }
  if (transformer.audioOutChannels !== audioVae.baseChannels) {
    throw new DiffusionConfigError(
      `LTX-2 transformer audio output channels ${transformer.audioOutChannels} must match audio VAE base channels ${audioVae.baseChannels}.`,
    );
  }
  if (
    audioVae.melBins !== null &&
    vocoder.inChannels !== audioVae.outputChannels * audioVae.melBins
  ) {
    throw new DiffusionConfigError(
      `LTX-2 vocoder input channels ${vocoder.inChannels} must match decoded audio width ${
        audioVae.outputChannels * audioVae.melBins
      }.`,
    );
  }
  if (
    vocoder.totalUpsampleFactor * audioVae.sampleRate !==
    vocoder.outputSamplingRate * audioVae.melHopLength
  ) {
    throw new DiffusionConfigError(
      `LTX-2 vocoder total upsample factor ${vocoder.totalUpsampleFactor} must match output sample rate ${vocoder.outputSamplingRate}, audio VAE hop length ${audioVae.melHopLength}, and audio VAE sample rate ${audioVae.sampleRate}.`,
    );
  }
}

async function loadLtx2ComponentConfigs(
  manifest: DiffusionSnapshotManifest,
): Promise<Ltx2ComponentConfigs> {
  const transformer = parseLtx2VideoTransformerConfig(
    await readComponentJson(
      componentConfigPath(manifest, "transformer"),
      "transformer/config.json",
    ),
  );
  const vae = parseLtx2VideoAutoencoderConfig(
    await readComponentJson(componentConfigPath(manifest, "vae"), "vae/config.json"),
  );
  const audioVae = parseLtx2AudioAutoencoderConfig(
    await readComponentJson(componentConfigPath(manifest, "audio_vae"), "audio_vae/config.json"),
  );
  const connectors = parseLtx2TextConnectorsConfig(
    await readComponentJson(componentConfigPath(manifest, "connectors"), "connectors/config.json"),
  );
  const vocoder = parseLtx2VocoderConfig(
    await readComponentJson(componentConfigPath(manifest, "vocoder"), "vocoder/config.json"),
  );
  const [temporalScale, spatialHeightScale, spatialWidthScale] = transformer.vaeScaleFactors;
  if (vae.latentChannels !== transformer.inChannels) {
    throw new DiffusionConfigError(
      `LTX-2 VAE latent channels ${vae.latentChannels} do not match transformer input channels ${transformer.inChannels}.`,
    );
  }
  if (vae.temporalCompressionRatio !== null && vae.temporalCompressionRatio !== temporalScale) {
    throw new DiffusionConfigError(
      `${fieldName("vae/config.json", "temporal_compression_ratio")} must match transformer vae_scale_factors[0].`,
    );
  }
  if (
    vae.spatialCompressionRatio !== null &&
    (vae.spatialCompressionRatio !== spatialHeightScale ||
      vae.spatialCompressionRatio !== spatialWidthScale)
  ) {
    throw new DiffusionConfigError(
      `${fieldName("vae/config.json", "spatial_compression_ratio")} must match transformer spatial vae_scale_factors.`,
    );
  }
  if (connectors.captionChannels !== transformer.captionChannels) {
    throw new DiffusionConfigError(
      `LTX-2 connector caption channels ${connectors.captionChannels} do not match transformer caption channels ${transformer.captionChannels}.`,
    );
  }
  validateLtx2AudioComponentAgreement(transformer, audioVae, vocoder);
  return { pipelineKind: "ltx2", transformer, vae, audioVae, connectors, vocoder };
}

/** Load LTX component configs from an inspected local snapshot manifest. */
export async function loadLtxComponentConfigs(
  manifest: DiffusionSnapshotManifest,
): Promise<LtxComponentConfigs> {
  if (manifest.modelIndex.kind === "ltx-video") {
    return loadLtxVideoComponentConfigs(manifest);
  }
  if (manifest.modelIndex.kind === "ltx2") {
    return loadLtx2ComponentConfigs(manifest);
  }
  throw new DiffusionConfigError(
    `LTX component configs do not support ${manifest.modelIndex.kind}.`,
  );
}
