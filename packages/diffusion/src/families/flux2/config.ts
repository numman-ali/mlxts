import { DiffusionConfigError } from "../../errors";
import type { DiffusionSnapshotManifest } from "../../pretrained/snapshot-manifest";
import {
  componentConfigPath,
  expectRecord,
  fieldName,
  optionalBoolean,
  optionalClassName,
  optionalExactString,
  optionalNullablePositiveInteger,
  optionalPositiveInteger,
  optionalPositiveIntegerList,
  optionalPositiveNumber,
  pipelineBooleanConfig,
  readComponentJson,
  rejectUnknownFields,
} from "./config-parsing";

export type Flux2KleinRopeAxes = readonly [number, number, number, number];

export type Flux2KleinVaePatchSize = readonly [number, number];

/** Autoencoder encoder block supported by the FLUX.2 Klein VAE path. */
export type Flux2KleinVaeDownBlockType = "DownEncoderBlock2D";

/** Autoencoder decoder block supported by the FLUX.2 Klein VAE path. */
export type Flux2KleinVaeUpBlockType = "UpDecoderBlock2D";

/** Package-native config for the FLUX.2 `Flux2Transformer2DModel` component. */
export type Flux2KleinTransformerConfig = {
  patchSize: number;
  inChannels: number;
  latentChannels: number;
  outChannels: number;
  numLayers: number;
  numSingleLayers: number;
  attentionHeadDim: number;
  numAttentionHeads: number;
  hiddenSize: number;
  mlpRatio: number;
  jointAttentionDim: number;
  timestepGuidanceChannels: number;
  axesDimsRope: Flux2KleinRopeAxes;
  ropeTheta: number;
  normEps: number;
  guidanceEmbeds: boolean;
  rawConfig: Record<string, unknown>;
};

/** Package-native config for the FLUX.2 `AutoencoderKLFlux2` component. */
export type Flux2KleinAutoencoderConfig = {
  inChannels: number;
  outChannels: number;
  latentChannels: number;
  latentChannelsOut: number;
  packedLatentChannels: number;
  useQuantConv: boolean;
  usePostQuantConv: boolean;
  blockOutChannels: readonly number[];
  decoderBlockOutChannels: readonly number[] | null;
  layersPerBlock: number;
  normNumGroups: number;
  forceUpcast: boolean;
  midBlockAddAttention: boolean;
  batchNormEps: number;
  batchNormMomentum: number;
  patchSize: Flux2KleinVaePatchSize;
  sampleSize: number;
  vaeScaleFactor: number;
  downBlockTypes: readonly Flux2KleinVaeDownBlockType[];
  upBlockTypes: readonly Flux2KleinVaeUpBlockType[];
  rawConfig: Record<string, unknown>;
};

/** Configs required before FLUX.2 Klein model construction can begin. */
export type Flux2KleinComponentConfigs = {
  pipelineKind: "flux2-klein";
  isDistilled: boolean;
  vae: Flux2KleinAutoencoderConfig;
  transformer: Flux2KleinTransformerConfig;
};

const FLUX2_TRANSFORMER_CONFIG_KEYS = new Set([
  "_class_name",
  "_diffusers_version",
  "_name_or_path",
  "attention_head_dim",
  "axes_dims_rope",
  "eps",
  "guidance_embeds",
  "in_channels",
  "joint_attention_dim",
  "mlp_ratio",
  "num_attention_heads",
  "num_layers",
  "num_single_layers",
  "out_channels",
  "patch_size",
  "rope_theta",
  "timestep_guidance_channels",
]);

const FLUX2_VAE_CONFIG_KEYS = new Set([
  "_class_name",
  "_diffusers_version",
  "_name_or_path",
  "act_fn",
  "batch_norm_eps",
  "batch_norm_momentum",
  "block_out_channels",
  "decoder_block_out_channels",
  "down_block_types",
  "force_upcast",
  "in_channels",
  "latent_channels",
  "layers_per_block",
  "mid_block_add_attention",
  "norm_num_groups",
  "out_channels",
  "patch_size",
  "sample_size",
  "up_block_types",
  "use_post_quant_conv",
  "use_quant_conv",
]);

function optionalDecoderBlockOutChannels(
  record: Record<string, unknown>,
  context: string,
  expectedLength: number,
): number[] | null {
  const value = record.decoder_block_out_channels;
  if (value === undefined || value === null) {
    return null;
  }
  const parsed = optionalPositiveIntegerList(record, "decoder_block_out_channels", context);
  if (parsed.length !== expectedLength) {
    throw new DiffusionConfigError(
      `${fieldName(context, "decoder_block_out_channels")} length must match block_out_channels.`,
    );
  }
  return parsed;
}

function optionalRopeAxes(
  record: Record<string, unknown>,
  key: string,
  context: string,
): Flux2KleinRopeAxes {
  const value = record[key] ?? [32, 32, 32, 32];
  if (!Array.isArray(value) || value.length !== 4) {
    throw new DiffusionConfigError(`${fieldName(context, key)} must be a 4-item integer array.`);
  }
  const axes = value.map((entry, index) => {
    if (typeof entry !== "number" || !Number.isInteger(entry) || entry <= 0) {
      throw new DiffusionConfigError(
        `${fieldName(context, `${key}[${index}]`)} must be a positive integer.`,
      );
    }
    if (entry % 2 !== 0) {
      throw new DiffusionConfigError(`${fieldName(context, `${key}[${index}]`)} must be even.`);
    }
    return entry;
  });
  const [first, second, third, fourth] = axes;
  if (first === undefined || second === undefined || third === undefined || fourth === undefined) {
    throw new DiffusionConfigError(`${fieldName(context, key)} must be a 4-item integer array.`);
  }
  return [first, second, third, fourth];
}

function optionalPatchSize(
  record: Record<string, unknown>,
  key: string,
  context: string,
): Flux2KleinVaePatchSize {
  const value = record[key] ?? [2, 2];
  if (!Array.isArray(value) || value.length !== 2) {
    throw new DiffusionConfigError(`${fieldName(context, key)} must be a 2-item integer array.`);
  }
  const [height, width] = value;
  if (
    typeof height !== "number" ||
    typeof width !== "number" ||
    !Number.isInteger(height) ||
    !Number.isInteger(width) ||
    height <= 0 ||
    width <= 0
  ) {
    throw new DiffusionConfigError(`${fieldName(context, key)} must contain positive integers.`);
  }
  if (height !== 2 || width !== 2) {
    throw new DiffusionConfigError(`${fieldName(context, key)} only supports [2, 2].`);
  }
  return [height, width];
}

function parseVaeDownBlockTypes(
  record: Record<string, unknown>,
  context: string,
  expectedLength: number,
): Flux2KleinVaeDownBlockType[] {
  const value =
    record.down_block_types ?? Array.from({ length: expectedLength }, () => "DownEncoderBlock2D");
  if (!Array.isArray(value) || value.length !== expectedLength) {
    throw new DiffusionConfigError(
      `${fieldName(context, "down_block_types")} must match block_out_channels length.`,
    );
  }
  return value.map((entry, index) => {
    if (entry !== "DownEncoderBlock2D") {
      throw new DiffusionConfigError(
        `${fieldName(context, `down_block_types[${index}]`)} is not supported yet.`,
      );
    }
    return entry;
  });
}

function parseVaeUpBlockTypes(
  record: Record<string, unknown>,
  context: string,
  expectedLength: number,
): Flux2KleinVaeUpBlockType[] {
  const value =
    record.up_block_types ?? Array.from({ length: expectedLength }, () => "UpDecoderBlock2D");
  if (!Array.isArray(value) || value.length !== expectedLength) {
    throw new DiffusionConfigError(
      `${fieldName(context, "up_block_types")} must match block_out_channels length.`,
    );
  }
  return value.map((entry, index) => {
    if (entry !== "UpDecoderBlock2D") {
      throw new DiffusionConfigError(
        `${fieldName(context, `up_block_types[${index}]`)} is not supported yet.`,
      );
    }
    return entry;
  });
}

/** Parse a Diffusers FLUX.2 Klein `Flux2Transformer2DModel` config into package-owned terms. */
export function parseFlux2KleinTransformerConfig(rawConfig: unknown): Flux2KleinTransformerConfig {
  const context = "transformer/config.json";
  const record = expectRecord(rawConfig, context);
  rejectUnknownFields(record, FLUX2_TRANSFORMER_CONFIG_KEYS, context);
  optionalClassName(record, "Flux2Transformer2DModel", context);

  const patchSize = optionalPositiveInteger(record, "patch_size", context, 1);
  const inChannels = optionalPositiveInteger(record, "in_channels", context, 128);
  const attentionHeadDim = optionalPositiveInteger(record, "attention_head_dim", context, 128);
  const numAttentionHeads = optionalPositiveInteger(record, "num_attention_heads", context, 24);
  const hiddenSize = attentionHeadDim * numAttentionHeads;
  const axesDimsRope = optionalRopeAxes(record, "axes_dims_rope", context);
  const ropeDim = axesDimsRope.reduce((sum, axis) => sum + axis, 0);
  if (ropeDim !== attentionHeadDim) {
    throw new DiffusionConfigError(
      `${fieldName(context, "axes_dims_rope")} sum must equal attention_head_dim.`,
    );
  }
  if (inChannels % 4 !== 0) {
    throw new DiffusionConfigError(`${fieldName(context, "in_channels")} must be divisible by 4.`);
  }

  return {
    patchSize,
    inChannels,
    latentChannels: inChannels / 4,
    outChannels: optionalNullablePositiveInteger(record, "out_channels", context, inChannels),
    numLayers: optionalPositiveInteger(record, "num_layers", context, 5),
    numSingleLayers: optionalPositiveInteger(record, "num_single_layers", context, 20),
    attentionHeadDim,
    numAttentionHeads,
    hiddenSize,
    mlpRatio: optionalPositiveNumber(record, "mlp_ratio", context, 3),
    jointAttentionDim: optionalPositiveInteger(record, "joint_attention_dim", context, 7680),
    timestepGuidanceChannels: optionalPositiveInteger(
      record,
      "timestep_guidance_channels",
      context,
      256,
    ),
    axesDimsRope,
    ropeTheta: optionalPositiveNumber(record, "rope_theta", context, 2000),
    normEps: optionalPositiveNumber(record, "eps", context, 1e-6),
    guidanceEmbeds: optionalBoolean(record, "guidance_embeds", context, true),
    rawConfig: record,
  };
}

/** Parse a Diffusers FLUX.2 Klein `AutoencoderKLFlux2` config into package-owned terms. */
export function parseFlux2KleinAutoencoderConfig(rawConfig: unknown): Flux2KleinAutoencoderConfig {
  const context = "vae/config.json";
  const record = expectRecord(rawConfig, context);
  rejectUnknownFields(record, FLUX2_VAE_CONFIG_KEYS, context);
  optionalClassName(record, "AutoencoderKLFlux2", context);
  optionalExactString(record, "act_fn", context, "silu");

  const blockOutChannels = optionalPositiveIntegerList(
    record,
    "block_out_channels",
    context,
    [128, 256, 512, 512],
  );
  const latentChannels = optionalPositiveInteger(record, "latent_channels", context, 32);
  const patchSize = optionalPatchSize(record, "patch_size", context);
  const vaeScaleFactor = 2 ** (blockOutChannels.length - 1);
  return {
    inChannels: optionalPositiveInteger(record, "in_channels", context, 3),
    outChannels: optionalPositiveInteger(record, "out_channels", context, 3),
    latentChannels,
    latentChannelsOut: 2 * latentChannels,
    packedLatentChannels: latentChannels * patchSize[0] * patchSize[1],
    useQuantConv: optionalBoolean(record, "use_quant_conv", context, true),
    usePostQuantConv: optionalBoolean(record, "use_post_quant_conv", context, true),
    blockOutChannels,
    decoderBlockOutChannels: optionalDecoderBlockOutChannels(
      record,
      context,
      blockOutChannels.length,
    ),
    layersPerBlock: optionalPositiveInteger(record, "layers_per_block", context, 2),
    normNumGroups: optionalPositiveInteger(record, "norm_num_groups", context, 32),
    forceUpcast: optionalBoolean(record, "force_upcast", context, true),
    midBlockAddAttention: optionalBoolean(record, "mid_block_add_attention", context, true),
    batchNormEps: optionalPositiveNumber(record, "batch_norm_eps", context, 1e-4),
    batchNormMomentum: optionalPositiveNumber(record, "batch_norm_momentum", context, 0.1),
    patchSize,
    sampleSize: optionalPositiveInteger(record, "sample_size", context, 1024),
    vaeScaleFactor,
    downBlockTypes: parseVaeDownBlockTypes(record, context, blockOutChannels.length),
    upBlockTypes: parseVaeUpBlockTypes(record, context, blockOutChannels.length),
    rawConfig: record,
  };
}

/** Load FLUX.2 Klein component configs from an inspected local snapshot manifest. */
export async function loadFlux2KleinComponentConfigs(
  manifest: DiffusionSnapshotManifest,
): Promise<Flux2KleinComponentConfigs> {
  if (manifest.modelIndex.kind !== "flux2-klein") {
    throw new DiffusionConfigError(
      `FLUX.2 Klein component configs do not support ${manifest.modelIndex.kind}.`,
    );
  }
  const vae = parseFlux2KleinAutoencoderConfig(
    await readComponentJson(componentConfigPath(manifest, "vae"), "vae/config.json"),
  );
  const transformer = parseFlux2KleinTransformerConfig(
    await readComponentJson(
      componentConfigPath(manifest, "transformer"),
      "transformer/config.json",
    ),
  );
  if (vae.packedLatentChannels !== transformer.inChannels) {
    throw new DiffusionConfigError(
      `FLUX.2 Klein VAE packed latent channels ${vae.packedLatentChannels} do not match transformer input channels ${transformer.inChannels}.`,
    );
  }
  return {
    pipelineKind: "flux2-klein",
    isDistilled: pipelineBooleanConfig(manifest, "is_distilled", false),
    vae,
    transformer,
  };
}
