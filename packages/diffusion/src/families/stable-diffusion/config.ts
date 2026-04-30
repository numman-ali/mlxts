import { basename } from "path";

import { DiffusionConfigError } from "../../errors";
import type { DiffusionSnapshotManifest } from "../../pretrained/snapshot-manifest";
import {
  expectRecord,
  fieldName,
  normalizeOptionalPositiveIntegerList,
  normalizePositiveIntegerList,
  optionalBoolean,
  optionalClassName,
  optionalFiniteNumber,
  optionalNullablePositiveInteger,
  optionalPositiveInteger,
  optionalSampleSize,
  rejectNonNull,
  rejectUnsupportedBoolean,
  rejectUnsupportedBooleanList,
  rejectUnsupportedString,
  requiredPositiveInteger,
  requiredPositiveIntegerList,
  requiredStringList,
  type StableDiffusionSampleSize,
} from "./config-parsing";

export type { StableDiffusionSampleSize };

/** Autoencoder encoder block supported by the initial Stable Diffusion VAE path. */
export type StableDiffusionVaeDownBlockType = "DownEncoderBlock2D";

/** Autoencoder decoder block supported by the initial Stable Diffusion VAE path. */
export type StableDiffusionVaeUpBlockType = "UpDecoderBlock2D";

/** UNet down block supported by the initial Stable Diffusion denoiser path. */
export type StableDiffusionUNetDownBlockType = "CrossAttnDownBlock2D" | "DownBlock2D";

/** UNet up block supported by the initial Stable Diffusion denoiser path. */
export type StableDiffusionUNetUpBlockType = "CrossAttnUpBlock2D" | "UpBlock2D";

/** Package-native config for the Stable Diffusion AutoencoderKL component. */
export type StableDiffusionAutoencoderConfig = {
  inChannels: number;
  outChannels: number;
  latentChannels: number;
  latentChannelsOut: number;
  blockOutChannels: readonly number[];
  layersPerBlock: number;
  normNumGroups: number;
  scalingFactor: number;
  sampleSize?: StableDiffusionSampleSize;
  downBlockTypes: readonly StableDiffusionVaeDownBlockType[];
  upBlockTypes: readonly StableDiffusionVaeUpBlockType[];
  forceUpcast: boolean;
  rawConfig: Record<string, unknown>;
};

/** Package-native config for the Stable Diffusion UNet2DConditionModel component. */
export type StableDiffusionUNetConfig = {
  sampleSize?: StableDiffusionSampleSize;
  inChannels: number;
  outChannels: number;
  convInKernel: number;
  convOutKernel: number;
  blockOutChannels: readonly number[];
  layersPerBlock: readonly number[];
  midBlockLayers: number;
  transformerLayersPerBlock: readonly number[];
  numAttentionHeads: readonly number[];
  crossAttentionDim: readonly number[];
  normNumGroups: number;
  normEps: number;
  downBlockTypes: readonly StableDiffusionUNetDownBlockType[];
  upBlockTypes: readonly StableDiffusionUNetUpBlockType[];
  additionEmbedType: "text_time" | null;
  additionTimeEmbedDim: number | null;
  projectionClassEmbeddingsInputDim: number | null;
  useLinearProjection: boolean;
  upcastAttention: boolean;
  flipSinToCos: boolean;
  freqShift: number;
  rawConfig: Record<string, unknown>;
};

/** Configs required before Stable Diffusion model construction can begin. */
export type StableDiffusionComponentConfigs = {
  pipelineKind: "stable-diffusion" | "stable-diffusion-xl";
  vae: StableDiffusionAutoencoderConfig;
  unet: StableDiffusionUNetConfig;
};

function parseVaeDownBlockTypes(
  record: Record<string, unknown>,
  context: string,
  expectedLength: number,
): StableDiffusionVaeDownBlockType[] {
  const values = requiredStringList(record, "down_block_types", context, expectedLength);
  return values.map((value) => {
    if (value === "DownEncoderBlock2D") {
      return value;
    }
    throw new DiffusionConfigError(`${fieldName(context, "down_block_types")} contains ${value}.`);
  });
}

function parseVaeUpBlockTypes(
  record: Record<string, unknown>,
  context: string,
  expectedLength: number,
): StableDiffusionVaeUpBlockType[] {
  const values = requiredStringList(record, "up_block_types", context, expectedLength);
  return values.map((value) => {
    if (value === "UpDecoderBlock2D") {
      return value;
    }
    throw new DiffusionConfigError(`${fieldName(context, "up_block_types")} contains ${value}.`);
  });
}

function parseUNetDownBlockTypes(
  record: Record<string, unknown>,
  context: string,
  expectedLength: number,
): StableDiffusionUNetDownBlockType[] {
  const values = requiredStringList(record, "down_block_types", context, expectedLength);
  return values.map((value) => {
    if (value === "CrossAttnDownBlock2D" || value === "DownBlock2D") {
      return value;
    }
    throw new DiffusionConfigError(`${fieldName(context, "down_block_types")} contains ${value}.`);
  });
}

function parseUNetUpBlockTypes(
  record: Record<string, unknown>,
  context: string,
  expectedLength: number,
): StableDiffusionUNetUpBlockType[] {
  const values = requiredStringList(record, "up_block_types", context, expectedLength);
  return values.map((value) => {
    if (value === "CrossAttnUpBlock2D" || value === "UpBlock2D") {
      return value;
    }
    throw new DiffusionConfigError(`${fieldName(context, "up_block_types")} contains ${value}.`);
  });
}

function parseAdditionEmbedType(
  record: Record<string, unknown>,
  context: string,
): "text_time" | null {
  const value = record.addition_embed_type;
  if (value === undefined || value === null) {
    return null;
  }
  if (value === "text_time") {
    return value;
  }
  throw new DiffusionConfigError(
    `${fieldName(context, "addition_embed_type")}="${String(value)}" is not supported yet.`,
  );
}

function rejectUnsupportedAutoencoderSemantics(
  record: Record<string, unknown>,
  context: string,
): void {
  rejectUnsupportedBoolean(record, "double_z", context, true);
  rejectUnsupportedBoolean(record, "mid_block_add_attention", context, true);
  rejectUnsupportedBoolean(record, "use_quant_conv", context, true);
  rejectUnsupportedBoolean(record, "use_post_quant_conv", context, true);
  rejectUnsupportedString(record, "act_fn", context, "silu");
  rejectUnsupportedString(record, "norm_type", context, "group");
  rejectNonNull(record, "shift_factor", context);
  rejectNonNull(record, "latents_mean", context);
  rejectNonNull(record, "latents_std", context);
}

function rejectUnsupportedUNetSemantics(record: Record<string, unknown>, context: string): void {
  rejectUnsupportedBoolean(record, "center_input_sample", context, false);
  rejectUnsupportedBoolean(record, "dual_cross_attention", context, false);
  rejectUnsupportedBooleanList(record, "only_cross_attention", context, false);
  rejectUnsupportedBooleanList(record, "mid_block_only_cross_attention", context, false);
  rejectUnsupportedBoolean(record, "class_embeddings_concat", context, false);
  rejectUnsupportedBoolean(record, "resnet_skip_time_act", context, false);
  rejectUnsupportedString(record, "time_embedding_type", context, "positional");
  rejectUnsupportedString(record, "act_fn", context, "silu");
  rejectUnsupportedString(record, "attention_type", context, "default");
  rejectUnsupportedString(record, "resnet_time_scale_shift", context, "default");
  rejectUnsupportedString(record, "mid_block_type", context, "UNetMidBlock2DCrossAttn");
  rejectNonNull(record, "class_embed_type", context);
  rejectNonNull(record, "num_class_embeds", context);
  rejectNonNull(record, "time_embedding_dim", context);
  rejectNonNull(record, "time_cond_proj_dim", context);
  rejectNonNull(record, "encoder_hid_dim", context);
  rejectNonNull(record, "encoder_hid_dim_type", context);
  rejectNonNull(record, "cross_attention_norm", context);
  const outScale = optionalFiniteNumber(record, "resnet_out_scale_factor", context, 1);
  if (outScale !== 1) {
    throw new DiffusionConfigError(
      `${fieldName(context, "resnet_out_scale_factor")} is not supported yet.`,
    );
  }
  const dropout = optionalFiniteNumber(record, "dropout", context, 0);
  if (dropout !== 0) {
    throw new DiffusionConfigError(`${fieldName(context, "dropout")} is not supported yet.`);
  }
  const downsamplePadding = optionalPositiveInteger(record, "downsample_padding", context, 1);
  if (downsamplePadding !== 1) {
    throw new DiffusionConfigError(
      `${fieldName(context, "downsample_padding")} is not supported yet.`,
    );
  }
  const midBlockScaleFactor = optionalFiniteNumber(record, "mid_block_scale_factor", context, 1);
  if (midBlockScaleFactor !== 1) {
    throw new DiffusionConfigError(
      `${fieldName(context, "mid_block_scale_factor")} is not supported yet.`,
    );
  }
}

/** Parse a Diffusers AutoencoderKL `config.json` into package-owned terms. */
export function parseStableDiffusionAutoencoderConfig(
  rawConfig: unknown,
): StableDiffusionAutoencoderConfig {
  const context = "vae/config.json";
  const record = expectRecord(rawConfig, context);
  optionalClassName(record, "AutoencoderKL", context);
  rejectUnsupportedAutoencoderSemantics(record, context);
  const blockOutChannels = requiredPositiveIntegerList(record, "block_out_channels", context);
  const latentChannels = requiredPositiveInteger(record, "latent_channels", context);
  const sampleSize = optionalSampleSize(record, "sample_size", context);
  const parsed: StableDiffusionAutoencoderConfig = {
    inChannels: requiredPositiveInteger(record, "in_channels", context),
    outChannels: requiredPositiveInteger(record, "out_channels", context),
    latentChannels,
    latentChannelsOut: 2 * latentChannels,
    blockOutChannels,
    layersPerBlock: requiredPositiveInteger(record, "layers_per_block", context),
    normNumGroups: requiredPositiveInteger(record, "norm_num_groups", context),
    scalingFactor: optionalFiniteNumber(record, "scaling_factor", context, 0.18215),
    downBlockTypes: parseVaeDownBlockTypes(record, context, blockOutChannels.length),
    upBlockTypes: parseVaeUpBlockTypes(record, context, blockOutChannels.length),
    forceUpcast: optionalBoolean(record, "force_upcast", context, true),
    rawConfig: record,
  };
  if (sampleSize !== undefined) {
    return { ...parsed, sampleSize };
  }
  return parsed;
}

/** Parse a Diffusers UNet2DConditionModel `config.json` into package-owned terms. */
export function parseStableDiffusionUNetConfig(rawConfig: unknown): StableDiffusionUNetConfig {
  const context = "unet/config.json";
  const record = expectRecord(rawConfig, context);
  optionalClassName(record, "UNet2DConditionModel", context);
  rejectUnsupportedUNetSemantics(record, context);
  rejectNonNull(record, "reverse_transformer_layers_per_block", context);
  const blockOutChannels = requiredPositiveIntegerList(record, "block_out_channels", context);
  const blockCount = blockOutChannels.length;
  const additionEmbedType = parseAdditionEmbedType(record, context);
  const additionTimeEmbedDim = optionalNullablePositiveInteger(
    record,
    "addition_time_embed_dim",
    context,
  );
  const projectionClassEmbeddingsInputDim = optionalNullablePositiveInteger(
    record,
    "projection_class_embeddings_input_dim",
    context,
  );
  if (additionEmbedType === "text_time") {
    if (additionTimeEmbedDim === null || projectionClassEmbeddingsInputDim === null) {
      throw new DiffusionConfigError(
        "unet/config.json text_time addition embedding requires time and projection dimensions.",
      );
    }
  }
  const sampleSize = optionalSampleSize(record, "sample_size", context);
  const parsed: StableDiffusionUNetConfig = {
    inChannels: requiredPositiveInteger(record, "in_channels", context),
    outChannels: requiredPositiveInteger(record, "out_channels", context),
    convInKernel: optionalPositiveInteger(record, "conv_in_kernel", context, 3),
    convOutKernel: optionalPositiveInteger(record, "conv_out_kernel", context, 3),
    blockOutChannels,
    layersPerBlock: normalizePositiveIntegerList(record, "layers_per_block", context, blockCount),
    midBlockLayers: 2,
    transformerLayersPerBlock: normalizeOptionalPositiveIntegerList(
      record,
      "transformer_layers_per_block",
      context,
      blockCount,
      1,
    ),
    numAttentionHeads: normalizePositiveIntegerList(
      record,
      record.num_attention_heads === undefined || record.num_attention_heads === null
        ? "attention_head_dim"
        : "num_attention_heads",
      context,
      blockCount,
    ),
    crossAttentionDim: normalizePositiveIntegerList(
      record,
      "cross_attention_dim",
      context,
      blockCount,
    ),
    normNumGroups: requiredPositiveInteger(record, "norm_num_groups", context),
    normEps: optionalFiniteNumber(record, "norm_eps", context, 1e-5),
    downBlockTypes: parseUNetDownBlockTypes(record, context, blockCount),
    upBlockTypes: parseUNetUpBlockTypes(record, context, blockCount),
    additionEmbedType,
    additionTimeEmbedDim,
    projectionClassEmbeddingsInputDim,
    useLinearProjection: optionalBoolean(record, "use_linear_projection", context, false),
    upcastAttention: optionalBoolean(record, "upcast_attention", context, false),
    flipSinToCos: optionalBoolean(record, "flip_sin_to_cos", context, true),
    freqShift: optionalFiniteNumber(record, "freq_shift", context, 0),
    rawConfig: record,
  };
  if (sampleSize !== undefined) {
    return { ...parsed, sampleSize };
  }
  return parsed;
}

async function readComponentJson(path: string, context: string): Promise<Record<string, unknown>> {
  const file = Bun.file(path);
  let rawConfig: unknown;
  try {
    rawConfig = await file.json();
  } catch {
    throw new DiffusionConfigError(`${context} must contain valid JSON: ${path}.`);
  }
  return expectRecord(rawConfig, context);
}

function componentConfigPath(
  manifest: DiffusionSnapshotManifest,
  componentName: "vae" | "unet",
): string {
  const component = manifest.components.find(
    (candidate) => candidate.name === componentName && candidate.enabled,
  );
  const path = component?.metadataPaths.find((candidate) => basename(candidate) === "config.json");
  if (path === undefined) {
    throw new DiffusionConfigError(
      `${componentName}/config.json is missing from the snapshot manifest.`,
    );
  }
  return path;
}

/** Load Stable Diffusion VAE and UNet configs from an inspected local snapshot manifest. */
export async function loadStableDiffusionComponentConfigs(
  manifest: DiffusionSnapshotManifest,
): Promise<StableDiffusionComponentConfigs> {
  if (
    manifest.modelIndex.kind !== "stable-diffusion" &&
    manifest.modelIndex.kind !== "stable-diffusion-xl"
  ) {
    throw new DiffusionConfigError(
      `Stable Diffusion component configs do not support ${manifest.modelIndex.kind}.`,
    );
  }
  const vaePath = componentConfigPath(manifest, "vae");
  const unetPath = componentConfigPath(manifest, "unet");
  const vae = parseStableDiffusionAutoencoderConfig(
    await readComponentJson(vaePath, "vae/config.json"),
  );
  const unet = parseStableDiffusionUNetConfig(
    await readComponentJson(unetPath, "unet/config.json"),
  );
  return {
    pipelineKind: manifest.modelIndex.kind,
    vae,
    unet,
  };
}
