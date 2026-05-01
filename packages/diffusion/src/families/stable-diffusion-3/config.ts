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
  readComponentJson,
  rejectUnknownFields,
} from "../flux2/config-parsing";

export type StableDiffusion3QkNorm = "rms_norm" | null;

/** Autoencoder encoder block supported by the SD3 VAE config path. */
export type StableDiffusion3VaeDownBlockType = "DownEncoderBlock2D";

/** Autoencoder decoder block supported by the SD3 VAE config path. */
export type StableDiffusion3VaeUpBlockType = "UpDecoderBlock2D";

/** Package-native config for the Stable Diffusion 3 `SD3Transformer2DModel`. */
export type StableDiffusion3TransformerConfig = {
  sampleSize: number;
  patchSize: number;
  inChannels: number;
  outChannels: number;
  numLayers: number;
  attentionHeadDim: number;
  numAttentionHeads: number;
  hiddenSize: number;
  jointAttentionDim: number;
  captionProjectionDim: number;
  pooledProjectionDim: number;
  posEmbedMaxSize: number;
  dualAttentionLayers: readonly number[];
  qkNorm: StableDiffusion3QkNorm;
  rawConfig: Record<string, unknown>;
};

/** Package-native config for the Stable Diffusion 3 AutoencoderKL component. */
export type StableDiffusion3AutoencoderConfig = {
  inChannels: number;
  outChannels: number;
  latentChannels: number;
  latentChannelsOut: number;
  useQuantConv: boolean;
  usePostQuantConv: boolean;
  blockOutChannels: readonly number[];
  layersPerBlock: number;
  normNumGroups: number;
  scalingFactor: number;
  shiftFactor: number;
  sampleSize: number;
  vaeScaleFactor: number;
  downBlockTypes: readonly StableDiffusion3VaeDownBlockType[];
  upBlockTypes: readonly StableDiffusion3VaeUpBlockType[];
  forceUpcast: boolean;
  rawConfig: Record<string, unknown>;
};

/** Configs required before Stable Diffusion 3 model construction can begin. */
export type StableDiffusion3ComponentConfigs = {
  pipelineKind: "stable-diffusion-3";
  vae: StableDiffusion3AutoencoderConfig;
  transformer: StableDiffusion3TransformerConfig;
};

const SD3_TRANSFORMER_CONFIG_KEYS = new Set([
  "_class_name",
  "_diffusers_version",
  "_name_or_path",
  "attention_head_dim",
  "caption_projection_dim",
  "dual_attention_layers",
  "in_channels",
  "joint_attention_dim",
  "num_attention_heads",
  "num_layers",
  "out_channels",
  "patch_size",
  "pooled_projection_dim",
  "pos_embed_max_size",
  "qk_norm",
  "sample_size",
]);

function describeConfigValue(value: unknown): string {
  if (Array.isArray(value)) {
    return "array";
  }
  return value === null ? "null" : typeof value;
}

function optionalQkNorm(record: Record<string, unknown>, context: string): StableDiffusion3QkNorm {
  const value = record.qk_norm;
  if (value === undefined || value === null) {
    return null;
  }
  if (value === "rms_norm") {
    return value;
  }
  throw new DiffusionConfigError(
    `${fieldName(context, "qk_norm")}="${String(value)}" is not supported yet.`,
  );
}

function optionalDualAttentionLayers(
  record: Record<string, unknown>,
  context: string,
  numLayers: number,
): number[] {
  const value = record.dual_attention_layers ?? [];
  if (!Array.isArray(value)) {
    throw new DiffusionConfigError(
      `${fieldName(context, "dual_attention_layers")} must be an integer array.`,
    );
  }
  const seen = new Set<number>();
  return value.map((entry, index) => {
    if (typeof entry !== "number" || !Number.isInteger(entry) || entry < 0) {
      throw new DiffusionConfigError(
        `${fieldName(context, `dual_attention_layers[${index}]`)} must be a non-negative integer.`,
      );
    }
    if (entry >= numLayers) {
      throw new DiffusionConfigError(
        `${fieldName(context, `dual_attention_layers[${index}]`)} must be less than num_layers.`,
      );
    }
    if (seen.has(entry)) {
      throw new DiffusionConfigError(
        `${fieldName(context, "dual_attention_layers")} must not contain duplicates.`,
      );
    }
    seen.add(entry);
    return entry;
  });
}

function optionalStringList(
  record: Record<string, unknown>,
  key: string,
  context: string,
  fallback: readonly string[],
): string[] {
  const value = record[key] ?? fallback;
  if (!Array.isArray(value) || value.length !== fallback.length) {
    throw new DiffusionConfigError(
      `${fieldName(context, key)} must be a ${fallback.length}-item string array.`,
    );
  }
  return value.map((entry, index) => {
    if (typeof entry !== "string") {
      throw new DiffusionConfigError(`${fieldName(context, `${key}[${index}]`)} must be a string.`);
    }
    return entry;
  });
}

function parseVaeDownBlockTypes(
  record: Record<string, unknown>,
  context: string,
  expectedLength: number,
): StableDiffusion3VaeDownBlockType[] {
  const values = optionalStringList(
    record,
    "down_block_types",
    context,
    Array.from({ length: expectedLength }, () => "DownEncoderBlock2D"),
  );
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
): StableDiffusion3VaeUpBlockType[] {
  const values = optionalStringList(
    record,
    "up_block_types",
    context,
    Array.from({ length: expectedLength }, () => "UpDecoderBlock2D"),
  );
  return values.map((value) => {
    if (value === "UpDecoderBlock2D") {
      return value;
    }
    throw new DiffusionConfigError(`${fieldName(context, "up_block_types")} contains ${value}.`);
  });
}

function rejectNonNull(record: Record<string, unknown>, key: string, context: string): void {
  const value = record[key];
  if (value !== undefined && value !== null) {
    throw new DiffusionConfigError(
      `${fieldName(context, key)} is not supported yet, got ${describeConfigValue(value)}.`,
    );
  }
}

/** Parse a Diffusers Stable Diffusion 3 `SD3Transformer2DModel` config. */
export function parseStableDiffusion3TransformerConfig(
  rawConfig: unknown,
): StableDiffusion3TransformerConfig {
  const context = "transformer/config.json";
  const record = expectRecord(rawConfig, context);
  rejectUnknownFields(record, SD3_TRANSFORMER_CONFIG_KEYS, context);
  optionalClassName(record, "SD3Transformer2DModel", context);

  const inChannels = optionalPositiveInteger(record, "in_channels", context, 16);
  const attentionHeadDim = optionalPositiveInteger(record, "attention_head_dim", context, 64);
  const numAttentionHeads = optionalPositiveInteger(record, "num_attention_heads", context, 18);
  const numLayers = optionalPositiveInteger(record, "num_layers", context, 18);
  return {
    sampleSize: optionalPositiveInteger(record, "sample_size", context, 128),
    patchSize: optionalPositiveInteger(record, "patch_size", context, 2),
    inChannels,
    outChannels: optionalNullablePositiveInteger(record, "out_channels", context, inChannels),
    numLayers,
    attentionHeadDim,
    numAttentionHeads,
    hiddenSize: attentionHeadDim * numAttentionHeads,
    jointAttentionDim: optionalPositiveInteger(record, "joint_attention_dim", context, 4096),
    captionProjectionDim: optionalPositiveInteger(record, "caption_projection_dim", context, 1152),
    pooledProjectionDim: optionalPositiveInteger(record, "pooled_projection_dim", context, 2048),
    posEmbedMaxSize: optionalPositiveInteger(record, "pos_embed_max_size", context, 96),
    dualAttentionLayers: optionalDualAttentionLayers(record, context, numLayers),
    qkNorm: optionalQkNorm(record, context),
    rawConfig: record,
  };
}

/** Parse a Diffusers Stable Diffusion 3 AutoencoderKL config. */
export function parseStableDiffusion3AutoencoderConfig(
  rawConfig: unknown,
): StableDiffusion3AutoencoderConfig {
  const context = "vae/config.json";
  const record = expectRecord(rawConfig, context);
  optionalClassName(record, "AutoencoderKL", context);
  optionalExactString(record, "act_fn", context, "silu");
  rejectNonNull(record, "latents_mean", context);
  rejectNonNull(record, "latents_std", context);

  const blockOutChannels = optionalPositiveIntegerList(record, "block_out_channels", context);
  const latentChannels = optionalPositiveInteger(record, "latent_channels", context, 16);
  return {
    inChannels: optionalPositiveInteger(record, "in_channels", context, 3),
    outChannels: optionalPositiveInteger(record, "out_channels", context, 3),
    latentChannels,
    latentChannelsOut: 2 * latentChannels,
    useQuantConv: optionalBoolean(record, "use_quant_conv", context, true),
    usePostQuantConv: optionalBoolean(record, "use_post_quant_conv", context, true),
    blockOutChannels,
    layersPerBlock: optionalPositiveInteger(record, "layers_per_block", context, 2),
    normNumGroups: optionalPositiveInteger(record, "norm_num_groups", context, 32),
    scalingFactor: optionalPositiveNumber(record, "scaling_factor", context, 1.5305),
    shiftFactor: optionalPositiveNumber(record, "shift_factor", context, 0.0609),
    sampleSize: optionalPositiveInteger(record, "sample_size", context, 1024),
    vaeScaleFactor: 2 ** (blockOutChannels.length - 1),
    downBlockTypes: parseVaeDownBlockTypes(record, context, blockOutChannels.length),
    upBlockTypes: parseVaeUpBlockTypes(record, context, blockOutChannels.length),
    forceUpcast: optionalBoolean(record, "force_upcast", context, true),
    rawConfig: record,
  };
}

/** Load Stable Diffusion 3 component configs from an inspected local snapshot manifest. */
export async function loadStableDiffusion3ComponentConfigs(
  manifest: DiffusionSnapshotManifest,
): Promise<StableDiffusion3ComponentConfigs> {
  if (manifest.modelIndex.kind !== "stable-diffusion-3") {
    throw new DiffusionConfigError(
      `Stable Diffusion 3 component configs do not support ${manifest.modelIndex.kind}.`,
    );
  }
  const vae = parseStableDiffusion3AutoencoderConfig(
    await readComponentJson(componentConfigPath(manifest, "vae"), "vae/config.json"),
  );
  const transformer = parseStableDiffusion3TransformerConfig(
    await readComponentJson(
      componentConfigPath(manifest, "transformer"),
      "transformer/config.json",
    ),
  );
  if (vae.latentChannels !== transformer.inChannels) {
    throw new DiffusionConfigError(
      `Stable Diffusion 3 VAE latent channels ${vae.latentChannels} do not match transformer input channels ${transformer.inChannels}.`,
    );
  }
  return {
    pipelineKind: "stable-diffusion-3",
    vae,
    transformer,
  };
}
