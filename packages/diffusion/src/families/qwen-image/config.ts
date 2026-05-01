import { basename } from "path";

import { DiffusionConfigError } from "../../errors";
import type { DiffusionSnapshotManifest } from "../../pretrained/snapshot-manifest";

export type QwenImageRopeAxes = readonly [number, number, number];

/** Package-native config for the Qwen-Image `QwenImageTransformer2DModel` component. */
export type QwenImageTransformerConfig = {
  patchSize: number;
  inChannels: number;
  outChannels: number;
  latentChannels: number;
  packedLatentChannels: number;
  numLayers: number;
  attentionHeadDim: number;
  numAttentionHeads: number;
  hiddenSize: number;
  jointAttentionDim: number;
  guidanceEmbeds: boolean;
  axesDimsRope: QwenImageRopeAxes;
  ropeTheta: number;
  zeroCondT: boolean;
  useAdditionalTCond: boolean;
  useLayer3dRope: boolean;
  rawConfig: Record<string, unknown>;
};

/** Package-native config for the Qwen-Image 3D causal VAE component. */
export type QwenImageAutoencoderConfig = {
  baseDim: number;
  latentChannels: number;
  latentChannelsOut: number;
  dimMultipliers: readonly number[];
  numResBlocks: number;
  attentionScales: readonly number[];
  temporalDownsample: readonly boolean[];
  temporalUpsample: readonly boolean[];
  dropout: number;
  inputChannels: number;
  latentsMean: readonly number[];
  latentsStd: readonly number[];
  spatialCompressionRatio: number;
  rawConfig: Record<string, unknown>;
};

/** Configs required before Qwen-Image model construction can begin. */
export type QwenImageComponentConfigs = {
  pipelineKind: "qwen-image";
  vae: QwenImageAutoencoderConfig;
  transformer: QwenImageTransformerConfig;
};

const DEFAULT_LATENTS_MEAN = [
  -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517,
  -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];

const DEFAULT_LATENTS_STD = [
  2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
  1.6382, 1.1253, 2.8251, 1.916,
];

function describeConfigValue(value: unknown): string {
  if (Array.isArray(value)) {
    return "array";
  }
  return value === null ? "null" : typeof value;
}

function fieldName(context: string, key: string): string {
  return `${context}.${key}`;
}

function expectRecord(value: unknown, context: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new DiffusionConfigError(`${context} must be a JSON object.`);
  }
  return Object.fromEntries(Object.entries(value));
}

function optionalClassName(
  record: Record<string, unknown>,
  expected: string,
  context: string,
): void {
  const value = record._class_name;
  if (value === undefined || value === null) {
    return;
  }
  if (value !== expected) {
    throw new DiffusionConfigError(
      `${fieldName(context, "_class_name")}="${String(value)}" is not supported yet.`,
    );
  }
}

function optionalPositiveInteger(
  record: Record<string, unknown>,
  key: string,
  context: string,
  fallback: number,
): number {
  const value = record[key];
  if (value === undefined || value === null) {
    return fallback;
  }
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) {
    throw new DiffusionConfigError(
      `${fieldName(context, key)} must be a positive integer, got ${describeConfigValue(value)}.`,
    );
  }
  return value;
}

function optionalFiniteNumber(
  record: Record<string, unknown>,
  key: string,
  context: string,
  fallback: number,
): number {
  const value = record[key];
  if (value === undefined || value === null) {
    return fallback;
  }
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new DiffusionConfigError(
      `${fieldName(context, key)} must be a finite number, got ${describeConfigValue(value)}.`,
    );
  }
  return value;
}

function optionalBoolean(
  record: Record<string, unknown>,
  key: string,
  context: string,
  fallback: boolean,
): boolean {
  const value = record[key];
  if (value === undefined || value === null) {
    return fallback;
  }
  if (typeof value !== "boolean") {
    throw new DiffusionConfigError(
      `${fieldName(context, key)} must be a boolean, got ${describeConfigValue(value)}.`,
    );
  }
  return value;
}

function optionalPositiveIntegerList(
  record: Record<string, unknown>,
  key: string,
  context: string,
  fallback: readonly number[],
): number[] {
  const value = record[key] ?? fallback;
  if (!Array.isArray(value) || value.length === 0) {
    throw new DiffusionConfigError(`${fieldName(context, key)} must be a non-empty integer array.`);
  }
  return value.map((entry, index) => {
    if (typeof entry !== "number" || !Number.isInteger(entry) || entry <= 0) {
      throw new DiffusionConfigError(
        `${fieldName(context, `${key}[${index}]`)} must be a positive integer.`,
      );
    }
    return entry;
  });
}

function optionalFiniteNumberList(
  record: Record<string, unknown>,
  key: string,
  context: string,
  fallback: readonly number[],
): number[] {
  const value = record[key] ?? fallback;
  if (!Array.isArray(value)) {
    throw new DiffusionConfigError(`${fieldName(context, key)} must be a finite number array.`);
  }
  return value.map((entry, index) => {
    if (typeof entry !== "number" || !Number.isFinite(entry)) {
      throw new DiffusionConfigError(
        `${fieldName(context, `${key}[${index}]`)} must be a finite number.`,
      );
    }
    return entry;
  });
}

function optionalBooleanList(
  record: Record<string, unknown>,
  key: string,
  context: string,
  fallback: readonly boolean[],
): boolean[] {
  const value = record[key] ?? fallback;
  if (!Array.isArray(value) || value.length === 0) {
    throw new DiffusionConfigError(`${fieldName(context, key)} must be a non-empty boolean array.`);
  }
  return value.map((entry, index) => {
    if (typeof entry !== "boolean") {
      throw new DiffusionConfigError(
        `${fieldName(context, `${key}[${index}]`)} must be a boolean.`,
      );
    }
    return entry;
  });
}

function optionalRopeAxes(
  record: Record<string, unknown>,
  key: string,
  context: string,
): QwenImageRopeAxes {
  const value = record[key] ?? [16, 56, 56];
  if (!Array.isArray(value) || value.length !== 3) {
    throw new DiffusionConfigError(`${fieldName(context, key)} must be a 3-item integer array.`);
  }
  const axes = value.map((entry, index) => {
    if (typeof entry !== "number" || !Number.isInteger(entry) || entry <= 0) {
      throw new DiffusionConfigError(
        `${fieldName(context, `${key}[${index}]`)} must be a positive integer.`,
      );
    }
    return entry;
  });
  const [first, second, third] = axes;
  if (first === undefined || second === undefined || third === undefined) {
    throw new DiffusionConfigError(`${fieldName(context, key)} must be a 3-item integer array.`);
  }
  return [first, second, third];
}

function reverseBooleanList(values: readonly boolean[]): boolean[] {
  return [...values].reverse();
}

function componentConfigPath(
  manifest: DiffusionSnapshotManifest,
  componentName: "transformer" | "vae",
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

/** Parse a Diffusers Qwen-Image transformer `config.json` into package-owned terms. */
export function parseQwenImageTransformerConfig(rawConfig: unknown): QwenImageTransformerConfig {
  const context = "transformer/config.json";
  const record = expectRecord(rawConfig, context);
  optionalClassName(record, "QwenImageTransformer2DModel", context);

  const patchSize = optionalPositiveInteger(record, "patch_size", context, 2);
  const inChannels = optionalPositiveInteger(record, "in_channels", context, 64);
  const outChannels = optionalPositiveInteger(record, "out_channels", context, 16);
  const packedLatentChannels = patchSize * patchSize * outChannels;
  if (packedLatentChannels !== inChannels) {
    throw new DiffusionConfigError(
      `${fieldName(context, "in_channels")} must equal patch_size^2 * out_channels.`,
    );
  }

  const attentionHeadDim = optionalPositiveInteger(record, "attention_head_dim", context, 128);
  const numAttentionHeads = optionalPositiveInteger(record, "num_attention_heads", context, 24);
  const axesDimsRope = optionalRopeAxes(record, "axes_dims_rope", context);
  const ropeDim = axesDimsRope.reduce((sum, axis) => sum + axis, 0);
  if (ropeDim !== attentionHeadDim) {
    throw new DiffusionConfigError(
      `${fieldName(context, "axes_dims_rope")} sum must equal attention_head_dim.`,
    );
  }

  return {
    patchSize,
    inChannels,
    outChannels,
    latentChannels: outChannels,
    packedLatentChannels,
    numLayers: optionalPositiveInteger(record, "num_layers", context, 60),
    attentionHeadDim,
    numAttentionHeads,
    hiddenSize: attentionHeadDim * numAttentionHeads,
    jointAttentionDim: optionalPositiveInteger(record, "joint_attention_dim", context, 3584),
    guidanceEmbeds: optionalBoolean(record, "guidance_embeds", context, false),
    axesDimsRope,
    ropeTheta: 10000,
    zeroCondT: optionalBoolean(record, "zero_cond_t", context, false),
    useAdditionalTCond: optionalBoolean(record, "use_additional_t_cond", context, false),
    useLayer3dRope: optionalBoolean(record, "use_layer3d_rope", context, false),
    rawConfig: record,
  };
}

/** Parse a Diffusers Qwen-Image `AutoencoderKLQwenImage` config into package-owned terms. */
export function parseQwenImageAutoencoderConfig(rawConfig: unknown): QwenImageAutoencoderConfig {
  const context = "vae/config.json";
  const record = expectRecord(rawConfig, context);
  optionalClassName(record, "AutoencoderKLQwenImage", context);

  const latentChannels = optionalPositiveInteger(record, "z_dim", context, 16);
  const dimMultipliers = optionalPositiveIntegerList(record, "dim_mult", context, [1, 2, 4, 4]);
  const temporalDownsample = optionalBooleanList(record, "temperal_downsample", context, [
    false,
    true,
    true,
  ]);
  if (temporalDownsample.length !== dimMultipliers.length - 1) {
    throw new DiffusionConfigError(
      `${fieldName(context, "temperal_downsample")} length must be one less than dim_mult length.`,
    );
  }
  const latentsMean = optionalFiniteNumberList(
    record,
    "latents_mean",
    context,
    DEFAULT_LATENTS_MEAN,
  );
  const latentsStd = optionalFiniteNumberList(record, "latents_std", context, DEFAULT_LATENTS_STD);
  if (latentsMean.length !== latentChannels || latentsStd.length !== latentChannels) {
    throw new DiffusionConfigError(
      `${fieldName(context, "latents_mean")} and ${fieldName(
        context,
        "latents_std",
      )} must match z_dim.`,
    );
  }

  return {
    baseDim: optionalPositiveInteger(record, "base_dim", context, 96),
    latentChannels,
    latentChannelsOut: 2 * latentChannels,
    dimMultipliers,
    numResBlocks: optionalPositiveInteger(record, "num_res_blocks", context, 2),
    attentionScales: optionalFiniteNumberList(record, "attn_scales", context, []),
    temporalDownsample,
    temporalUpsample: reverseBooleanList(temporalDownsample),
    dropout: optionalFiniteNumber(record, "dropout", context, 0),
    inputChannels: optionalPositiveInteger(record, "input_channels", context, 3),
    latentsMean,
    latentsStd,
    spatialCompressionRatio: 2 ** temporalDownsample.length,
    rawConfig: record,
  };
}

/** Load Qwen-Image component configs from an inspected local snapshot manifest. */
export async function loadQwenImageComponentConfigs(
  manifest: DiffusionSnapshotManifest,
): Promise<QwenImageComponentConfigs> {
  if (manifest.modelIndex.kind !== "qwen-image") {
    throw new DiffusionConfigError(
      `Qwen-Image component configs do not support ${manifest.modelIndex.kind}.`,
    );
  }
  return {
    pipelineKind: "qwen-image",
    vae: parseQwenImageAutoencoderConfig(
      await readComponentJson(componentConfigPath(manifest, "vae"), "vae/config.json"),
    ),
    transformer: parseQwenImageTransformerConfig(
      await readComponentJson(
        componentConfigPath(manifest, "transformer"),
        "transformer/config.json",
      ),
    ),
  };
}
