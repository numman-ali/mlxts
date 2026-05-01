import { basename } from "path";

import { DiffusionConfigError } from "../../errors";
import type { DiffusionSnapshotManifest } from "../../pretrained/snapshot-manifest";
import { type FluxAutoencoderConfig, parseFluxAutoencoderConfig } from "../flux/config";

export type ZImageRopeAxes = readonly [number, number, number];

export type ZImageRopeAxisLengths = readonly [number, number, number];

export const Z_IMAGE_SEQUENCE_MULTIPLE = 32;
export const Z_IMAGE_LATENT_PAD_DIM = 64;

/** Patch geometry supported by a Z-Image transformer config. */
export type ZImagePatchGeometry = {
  patchSize: number;
  framePatchSize: number;
  packedLatentChannels: number;
};

/** Package-native config for the Z-Image `ZImageTransformer2DModel` component. */
export type ZImageTransformerConfig = {
  patchGeometries: readonly ZImagePatchGeometry[];
  inChannels: number;
  outChannels: number;
  hiddenSize: number;
  numLayers: number;
  numRefinerLayers: number;
  numAttentionHeads: number;
  numKeyValueHeads: number;
  attentionHeadDim: number;
  normEps: number;
  qkNorm: boolean;
  captionFeatureDim: number;
  siglipFeatureDim: null;
  ropeTheta: number;
  timestepScale: number;
  sequenceMultiple: typeof Z_IMAGE_SEQUENCE_MULTIPLE;
  latentPadDim: typeof Z_IMAGE_LATENT_PAD_DIM;
  axesDims: ZImageRopeAxes;
  axesLens: ZImageRopeAxisLengths;
  rawConfig: Record<string, unknown>;
};

export type ZImageAutoencoderConfig = FluxAutoencoderConfig;

/** Configs required before Z-Image model construction can begin. */
export type ZImageComponentConfigs = {
  pipelineKind: "z-image";
  vae: ZImageAutoencoderConfig;
  transformer: ZImageTransformerConfig;
};

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

function optionalNull(record: Record<string, unknown>, key: string, context: string): null {
  const value = record[key];
  if (value !== undefined && value !== null) {
    throw new DiffusionConfigError(`${fieldName(context, key)} is not supported yet.`);
  }
  return null;
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

function optionalRopeAxes(
  record: Record<string, unknown>,
  key: string,
  context: string,
): ZImageRopeAxes {
  const value = record[key] ?? [32, 48, 48];
  if (!Array.isArray(value) || value.length !== 3) {
    throw new DiffusionConfigError(`${fieldName(context, key)} must be a 3-item integer array.`);
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
  const [first, second, third] = axes;
  if (first === undefined || second === undefined || third === undefined) {
    throw new DiffusionConfigError(`${fieldName(context, key)} must be a 3-item integer array.`);
  }
  return [first, second, third];
}

function optionalRopeAxisLengths(
  record: Record<string, unknown>,
  key: string,
  context: string,
): ZImageRopeAxisLengths {
  const value = record[key] ?? [1024, 512, 512];
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

function parsePatchGeometries(
  record: Record<string, unknown>,
  context: string,
  inChannels: number,
): ZImagePatchGeometry[] {
  const patchSizes = optionalPositiveIntegerList(record, "all_patch_size", context, [2]);
  const framePatchSizes = optionalPositiveIntegerList(record, "all_f_patch_size", context, [1]);
  if (patchSizes.length !== framePatchSizes.length) {
    throw new DiffusionConfigError(
      `${fieldName(context, "all_patch_size")} length must equal ${fieldName(
        context,
        "all_f_patch_size",
      )} length.`,
    );
  }
  return patchSizes.map((patchSize, index) => {
    const framePatchSize = framePatchSizes[index];
    if (framePatchSize === undefined) {
      throw new DiffusionConfigError(`${fieldName(context, "all_f_patch_size")} is malformed.`);
    }
    return {
      patchSize,
      framePatchSize,
      packedLatentChannels: patchSize * patchSize * framePatchSize * inChannels,
    };
  });
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

/** Parse a Diffusers Z-Image transformer `config.json` into package-owned terms. */
export function parseZImageTransformerConfig(rawConfig: unknown): ZImageTransformerConfig {
  const context = "transformer/config.json";
  const record = expectRecord(rawConfig, context);
  optionalClassName(record, "ZImageTransformer2DModel", context);

  const inChannels = optionalPositiveInteger(record, "in_channels", context, 16);
  const hiddenSize = optionalPositiveInteger(record, "dim", context, 3840);
  const numAttentionHeads = optionalPositiveInteger(record, "n_heads", context, 30);
  if (hiddenSize % numAttentionHeads !== 0) {
    throw new DiffusionConfigError(`${fieldName(context, "dim")} must divide n_heads evenly.`);
  }
  const attentionHeadDim = hiddenSize / numAttentionHeads;
  const axesDims = optionalRopeAxes(record, "axes_dims", context);
  const ropeDim = axesDims.reduce((sum, axis) => sum + axis, 0);
  if (ropeDim !== attentionHeadDim) {
    throw new DiffusionConfigError(
      `${fieldName(context, "axes_dims")} sum must equal dim/n_heads.`,
    );
  }

  const numKeyValueHeads = optionalPositiveInteger(
    record,
    "n_kv_heads",
    context,
    numAttentionHeads,
  );
  if (numAttentionHeads % numKeyValueHeads !== 0) {
    throw new DiffusionConfigError(
      `${fieldName(context, "n_heads")} must be divisible by n_kv_heads.`,
    );
  }

  return {
    patchGeometries: parsePatchGeometries(record, context, inChannels),
    inChannels,
    outChannels: inChannels,
    hiddenSize,
    numLayers: optionalPositiveInteger(record, "n_layers", context, 30),
    numRefinerLayers: optionalPositiveInteger(record, "n_refiner_layers", context, 2),
    numAttentionHeads,
    numKeyValueHeads,
    attentionHeadDim,
    normEps: optionalFiniteNumber(record, "norm_eps", context, 1e-5),
    qkNorm: optionalBoolean(record, "qk_norm", context, true),
    captionFeatureDim: optionalPositiveInteger(record, "cap_feat_dim", context, 2560),
    siglipFeatureDim: optionalNull(record, "siglip_feat_dim", context),
    ropeTheta: optionalFiniteNumber(record, "rope_theta", context, 256),
    timestepScale: optionalFiniteNumber(record, "t_scale", context, 1000),
    sequenceMultiple: Z_IMAGE_SEQUENCE_MULTIPLE,
    latentPadDim: Z_IMAGE_LATENT_PAD_DIM,
    axesDims,
    axesLens: optionalRopeAxisLengths(record, "axes_lens", context),
    rawConfig: record,
  };
}

/** Parse a Diffusers Z-Image `AutoencoderKL` config into package-owned terms. */
export function parseZImageAutoencoderConfig(rawConfig: unknown): ZImageAutoencoderConfig {
  return parseFluxAutoencoderConfig(rawConfig);
}

/** Load Z-Image component configs from an inspected local snapshot manifest. */
export async function loadZImageComponentConfigs(
  manifest: DiffusionSnapshotManifest,
): Promise<ZImageComponentConfigs> {
  if (manifest.modelIndex.kind !== "z-image") {
    throw new DiffusionConfigError(
      `Z-Image component configs do not support ${manifest.modelIndex.kind}.`,
    );
  }
  return {
    pipelineKind: "z-image",
    vae: parseZImageAutoencoderConfig(
      await readComponentJson(componentConfigPath(manifest, "vae"), "vae/config.json"),
    ),
    transformer: parseZImageTransformerConfig(
      await readComponentJson(
        componentConfigPath(manifest, "transformer"),
        "transformer/config.json",
      ),
    ),
  };
}
