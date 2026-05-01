import { basename } from "path";

import { DiffusionConfigError } from "../../errors";
import type { DiffusionSnapshotManifest } from "../../pretrained/snapshot-manifest";

export type FluxRopeAxes = readonly [number, number, number];

/** Autoencoder encoder block supported by the FLUX VAE path. */
export type FluxVaeDownBlockType = "DownEncoderBlock2D";

/** Autoencoder decoder block supported by the FLUX VAE path. */
export type FluxVaeUpBlockType = "UpDecoderBlock2D";

/** Package-native config for the FLUX.1 `FluxTransformer2DModel` component. */
export type FluxTransformerConfig = {
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
  pooledProjectionDim: number;
  guidanceEmbeds: boolean;
  axesDimsRope: FluxRopeAxes;
  ropeTheta: number;
  qkvBias: boolean;
  rawConfig: Record<string, unknown>;
};

/** Package-native config for the FLUX.1 AutoencoderKL component. */
export type FluxAutoencoderConfig = {
  inChannels: number;
  outChannels: number;
  latentChannels: number;
  latentChannelsOut: number;
  blockOutChannels: readonly number[];
  layersPerBlock: number;
  normNumGroups: number;
  scalingFactor: number;
  shiftFactor: number;
  vaeScaleFactor: number;
  downBlockTypes: readonly FluxVaeDownBlockType[];
  upBlockTypes: readonly FluxVaeUpBlockType[];
  forceUpcast: boolean;
  rawConfig: Record<string, unknown>;
};

/** Configs required before FLUX.1 model construction can begin. */
export type FluxComponentConfigs = {
  pipelineKind: "flux";
  vae: FluxAutoencoderConfig;
  transformer: FluxTransformerConfig;
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

const FLUX_TRANSFORMER_CONFIG_KEYS = new Set([
  "_class_name",
  "_diffusers_version",
  "_name_or_path",
  "attention_head_dim",
  "axes_dims_rope",
  "guidance_embeds",
  "in_channels",
  "joint_attention_dim",
  "num_attention_heads",
  "num_layers",
  "num_single_layers",
  "out_channels",
  "patch_size",
  "pooled_projection_dim",
]);

function expectRecord(value: unknown, context: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new DiffusionConfigError(`${context} must be a JSON object.`);
  }
  return Object.fromEntries(Object.entries(value));
}

function rejectUnknownFields(
  record: Record<string, unknown>,
  knownKeys: ReadonlySet<string>,
  context: string,
): void {
  for (const key of Object.keys(record)) {
    if (!knownKeys.has(key)) {
      throw new DiffusionConfigError(`${fieldName(context, key)} is not supported yet.`);
    }
  }
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

function optionalExactPositiveInteger(
  record: Record<string, unknown>,
  key: string,
  context: string,
  expected: number,
): number {
  const value = record[key];
  if (value === undefined || value === null) {
    return expected;
  }
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) {
    throw new DiffusionConfigError(
      `${fieldName(context, key)} must be a positive integer, got ${describeConfigValue(value)}.`,
    );
  }
  if (value !== expected) {
    throw new DiffusionConfigError(
      `${fieldName(context, key)}=${value} is not supported; expected ${expected}.`,
    );
  }
  return value;
}

function optionalNullableExactPositiveInteger(
  record: Record<string, unknown>,
  key: string,
  context: string,
  expected: number,
): number {
  const value = record[key];
  if (value === undefined || value === null) {
    return expected;
  }
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) {
    throw new DiffusionConfigError(
      `${fieldName(context, key)} must be a positive integer or null, got ${describeConfigValue(
        value,
      )}.`,
    );
  }
  if (value !== expected) {
    throw new DiffusionConfigError(
      `${fieldName(context, key)}=${value} is not supported; expected null or ${expected}.`,
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

function optionalExactFiniteNumber(
  record: Record<string, unknown>,
  key: string,
  context: string,
  expected: number,
): number {
  const value = record[key];
  if (value === undefined || value === null) {
    return expected;
  }
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new DiffusionConfigError(
      `${fieldName(context, key)} must be a finite number, got ${describeConfigValue(value)}.`,
    );
  }
  if (value !== expected) {
    throw new DiffusionConfigError(
      `${fieldName(context, key)}=${value} is not supported; expected ${expected}.`,
    );
  }
  return value;
}

function requiredPositiveInteger(
  record: Record<string, unknown>,
  key: string,
  context: string,
): number {
  const value = record[key];
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) {
    throw new DiffusionConfigError(
      `${fieldName(context, key)} must be a positive integer, got ${describeConfigValue(value)}.`,
    );
  }
  return value;
}

function requiredPositiveIntegerList(
  record: Record<string, unknown>,
  key: string,
  context: string,
): number[] {
  const value = record[key];
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
): FluxVaeDownBlockType[] {
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
): FluxVaeUpBlockType[] {
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
    throw new DiffusionConfigError(`${fieldName(context, key)} is not supported yet.`);
  }
}

function optionalRopeAxes(
  record: Record<string, unknown>,
  key: string,
  context: string,
): FluxRopeAxes {
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
  const parsed: FluxRopeAxes = [first, second, third];
  if (parsed[0] !== 16 || parsed[1] !== 56 || parsed[2] !== 56) {
    throw new DiffusionConfigError(
      `${fieldName(context, key)} only supports FLUX.1 axes [16, 56, 56].`,
    );
  }
  return parsed;
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

/** Parse a Diffusers FLUX.1 AutoencoderKL `config.json` into package-owned terms. */
export function parseFluxAutoencoderConfig(rawConfig: unknown): FluxAutoencoderConfig {
  const context = "vae/config.json";
  const record = expectRecord(rawConfig, context);
  optionalClassName(record, "AutoencoderKL", context);
  rejectNonNull(record, "latents_mean", context);
  rejectNonNull(record, "latents_std", context);
  rejectNonNull(record, "patch_size", context);

  const blockOutChannels = requiredPositiveIntegerList(record, "block_out_channels", context);
  const vaeScaleFactor = 2 ** (blockOutChannels.length - 1);
  if (vaeScaleFactor !== 8) {
    throw new DiffusionConfigError(
      `${fieldName(context, "block_out_channels")} must imply 8x VAE scale.`,
    );
  }
  const latentChannels = optionalExactPositiveInteger(record, "latent_channels", context, 16);
  return {
    inChannels: optionalExactPositiveInteger(record, "in_channels", context, 3),
    outChannels: optionalExactPositiveInteger(record, "out_channels", context, 3),
    latentChannels,
    latentChannelsOut: 2 * latentChannels,
    blockOutChannels,
    layersPerBlock: optionalExactPositiveInteger(record, "layers_per_block", context, 2),
    normNumGroups: requiredPositiveInteger(record, "norm_num_groups", context),
    scalingFactor: optionalExactFiniteNumber(record, "scaling_factor", context, 0.3611),
    shiftFactor: optionalExactFiniteNumber(record, "shift_factor", context, 0.1159),
    vaeScaleFactor,
    downBlockTypes: parseVaeDownBlockTypes(record, context, blockOutChannels.length),
    upBlockTypes: parseVaeUpBlockTypes(record, context, blockOutChannels.length),
    forceUpcast: optionalBoolean(record, "force_upcast", context, false),
    rawConfig: record,
  };
}

/** Parse a Diffusers FLUX.1 transformer `config.json` into package-owned terms. */
export function parseFluxTransformerConfig(rawConfig: unknown): FluxTransformerConfig {
  const context = "transformer/config.json";
  const record = expectRecord(rawConfig, context);
  rejectUnknownFields(record, FLUX_TRANSFORMER_CONFIG_KEYS, context);
  optionalClassName(record, "FluxTransformer2DModel", context);

  const inChannels = optionalExactPositiveInteger(record, "in_channels", context, 64);
  const attentionHeadDim = optionalExactPositiveInteger(record, "attention_head_dim", context, 128);
  const numAttentionHeads = optionalExactPositiveInteger(
    record,
    "num_attention_heads",
    context,
    24,
  );
  const hiddenSize = attentionHeadDim * numAttentionHeads;
  const axesDimsRope = optionalRopeAxes(record, "axes_dims_rope", context);
  const ropeDim = axesDimsRope.reduce((sum, axis) => sum + axis, 0);
  if (ropeDim !== attentionHeadDim) {
    throw new DiffusionConfigError(
      `${fieldName(context, "axes_dims_rope")} sum must equal attention_head_dim.`,
    );
  }

  return {
    patchSize: optionalExactPositiveInteger(record, "patch_size", context, 1),
    inChannels,
    latentChannels: inChannels / 4,
    outChannels: optionalNullableExactPositiveInteger(record, "out_channels", context, inChannels),
    numLayers: optionalExactPositiveInteger(record, "num_layers", context, 19),
    numSingleLayers: optionalExactPositiveInteger(record, "num_single_layers", context, 38),
    attentionHeadDim,
    numAttentionHeads,
    hiddenSize,
    mlpRatio: 4,
    jointAttentionDim: optionalExactPositiveInteger(record, "joint_attention_dim", context, 4096),
    pooledProjectionDim: optionalExactPositiveInteger(
      record,
      "pooled_projection_dim",
      context,
      768,
    ),
    guidanceEmbeds: optionalBoolean(record, "guidance_embeds", context, false),
    axesDimsRope,
    ropeTheta: 10000,
    qkvBias: true,
    rawConfig: record,
  };
}

/** Load FLUX.1 component configs from an inspected local snapshot manifest. */
export async function loadFluxComponentConfigs(
  manifest: DiffusionSnapshotManifest,
): Promise<FluxComponentConfigs> {
  if (manifest.modelIndex.kind !== "flux") {
    throw new DiffusionConfigError(
      `Flux component configs do not support ${manifest.modelIndex.kind}.`,
    );
  }
  return {
    pipelineKind: "flux",
    vae: parseFluxAutoencoderConfig(
      await readComponentJson(componentConfigPath(manifest, "vae"), "vae/config.json"),
    ),
    transformer: parseFluxTransformerConfig(
      await readComponentJson(
        componentConfigPath(manifest, "transformer"),
        "transformer/config.json",
      ),
    ),
  };
}
