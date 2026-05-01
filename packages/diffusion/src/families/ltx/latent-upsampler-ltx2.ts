import type { MxArray } from "@mlxts/core";
import {
  conv2d,
  formatShape,
  MxArray as MxArrayCtor,
  reshape,
  retainArray,
  slice,
  transpose,
} from "@mlxts/core";
import { Conv2d, Conv3d, GroupNorm, Module, silu } from "@mlxts/nn";

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
  expectLtxVideoVaeVolume,
  ltxVideoBcfhwToBfhwc,
  ltxVideoBfhwcToBcfhw,
} from "./autoencoder-volume";
import {
  LtxVideoLatentUpsamplerResBlock2d,
  LtxVideoLatentUpsamplerResBlock3d,
  pixelShuffleLtxLatents1d,
  pixelShuffleLtxLatents3d,
} from "./latent-upsampler";
import { checkedModule } from "./tensor-utils";

export type Ltx2LatentUpsamplerDims = 2 | 3;

/** Package-native config for Diffusers `LTX2LatentUpsamplerModel`. */
export type Ltx2LatentUpsamplerConfig = {
  inChannels: number;
  midChannels: number;
  numBlocksPerStage: number;
  dims: Ltx2LatentUpsamplerDims;
  spatialUpsample: boolean;
  temporalUpsample: boolean;
  rationalSpatialScale: 0.75 | 1.5 | 2 | 4;
  useRationalResampler: boolean;
  rawConfig: Record<string, unknown>;
};

const LTX2_LATENT_UPSAMPLER_KEYS = new Set([
  "_class_name",
  "_diffusers_version",
  "_name_or_path",
  "dims",
  "in_channels",
  "mid_channels",
  "num_blocks_per_stage",
  "rational_spatial_scale",
  "spatial_upsample",
  "temporal_upsample",
  "use_rational_resampler",
]);

const GROUP_NORM_GROUPS = 32;
const RATIONAL_SCALES = new Map<
  Ltx2LatentUpsamplerConfig["rationalSpatialScale"],
  [number, number]
>([
  [0.75, [3, 4]],
  [1.5, [3, 2]],
  [2, [2, 1]],
  [4, [4, 1]],
]);

function optionalNonNegativeInteger(
  record: Record<string, unknown>,
  key: string,
  context: string,
  fallback: number,
): number {
  const value = record[key];
  if (value === undefined || value === null) {
    return fallback;
  }
  if (typeof value !== "number" || !Number.isInteger(value) || value < 0) {
    throw new DiffusionConfigError(`${fieldName(context, key)} must be a non-negative integer.`);
  }
  return value;
}

function parseDims(record: Record<string, unknown>, context: string): Ltx2LatentUpsamplerDims {
  const dims = optionalPositiveInteger(record, "dims", context, 3);
  if (dims !== 2 && dims !== 3) {
    throw new DiffusionConfigError(`${fieldName(context, "dims")} must be 2 or 3.`);
  }
  return dims;
}

function parseRationalScale(
  record: Record<string, unknown>,
  context: string,
): Ltx2LatentUpsamplerConfig["rationalSpatialScale"] {
  const scale = optionalPositiveNumber(record, "rational_spatial_scale", context, 2);
  if (scale === 0.75 || scale === 1.5 || scale === 2 || scale === 4) {
    return scale;
  }
  throw new DiffusionConfigError(
    `${fieldName(context, "rational_spatial_scale")} must be one of 0.75, 1.5, 2, or 4.`,
  );
}

function validateConfig(config: Ltx2LatentUpsamplerConfig): void {
  if (!config.spatialUpsample && !config.temporalUpsample) {
    throw new Error("Ltx2LatentUpsamplerModel: one upsample axis must be enabled.");
  }
  if (config.dims === 2 && config.temporalUpsample) {
    throw new Error("Ltx2LatentUpsamplerModel: temporal upsampling requires dims=3.");
  }
  if (config.midChannels % GROUP_NORM_GROUPS !== 0) {
    throw new Error("Ltx2LatentUpsamplerModel: midChannels must be divisible by 32.");
  }
  if (!config.useRationalResampler && config.rationalSpatialScale !== 2) {
    throw new Error("Ltx2LatentUpsamplerModel: non-rational spatial upsampling uses scale 2.");
  }
}

/** Parse a Diffusers `LTX2LatentUpsamplerModel` config. */
export function parseLtx2LatentUpsamplerConfig(rawConfig: unknown): Ltx2LatentUpsamplerConfig {
  const context = "latent_upsampler/config.json";
  const record = expectRecord(rawConfig, context);
  rejectUnknownFields(record, LTX2_LATENT_UPSAMPLER_KEYS, context);
  optionalClassName(record, "LTX2LatentUpsamplerModel", context);
  const parsed: Ltx2LatentUpsamplerConfig = {
    inChannels: optionalPositiveInteger(record, "in_channels", context, 128),
    midChannels: optionalPositiveInteger(record, "mid_channels", context, 1024),
    numBlocksPerStage: optionalNonNegativeInteger(record, "num_blocks_per_stage", context, 4),
    dims: parseDims(record, context),
    spatialUpsample: optionalBoolean(record, "spatial_upsample", context, true),
    temporalUpsample: optionalBoolean(record, "temporal_upsample", context, false),
    rationalSpatialScale: parseRationalScale(record, context),
    useRationalResampler: optionalBoolean(record, "use_rational_resampler", context, true),
    rawConfig: record,
  };
  validateConfig(parsed);
  return parsed;
}

function expectBcfhw(x: MxArray, owner: string): readonly [number, number, number, number, number] {
  const [batch, channels, frames, height, width] = x.shape;
  if (
    x.shape.length !== 5 ||
    batch === undefined ||
    channels === undefined ||
    frames === undefined ||
    height === undefined ||
    width === undefined
  ) {
    throw new Error(`${owner}: expected BCFHW volume, got ${formatShape(x.shape)}.`);
  }
  return [batch, channels, frames, height, width];
}

function expectBhwc(x: MxArray, owner: string): readonly [number, number, number, number] {
  const [batch, height, width, channels] = x.shape;
  if (
    x.shape.length !== 4 ||
    batch === undefined ||
    height === undefined ||
    width === undefined ||
    channels === undefined
  ) {
    throw new Error(`${owner}: expected BHWC image batch, got ${formatShape(x.shape)}.`);
  }
  return [batch, height, width, channels];
}

function bfhwcToFrameBatch(x: MxArray): MxArray {
  const [batch, frames, height, width, channels] = expectLtxVideoVaeVolume(x, "bfhwcToFrameBatch");
  return reshape(x, [batch * frames, height, width, channels]);
}

function frameBatchToBfhwc(x: MxArray, batch: number, frames: number): MxArray {
  const [flatBatch, height, width, channels] = expectBhwc(x, "frameBatchToBfhwc");
  if (flatBatch !== batch * frames) {
    throw new Error("frameBatchToBfhwc: flat batch does not match video batch and frames.");
  }
  return reshape(x, [batch, frames, height, width, channels]);
}

function dropFirstFrame(x: MxArray): MxArray {
  const [batch, frames, height, width, channels] = expectLtxVideoVaeVolume(x, "dropFirstFrame");
  return slice(x, [0, 1, 0, 0, 0], [batch, frames, height, width, channels]);
}

/** Diffusers-compatible spatial PixelShuffleND with configurable rational numerator. */
export function pixelShuffleLtx2Latents2d(x: MxArray, upscaleFactor = 2): MxArray {
  const [batch, height, width, packedChannels] = expectBhwc(x, "pixelShuffleLtx2Latents2d");
  const upscaleArea = upscaleFactor * upscaleFactor;
  if (!Number.isInteger(upscaleFactor) || upscaleFactor <= 0) {
    throw new Error("pixelShuffleLtx2Latents2d: upscaleFactor must be a positive integer.");
  }
  if (packedChannels % upscaleArea !== 0) {
    throw new Error("pixelShuffleLtx2Latents2d: channels must be divisible by upscale area.");
  }
  const channels = packedChannels / upscaleArea;
  using grid = reshape(x, [batch, height, width, channels, upscaleFactor, upscaleFactor]);
  using shuffled = transpose(grid, [0, 1, 4, 2, 5, 3]);
  return reshape(shuffled, [batch, height * upscaleFactor, width * upscaleFactor, channels]);
}

function binomialKernelValues(
  channels: number,
  kernelSize: number,
  dtype: MxArray["dtype"],
): MxArray {
  const row = kernelSize === 5 ? [1, 4, 6, 4, 1] : [1, 2, 1];
  const normalizer = row.reduce((sum, value) => sum + value, 0) ** 2;
  const values = new Float32Array(channels * kernelSize * kernelSize);
  for (let channel = 0; channel < channels; channel += 1) {
    for (let y = 0; y < kernelSize; y += 1) {
      for (let x = 0; x < kernelSize; x += 1) {
        values[(channel * kernelSize + y) * kernelSize + x] =
          ((row[y] ?? 0) * (row[x] ?? 0)) / normalizer;
      }
    }
  }
  return MxArrayCtor.fromData(values, [channels, kernelSize, kernelSize, 1], dtype);
}

function blurDownsampleSpatial(x: MxArray, channels: number, stride: number): MxArray {
  if (stride === 1) {
    return retainArray(x);
  }
  const kernelSize = 5;
  using weight = binomialKernelValues(channels, kernelSize, x.dtype);
  return conv2d(x, weight, [stride, stride], [2, 2], [1, 1], channels);
}

export class Ltx2LatentSpatialUpsampler extends Module {
  conv: Conv2d;
  #upscaleFactor: number;

  constructor(channels: number, upscaleFactor = 2) {
    super();
    this.#upscaleFactor = upscaleFactor;
    this.conv = new Conv2d(channels, channels * upscaleFactor * upscaleFactor, 3, 1, 1);
  }

  forward(x: MxArray): MxArray {
    using projected = this.conv.forward(x);
    return pixelShuffleLtx2Latents2d(projected, this.#upscaleFactor);
  }
}

export class Ltx2LatentSpatialRationalResampler extends Module {
  conv: Conv2d;
  #channels: number;
  #numerator: number;
  #denominator: number;

  constructor(channels: number, scale: Ltx2LatentUpsamplerConfig["rationalSpatialScale"]) {
    super();
    const pair = RATIONAL_SCALES.get(scale);
    if (pair === undefined) {
      throw new Error("Ltx2LatentSpatialRationalResampler: unsupported scale.");
    }
    const [numerator, denominator] = pair;
    this.#channels = channels;
    this.#numerator = numerator;
    this.#denominator = denominator;
    this.conv = new Conv2d(channels, channels * numerator * numerator, 3, 1, 1);
  }

  forward(x: MxArray): MxArray {
    using projected = this.conv.forward(x);
    using upsampled = pixelShuffleLtx2Latents2d(projected, this.#numerator);
    return blurDownsampleSpatial(upsampled, this.#channels, this.#denominator);
  }
}

export class Ltx2LatentTemporalUpsampler extends Module {
  conv: Conv3d;
  #spatial: boolean;

  constructor(channels: number, spatial: boolean) {
    super();
    this.#spatial = spatial;
    this.conv = new Conv3d(channels, channels * (spatial ? 8 : 2), 3, 1, 1);
  }

  forward(x: MxArray): MxArray {
    using projected = this.conv.forward(x);
    using shuffled = this.#spatial
      ? pixelShuffleLtxLatents3d(projected)
      : pixelShuffleLtxLatents1d(projected);
    return dropFirstFrame(shuffled);
  }
}

/** Diffusers `LTX2LatentUpsamplerModel` with BCFHW tensor boundaries. */
export class Ltx2LatentUpsamplerModel extends Module {
  initialConv: Conv2d | Conv3d;
  initialNorm: GroupNorm;
  resBlocks: (LtxVideoLatentUpsamplerResBlock2d | LtxVideoLatentUpsamplerResBlock3d)[];
  upsampler:
    | Ltx2LatentSpatialUpsampler
    | Ltx2LatentSpatialRationalResampler
    | Ltx2LatentTemporalUpsampler;
  postUpsampleResBlocks: (LtxVideoLatentUpsamplerResBlock2d | LtxVideoLatentUpsamplerResBlock3d)[];
  finalConv: Conv2d | Conv3d;
  #config: Ltx2LatentUpsamplerConfig;

  constructor(config: Ltx2LatentUpsamplerConfig) {
    super();
    validateConfig(config);
    this.#config = config;
    if (config.dims === 2) {
      this.initialConv = new Conv2d(config.inChannels, config.midChannels, 3, 1, 1);
      this.resBlocks = Array.from(
        { length: config.numBlocksPerStage },
        () => new LtxVideoLatentUpsamplerResBlock2d(config.midChannels),
      );
      this.upsampler = config.useRationalResampler
        ? new Ltx2LatentSpatialRationalResampler(config.midChannels, config.rationalSpatialScale)
        : new Ltx2LatentSpatialUpsampler(config.midChannels);
      this.postUpsampleResBlocks = Array.from(
        { length: config.numBlocksPerStage },
        () => new LtxVideoLatentUpsamplerResBlock2d(config.midChannels),
      );
      this.finalConv = new Conv2d(config.midChannels, config.inChannels, 3, 1, 1);
    } else {
      this.initialConv = new Conv3d(config.inChannels, config.midChannels, 3, 1, 1);
      this.resBlocks = Array.from(
        { length: config.numBlocksPerStage },
        () => new LtxVideoLatentUpsamplerResBlock3d(config.midChannels),
      );
      this.upsampler = this.create3dUpsampler(config);
      this.postUpsampleResBlocks = Array.from(
        { length: config.numBlocksPerStage },
        () => new LtxVideoLatentUpsamplerResBlock3d(config.midChannels),
      );
      this.finalConv = new Conv3d(config.midChannels, config.inChannels, 3, 1, 1);
    }
    this.initialNorm = new GroupNorm(GROUP_NORM_GROUPS, config.midChannels);
  }

  get config(): Ltx2LatentUpsamplerConfig {
    return this.#config;
  }

  private create3dUpsampler(
    config: Ltx2LatentUpsamplerConfig,
  ): Ltx2LatentSpatialUpsampler | Ltx2LatentSpatialRationalResampler | Ltx2LatentTemporalUpsampler {
    if (config.temporalUpsample) {
      return new Ltx2LatentTemporalUpsampler(config.midChannels, config.spatialUpsample);
    }
    return config.useRationalResampler
      ? new Ltx2LatentSpatialRationalResampler(config.midChannels, config.rationalSpatialScale)
      : new Ltx2LatentSpatialUpsampler(config.midChannels);
  }

  forward(latents: MxArray): MxArray {
    const [batch, channels, frames] = expectBcfhw(latents, "Ltx2LatentUpsamplerModel.forward");
    if (channels !== this.#config.inChannels) {
      throw new Error(
        `Ltx2LatentUpsamplerModel.forward: expected ${this.#config.inChannels} channels, got ${channels}.`,
      );
    }
    if (this.#config.dims === 2) {
      return this.forward2d(latents, batch, frames);
    }
    return this.forward3d(latents, batch, frames);
  }

  #runResBlocks(
    hiddenStates: MxArray,
    blocks: readonly (LtxVideoLatentUpsamplerResBlock2d | LtxVideoLatentUpsamplerResBlock3d)[],
    owner: string,
  ): MxArray {
    let hidden = retainArray(hiddenStates);
    try {
      for (let index = 0; index < blocks.length; index += 1) {
        const block = checkedModule(blocks, index, owner);
        const next = block.forward(hidden);
        hidden.free();
        hidden = next;
      }
      return retainArray(hidden);
    } finally {
      hidden.free();
    }
  }

  private forward2d(latents: MxArray, batch: number, frames: number): MxArray {
    using bfhwc = ltxVideoBcfhwToBfhwc(latents);
    using frameBatch = bfhwcToFrameBatch(bfhwc);
    let hidden = this.initialConv.forward(frameBatch);
    try {
      const processed = this.runSharedStages(hidden, "Ltx2LatentUpsamplerModel");
      hidden.free();
      hidden = processed;
      using video = frameBatchToBfhwc(hidden, batch, frames);
      return ltxVideoBfhwcToBcfhw(video);
    } finally {
      hidden.free();
    }
  }

  private forward3d(latents: MxArray, batch: number, frames: number): MxArray {
    using bfhwc = ltxVideoBcfhwToBfhwc(latents);
    let hidden = this.initialConv.forward(bfhwc);
    try {
      const normalized = this.initialNorm.forward(hidden);
      hidden.free();
      hidden = normalized;
      const activated = silu(hidden);
      hidden.free();
      hidden = activated;
      const pre = this.#runResBlocks(hidden, this.resBlocks, "Ltx2LatentUpsamplerModel.resBlocks");
      hidden.free();
      hidden = pre;
      const upsampled = this.#config.temporalUpsample
        ? this.upsampler.forward(hidden)
        : this.forwardSpatialFrames(hidden, batch, frames);
      hidden.free();
      hidden = upsampled;
      const post = this.#runResBlocks(
        hidden,
        this.postUpsampleResBlocks,
        "Ltx2LatentUpsamplerModel.postUpsampleResBlocks",
      );
      hidden.free();
      hidden = post;
      const projected = this.finalConv.forward(hidden);
      hidden.free();
      hidden = projected;
      return ltxVideoBfhwcToBcfhw(hidden);
    } finally {
      hidden.free();
    }
  }

  private runSharedStages(hiddenStates: MxArray, owner: string): MxArray {
    let hidden = retainArray(hiddenStates);
    try {
      const normalized = this.initialNorm.forward(hidden);
      hidden.free();
      hidden = normalized;
      const activated = silu(hidden);
      hidden.free();
      hidden = activated;
      const pre = this.#runResBlocks(hidden, this.resBlocks, `${owner}.resBlocks`);
      hidden.free();
      hidden = pre;
      const upsampled = this.upsampler.forward(hidden);
      hidden.free();
      hidden = upsampled;
      const post = this.#runResBlocks(
        hidden,
        this.postUpsampleResBlocks,
        `${owner}.postUpsampleResBlocks`,
      );
      hidden.free();
      hidden = post;
      const projected = this.finalConv.forward(hidden);
      hidden.free();
      hidden = projected;
      return retainArray(hidden);
    } finally {
      hidden.free();
    }
  }

  private forwardSpatialFrames(hidden: MxArray, batch: number, frames: number): MxArray {
    using frameBatch = bfhwcToFrameBatch(hidden);
    using spatial = this.upsampler.forward(frameBatch);
    return frameBatchToBfhwc(spatial, batch, frames);
  }
}
