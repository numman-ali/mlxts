import type { MxArray } from "@mlxts/core";
import { add, formatShape, reshape, retainArray, slice, transpose } from "@mlxts/core";
import { Conv2d, Conv3d, GroupNorm, Module, silu } from "@mlxts/nn";

import { DiffusionConfigError } from "../../errors";
import {
  expectRecord,
  fieldName,
  optionalBoolean,
  optionalClassName,
  optionalPositiveInteger,
  rejectUnknownFields,
} from "../flux2/config-parsing";
import {
  expectLtxVideoVaeVolume,
  ltxVideoBcfhwToBfhwc,
  ltxVideoBfhwcToBcfhw,
} from "./autoencoder-volume";
import { checkedModule } from "./tensor-utils";

export type LtxVideoLatentUpsamplerDims = 2 | 3;

/** Package-native config for Diffusers `LTXLatentUpsamplerModel`. */
export type LtxVideoLatentUpsamplerConfig = {
  inChannels: number;
  midChannels: number;
  numBlocksPerStage: number;
  dims: LtxVideoLatentUpsamplerDims;
  spatialUpsample: boolean;
  temporalUpsample: boolean;
  rawConfig: Record<string, unknown>;
};

const LTX_LATENT_UPSAMPLER_KEYS = new Set([
  "_class_name",
  "_diffusers_version",
  "_name_or_path",
  "dims",
  "in_channels",
  "mid_channels",
  "num_blocks_per_stage",
  "spatial_upsample",
  "temporal_upsample",
]);

const GROUP_NORM_GROUPS = 32;

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

function parseDims(record: Record<string, unknown>, context: string): LtxVideoLatentUpsamplerDims {
  const dims = optionalPositiveInteger(record, "dims", context, 3);
  if (dims !== 2 && dims !== 3) {
    throw new DiffusionConfigError(`${fieldName(context, "dims")} must be 2 or 3.`);
  }
  return dims;
}

function validateConfig(config: LtxVideoLatentUpsamplerConfig): void {
  if (!config.spatialUpsample && !config.temporalUpsample) {
    throw new Error("LtxVideoLatentUpsamplerModel: one upsample axis must be enabled.");
  }
  if (config.dims === 2 && config.temporalUpsample) {
    throw new Error("LtxVideoLatentUpsamplerModel: temporal upsampling requires dims=3.");
  }
  if (config.midChannels % GROUP_NORM_GROUPS !== 0) {
    throw new Error("LtxVideoLatentUpsamplerModel: midChannels must be divisible by 32.");
  }
}

/** Parse a Diffusers `LTXLatentUpsamplerModel` config. */
export function parseLtxVideoLatentUpsamplerConfig(
  rawConfig: unknown,
): LtxVideoLatentUpsamplerConfig {
  const context = "latent_upsampler/config.json";
  const record = expectRecord(rawConfig, context);
  rejectUnknownFields(record, LTX_LATENT_UPSAMPLER_KEYS, context);
  optionalClassName(record, "LTXLatentUpsamplerModel", context);
  const parsed: LtxVideoLatentUpsamplerConfig = {
    inChannels: optionalPositiveInteger(record, "in_channels", context, 128),
    midChannels: optionalPositiveInteger(record, "mid_channels", context, 512),
    numBlocksPerStage: optionalNonNegativeInteger(record, "num_blocks_per_stage", context, 4),
    dims: parseDims(record, context),
    spatialUpsample: optionalBoolean(record, "spatial_upsample", context, true),
    temporalUpsample: optionalBoolean(record, "temporal_upsample", context, false),
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

export function pixelShuffleLtxLatents2d(x: MxArray): MxArray {
  const [batch, height, width, packedChannels] = expectBhwc(x, "pixelShuffleLtxLatents2d");
  if (packedChannels % 4 !== 0) {
    throw new Error("pixelShuffleLtxLatents2d: channels must be divisible by 4.");
  }
  const channels = packedChannels / 4;
  using grid = reshape(x, [batch, height, width, channels, 2, 2]);
  using shuffled = transpose(grid, [0, 1, 4, 2, 5, 3]);
  return reshape(shuffled, [batch, height * 2, width * 2, channels]);
}

export function pixelShuffleLtxLatents1d(x: MxArray): MxArray {
  const [batch, frames, height, width, packedChannels] = expectLtxVideoVaeVolume(
    x,
    "pixelShuffleLtxLatents1d",
  );
  if (packedChannels % 2 !== 0) {
    throw new Error("pixelShuffleLtxLatents1d: channels must be divisible by 2.");
  }
  const channels = packedChannels / 2;
  using grid = reshape(x, [batch, frames, height, width, channels, 2]);
  using shuffled = transpose(grid, [0, 1, 5, 2, 3, 4]);
  return reshape(shuffled, [batch, frames * 2, height, width, channels]);
}

export function pixelShuffleLtxLatents3d(x: MxArray): MxArray {
  const [batch, frames, height, width, packedChannels] = expectLtxVideoVaeVolume(
    x,
    "pixelShuffleLtxLatents3d",
  );
  if (packedChannels % 8 !== 0) {
    throw new Error("pixelShuffleLtxLatents3d: channels must be divisible by 8.");
  }
  const channels = packedChannels / 8;
  using grid = reshape(x, [batch, frames, height, width, channels, 2, 2, 2]);
  using shuffled = transpose(grid, [0, 1, 5, 2, 6, 3, 7, 4]);
  return reshape(shuffled, [batch, frames * 2, height * 2, width * 2, channels]);
}

function dropFirstFrame(x: MxArray): MxArray {
  const [batch, frames, height, width, channels] = expectLtxVideoVaeVolume(x, "dropFirstFrame");
  return slice(x, [0, 1, 0, 0, 0], [batch, frames, height, width, channels]);
}

export class LtxVideoLatentUpsamplerResBlock2d extends Module {
  conv1: Conv2d;
  norm1: GroupNorm;
  conv2: Conv2d;
  norm2: GroupNorm;

  constructor(channels: number, midChannels = channels) {
    super();
    this.conv1 = new Conv2d(channels, midChannels, 3, 1, 1);
    this.norm1 = new GroupNorm(GROUP_NORM_GROUPS, midChannels);
    this.conv2 = new Conv2d(midChannels, channels, 3, 1, 1);
    this.norm2 = new GroupNorm(GROUP_NORM_GROUPS, channels);
  }

  forward(x: MxArray): MxArray {
    using first = this.conv1.forward(x);
    using firstNorm = this.norm1.forward(first);
    using firstActivated = silu(firstNorm);
    using second = this.conv2.forward(firstActivated);
    using secondNorm = this.norm2.forward(second);
    using residual = add(secondNorm, x);
    return silu(residual);
  }
}

export class LtxVideoLatentUpsamplerResBlock3d extends Module {
  conv1: Conv3d;
  norm1: GroupNorm;
  conv2: Conv3d;
  norm2: GroupNorm;

  constructor(channels: number, midChannels = channels) {
    super();
    this.conv1 = new Conv3d(channels, midChannels, 3, 1, 1);
    this.norm1 = new GroupNorm(GROUP_NORM_GROUPS, midChannels);
    this.conv2 = new Conv3d(midChannels, channels, 3, 1, 1);
    this.norm2 = new GroupNorm(GROUP_NORM_GROUPS, channels);
  }

  forward(x: MxArray): MxArray {
    using first = this.conv1.forward(x);
    using firstNorm = this.norm1.forward(first);
    using firstActivated = silu(firstNorm);
    using second = this.conv2.forward(firstActivated);
    using secondNorm = this.norm2.forward(second);
    using residual = add(secondNorm, x);
    return silu(residual);
  }
}

export class LtxVideoLatentSpatialUpsampler extends Module {
  conv: Conv2d;

  constructor(channels: number) {
    super();
    this.conv = new Conv2d(channels, channels * 4, 3, 1, 1);
  }

  forward(x: MxArray): MxArray {
    using projected = this.conv.forward(x);
    return pixelShuffleLtxLatents2d(projected);
  }
}

export class LtxVideoLatentTemporalUpsampler extends Module {
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

/** Diffusers `LTXLatentUpsamplerModel` with BCFHW tensor boundaries. */
export class LtxVideoLatentUpsamplerModel extends Module {
  initialConv: Conv2d | Conv3d;
  initialNorm: GroupNorm;
  resBlocks: (LtxVideoLatentUpsamplerResBlock2d | LtxVideoLatentUpsamplerResBlock3d)[];
  upsampler: LtxVideoLatentSpatialUpsampler | LtxVideoLatentTemporalUpsampler;
  postUpsampleResBlocks: (LtxVideoLatentUpsamplerResBlock2d | LtxVideoLatentUpsamplerResBlock3d)[];
  finalConv: Conv2d | Conv3d;
  #config: LtxVideoLatentUpsamplerConfig;

  constructor(config: LtxVideoLatentUpsamplerConfig) {
    super();
    validateConfig(config);
    this.#config = config;
    if (config.dims === 2) {
      this.initialConv = new Conv2d(config.inChannels, config.midChannels, 3, 1, 1);
      this.resBlocks = Array.from(
        { length: config.numBlocksPerStage },
        () => new LtxVideoLatentUpsamplerResBlock2d(config.midChannels),
      );
      this.upsampler = new LtxVideoLatentSpatialUpsampler(config.midChannels);
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
      this.upsampler = config.temporalUpsample
        ? new LtxVideoLatentTemporalUpsampler(config.midChannels, config.spatialUpsample)
        : new LtxVideoLatentSpatialUpsampler(config.midChannels);
      this.postUpsampleResBlocks = Array.from(
        { length: config.numBlocksPerStage },
        () => new LtxVideoLatentUpsamplerResBlock3d(config.midChannels),
      );
      this.finalConv = new Conv3d(config.midChannels, config.inChannels, 3, 1, 1);
    }
    this.initialNorm = new GroupNorm(GROUP_NORM_GROUPS, config.midChannels);
  }

  get config(): LtxVideoLatentUpsamplerConfig {
    return this.#config;
  }

  forward(latents: MxArray): MxArray {
    const [batch, channels, frames] = expectBcfhw(latents, "LtxVideoLatentUpsamplerModel.forward");
    if (channels !== this.#config.inChannels) {
      throw new Error(
        `LtxVideoLatentUpsamplerModel.forward: expected ${this.#config.inChannels} channels, got ${channels}.`,
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
      const normalized = this.initialNorm.forward(hidden);
      hidden.free();
      hidden = normalized;
      const activated = silu(hidden);
      hidden.free();
      hidden = activated;
      const pre = this.#runResBlocks(
        hidden,
        this.resBlocks,
        "LtxVideoLatentUpsamplerModel.resBlocks",
      );
      hidden.free();
      hidden = pre;
      const upsampled = this.upsampler.forward(hidden);
      hidden.free();
      hidden = upsampled;
      const post = this.#runResBlocks(
        hidden,
        this.postUpsampleResBlocks,
        "LtxVideoLatentUpsamplerModel.postUpsampleResBlocks",
      );
      hidden.free();
      hidden = post;
      const projected = this.finalConv.forward(hidden);
      hidden.free();
      hidden = projected;
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
      const pre = this.#runResBlocks(
        hidden,
        this.resBlocks,
        "LtxVideoLatentUpsamplerModel.resBlocks",
      );
      hidden.free();
      hidden = pre;
      let upsampled: MxArray;
      if (this.#config.temporalUpsample) {
        upsampled = this.upsampler.forward(hidden);
      } else {
        using frameBatch = bfhwcToFrameBatch(hidden);
        using spatial = this.upsampler.forward(frameBatch);
        upsampled = frameBatchToBfhwc(spatial, batch, frames);
      }
      hidden.free();
      hidden = upsampled;
      const post = this.#runResBlocks(
        hidden,
        this.postUpsampleResBlocks,
        "LtxVideoLatentUpsamplerModel.postUpsampleResBlocks",
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
}
