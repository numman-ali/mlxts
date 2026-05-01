/**
 * Stable Diffusion AutoencoderKL modules.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { add, exp, formatShape, maximum, minimum, multiply, random, split } from "@mlxts/core";
import { Conv2d, GroupNorm, Module, silu } from "@mlxts/nn";
import {
  StableDiffusionVaeDownEncoderBlock2d,
  StableDiffusionVaeMidBlock2d,
  StableDiffusionVaeUpDecoderBlock2d,
} from "./autoencoder-blocks";
import type { StableDiffusionAutoencoderConfig } from "./config";

/** Posterior distribution produced by the Stable Diffusion AutoencoderKL encoder. */
export class StableDiffusionVaePosterior implements Disposable {
  readonly mean: MxArray;
  readonly logVariance: MxArray;

  constructor(mean: MxArray, logVariance: MxArray) {
    this.mean = mean;
    this.logVariance = logVariance;
  }

  /** Return the deterministic posterior mode. */
  mode(): MxArray {
    return add(this.mean, 0);
  }

  /** Draw a latent sample from the diagonal Gaussian posterior. */
  sample(rngKey?: MxArray): MxArray {
    using clippedHigh = minimum(this.logVariance, 20);
    using clipped = maximum(clippedHigh, -30);
    using halfLogVariance = multiply(clipped, 0.5);
    using deviation = exp(halfLogVariance);
    using noise = random.normal([...this.mean.shape], this.mean.dtype, 0, 1, rngKey);
    using scaledNoise = multiply(noise, deviation);
    return add(this.mean, scaledNoise);
  }

  [Symbol.dispose](): void {
    this.mean.free();
    this.logVariance.free();
  }
}

function blockChannels(channels: readonly number[], index: number, owner: string): number {
  const value = channels[index];
  if (value === undefined) {
    throw new Error(`${owner}: missing block channel at index ${index}.`);
  }
  return value;
}

function lastBlockChannel(channels: readonly number[], owner: string): number {
  const value = channels[channels.length - 1];
  if (value === undefined) {
    throw new Error(`${owner}: blockOutChannels must not be empty.`);
  }
  return value;
}

function decoderBlockChannels(config: StableDiffusionAutoencoderConfig): readonly number[] {
  return config.decoderBlockOutChannels ?? config.blockOutChannels;
}

function validateAutoencoderConfig(config: StableDiffusionAutoencoderConfig): void {
  if (config.blockOutChannels.length === 0) {
    throw new Error("StableDiffusionAutoencoderKL: blockOutChannels must not be empty.");
  }
  const decoderChannels = decoderBlockChannels(config);
  if (decoderChannels.length === 0) {
    throw new Error("StableDiffusionAutoencoderKL: decoderBlockOutChannels must not be empty.");
  }
  if (config.downBlockTypes.length !== config.blockOutChannels.length) {
    throw new Error("StableDiffusionAutoencoderKL: downBlockTypes must match blockOutChannels.");
  }
  if (config.upBlockTypes.length !== decoderChannels.length) {
    throw new Error(
      "StableDiffusionAutoencoderKL: upBlockTypes must match decoder block channels.",
    );
  }
  if (config.latentChannelsOut !== config.latentChannels * 2) {
    throw new Error("StableDiffusionAutoencoderKL: latentChannelsOut must be 2 * latentChannels.");
  }
}

function checkedDownBlock(
  blocks: readonly StableDiffusionVaeDownEncoderBlock2d[],
  index: number,
): StableDiffusionVaeDownEncoderBlock2d {
  const block = blocks[index];
  if (block === undefined) {
    throw new Error(`StableDiffusionVaeEncoder.forward: missing down block at index ${index}.`);
  }
  return block;
}

function checkedUpBlock(
  blocks: readonly StableDiffusionVaeUpDecoderBlock2d[],
  index: number,
): StableDiffusionVaeUpDecoderBlock2d {
  const block = blocks[index];
  if (block === undefined) {
    throw new Error(`StableDiffusionVaeDecoder.forward: missing up block at index ${index}.`);
  }
  return block;
}

/** Encoder side of a Stable Diffusion AutoencoderKL component. */
export class StableDiffusionVaeEncoder extends Module {
  convIn: Conv2d;
  downBlocks: StableDiffusionVaeDownEncoderBlock2d[];
  midBlock: StableDiffusionVaeMidBlock2d;
  convNormOut: GroupNorm;
  convOut: Conv2d;

  constructor(config: StableDiffusionAutoencoderConfig) {
    super();
    validateAutoencoderConfig(config);
    const firstChannels = blockChannels(config.blockOutChannels, 0, "StableDiffusionVaeEncoder");
    const finalChannels = lastBlockChannel(config.blockOutChannels, "StableDiffusionVaeEncoder");
    this.convIn = new Conv2d(config.inChannels, firstChannels, 3, 1, 1);
    this.downBlocks = config.blockOutChannels.map((outChannels, index) => {
      const inChannels =
        index === 0
          ? firstChannels
          : blockChannels(config.blockOutChannels, index - 1, "StableDiffusionVaeEncoder");
      return new StableDiffusionVaeDownEncoderBlock2d(
        inChannels,
        outChannels,
        config.layersPerBlock,
        config.normNumGroups,
        index < config.blockOutChannels.length - 1,
      );
    });
    this.midBlock = new StableDiffusionVaeMidBlock2d(finalChannels, config.normNumGroups);
    this.convNormOut = new GroupNorm(config.normNumGroups, finalChannels, 1e-6);
    this.convOut = new Conv2d(finalChannels, config.latentChannelsOut, 3, 1, 1);
  }

  /** Encode an NHWC image tensor into raw posterior moments. */
  forward(x: MxArray): MxArray {
    let hidden = this.convIn.forward(x);
    try {
      for (let index = 0; index < this.downBlocks.length; index += 1) {
        const block = checkedDownBlock(this.downBlocks, index);
        const nextHidden = block.forward(hidden);
        hidden.free();
        hidden = nextHidden;
      }

      const mid = this.midBlock.forward(hidden);
      hidden.free();
      hidden = mid;
      const normalized = this.convNormOut.forward(hidden);
      hidden.free();
      hidden = normalized;
      const activated = silu(hidden);
      hidden.free();
      hidden = activated;
      const output = this.convOut.forward(hidden);
      hidden.free();
      return output;
    } catch (error) {
      hidden.free();
      throw error;
    }
  }
}

/** Decoder side of a Stable Diffusion AutoencoderKL component. */
export class StableDiffusionVaeDecoder extends Module {
  convIn: Conv2d;
  midBlock: StableDiffusionVaeMidBlock2d;
  upBlocks: StableDiffusionVaeUpDecoderBlock2d[];
  convNormOut: GroupNorm;
  convOut: Conv2d;

  constructor(config: StableDiffusionAutoencoderConfig) {
    super();
    validateAutoencoderConfig(config);
    const decoderChannels = decoderBlockChannels(config);
    const firstChannels = blockChannels(decoderChannels, 0, "StableDiffusionVaeDecoder");
    const finalChannels = lastBlockChannel(decoderChannels, "StableDiffusionVaeDecoder");
    const reversedChannels = [...decoderChannels].reverse();
    this.convIn = new Conv2d(config.latentChannels, finalChannels, 3, 1, 1);
    this.midBlock = new StableDiffusionVaeMidBlock2d(finalChannels, config.normNumGroups);
    this.upBlocks = reversedChannels.map((outChannels, index) => {
      const inChannels =
        index === 0
          ? finalChannels
          : blockChannels(reversedChannels, index - 1, "StableDiffusionVaeDecoder");
      return new StableDiffusionVaeUpDecoderBlock2d(
        inChannels,
        outChannels,
        config.layersPerBlock + 1,
        config.normNumGroups,
        index < reversedChannels.length - 1,
      );
    });
    this.convNormOut = new GroupNorm(config.normNumGroups, firstChannels, 1e-6);
    this.convOut = new Conv2d(firstChannels, config.outChannels, 3, 1, 1);
  }

  /** Decode an NHWC latent tensor into an NHWC image tensor. */
  forward(z: MxArray): MxArray {
    let hidden = this.convIn.forward(z);
    try {
      const mid = this.midBlock.forward(hidden);
      hidden.free();
      hidden = mid;

      for (let index = 0; index < this.upBlocks.length; index += 1) {
        const block = checkedUpBlock(this.upBlocks, index);
        const nextHidden = block.forward(hidden);
        hidden.free();
        hidden = nextHidden;
      }

      const normalized = this.convNormOut.forward(hidden);
      hidden.free();
      hidden = normalized;
      const activated = silu(hidden);
      hidden.free();
      hidden = activated;
      const output = this.convOut.forward(hidden);
      hidden.free();
      return output;
    } catch (error) {
      hidden.free();
      throw error;
    }
  }
}

/** Stable Diffusion AutoencoderKL module with raw encode/decode component semantics. */
export class StableDiffusionAutoencoderKL extends Module {
  encoder: StableDiffusionVaeEncoder;
  decoder: StableDiffusionVaeDecoder;
  quantConv: Conv2d | null;
  postQuantConv: Conv2d | null;
  #scalingFactor: number;
  #latentChannels: number;
  #vaeScaleFactor: number;

  constructor(config: StableDiffusionAutoencoderConfig) {
    super();
    validateAutoencoderConfig(config);
    this.encoder = new StableDiffusionVaeEncoder(config);
    this.decoder = new StableDiffusionVaeDecoder(config);
    this.quantConv = config.useQuantConv
      ? new Conv2d(config.latentChannelsOut, config.latentChannelsOut, 1, 1, 0)
      : null;
    this.postQuantConv = config.usePostQuantConv
      ? new Conv2d(config.latentChannels, config.latentChannels, 1, 1, 0)
      : null;
    this.#scalingFactor = config.scalingFactor;
    this.#latentChannels = config.latentChannels;
    this.#vaeScaleFactor = 2 ** (config.blockOutChannels.length - 1);
  }

  get scalingFactor(): number {
    return this.#scalingFactor;
  }

  get latentChannels(): number {
    return this.#latentChannels;
  }

  get vaeScaleFactor(): number {
    return this.#vaeScaleFactor;
  }

  /** Encode an NHWC image tensor into the raw posterior moment tensor. */
  encodeMoments(x: MxArray): MxArray {
    const encoded = this.encoder.forward(x);
    let ownsEncoded = true;
    try {
      if (this.quantConv === null) {
        ownsEncoded = false;
        return encoded;
      }
      return this.quantConv.forward(encoded);
    } finally {
      if (ownsEncoded) {
        encoded.free();
      }
    }
  }

  /** Split the raw posterior moment tensor into mean and log-variance tensors. */
  splitMoments(moments: MxArray): StableDiffusionVaePosterior {
    const channels = moments.shape[moments.shape.length - 1];
    if (channels !== this.#latentChannels * 2) {
      throw new Error(
        `StableDiffusionAutoencoderKL.splitMoments: expected last dimension ${this.#latentChannels * 2}, got ${channels ?? "undefined"} for shape ${formatShape(moments.shape)}.`,
      );
    }
    const parts = split(moments, 2, moments.shape.length - 1);
    if (parts.length !== 2) {
      for (const part of parts) {
        part.free();
      }
      throw new Error("StableDiffusionAutoencoderKL.splitMoments: split did not return two parts.");
    }
    const mean = parts[0];
    const logVariance = parts[1];
    if (mean === undefined || logVariance === undefined) {
      for (const part of parts) {
        part.free();
      }
      throw new Error("StableDiffusionAutoencoderKL.splitMoments: split returned empty parts.");
    }
    return new StableDiffusionVaePosterior(mean, logVariance);
  }

  /** Encode an NHWC image tensor into an unscaled posterior distribution. */
  encode(x: MxArray): StableDiffusionVaePosterior {
    using moments = this.encodeMoments(x);
    return this.splitMoments(moments);
  }

  /** Decode a raw NHWC latent tensor into an NHWC image tensor. */
  decode(z: MxArray): MxArray {
    if (this.postQuantConv === null) {
      return this.decoder.forward(z);
    }
    using projected = this.postQuantConv.forward(z);
    return this.decoder.forward(projected);
  }

  /** Reconstruct through the posterior mode without stochastic sampling. */
  forward(x: MxArray): MxArray {
    using posterior = this.encode(x);
    using latents = posterior.mode();
    return this.decode(latents);
  }
}
