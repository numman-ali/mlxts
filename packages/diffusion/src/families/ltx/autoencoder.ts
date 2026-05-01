import type { MxArray } from "@mlxts/core";
import { formatShape, reshape, transpose } from "@mlxts/core";
import { Module, silu } from "@mlxts/nn";

import {
  LtxVideoMidBlock3d,
  LtxVideoUpBlock3d,
  LtxVideoVaeRMSNorm,
  ltxVideoDecoderLayout,
} from "./autoencoder-blocks";
import {
  expectLtxVideoVaeVolume,
  LtxVideoCausalConv3d,
  ltxVideoBcfhwToBfhwc,
  ltxVideoBfhwcToBcfhw,
} from "./autoencoder-volume";
import type { LtxVideoAutoencoderConfig } from "./config";
import { checkedModule } from "./tensor-utils";

function expectLatentStats(values: readonly number[], channels: number, name: string): void {
  if (values.length !== channels) {
    throw new Error(`${name} length must be ${channels}, got ${values.length}.`);
  }
  if (values.some((value) => !Number.isFinite(value))) {
    throw new Error(`${name} values must be finite.`);
  }
}

function compressionRatio(
  explicitRatio: number | null,
  patchSize: number,
  scaling: readonly boolean[],
): number {
  return explicitRatio ?? patchSize * 2 ** scaling.filter(Boolean).length;
}

/** Unpatch packed decoder channels into BFHWC sample layout. */
export function unpatchLtxVideoDecoderOutput(
  hiddenStates: MxArray,
  outputChannels: number,
  patchSize: number,
  patchSizeT: number,
): MxArray {
  const [batch, frames, height, width, channels] = expectLtxVideoVaeVolume(
    hiddenStates,
    "unpatchLtxVideoDecoderOutput",
  );
  const packedChannels = outputChannels * patchSizeT * patchSize * patchSize;
  if (channels !== packedChannels) {
    throw new Error(
      `unpatchLtxVideoDecoderOutput: expected ${packedChannels} channels, got ${channels} for ${formatShape(
        hiddenStates.shape,
      )}.`,
    );
  }
  using grid = reshape(hiddenStates, [
    batch,
    frames,
    height,
    width,
    outputChannels,
    patchSizeT,
    patchSize,
    patchSize,
  ]);
  using unpacked = transpose(grid, [0, 1, 5, 2, 7, 3, 6, 4]);
  return reshape(unpacked, [
    batch,
    frames * patchSizeT,
    height * patchSize,
    width * patchSize,
    outputChannels,
  ]);
}

/** Decoder side of Diffusers `AutoencoderKLLTXVideo`. */
export class LtxVideoDecoder3d extends Module {
  convIn: LtxVideoCausalConv3d;
  midBlock: LtxVideoMidBlock3d;
  upBlocks: LtxVideoUpBlock3d[];
  normOut: LtxVideoVaeRMSNorm;
  convOut: LtxVideoCausalConv3d;
  #outChannels: number;
  #patchSize: number;
  #patchSizeT: number;

  constructor(config: LtxVideoAutoencoderConfig) {
    super();
    const layout = ltxVideoDecoderLayout(config);
    const firstChannel = layout.channels[0];
    if (firstChannel === undefined) {
      throw new Error("LtxVideoDecoder3d: decoder channels must not be empty.");
    }
    this.convIn = new LtxVideoCausalConv3d(
      config.latentChannels,
      firstChannel,
      3,
      1,
      1,
      config.decoderCausal,
    );
    this.midBlock = new LtxVideoMidBlock3d(
      firstChannel,
      layout.layers[0] ?? 0,
      config.decoderCausal,
    );
    this.upBlocks = [];
    let outputChannel = firstChannel;
    for (let index = 0; index < layout.channels.length; index += 1) {
      const inputChannel = outputChannel;
      outputChannel = checkedLayoutChannel(layout.channels, index, "LtxVideoDecoder3d");
      const numLayers = checkedLayoutChannel(layout.layers, index + 1, "LtxVideoDecoder3d layers");
      const scale = checkedLayoutBoolean(layout.scaling, index, "LtxVideoDecoder3d scaling");
      this.upBlocks.push(
        new LtxVideoUpBlock3d(inputChannel, outputChannel, numLayers, scale, config.decoderCausal),
      );
    }
    this.normOut = new LtxVideoVaeRMSNorm(outputChannel);
    this.convOut = new LtxVideoCausalConv3d(
      outputChannel,
      config.outChannels * config.patchSizeT * config.patchSize * config.patchSize,
      3,
      1,
      1,
      config.decoderCausal,
    );
    this.#outChannels = config.outChannels;
    this.#patchSize = config.patchSize;
    this.#patchSizeT = config.patchSizeT;
  }

  forward(z: MxArray): MxArray {
    let hidden = this.convIn.forward(z);
    try {
      const mid = this.midBlock.forward(hidden);
      hidden.free();
      hidden = mid;
      for (let index = 0; index < this.upBlocks.length; index += 1) {
        const block = checkedModule(this.upBlocks, index, "LtxVideoDecoder3d.forward");
        const next = block.forward(hidden);
        hidden.free();
        hidden = next;
      }
      const normalized = this.normOut.forward(hidden);
      hidden.free();
      hidden = normalized;
      const activated = silu(hidden);
      hidden.free();
      hidden = activated;
      const projected = this.convOut.forward(hidden);
      hidden.free();
      hidden = projected;
      return unpatchLtxVideoDecoderOutput(
        hidden,
        this.#outChannels,
        this.#patchSize,
        this.#patchSizeT,
      );
    } finally {
      hidden.free();
    }
  }
}

function checkedLayoutChannel(values: readonly number[], index: number, owner: string): number {
  const value = values[index];
  if (value === undefined) {
    throw new Error(`${owner}: missing numeric layout value at index ${index}.`);
  }
  return value;
}

function checkedLayoutBoolean(values: readonly boolean[], index: number, owner: string): boolean {
  const value = values[index];
  if (value === undefined) {
    throw new Error(`${owner}: missing boolean layout value at index ${index}.`);
  }
  return value;
}

/** Decode-only LTX AutoencoderKLLTXVideo surface with BCFHW tensor boundaries. */
export class LtxVideoAutoencoderKL extends Module {
  decoder: LtxVideoDecoder3d;
  #latentChannels: number;
  #scalingFactor: number;
  #spatialCompressionRatio: number;
  #temporalCompressionRatio: number;
  #latentsMean: readonly number[];
  #latentsStd: readonly number[];

  constructor(config: LtxVideoAutoencoderConfig) {
    super();
    this.decoder = new LtxVideoDecoder3d(config);
    this.#latentChannels = config.latentChannels;
    this.#scalingFactor = config.scalingFactor;
    this.#spatialCompressionRatio = compressionRatio(
      config.spatialCompressionRatio,
      config.patchSize,
      config.spatioTemporalScaling,
    );
    this.#temporalCompressionRatio = compressionRatio(
      config.temporalCompressionRatio,
      config.patchSizeT,
      config.spatioTemporalScaling,
    );
    this.#latentsMean = Array.from({ length: config.latentChannels }, () => 0);
    this.#latentsStd = Array.from({ length: config.latentChannels }, () => 1);
  }

  get latentChannels(): number {
    return this.#latentChannels;
  }

  get scalingFactor(): number {
    return this.#scalingFactor;
  }

  get spatialCompressionRatio(): number {
    return this.#spatialCompressionRatio;
  }

  get temporalCompressionRatio(): number {
    return this.#temporalCompressionRatio;
  }

  get latentsMean(): readonly number[] {
    return this.#latentsMean;
  }

  get latentsStd(): readonly number[] {
    return this.#latentsStd;
  }

  /** Replace the latent mean/std buffers loaded from Diffusers checkpoints. */
  setLatentStats(mean: readonly number[], std: readonly number[]): void {
    expectLatentStats(mean, this.#latentChannels, "latentsMean");
    expectLatentStats(std, this.#latentChannels, "latentsStd");
    this.#latentsMean = [...mean];
    this.#latentsStd = [...std];
  }

  /** Decode BCFHW denormalized latent videos into BCFHW sample videos. */
  decodeRaw(latents: MxArray): MxArray {
    const [, channels] = expectBcfhw(latents, "LtxVideoAutoencoderKL.decodeRaw");
    if (channels !== this.#latentChannels) {
      throw new Error(
        `LtxVideoAutoencoderKL.decodeRaw: expected ${this.#latentChannels} latent channels, got ${channels}.`,
      );
    }
    using internal = ltxVideoBcfhwToBfhwc(latents);
    using decoded = this.decoder.forward(internal);
    return ltxVideoBfhwcToBcfhw(decoded);
  }

  forward(latents: MxArray): MxArray {
    return this.decodeRaw(latents);
  }
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
