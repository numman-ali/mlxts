import type { MxArray } from "@mlxts/core";
import { formatShape, reshape, transpose } from "@mlxts/core";
import { Module, silu } from "@mlxts/nn";
import { LtxVideoVaeRMSNorm } from "./autoencoder-blocks";
import {
  Ltx2VideoCausalConv3d,
  Ltx2VideoMidBlock3d,
  Ltx2VideoUpBlock3d,
  ltx2VideoDecoderLayout,
} from "./autoencoder-ltx2-blocks";
import {
  expectLtxVideoVaeVolume,
  ltxVideoBcfhwToBfhwc,
  ltxVideoBfhwcToBcfhw,
} from "./autoencoder-volume";
import type { Ltx2VideoAutoencoderConfig } from "./config";
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

/** Unpatch packed LTX-2 decoder channels into BFHWC sample layout. */
export function unpatchLtx2VideoDecoderOutput(
  hiddenStates: MxArray,
  outputChannels: number,
  patchSize: number,
  patchSizeT: number,
): MxArray {
  const [batch, frames, height, width, channels] = expectLtxVideoVaeVolume(
    hiddenStates,
    "unpatchLtx2VideoDecoderOutput",
  );
  const packedChannels = outputChannels * patchSizeT * patchSize * patchSize;
  if (channels !== packedChannels) {
    throw new Error(
      `unpatchLtx2VideoDecoderOutput: expected ${packedChannels} channels, got ${channels} for ${formatShape(
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

/** Decoder side of Diffusers `AutoencoderKLLTX2Video`. */
export class Ltx2VideoDecoder3d extends Module {
  convIn: Ltx2VideoCausalConv3d;
  midBlock: Ltx2VideoMidBlock3d;
  upBlocks: Ltx2VideoUpBlock3d[];
  normOut: LtxVideoVaeRMSNorm;
  convOut: Ltx2VideoCausalConv3d;
  #outChannels: number;
  #patchSize: number;
  #patchSizeT: number;
  #defaultCausal: boolean;

  constructor(config: Ltx2VideoAutoencoderConfig) {
    super();
    const layout = ltx2VideoDecoderLayout(config);
    const firstChannel = layout.channels[0];
    if (firstChannel === undefined) {
      throw new Error("Ltx2VideoDecoder3d: decoder channels must not be empty.");
    }
    this.convIn = new Ltx2VideoCausalConv3d(
      config.latentChannels,
      firstChannel,
      3,
      1,
      1,
      config.decoderCausal,
      config.decoderSpatialPaddingMode,
    );
    this.midBlock = new Ltx2VideoMidBlock3d(
      firstChannel,
      checkedLayoutNumber(layout.layers, 0, "Ltx2VideoDecoder3d layers"),
      config.decoderSpatialPaddingMode,
    );
    this.upBlocks = [];
    let outputChannel = firstChannel;
    for (let index = 0; index < layout.channels.length; index += 1) {
      const factor = checkedLayoutNumber(layout.factors, index, "Ltx2VideoDecoder3d factors");
      const inputChannel = outputChannel / factor;
      outputChannel = checkedLayoutNumber(layout.channels, index, "Ltx2VideoDecoder3d") / factor;
      if (!Number.isInteger(inputChannel) || !Number.isInteger(outputChannel)) {
        throw new Error("Ltx2VideoDecoder3d: decoder channels must divide by upsample factors.");
      }
      const numLayers = checkedLayoutNumber(layout.layers, index + 1, "Ltx2VideoDecoder3d layers");
      const scale = checkedLayoutBoolean(layout.scaling, index, "Ltx2VideoDecoder3d scaling");
      const residual = checkedLayoutBoolean(layout.residual, index, "Ltx2VideoDecoder3d residual");
      this.upBlocks.push(
        new Ltx2VideoUpBlock3d(
          inputChannel,
          outputChannel,
          numLayers,
          scale,
          residual,
          factor,
          config.decoderSpatialPaddingMode,
        ),
      );
    }
    this.normOut = new LtxVideoVaeRMSNorm(outputChannel);
    this.convOut = new Ltx2VideoCausalConv3d(
      outputChannel,
      config.outChannels * config.patchSizeT * config.patchSize * config.patchSize,
      3,
      1,
      1,
      config.decoderCausal,
      config.decoderSpatialPaddingMode,
    );
    this.#outChannels = config.outChannels;
    this.#patchSize = config.patchSize;
    this.#patchSizeT = config.patchSizeT;
    this.#defaultCausal = config.decoderCausal;
  }

  forward(z: MxArray): MxArray {
    return this.run(z);
  }

  run(z: MxArray, causal = this.#defaultCausal): MxArray {
    let hidden = this.convIn.run(z, causal);
    try {
      const mid = this.midBlock.run(hidden, causal);
      hidden.free();
      hidden = mid;
      for (let index = 0; index < this.upBlocks.length; index += 1) {
        const block = checkedModule(this.upBlocks, index, "Ltx2VideoDecoder3d.forward");
        const next = block.run(hidden, causal);
        hidden.free();
        hidden = next;
      }
      const normalized = this.normOut.forward(hidden);
      hidden.free();
      hidden = normalized;
      const activated = silu(hidden);
      hidden.free();
      hidden = activated;
      const projected = this.convOut.run(hidden, causal);
      hidden.free();
      hidden = projected;
      return unpatchLtx2VideoDecoderOutput(
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

/** Decode-only LTX-2 AutoencoderKLLTX2Video surface with BCFHW tensor boundaries. */
export class Ltx2VideoAutoencoderKL extends Module {
  decoder: Ltx2VideoDecoder3d;
  #latentChannels: number;
  #scalingFactor: number;
  #spatialCompressionRatio: number;
  #temporalCompressionRatio: number;
  #latentsMean: readonly number[];
  #latentsStd: readonly number[];

  constructor(config: Ltx2VideoAutoencoderConfig) {
    super();
    this.decoder = new Ltx2VideoDecoder3d(config);
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
  decodeRaw(latents: MxArray, causal?: boolean): MxArray {
    const [, channels] = expectBcfhw(latents, "Ltx2VideoAutoencoderKL.decodeRaw");
    if (channels !== this.#latentChannels) {
      throw new Error(
        `Ltx2VideoAutoencoderKL.decodeRaw: expected ${this.#latentChannels} latent channels, got ${channels}.`,
      );
    }
    using internal = ltxVideoBcfhwToBfhwc(latents);
    using decoded = this.decoder.run(internal, causal);
    return ltxVideoBfhwcToBcfhw(decoded);
  }

  forward(latents: MxArray): MxArray {
    return this.decodeRaw(latents);
  }
}

function checkedLayoutNumber(values: readonly number[], index: number, owner: string): number {
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
