import { formatShape, type MxArray, pad, retainArray, slice, transpose } from "@mlxts/core";
import { Conv2d, Module, silu } from "@mlxts/nn";

import {
  Ltx2AudioCausalConv2d,
  Ltx2AudioMidBlock,
  Ltx2AudioPixelNorm,
  Ltx2AudioUpStage,
} from "./autoencoder-ltx2-audio-blocks";
import type { Ltx2AudioAutoencoderConfig } from "./config";
import { checkedModule } from "./tensor-utils";

const LTX2_AUDIO_DOWNSAMPLE_FACTOR = 4;

function expectBclm(x: MxArray, owner: string): readonly [number, number, number, number] {
  const [batch, channels, length, melBins] = x.shape;
  if (
    x.shape.length !== 4 ||
    batch === undefined ||
    channels === undefined ||
    length === undefined ||
    melBins === undefined
  ) {
    throw new Error(`${owner}: expected BCLM rank-4 tensor, got ${formatShape(x.shape)}.`);
  }
  return [batch, channels, length, melBins];
}

function expectBlmc(x: MxArray, owner: string): readonly [number, number, number, number] {
  const [batch, length, melBins, channels] = x.shape;
  if (
    x.shape.length !== 4 ||
    batch === undefined ||
    length === undefined ||
    melBins === undefined ||
    channels === undefined
  ) {
    throw new Error(`${owner}: expected BLMC rank-4 tensor, got ${formatShape(x.shape)}.`);
  }
  return [batch, length, melBins, channels];
}

function bclmToBlmc(x: MxArray): MxArray {
  return transpose(x, [0, 2, 3, 1]);
}

function blmcToBclm(x: MxArray): MxArray {
  return transpose(x, [0, 3, 1, 2]);
}

function reverse<T>(values: readonly T[]): T[] {
  return [...values].reverse();
}

function hasResolution(value: readonly number[] | null, resolution: number): boolean {
  return value?.includes(resolution) === true;
}

function expectLatentStats(values: readonly number[], size: number, name: string): void {
  if (values.length !== size) {
    throw new Error(`${name} length must be ${size}, got ${values.length}.`);
  }
  if (values.some((value) => !Number.isFinite(value))) {
    throw new Error(`${name} values must be finite.`);
  }
}

function cropOrPadDecoded(
  decoded: MxArray,
  targetChannels: number,
  targetLength: number,
  targetMelBins: number,
): MxArray {
  const [batch, length, melBins, channels] = expectBlmc(decoded, "cropOrPadDecoded");
  const croppedLength = Math.min(length, targetLength);
  const croppedMelBins = Math.min(melBins, targetMelBins);
  const croppedChannels = Math.min(channels, targetChannels);
  using cropped = slice(
    decoded,
    [0, 0, 0, 0],
    [batch, croppedLength, croppedMelBins, croppedChannels],
  );
  const lengthPadding = targetLength - croppedLength;
  const melPadding = targetMelBins - croppedMelBins;
  const channelPadding = targetChannels - croppedChannels;
  if (lengthPadding > 0 || melPadding > 0 || channelPadding > 0) {
    using padded = pad(cropped, [
      [0, 0],
      [0, Math.max(lengthPadding, 0)],
      [0, Math.max(melPadding, 0)],
      [0, Math.max(channelPadding, 0)],
    ]);
    return slice(padded, [0, 0, 0, 0], [batch, targetLength, targetMelBins, targetChannels]);
  }
  return retainArray(cropped);
}

/** Decoder side of Diffusers `AutoencoderKLLTX2Audio`. */
export class Ltx2AudioDecoder2d extends Module {
  convIn: Ltx2AudioCausalConv2d | Conv2d;
  mid: Ltx2AudioMidBlock;
  up: Ltx2AudioUpStage[];
  normOut: Ltx2AudioPixelNorm;
  convOut: Ltx2AudioCausalConv2d | Conv2d;
  #latentChannels: number;
  #outputChannels: number;
  #melBins: number | null;
  #causal: boolean;

  constructor(config: Ltx2AudioAutoencoderConfig) {
    super();
    if (config.normType !== "pixel") {
      throw new Error("Ltx2AudioDecoder2d: only pixel normalization is supported.");
    }
    const channelMultipliers = config.chMult;
    const lastMultiplier = channelMultipliers[channelMultipliers.length - 1];
    if (lastMultiplier === undefined) {
      throw new Error("Ltx2AudioDecoder2d: chMult must not be empty.");
    }
    const baseBlockChannels = config.baseChannels * lastMultiplier;
    this.convIn =
      config.causalityAxis === null
        ? new Conv2d(config.latentChannels, baseBlockChannels, 3, 1, 1)
        : new Ltx2AudioCausalConv2d(
            config.latentChannels,
            baseBlockChannels,
            3,
            1,
            1,
            1,
            true,
            config.causalityAxis,
          );
    this.mid = new Ltx2AudioMidBlock(
      baseBlockChannels,
      config.dropout,
      config.causalityAxis,
      config.midBlockAddAttention,
    );
    this.up = [];
    let blockInputChannels = baseBlockChannels;
    let currentResolution = config.resolution / 2 ** (channelMultipliers.length - 1);
    const reversedMultipliers = reverse(channelMultipliers);
    for (let index = 0; index < reversedMultipliers.length; index += 1) {
      const multiplier = checkedNumber(reversedMultipliers, index, "Ltx2AudioDecoder2d.chMult");
      const outputChannels = config.baseChannels * multiplier;
      const stage = new Ltx2AudioUpStage({
        inputChannels: blockInputChannels,
        outputChannels,
        numBlocks: config.numResBlocks + 1,
        useAttention: hasResolution(config.attnResolutions, currentResolution),
        hasUpsample: index !== reversedMultipliers.length - 1,
        dropout: config.dropout,
        causalityAxis: config.causalityAxis,
      });
      this.up.unshift(stage);
      blockInputChannels = outputChannels;
      if (index !== reversedMultipliers.length - 1) {
        currentResolution *= 2;
      }
    }
    this.normOut = new Ltx2AudioPixelNorm();
    this.convOut =
      config.causalityAxis === null
        ? new Conv2d(blockInputChannels, config.outputChannels, 3, 1, 1)
        : new Ltx2AudioCausalConv2d(
            blockInputChannels,
            config.outputChannels,
            3,
            1,
            1,
            1,
            true,
            config.causalityAxis,
          );
    this.#latentChannels = config.latentChannels;
    this.#outputChannels = config.outputChannels;
    this.#melBins = config.melBins;
    this.#causal = config.causalityAxis !== null;
  }

  forward(latents: MxArray): MxArray {
    const [, latentLength, latentMelBins] = expectBlmc(latents, "Ltx2AudioDecoder2d.forward");
    let hidden = this.convIn.forward(latents);
    try {
      const mid = this.mid.forward(hidden);
      hidden.free();
      hidden = mid;
      for (let index = this.up.length - 1; index >= 0; index -= 1) {
        const stage = checkedModule(this.up, index, "Ltx2AudioDecoder2d.forward");
        const next = stage.forward(hidden);
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
      const targetLength = this.#causal
        ? Math.max(
            latentLength * LTX2_AUDIO_DOWNSAMPLE_FACTOR - (LTX2_AUDIO_DOWNSAMPLE_FACTOR - 1),
            1,
          )
        : latentLength * LTX2_AUDIO_DOWNSAMPLE_FACTOR;
      return cropOrPadDecoded(
        hidden,
        this.#outputChannels,
        targetLength,
        this.#melBins ?? latentMelBins,
      );
    } finally {
      hidden.free();
    }
  }

  get latentChannels(): number {
    return this.#latentChannels;
  }
}

/** Decode-only LTX-2 AutoencoderKLLTX2Audio surface with BCLM tensor boundaries. */
export class Ltx2AudioAutoencoderKL extends Module {
  decoder: Ltx2AudioDecoder2d;
  #latentChannels: number;
  #latentStatSize: number;
  #melCompressionRatio: number;
  #temporalCompressionRatio: number;
  #latentsMean: readonly number[];
  #latentsStd: readonly number[];

  constructor(config: Ltx2AudioAutoencoderConfig) {
    super();
    this.decoder = new Ltx2AudioDecoder2d(config);
    this.#latentChannels = config.latentChannels;
    this.#latentStatSize = config.baseChannels;
    this.#melCompressionRatio = config.melCompressionRatio;
    this.#temporalCompressionRatio = config.temporalCompressionRatio;
    this.#latentsMean = Array.from({ length: this.#latentStatSize }, () => 0);
    this.#latentsStd = Array.from({ length: this.#latentStatSize }, () => 1);
  }

  get latentChannels(): number {
    return this.#latentChannels;
  }

  get latentStatSize(): number {
    return this.#latentStatSize;
  }

  get melCompressionRatio(): number {
    return this.#melCompressionRatio;
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

  /** Replace the packed latent mean/std buffers loaded from Diffusers checkpoints. */
  setLatentStats(mean: readonly number[], std: readonly number[]): void {
    expectLatentStats(mean, this.#latentStatSize, "latentsMean");
    expectLatentStats(std, this.#latentStatSize, "latentsStd");
    this.#latentsMean = [...mean];
    this.#latentsStd = [...std];
  }

  /** Decode BCLM denormalized latent spectrograms into BCLM mel spectrograms. */
  decodeRaw(latents: MxArray): MxArray {
    const [, channels] = expectBclm(latents, "Ltx2AudioAutoencoderKL.decodeRaw");
    if (channels !== this.#latentChannels) {
      throw new Error(
        `Ltx2AudioAutoencoderKL.decodeRaw: expected ${this.#latentChannels} latent channels, got ${channels}.`,
      );
    }
    using internal = bclmToBlmc(latents);
    using decoded = this.decoder.forward(internal);
    return blmcToBclm(decoded);
  }

  forward(latents: MxArray): MxArray {
    return this.decodeRaw(latents);
  }
}

function checkedNumber(values: readonly number[], index: number, owner: string): number {
  const value = values[index];
  if (value === undefined) {
    throw new Error(`${owner}: missing numeric value at index ${index}.`);
  }
  return value;
}
