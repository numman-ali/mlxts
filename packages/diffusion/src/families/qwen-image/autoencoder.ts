import { type MxArray, split } from "@mlxts/core";
import { Module, silu } from "@mlxts/nn";
import {
  QwenImageAttentionBlock,
  QwenImageCausalConv3d,
  QwenImageMidBlock,
  QwenImageResample,
  type QwenImageResampleMode,
  QwenImageResidualBlock,
  QwenImageRMSNorm,
  QwenImageUpBlock,
  qwenImageNcfhwToNdhwc,
  qwenImageNdhwcToNcfhw,
} from "./autoencoder-blocks";
import type { QwenImageAutoencoderConfig } from "./config";

type QwenImageDownLayer = QwenImageResidualBlock | QwenImageAttentionBlock | QwenImageResample;

function blockChannel(channels: readonly number[], index: number, owner: string): number {
  const value = channels[index];
  if (value === undefined) {
    throw new Error(`${owner}: missing channel at index ${index}.`);
  }
  return value;
}

function lastValue(values: readonly number[], owner: string): number {
  const value = values[values.length - 1];
  if (value === undefined) {
    throw new Error(`${owner}: dimMultipliers must not be empty.`);
  }
  return value;
}

function validateConfig(config: QwenImageAutoencoderConfig): void {
  if (config.dimMultipliers.length === 0) {
    throw new Error("QwenImageAutoencoderKL: dimMultipliers must not be empty.");
  }
  if (config.latentChannelsOut !== config.latentChannels * 2) {
    throw new Error("QwenImageAutoencoderKL: latentChannelsOut must be 2 * latentChannels.");
  }
  if (config.temporalDownsample.length !== config.dimMultipliers.length - 1) {
    throw new Error("QwenImageAutoencoderKL: temporalDownsample length is invalid.");
  }
  if (config.temporalUpsample.length !== config.temporalDownsample.length) {
    throw new Error("QwenImageAutoencoderKL: temporalUpsample must mirror temporalDownsample.");
  }
}

function encoderChannels(config: QwenImageAutoencoderConfig): number[] {
  return [1, ...config.dimMultipliers].map((multiplier) => config.baseDim * multiplier);
}

function decoderChannels(config: QwenImageAutoencoderConfig): number[] {
  return [
    lastValue(config.dimMultipliers, "QwenImageDecoder3d"),
    ...config.dimMultipliers.toReversed(),
  ].map((multiplier) => config.baseDim * multiplier);
}

function downsampleMode(
  config: QwenImageAutoencoderConfig,
  index: number,
): Extract<QwenImageResampleMode, "downsample2d" | "downsample3d"> {
  return config.temporalDownsample[index] === true ? "downsample3d" : "downsample2d";
}

function upsampleMode(
  config: QwenImageAutoencoderConfig,
  index: number,
): Extract<QwenImageResampleMode, "upsample2d" | "upsample3d"> {
  return config.temporalUpsample[index] === true ? "upsample3d" : "upsample2d";
}

/** Encoder side of the Qwen-Image 3D causal VAE. */
export class QwenImageEncoder3d extends Module {
  convIn: QwenImageCausalConv3d;
  downBlocks: QwenImageDownLayer[];
  midBlock: QwenImageMidBlock;
  normOut: QwenImageRMSNorm;
  convOut: QwenImageCausalConv3d;

  constructor(config: QwenImageAutoencoderConfig) {
    super();
    validateConfig(config);
    const channels = encoderChannels(config);
    const firstChannels = blockChannel(channels, 0, "QwenImageEncoder3d");
    this.convIn = new QwenImageCausalConv3d(config.inputChannels, firstChannels, 3, 1, 1);
    this.downBlocks = [];

    let scale = 1.0;
    for (let index = 0; index < channels.length - 1; index += 1) {
      let inputChannels = blockChannel(channels, index, "QwenImageEncoder3d");
      const outputChannels = blockChannel(channels, index + 1, "QwenImageEncoder3d");
      for (let blockIndex = 0; blockIndex < config.numResBlocks; blockIndex += 1) {
        this.downBlocks.push(
          new QwenImageResidualBlock(inputChannels, outputChannels, config.dropout),
        );
        if (config.attentionScales.includes(scale)) {
          this.downBlocks.push(new QwenImageAttentionBlock(outputChannels));
        }
        inputChannels = outputChannels;
      }

      if (index !== config.dimMultipliers.length - 1) {
        this.downBlocks.push(new QwenImageResample(outputChannels, downsampleMode(config, index)));
        scale /= 2.0;
      }
    }

    const finalChannels = blockChannel(channels, channels.length - 1, "QwenImageEncoder3d");
    this.midBlock = new QwenImageMidBlock(finalChannels, config.dropout, 1);
    this.normOut = new QwenImageRMSNorm(finalChannels);
    this.convOut = new QwenImageCausalConv3d(finalChannels, config.latentChannelsOut, 3, 1, 1);
  }

  override forward(x: MxArray): MxArray {
    let hidden = this.convIn.forward(x);
    try {
      for (let index = 0; index < this.downBlocks.length; index += 1) {
        const block = this.downBlocks[index];
        if (block === undefined) {
          throw new Error(`QwenImageEncoder3d.forward: missing down block at index ${index}.`);
        }
        const next = block.forward(hidden);
        hidden.free();
        hidden = next;
      }

      const mid = this.midBlock.forward(hidden);
      hidden.free();
      hidden = mid;
      const normalized = this.normOut.forward(hidden);
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

/** Decoder side of the Qwen-Image 3D causal VAE. */
export class QwenImageDecoder3d extends Module {
  convIn: QwenImageCausalConv3d;
  midBlock: QwenImageMidBlock;
  upBlocks: QwenImageUpBlock[];
  normOut: QwenImageRMSNorm;
  convOut: QwenImageCausalConv3d;

  constructor(config: QwenImageAutoencoderConfig) {
    super();
    validateConfig(config);
    const channels = decoderChannels(config);
    const firstChannels = blockChannel(channels, 0, "QwenImageDecoder3d");
    this.convIn = new QwenImageCausalConv3d(config.latentChannels, firstChannels, 3, 1, 1);
    this.midBlock = new QwenImageMidBlock(firstChannels, config.dropout, 1);
    this.upBlocks = [];

    for (let index = 0; index < channels.length - 1; index += 1) {
      const rawInputChannels = blockChannel(channels, index, "QwenImageDecoder3d");
      const inputChannels = index === 0 ? rawInputChannels : rawInputChannels / 2;
      const outputChannels = blockChannel(channels, index + 1, "QwenImageDecoder3d");
      const mode = index === config.dimMultipliers.length - 1 ? null : upsampleMode(config, index);
      this.upBlocks.push(
        new QwenImageUpBlock(
          inputChannels,
          outputChannels,
          config.numResBlocks,
          config.dropout,
          mode,
        ),
      );
    }

    const outputChannels = blockChannel(channels, channels.length - 1, "QwenImageDecoder3d");
    this.normOut = new QwenImageRMSNorm(outputChannels);
    this.convOut = new QwenImageCausalConv3d(outputChannels, config.inputChannels, 3, 1, 1);
  }

  override forward(x: MxArray): MxArray {
    let hidden = this.convIn.forward(x);
    try {
      const mid = this.midBlock.forward(hidden);
      hidden.free();
      hidden = mid;

      for (let index = 0; index < this.upBlocks.length; index += 1) {
        const block = this.upBlocks[index];
        if (block === undefined) {
          throw new Error(`QwenImageDecoder3d.forward: missing up block at index ${index}.`);
        }
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
      const output = this.convOut.forward(hidden);
      hidden.free();
      return output;
    } catch (error) {
      hidden.free();
      throw error;
    }
  }
}

/** Qwen-Image AutoencoderKL module with channel-first public tensor boundaries. */
export class QwenImageAutoencoderKL extends Module {
  encoder: QwenImageEncoder3d;
  quantConv: QwenImageCausalConv3d;
  postQuantConv: QwenImageCausalConv3d;
  decoder: QwenImageDecoder3d;
  #latentChannels: number;
  #spatialCompressionRatio: number;
  #latentsMean: readonly number[];
  #latentsStd: readonly number[];

  constructor(config: QwenImageAutoencoderConfig) {
    super();
    validateConfig(config);
    this.encoder = new QwenImageEncoder3d(config);
    this.quantConv = new QwenImageCausalConv3d(
      config.latentChannelsOut,
      config.latentChannelsOut,
      1,
    );
    this.postQuantConv = new QwenImageCausalConv3d(config.latentChannels, config.latentChannels, 1);
    this.decoder = new QwenImageDecoder3d(config);
    this.#latentChannels = config.latentChannels;
    this.#spatialCompressionRatio = config.spatialCompressionRatio;
    this.#latentsMean = config.latentsMean;
    this.#latentsStd = config.latentsStd;
  }

  get latentChannels(): number {
    return this.#latentChannels;
  }

  get spatialCompressionRatio(): number {
    return this.#spatialCompressionRatio;
  }

  get latentsMean(): readonly number[] {
    return this.#latentsMean;
  }

  get latentsStd(): readonly number[] {
    return this.#latentsStd;
  }

  /** Encode an NCFHW sample volume into NCFHW posterior moments. */
  encodeMoments(sample: MxArray): MxArray {
    using internal = qwenImageNcfhwToNdhwc(sample);
    using encoded = this.encoder.forward(internal);
    using moments = this.quantConv.forward(encoded);
    return qwenImageNdhwcToNcfhw(moments);
  }

  /** Encode an NCFHW sample volume into the deterministic posterior mode. */
  encodeRaw(sample: MxArray): MxArray {
    using moments = this.encodeMoments(sample);
    const parts = split(moments, 2, 1);
    const mean = parts[0];
    const logVariance = parts[1];
    if (mean === undefined || logVariance === undefined) {
      for (const part of parts) {
        part.free();
      }
      throw new Error("QwenImageAutoencoderKL.encodeRaw: expected mean/log-variance moments.");
    }

    logVariance.free();
    return mean;
  }

  /** Decode an NCFHW latent volume into an NCFHW sample volume. */
  decodeRaw(latents: MxArray): MxArray {
    using internal = qwenImageNcfhwToNdhwc(latents);
    using projected = this.postQuantConv.forward(internal);
    using decoded = this.decoder.forward(projected);
    return qwenImageNdhwcToNcfhw(decoded);
  }

  /** Reconstruct an NCFHW sample volume through the deterministic posterior mode. */
  forward(sample: MxArray): MxArray {
    using posteriorMode = this.encodeRaw(sample);
    return this.decodeRaw(posteriorMode);
  }
}
