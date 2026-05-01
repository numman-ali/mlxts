import { formatShape, type MxArray, split, transpose } from "@mlxts/core";

import {
  StableDiffusionAutoencoderKL,
  StableDiffusionVaePosterior,
} from "../stable-diffusion/autoencoder";
import type { StableDiffusionAutoencoderConfig } from "../stable-diffusion/config";
import type { Flux2KleinAutoencoderConfig } from "./config";

function stableConfigForFlux2KleinAutoencoder(
  config: Flux2KleinAutoencoderConfig,
): StableDiffusionAutoencoderConfig {
  const stableConfig: StableDiffusionAutoencoderConfig = {
    inChannels: config.inChannels,
    outChannels: config.outChannels,
    latentChannels: config.latentChannels,
    latentChannelsOut: config.latentChannelsOut,
    useQuantConv: config.useQuantConv,
    usePostQuantConv: config.usePostQuantConv,
    blockOutChannels: config.blockOutChannels,
    layersPerBlock: config.layersPerBlock,
    normNumGroups: config.normNumGroups,
    scalingFactor: 1,
    sampleSize: config.sampleSize,
    downBlockTypes: config.downBlockTypes,
    upBlockTypes: config.upBlockTypes,
    forceUpcast: config.forceUpcast,
    rawConfig: config.rawConfig,
  };
  if (config.decoderBlockOutChannels !== null) {
    stableConfig.decoderBlockOutChannels = config.decoderBlockOutChannels;
  }
  return stableConfig;
}

function defaultStats(length: number, value: number): number[] {
  return Array.from({ length }, () => value);
}

function validateStats(values: readonly number[], expectedLength: number, name: string): void {
  if (values.length !== expectedLength) {
    throw new Error(`${name} length must be ${expectedLength}, got ${values.length}.`);
  }
  if (values.some((value) => !Number.isFinite(value))) {
    throw new Error(`${name} values must be finite.`);
  }
}

/** FLUX.2 Klein AutoencoderKLFlux2 with NCHW public tensor boundaries. */
export class Flux2KleinAutoencoderKL extends StableDiffusionAutoencoderKL {
  #batchNormMean: readonly number[];
  #batchNormVar: readonly number[];
  #batchNormEps: number;
  #batchNormMomentum: number;
  #patchSize: number;
  #packedLatentChannels: number;

  constructor(config: Flux2KleinAutoencoderConfig) {
    super(stableConfigForFlux2KleinAutoencoder(config));
    this.#batchNormEps = config.batchNormEps;
    this.#batchNormMomentum = config.batchNormMomentum;
    this.#patchSize = config.patchSize[0];
    this.#packedLatentChannels = config.packedLatentChannels;
    this.#batchNormMean = defaultStats(config.packedLatentChannels, 0);
    this.#batchNormVar = defaultStats(config.packedLatentChannels, 1);
  }

  get batchNormMean(): readonly number[] {
    return this.#batchNormMean;
  }

  get batchNormVar(): readonly number[] {
    return this.#batchNormVar;
  }

  get batchNormEps(): number {
    return this.#batchNormEps;
  }

  get batchNormMomentum(): number {
    return this.#batchNormMomentum;
  }

  get patchSize(): number {
    return this.#patchSize;
  }

  get packedLatentChannels(): number {
    return this.#packedLatentChannels;
  }

  /** Replace the non-affine VAE batch-norm statistics loaded from checkpoint buffers. */
  setBatchNormStats(mean: readonly number[], variance: readonly number[]): void {
    validateStats(mean, this.#packedLatentChannels, "batchNormMean");
    validateStats(variance, this.#packedLatentChannels, "batchNormVar");
    this.#batchNormMean = [...mean];
    this.#batchNormVar = [...variance];
  }

  /** Encode an NCHW image tensor into NCHW posterior moments. */
  override encodeMoments(x: MxArray): MxArray {
    using channelLast = transpose(x, [0, 2, 3, 1]);
    using moments = super.encodeMoments(channelLast);
    return transpose(moments, [0, 3, 1, 2]);
  }

  /** Split an NCHW posterior moment tensor into mean and log-variance tensors. */
  override splitMoments(moments: MxArray): StableDiffusionVaePosterior {
    const channels = moments.shape[1];
    if (channels !== this.latentChannels * 2) {
      throw new Error(
        `Flux2KleinAutoencoderKL.splitMoments: expected channel dimension ${
          this.latentChannels * 2
        }, got ${channels ?? "undefined"} for shape ${formatShape(moments.shape)}.`,
      );
    }
    const parts = split(moments, 2, 1);
    if (parts.length !== 2) {
      for (const part of parts) {
        part.free();
      }
      throw new Error("Flux2KleinAutoencoderKL.splitMoments: split did not return two parts.");
    }
    const mean = parts[0];
    const logVariance = parts[1];
    if (mean === undefined || logVariance === undefined) {
      for (const part of parts) {
        part.free();
      }
      throw new Error("Flux2KleinAutoencoderKL.splitMoments: split returned empty parts.");
    }
    return new StableDiffusionVaePosterior(mean, logVariance);
  }

  /** Encode an NCHW image tensor into an unscaled posterior distribution. */
  override encode(x: MxArray): StableDiffusionVaePosterior {
    using moments = this.encodeMoments(x);
    return this.splitMoments(moments);
  }

  /** Decode an NCHW latent tensor into an NCHW image tensor. */
  override decode(z: MxArray): MxArray {
    using channelLast = transpose(z, [0, 2, 3, 1]);
    using decoded = super.decode(channelLast);
    return transpose(decoded, [0, 3, 1, 2]);
  }

  /** Reconstruct an NCHW image through the posterior mode. */
  override forward(x: MxArray): MxArray {
    using posterior = this.encode(x);
    using latents = posterior.mode();
    return this.decode(latents);
  }
}
