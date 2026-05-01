import {
  add,
  array,
  divide,
  formatShape,
  type MxArray,
  maximum,
  minimum,
  multiply,
  reshape,
  slice,
  squeeze,
  transpose,
} from "@mlxts/core";

import { unpackQwenImageLatents } from "./latents";

/** VAE decoder surface required by Qwen-Image latent image decoding. */
export type QwenImageLatentDecoder = {
  readonly latentChannels: number;
  readonly latentsMean: readonly number[];
  readonly latentsStd: readonly number[];
  decodeRaw(latents: MxArray): MxArray;
};

function validateLatentStats(vae: QwenImageLatentDecoder): void {
  if (
    vae.latentsMean.length !== vae.latentChannels ||
    vae.latentsStd.length !== vae.latentChannels
  ) {
    throw new Error(
      "decodeQwenImageLatents: VAE latent mean/std lengths must match latent channels.",
    );
  }
}

function expectSingleFrameSample(sample: MxArray): readonly [number, number, number, number] {
  const [batch, channels, frames, height, width] = sample.shape;
  if (
    sample.shape.length !== 5 ||
    batch === undefined ||
    channels === undefined ||
    frames === undefined ||
    height === undefined ||
    width === undefined
  ) {
    throw new Error(
      `decodeQwenImageLatents: expected decoded NCFHW sample, got ${formatShape(sample.shape)}.`,
    );
  }
  if (frames !== 1) {
    throw new Error(`decodeQwenImageLatents: expected one decoded frame, got ${frames}.`);
  }
  return [batch, channels, height, width];
}

function latentStatsTensor(
  values: readonly number[],
  channels: number,
  dtype: MxArray["dtype"],
): MxArray {
  using vector = array([...values], dtype);
  return reshape(vector, [1, channels, 1, 1, 1]);
}

/** Decode packed Qwen-Image latents into an NHWC image tensor in the `0..1` range. */
export function decodeQwenImageLatents(
  vae: QwenImageLatentDecoder,
  packedLatents: MxArray,
  latentHeight: number,
  latentWidth: number,
  patchSize = 2,
): MxArray {
  validateLatentStats(vae);

  using unpacked = unpackQwenImageLatents(packedLatents, latentHeight, latentWidth, patchSize);
  using std = latentStatsTensor(vae.latentsStd, vae.latentChannels, unpacked.dtype);
  using mean = latentStatsTensor(vae.latentsMean, vae.latentChannels, unpacked.dtype);
  using scaled = multiply(unpacked, std);
  using shifted = add(scaled, mean);
  using decoded = vae.decodeRaw(shifted);
  const [batch, channels, height, width] = expectSingleFrameSample(decoded);
  using firstFrame = slice(decoded, [0, 0, 0, 0, 0], [batch, channels, 1, height, width]);
  using imageNchw = squeeze(firstFrame, 2);
  using imageNhwc = transpose(imageNchw, [0, 2, 3, 1]);
  using positive = add(imageNhwc, 1);
  using normalized = divide(positive, 2);
  using lowerClamped = maximum(normalized, 0);
  return minimum(lowerClamped, 1);
}
