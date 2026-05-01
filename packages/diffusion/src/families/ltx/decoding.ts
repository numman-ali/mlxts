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
  transpose,
} from "@mlxts/core";

import { unpackLtx2AudioLatents, unpackLtxVideoLatents } from "./latents";

/** VAE decoder surface required by classic LTX video latent decoding. */
export type LtxVideoLatentDecoder = {
  readonly latentChannels: number;
  readonly latentsMean: readonly number[];
  readonly latentsStd: readonly number[];
  readonly scalingFactor: number;
  decodeRaw(latents: MxArray): MxArray;
};

/** Audio VAE decoder surface required by LTX-2 audio latent decoding. */
export type Ltx2AudioLatentDecoder = {
  readonly latentStatSize: number;
  readonly latentsMean: readonly number[];
  readonly latentsStd: readonly number[];
  decodeRaw(latents: MxArray): MxArray;
};

function validateLatentStats(vae: LtxVideoLatentDecoder): void {
  if (
    vae.latentsMean.length !== vae.latentChannels ||
    vae.latentsStd.length !== vae.latentChannels
  ) {
    throw new Error("decodeLtxVideoLatents: VAE latent mean/std lengths must match channels.");
  }
  if (vae.latentsMean.some((value) => !Number.isFinite(value))) {
    throw new Error("decodeLtxVideoLatents: VAE latent mean values must be finite.");
  }
  if (vae.latentsStd.some((value) => !Number.isFinite(value))) {
    throw new Error("decodeLtxVideoLatents: VAE latent std values must be finite.");
  }
  if (!Number.isFinite(vae.scalingFactor) || vae.scalingFactor <= 0) {
    throw new Error("decodeLtxVideoLatents: VAE scalingFactor must be positive.");
  }
}

function latentStatsTensor(
  values: readonly number[],
  channels: number,
  dtype: MxArray["dtype"],
): MxArray {
  using vector = array([...values], dtype);
  return reshape(vector, [1, channels, 1, 1, 1]);
}

function audioLatentStatsTensor(
  values: readonly number[],
  channels: number,
  dtype: MxArray["dtype"],
): MxArray {
  using vector = array([...values], dtype);
  return reshape(vector, [1, 1, channels]);
}

function expectDecodedVideo(video: MxArray): readonly [number, number, number, number, number] {
  const [batch, channels, frames, height, width] = video.shape;
  if (
    video.shape.length !== 5 ||
    batch === undefined ||
    channels === undefined ||
    frames === undefined ||
    height === undefined ||
    width === undefined
  ) {
    throw new Error(
      `decodeLtxVideoLatents: expected decoded BCFHW sample, got ${formatShape(video.shape)}.`,
    );
  }
  return [batch, channels, frames, height, width];
}

function expectDecodedAudio(audio: MxArray): readonly [number, number, number, number] {
  const [batch, channels, length, melBins] = audio.shape;
  if (
    audio.shape.length !== 4 ||
    batch === undefined ||
    channels === undefined ||
    length === undefined ||
    melBins === undefined
  ) {
    throw new Error(
      `decodeLtx2AudioLatents: expected decoded BCLM sample, got ${formatShape(audio.shape)}.`,
    );
  }
  return [batch, channels, length, melBins];
}

function validateAudioLatentStats(vae: Ltx2AudioLatentDecoder): void {
  if (
    vae.latentsMean.length !== vae.latentStatSize ||
    vae.latentsStd.length !== vae.latentStatSize
  ) {
    throw new Error(
      "decodeLtx2AudioLatents: VAE latent mean/std lengths must match packed features.",
    );
  }
  if (vae.latentsMean.some((value) => !Number.isFinite(value))) {
    throw new Error("decodeLtx2AudioLatents: VAE latent mean values must be finite.");
  }
  if (vae.latentsStd.some((value) => !Number.isFinite(value))) {
    throw new Error("decodeLtx2AudioLatents: VAE latent std values must be finite.");
  }
}

function expectPackedAudioLatents(packedLatents: MxArray): readonly [number, number, number] {
  const [batch, sequenceLength, featureSize] = packedLatents.shape;
  if (
    packedLatents.shape.length !== 3 ||
    batch === undefined ||
    sequenceLength === undefined ||
    featureSize === undefined
  ) {
    throw new Error(
      `denormalizeLtx2AudioLatents: expected packed BLD audio latents, got ${formatShape(
        packedLatents.shape,
      )}.`,
    );
  }
  return [batch, sequenceLength, featureSize];
}

/** Apply Diffusers LTX channelwise latent denormalization to BCFHW latents. */
export function denormalizeLtxVideoLatents(vae: LtxVideoLatentDecoder, latents: MxArray): MxArray {
  validateLatentStats(vae);
  using std = latentStatsTensor(vae.latentsStd, vae.latentChannels, latents.dtype);
  using mean = latentStatsTensor(vae.latentsMean, vae.latentChannels, latents.dtype);
  using scaled = multiply(latents, std);
  using unscaled = divide(scaled, vae.scalingFactor);
  return add(unscaled, mean);
}

/** Apply Diffusers LTX-2 audio denormalization to packed audio latent tokens. */
export function denormalizeLtx2AudioLatents(
  vae: Ltx2AudioLatentDecoder,
  packedLatents: MxArray,
): MxArray {
  validateAudioLatentStats(vae);
  const [, , featureSize] = expectPackedAudioLatents(packedLatents);
  if (featureSize !== vae.latentStatSize) {
    throw new Error(
      `denormalizeLtx2AudioLatents: expected ${vae.latentStatSize} packed features, got ${featureSize}.`,
    );
  }
  using std = audioLatentStatsTensor(vae.latentsStd, vae.latentStatSize, packedLatents.dtype);
  using mean = audioLatentStatsTensor(vae.latentsMean, vae.latentStatSize, packedLatents.dtype);
  using scaled = multiply(packedLatents, std);
  return add(scaled, mean);
}

/** Decode packed LTX video latents into a BFHWC video tensor in the `0..1` range. */
export function decodeLtxVideoLatents(
  vae: LtxVideoLatentDecoder,
  packedLatents: MxArray,
  latentFrames: number,
  latentHeight: number,
  latentWidth: number,
  patchSize = 1,
  patchSizeT = 1,
): MxArray {
  using unpacked = unpackLtxVideoLatents(
    packedLatents,
    latentFrames,
    latentHeight,
    latentWidth,
    patchSize,
    patchSizeT,
  );
  using denormalized = denormalizeLtxVideoLatents(vae, unpacked);
  using decoded = vae.decodeRaw(denormalized);
  const [, , frames, height, width] = expectDecodedVideo(decoded);
  using bfhwc = transpose(decoded, [0, 2, 3, 4, 1]);
  using positive = add(bfhwc, 1);
  using normalized = divide(positive, 2);
  using lowerClamped = maximum(normalized, 0);
  const video = minimum(lowerClamped, 1);
  if (video.shape[1] !== frames || video.shape[2] !== height || video.shape[3] !== width) {
    video.free();
    throw new Error("decodeLtxVideoLatents: decoded video shape changed unexpectedly.");
  }
  return video;
}

/** Decode packed LTX-2 audio latents into a BCLM mel spectrogram tensor. */
export function decodeLtx2AudioLatents(
  vae: Ltx2AudioLatentDecoder,
  packedLatents: MxArray,
  latentLength: number,
  latentMelBins: number,
  patchSize?: number,
  patchSizeT?: number,
): MxArray {
  using denormalized = denormalizeLtx2AudioLatents(vae, packedLatents);
  using unpacked = unpackLtx2AudioLatents(
    denormalized,
    latentLength,
    latentMelBins,
    patchSize,
    patchSizeT,
  );
  const decoded = vae.decodeRaw(unpacked);
  try {
    expectDecodedAudio(decoded);
    return decoded;
  } catch (error) {
    decoded.free();
    throw error;
  }
}
