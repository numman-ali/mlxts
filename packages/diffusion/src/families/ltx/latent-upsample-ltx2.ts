import { add, array, divide, type MxArray, multiply, reshape } from "@mlxts/core";

import type { Ltx2LatentUpsamplerModel } from "./latent-upsampler-ltx2";
import { packLtxVideoLatents, unpackLtxVideoLatents } from "./latents";

/** Latent normalization contract used when LTX-2 supplied latents are normalized. */
export type Ltx2VideoLatentNormalizer = {
  readonly latentChannels: number;
  readonly latentsMean: readonly number[];
  readonly latentsStd: readonly number[];
  readonly scalingFactor: number;
};

export type Ltx2VideoLatentUpsampleOptions = {
  readonly normalizer?: Ltx2VideoLatentNormalizer;
  readonly latentsNormalized?: boolean;
};

function validateLatentNormalizer(normalizer: Ltx2VideoLatentNormalizer, owner: string): void {
  if (
    normalizer.latentsMean.length !== normalizer.latentChannels ||
    normalizer.latentsStd.length !== normalizer.latentChannels
  ) {
    throw new Error(`${owner}: latent mean/std lengths must match channels.`);
  }
  if (normalizer.latentsMean.some((value) => !Number.isFinite(value))) {
    throw new Error(`${owner}: latent mean values must be finite.`);
  }
  if (normalizer.latentsStd.some((value) => !Number.isFinite(value))) {
    throw new Error(`${owner}: latent std values must be finite.`);
  }
  if (!Number.isFinite(normalizer.scalingFactor) || normalizer.scalingFactor <= 0) {
    throw new Error(`${owner}: scalingFactor must be positive.`);
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

/** Denormalize LTX-2 video latents before supplying them to the sidecar upsampler. */
export function denormalizeLtx2VideoUpsamplerLatents(
  normalizer: Ltx2VideoLatentNormalizer,
  latents: MxArray,
): MxArray {
  validateLatentNormalizer(normalizer, "denormalizeLtx2VideoUpsamplerLatents");
  using std = latentStatsTensor(normalizer.latentsStd, normalizer.latentChannels, latents.dtype);
  using mean = latentStatsTensor(normalizer.latentsMean, normalizer.latentChannels, latents.dtype);
  using scaled = multiply(latents, std);
  using unscaled = divide(scaled, normalizer.scalingFactor);
  return add(unscaled, mean);
}

/** Run the LTX-2 sidecar upsampler over BCFHW latents. */
export function upsampleLtx2VideoLatents(
  upsampler: Ltx2LatentUpsamplerModel,
  latents: MxArray,
  options: Ltx2VideoLatentUpsampleOptions = {},
): MxArray {
  if (options.latentsNormalized === true) {
    if (options.normalizer === undefined) {
      throw new Error("upsampleLtx2VideoLatents: normalized latents require a normalizer.");
    }
    using denormalized = denormalizeLtx2VideoUpsamplerLatents(options.normalizer, latents);
    return upsampler.forward(denormalized);
  }
  return upsampler.forward(latents);
}

/** Upsample packed LTX-2 video latents and return a packed token sequence. */
export function upsamplePackedLtx2VideoLatents(
  upsampler: Ltx2LatentUpsamplerModel,
  packedLatents: MxArray,
  latentFrames: number,
  latentHeight: number,
  latentWidth: number,
  patchSize = 1,
  patchSizeT = 1,
  options: Ltx2VideoLatentUpsampleOptions = {},
): MxArray {
  using unpacked = unpackLtxVideoLatents(
    packedLatents,
    latentFrames,
    latentHeight,
    latentWidth,
    patchSize,
    patchSizeT,
  );
  using upsampled = upsampleLtx2VideoLatents(upsampler, unpacked, options);
  return packLtxVideoLatents(upsampled, patchSize, patchSizeT);
}
