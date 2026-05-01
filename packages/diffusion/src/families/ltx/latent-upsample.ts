import { add, array, divide, type MxArray, multiply, reshape, subtract } from "@mlxts/core";
import type { LtxVideoLatentUpsamplerModel } from "./latent-upsampler";
import { packLtxVideoLatents, unpackLtxVideoLatents } from "./latents";

/** Latent normalization contract shared by classic LTX video VAE and upsampler paths. */
export type LtxVideoLatentNormalizer = {
  readonly latentChannels: number;
  readonly latentsMean: readonly number[];
  readonly latentsStd: readonly number[];
  readonly scalingFactor: number;
};

function validateLatentNormalizer(normalizer: LtxVideoLatentNormalizer, owner: string): void {
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

function denormalizeUpsamplerLatents(
  normalizer: LtxVideoLatentNormalizer,
  latents: MxArray,
): MxArray {
  validateLatentNormalizer(normalizer, "upsampleLtxVideoLatents");
  using std = latentStatsTensor(normalizer.latentsStd, normalizer.latentChannels, latents.dtype);
  using mean = latentStatsTensor(normalizer.latentsMean, normalizer.latentChannels, latents.dtype);
  using scaled = multiply(latents, std);
  using unscaled = divide(scaled, normalizer.scalingFactor);
  return add(unscaled, mean);
}

function normalizeUpsamplerLatents(
  normalizer: LtxVideoLatentNormalizer,
  latents: MxArray,
): MxArray {
  validateLatentNormalizer(normalizer, "upsampleLtxVideoLatents");
  using std = latentStatsTensor(normalizer.latentsStd, normalizer.latentChannels, latents.dtype);
  using mean = latentStatsTensor(normalizer.latentsMean, normalizer.latentChannels, latents.dtype);
  using centered = subtract(latents, mean);
  using scaled = multiply(centered, normalizer.scalingFactor);
  return divide(scaled, std);
}

/** Run the classic LTX sidecar upsampler over normalized BCFHW latents. */
export function upsampleLtxVideoLatents(
  upsampler: LtxVideoLatentUpsamplerModel,
  normalizer: LtxVideoLatentNormalizer,
  normalizedLatents: MxArray,
): MxArray {
  using denormalized = denormalizeUpsamplerLatents(normalizer, normalizedLatents);
  using upsampled = upsampler.forward(denormalized);
  return normalizeUpsamplerLatents(normalizer, upsampled);
}

/** Upsample packed classic LTX video latents and return a packed token sequence. */
export function upsamplePackedLtxVideoLatents(
  upsampler: LtxVideoLatentUpsamplerModel,
  normalizer: LtxVideoLatentNormalizer,
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
  using upsampled = upsampleLtxVideoLatents(upsampler, normalizer, unpacked);
  return packLtxVideoLatents(upsampled, patchSize, patchSizeT);
}
