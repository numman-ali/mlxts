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
  retainArray,
  sqrt,
  transpose,
} from "@mlxts/core";

import { unpackFlux2Latents, unpatchifyFlux2VaeLatents } from "./latents";

/** VAE decoder surface required by FLUX.2 Klein latent image decoding. */
export type Flux2KleinLatentDecoder = {
  readonly latentChannels: number;
  readonly batchNormMean: readonly number[];
  readonly batchNormVar: readonly number[];
  readonly batchNormEps: number;
  readonly vaeScaleFactor?: number;
  readonly patchSize?: number;
  decode(latents: MxArray): MxArray;
};

function assertPositiveInteger(name: string, value: number): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
}

function assertPositiveFinite(name: string, value: number): void {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${name} must be a positive finite number.`);
  }
}

function validateLatentStats(vae: Flux2KleinLatentDecoder): number {
  const patchSize = vae.patchSize ?? 2;
  assertPositiveInteger("vae.patchSize", patchSize);
  const packedChannels = vae.latentChannels * patchSize * patchSize;
  if (vae.batchNormMean.length !== packedChannels || vae.batchNormVar.length !== packedChannels) {
    throw new Error(
      "decodeFlux2KleinLatents: VAE batch-norm mean/var lengths must match packed latent channels.",
    );
  }
  assertPositiveFinite("vae.batchNormEps", vae.batchNormEps);
  return patchSize;
}

function statsTensor(
  values: readonly number[],
  channels: number,
  dtype: MxArray["dtype"],
): MxArray {
  using vector = array([...values], dtype);
  return reshape(vector, [1, channels, 1, 1]);
}

function expectDecodedNchwImage(image: MxArray): readonly [number, number, number, number] {
  const [batch, channels, height, width] = image.shape;
  if (
    image.shape.length !== 4 ||
    batch === undefined ||
    channels === undefined ||
    height === undefined ||
    width === undefined
  ) {
    throw new Error(
      `decodeFlux2KleinLatents: expected decoded NCHW image, got ${formatShape(image.shape)}.`,
    );
  }
  return [batch, channels, height, width];
}

/** Decode packed FLUX.2 Klein latents into an NHWC image tensor in the `0..1` range. */
export function decodeFlux2KleinLatents(
  vae: Flux2KleinLatentDecoder,
  packedLatents: MxArray,
  packedHeight: number,
  packedWidth: number,
): MxArray {
  const patchSize = validateLatentStats(vae);
  using packedMap = unpackFlux2Latents(packedLatents, packedHeight, packedWidth);
  const packedChannels = vae.latentChannels * patchSize * patchSize;
  using mean = statsTensor(vae.batchNormMean, packedChannels, packedMap.dtype);
  using variance = statsTensor(vae.batchNormVar, packedChannels, packedMap.dtype);
  using stabilizedVariance = add(variance, vae.batchNormEps);
  using std = sqrt(stabilizedVariance);
  using scaled = multiply(packedMap, std);
  using shifted = add(scaled, mean);
  using vaeLatents = unpatchifyFlux2VaeLatents(shifted, patchSize);
  using decoded = vae.decode(vaeLatents);
  const [, , height, width] = expectDecodedNchwImage(decoded);
  using imageNhwc = transpose(decoded, [0, 2, 3, 1]);
  using positive = add(imageNhwc, 1);
  using normalized = divide(positive, 2);
  using lowerClamped = maximum(normalized, 0);
  using image = minimum(lowerClamped, 1);
  if (image.shape[1] !== height || image.shape[2] !== width) {
    throw new Error("decodeFlux2KleinLatents: decoded image shape changed unexpectedly.");
  }
  return retainArray(image);
}
