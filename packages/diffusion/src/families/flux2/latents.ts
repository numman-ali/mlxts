import { type DType, MxArray, reshape, transpose } from "@mlxts/core";

import type { FlowMatchEulerScheduler } from "../../schedulers/flow-match-euler";

export type Flux2InitialLatentOptions = {
  scheduler: FlowMatchEulerScheduler;
  batchSize: number;
  height: number;
  width: number;
  latentChannels: number;
  vaeScaleFactor?: number;
  patchSize?: number;
  dtype?: DType;
  rngKey?: MxArray;
};

function expectPositiveInteger(value: number, name: string): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
}

function expectDivisible(value: number, divisor: number, name: string): void {
  if (value % divisor !== 0) {
    throw new Error(`${name} must be divisible by ${divisor}.`);
  }
}

function expectNchwLatents(
  latents: MxArray,
  owner: string,
): readonly [number, number, number, number] {
  const [batchSize, channels, height, width] = latents.shape;
  if (
    latents.shape.length !== 4 ||
    batchSize === undefined ||
    channels === undefined ||
    height === undefined ||
    width === undefined
  ) {
    throw new Error(`${owner}: expected NCHW rank 4 latents.`);
  }
  expectPositiveInteger(batchSize, "batchSize");
  expectPositiveInteger(channels, "channels");
  expectPositiveInteger(height, "height");
  expectPositiveInteger(width, "width");
  return [batchSize, channels, height, width];
}

function expectPackedSequence(
  latents: MxArray,
  packedHeight: number,
  packedWidth: number,
): readonly [number, number, number] {
  const [batchSize, length, channels] = latents.shape;
  if (
    latents.shape.length !== 3 ||
    batchSize === undefined ||
    length === undefined ||
    channels === undefined
  ) {
    throw new Error("Packed FLUX.2 latents must have rank 3 shape.");
  }
  expectPositiveInteger(batchSize, "batchSize");
  expectPositiveInteger(length, "length");
  expectPositiveInteger(channels, "channels");
  expectPositiveInteger(packedHeight, "packedHeight");
  expectPositiveInteger(packedWidth, "packedWidth");
  const expectedLength = packedHeight * packedWidth;
  if (length !== expectedLength) {
    throw new Error(`Packed FLUX.2 latents require ${expectedLength} tokens, got ${length}.`);
  }
  return [batchSize, length, channels];
}

/** NCHW latent map shape used by the FLUX.2 Klein denoising loop. */
export function flux2LatentMapShape(
  batchSize: number,
  height: number,
  width: number,
  latentChannels: number,
  vaeScaleFactor = 8,
  patchSize = 2,
): [number, number, number, number] {
  expectPositiveInteger(batchSize, "batchSize");
  expectPositiveInteger(height, "height");
  expectPositiveInteger(width, "width");
  expectPositiveInteger(latentChannels, "latentChannels");
  expectPositiveInteger(vaeScaleFactor, "vaeScaleFactor");
  expectPositiveInteger(patchSize, "patchSize");
  const multiple = vaeScaleFactor * patchSize;
  expectDivisible(height, multiple, "height");
  expectDivisible(width, multiple, "width");
  return [batchSize, latentChannels * patchSize * patchSize, height / multiple, width / multiple];
}

/** Packed FLUX.2 latent sequence shape for a generated image size. */
export function flux2PackedLatentShape(
  batchSize: number,
  height: number,
  width: number,
  latentChannels: number,
  vaeScaleFactor = 8,
  patchSize = 2,
): [number, number, number] {
  const [, channels, packedHeight, packedWidth] = flux2LatentMapShape(
    batchSize,
    height,
    width,
    latentChannels,
    vaeScaleFactor,
    patchSize,
  );
  return [batchSize, packedHeight * packedWidth, channels];
}

/** Pack NCHW FLUX.2 latent maps into row-major token sequences. */
export function packFlux2Latents(latents: MxArray): MxArray {
  const [batchSize, channels, height, width] = expectNchwLatents(latents, "packFlux2Latents");
  using flattened = reshape(latents, [batchSize, channels, height * width]);
  return transpose(flattened, [0, 2, 1]);
}

/** Unpack row-major FLUX.2 latent token sequences into NCHW latent maps. */
export function unpackFlux2Latents(
  latents: MxArray,
  packedHeight: number,
  packedWidth: number,
): MxArray {
  const [batchSize, , channels] = expectPackedSequence(latents, packedHeight, packedWidth);
  using channelLast = transpose(latents, [0, 2, 1]);
  return reshape(channelLast, [batchSize, channels, packedHeight, packedWidth]);
}

/** Convert VAE latent samples from NCHW to the FLUX.2 2x2 packed latent map. */
export function patchifyFlux2VaeLatents(latents: MxArray, patchSize = 2): MxArray {
  const [batchSize, channels, height, width] = expectNchwLatents(
    latents,
    "patchifyFlux2VaeLatents",
  );
  expectPositiveInteger(patchSize, "patchSize");
  expectDivisible(height, patchSize, "height");
  expectDivisible(width, patchSize, "width");
  using grid = reshape(latents, [
    batchSize,
    channels,
    height / patchSize,
    patchSize,
    width / patchSize,
    patchSize,
  ]);
  using packed = transpose(grid, [0, 1, 3, 5, 2, 4]);
  return reshape(packed, [
    batchSize,
    channels * patchSize * patchSize,
    height / patchSize,
    width / patchSize,
  ]);
}

/** Convert FLUX.2 2x2 packed latent maps back into VAE NCHW latent samples. */
export function unpatchifyFlux2VaeLatents(latents: MxArray, patchSize = 2): MxArray {
  const [batchSize, channels, height, width] = expectNchwLatents(
    latents,
    "unpatchifyFlux2VaeLatents",
  );
  expectPositiveInteger(patchSize, "patchSize");
  const patchArea = patchSize * patchSize;
  expectDivisible(channels, patchArea, "channels");
  using grid = reshape(latents, [
    batchSize,
    channels / patchArea,
    patchSize,
    patchSize,
    height,
    width,
  ]);
  using unpatched = transpose(grid, [0, 1, 4, 2, 5, 3]);
  return reshape(unpatched, [
    batchSize,
    channels / patchArea,
    height * patchSize,
    width * patchSize,
  ]);
}

/** Create FLUX.2 text position ids with columns `[time, height, width, token]`. */
export function createFlux2TextIds(sequenceLength: number, dtype: DType = "int32"): MxArray {
  expectPositiveInteger(sequenceLength, "sequenceLength");
  const values = new Int32Array(sequenceLength * 4);
  for (let index = 0; index < sequenceLength; index += 1) {
    values[index * 4 + 3] = index;
  }
  return MxArray.fromData(values, [sequenceLength, 4], dtype);
}

/** Create FLUX.2 latent position ids with columns `[time, height, width, token]`. */
export function createFlux2LatentIds(
  packedHeight: number,
  packedWidth: number,
  dtype: DType = "int32",
): MxArray {
  expectPositiveInteger(packedHeight, "packedHeight");
  expectPositiveInteger(packedWidth, "packedWidth");
  const values = new Int32Array(packedHeight * packedWidth * 4);
  let offset = 0;
  for (let row = 0; row < packedHeight; row += 1) {
    for (let column = 0; column < packedWidth; column += 1) {
      values[offset] = 0;
      values[offset + 1] = row;
      values[offset + 2] = column;
      values[offset + 3] = 0;
      offset += 4;
    }
  }
  return MxArray.fromData(values, [packedHeight * packedWidth, 4], dtype);
}

/** Create packed FLUX.2 initial noise latents for text-to-image sampling. */
export function createFlux2InitialLatents(options: Flux2InitialLatentOptions): MxArray {
  const shape = flux2LatentMapShape(
    options.batchSize,
    options.height,
    options.width,
    options.latentChannels,
    options.vaeScaleFactor,
    options.patchSize,
  );
  using latents = options.scheduler.samplePrior(shape, options.dtype ?? "float32", options.rngKey);
  return packFlux2Latents(latents);
}
