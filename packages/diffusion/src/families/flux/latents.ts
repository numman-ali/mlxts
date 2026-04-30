import { type DType, MxArray, reshape, transpose } from "@mlxts/core";

function expectPositiveInteger(value: number, name: string): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
}

function expectEvenPositiveInteger(value: number, name: string): void {
  expectPositiveInteger(value, name);
  if (value % 2 !== 0) {
    throw new Error(`${name} must be divisible by 2.`);
  }
}

function expectNhwcLatents(latents: MxArray): readonly [number, number, number, number] {
  const [batchSize, height, width, channels] = latents.shape;
  if (
    latents.shape.length !== 4 ||
    batchSize === undefined ||
    height === undefined ||
    width === undefined ||
    channels === undefined
  ) {
    throw new Error("Flux latents must have NHWC rank 4 shape.");
  }
  expectPositiveInteger(batchSize, "batchSize");
  expectEvenPositiveInteger(height, "height");
  expectEvenPositiveInteger(width, "width");
  expectPositiveInteger(channels, "channels");
  return [batchSize, height, width, channels];
}

function expectPackedLatents(
  latents: MxArray,
  latentHeight: number,
  latentWidth: number,
): readonly [number, number, number] {
  const [batchSize, patches, packedChannels] = latents.shape;
  if (
    latents.shape.length !== 3 ||
    batchSize === undefined ||
    patches === undefined ||
    packedChannels === undefined
  ) {
    throw new Error("Packed Flux latents must have rank 3 shape.");
  }
  expectPositiveInteger(batchSize, "batchSize");
  expectEvenPositiveInteger(latentHeight, "latentHeight");
  expectEvenPositiveInteger(latentWidth, "latentWidth");
  expectPositiveInteger(packedChannels, "packedChannels");
  if (packedChannels % 4 !== 0) {
    throw new Error("packedChannels must be divisible by 4.");
  }
  const expectedPatches = (latentHeight / 2) * (latentWidth / 2);
  if (patches !== expectedPatches) {
    throw new Error(`Packed Flux latents require ${expectedPatches} patches, got ${patches}.`);
  }
  return [batchSize, patches, packedChannels];
}

/** Packed Flux latent tensor shape for an NHWC latent image. */
export function fluxPackedLatentShape(
  batchSize: number,
  latentHeight: number,
  latentWidth: number,
  latentChannels: number,
): [number, number, number] {
  expectPositiveInteger(batchSize, "batchSize");
  expectEvenPositiveInteger(latentHeight, "latentHeight");
  expectEvenPositiveInteger(latentWidth, "latentWidth");
  expectPositiveInteger(latentChannels, "latentChannels");
  return [batchSize, (latentHeight / 2) * (latentWidth / 2), latentChannels * 4];
}

/** Pack NHWC Flux latent images into 2x2 patch sequences. */
export function packFluxLatents(latents: MxArray): MxArray {
  const [batchSize, height, width, channels] = expectNhwcLatents(latents);
  using grid = reshape(latents, [batchSize, height / 2, 2, width / 2, 2, channels]);
  using patchMajor = transpose(grid, [0, 1, 3, 5, 2, 4]);
  return reshape(patchMajor, fluxPackedLatentShape(batchSize, height, width, channels));
}

/** Unpack 2x2 Flux patch sequences back into NHWC latent images. */
export function unpackFluxLatents(
  latents: MxArray,
  latentHeight: number,
  latentWidth: number,
): MxArray {
  const [batchSize, , packedChannels] = expectPackedLatents(latents, latentHeight, latentWidth);
  const channels = packedChannels / 4;
  using grid = reshape(latents, [batchSize, latentHeight / 2, latentWidth / 2, channels, 2, 2]);
  using nhwcGrid = transpose(grid, [0, 1, 4, 2, 5, 3]);
  return reshape(nhwcGrid, [batchSize, latentHeight, latentWidth, channels]);
}

/** Create Flux image position ids for packed latent patches. */
export function createFluxLatentImageIds(
  packedHeight: number,
  packedWidth: number,
  dtype: DType = "int32",
): MxArray {
  expectPositiveInteger(packedHeight, "packedHeight");
  expectPositiveInteger(packedWidth, "packedWidth");

  const patchCount = packedHeight * packedWidth;
  const values = new Int32Array(patchCount * 3);
  let offset = 0;
  for (let row = 0; row < packedHeight; row += 1) {
    for (let column = 0; column < packedWidth; column += 1) {
      values[offset] = 0;
      values[offset + 1] = row;
      values[offset + 2] = column;
      offset += 3;
    }
  }
  return MxArray.fromData(values, [patchCount, 3], dtype);
}
