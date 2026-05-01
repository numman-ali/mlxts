import { type DType, type MxArray, reshape, transpose } from "@mlxts/core";

import type { FlowMatchEulerScheduler } from "../../schedulers/flow-match-euler";

export type QwenImageInitialLatentOptions = {
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

export type QwenImageRopeImageShape = readonly [frames: number, height: number, width: number];

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

function resolveVaeScaleFactor(vaeScaleFactor: number | undefined): number {
  const resolved = vaeScaleFactor ?? 8;
  expectPositiveInteger(resolved, "vaeScaleFactor");
  return resolved;
}

function resolvePatchSize(patchSize: number | undefined): number {
  const resolved = patchSize ?? 2;
  expectPositiveInteger(resolved, "patchSize");
  return resolved;
}

function expectQwenImageLatents(
  latents: MxArray,
  patchSize: number,
): readonly [number, number, number, number] {
  if (latents.shape.length === 4) {
    const [batchSize, channels, height, width] = latents.shape;
    if (
      batchSize === undefined ||
      channels === undefined ||
      height === undefined ||
      width === undefined
    ) {
      throw new Error("Qwen-Image latents must have NCHW or NCFHW shape.");
    }
    expectPositiveInteger(batchSize, "batchSize");
    expectPositiveInteger(channels, "channels");
    expectPositiveInteger(height, "height");
    expectPositiveInteger(width, "width");
    expectDivisible(height, patchSize, "height");
    expectDivisible(width, patchSize, "width");
    return [batchSize, channels, height, width];
  }

  const [batchSize, channels, frames, height, width] = latents.shape;
  if (
    latents.shape.length !== 5 ||
    batchSize === undefined ||
    channels === undefined ||
    frames === undefined ||
    height === undefined ||
    width === undefined
  ) {
    throw new Error("Qwen-Image latents must have NCHW or NCFHW shape.");
  }
  expectPositiveInteger(batchSize, "batchSize");
  expectPositiveInteger(channels, "channels");
  if (frames !== 1) {
    throw new Error("Qwen-Image base latents require a single frame.");
  }
  expectPositiveInteger(height, "height");
  expectPositiveInteger(width, "width");
  expectDivisible(height, patchSize, "height");
  expectDivisible(width, patchSize, "width");
  return [batchSize, channels, height, width];
}

function expectPackedLatents(
  latents: MxArray,
  latentHeight: number,
  latentWidth: number,
  patchSize: number,
): readonly [number, number, number] {
  const [batchSize, patches, packedChannels] = latents.shape;
  if (
    latents.shape.length !== 3 ||
    batchSize === undefined ||
    patches === undefined ||
    packedChannels === undefined
  ) {
    throw new Error("Packed Qwen-Image latents must have rank 3 shape.");
  }
  expectPositiveInteger(batchSize, "batchSize");
  expectPositiveInteger(patches, "patches");
  expectPositiveInteger(packedChannels, "packedChannels");
  expectDivisible(latentHeight, patchSize, "latentHeight");
  expectDivisible(latentWidth, patchSize, "latentWidth");
  const patchArea = patchSize * patchSize;
  expectDivisible(packedChannels, patchArea, "packedChannels");
  const expectedPatches = (latentHeight / patchSize) * (latentWidth / patchSize);
  if (patches !== expectedPatches) {
    throw new Error(
      `Packed Qwen-Image latents require ${expectedPatches} patches, got ${patches}.`,
    );
  }
  return [batchSize, patches, packedChannels];
}

/** Return the NCFHW latent shape for a Qwen-Image image size and VAE scale factor. */
export function qwenImageLatentShape(
  options: Omit<QwenImageInitialLatentOptions, "scheduler" | "dtype" | "rngKey">,
): readonly [number, number, number, number, number] {
  expectPositiveInteger(options.batchSize, "batchSize");
  expectPositiveInteger(options.height, "height");
  expectPositiveInteger(options.width, "width");
  expectPositiveInteger(options.latentChannels, "latentChannels");
  const vaeScaleFactor = resolveVaeScaleFactor(options.vaeScaleFactor);
  const patchSize = resolvePatchSize(options.patchSize);
  const imageMultiple = vaeScaleFactor * patchSize;
  expectDivisible(options.height, imageMultiple, "height");
  expectDivisible(options.width, imageMultiple, "width");
  return [
    options.batchSize,
    options.latentChannels,
    1,
    options.height / vaeScaleFactor,
    options.width / vaeScaleFactor,
  ];
}

/** Packed Qwen-Image latent tensor shape for an NCHW or NCFHW latent image. */
export function qwenImagePackedLatentShape(
  batchSize: number,
  latentHeight: number,
  latentWidth: number,
  latentChannels: number,
  patchSize = 2,
): [number, number, number] {
  expectPositiveInteger(batchSize, "batchSize");
  expectPositiveInteger(latentHeight, "latentHeight");
  expectPositiveInteger(latentWidth, "latentWidth");
  expectPositiveInteger(latentChannels, "latentChannels");
  expectPositiveInteger(patchSize, "patchSize");
  expectDivisible(latentHeight, patchSize, "latentHeight");
  expectDivisible(latentWidth, patchSize, "latentWidth");
  return [
    batchSize,
    (latentHeight / patchSize) * (latentWidth / patchSize),
    latentChannels * patchSize * patchSize,
  ];
}

/** Pack NCHW or NCFHW Qwen-Image latent images into patch sequences. */
export function packQwenImageLatents(latents: MxArray, patchSize = 2): MxArray {
  expectPositiveInteger(patchSize, "patchSize");
  const [batchSize, channels, height, width] = expectQwenImageLatents(latents, patchSize);
  using grid = reshape(latents, [
    batchSize,
    channels,
    height / patchSize,
    patchSize,
    width / patchSize,
    patchSize,
  ]);
  using patchMajor = transpose(grid, [0, 2, 4, 1, 3, 5]);
  return reshape(
    patchMajor,
    qwenImagePackedLatentShape(batchSize, height, width, channels, patchSize),
  );
}

/** Unpack Qwen-Image patch sequences back into NCFHW latent images. */
export function unpackQwenImageLatents(
  latents: MxArray,
  latentHeight: number,
  latentWidth: number,
  patchSize = 2,
): MxArray {
  expectPositiveInteger(patchSize, "patchSize");
  const [batchSize, , packedChannels] = expectPackedLatents(
    latents,
    latentHeight,
    latentWidth,
    patchSize,
  );
  const channels = packedChannels / (patchSize * patchSize);
  using grid = reshape(latents, [
    batchSize,
    latentHeight / patchSize,
    latentWidth / patchSize,
    channels,
    patchSize,
    patchSize,
  ]);
  using channelFirstGrid = transpose(grid, [0, 3, 1, 4, 2, 5]);
  return reshape(channelFirstGrid, [batchSize, channels, 1, latentHeight, latentWidth]);
}

/** Create packed Qwen-Image initial noise latents for text-to-image sampling. */
export function createQwenImageInitialLatents(options: QwenImageInitialLatentOptions): MxArray {
  const shape = qwenImageLatentShape(options);
  using latents = options.scheduler.samplePrior(
    [...shape],
    options.dtype ?? "float32",
    options.rngKey,
  );
  return packQwenImageLatents(latents, options.patchSize);
}

/** Return the `[frames, height, width]` image shape consumed by Qwen-Image RoPE. */
export function qwenImageRopeImageShape(
  options: Omit<QwenImageInitialLatentOptions, "scheduler" | "dtype" | "rngKey">,
): QwenImageRopeImageShape {
  const [, , , latentHeight, latentWidth] = qwenImageLatentShape(options);
  const patchSize = resolvePatchSize(options.patchSize);
  return [1, latentHeight / patchSize, latentWidth / patchSize];
}
