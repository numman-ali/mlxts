import { type DType, type MxArray, random, reshape, transpose } from "@mlxts/core";

import type { FlowMatchEulerScheduler } from "../../schedulers/flow-match-euler";
import { assertSequence3d } from "./tensor-utils";

export type StableDiffusion3InitialLatentOptions = {
  scheduler: FlowMatchEulerScheduler;
  batchSize: number;
  height: number;
  width: number;
  latentChannels: number;
  vaeScaleFactor?: number;
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

/** Return the NHWC latent shape for a Stable Diffusion 3 image size. */
export function stableDiffusion3LatentShape(
  options: Omit<StableDiffusion3InitialLatentOptions, "scheduler" | "dtype" | "rngKey">,
): readonly [number, number, number, number] {
  expectPositiveInteger(options.batchSize, "batchSize");
  expectPositiveInteger(options.height, "height");
  expectPositiveInteger(options.width, "width");
  expectPositiveInteger(options.latentChannels, "latentChannels");
  const vaeScaleFactor = options.vaeScaleFactor ?? 8;
  expectPositiveInteger(vaeScaleFactor, "vaeScaleFactor");
  expectDivisible(options.height, vaeScaleFactor, "height");
  expectDivisible(options.width, vaeScaleFactor, "width");
  return [
    options.batchSize,
    options.height / vaeScaleFactor,
    options.width / vaeScaleFactor,
    options.latentChannels,
  ];
}

/** Create FlowMatch initial noise latents for Stable Diffusion 3 text-to-image sampling. */
export function createStableDiffusion3InitialLatents(
  options: StableDiffusion3InitialLatentOptions,
): MxArray {
  const shape = stableDiffusion3LatentShape(options);
  const dtype = options.dtype ?? "float32";
  using noise = random.normal([...shape], dtype, 0, 1, options.rngKey);
  return options.scheduler.scaleInitialNoise(noise);
}

/** Unpack SD3 patch-token predictions into NHWC latent images. */
export function unpatchifyStableDiffusion3Latents(
  patches: MxArray,
  latentHeight: number,
  latentWidth: number,
  patchSize: number,
  outChannels: number,
): MxArray {
  expectPositiveInteger(latentHeight, "latentHeight");
  expectPositiveInteger(latentWidth, "latentWidth");
  expectPositiveInteger(patchSize, "patchSize");
  expectPositiveInteger(outChannels, "outChannels");
  expectDivisible(latentHeight, patchSize, "latentHeight");
  expectDivisible(latentWidth, patchSize, "latentWidth");
  const shape = assertSequence3d(patches, "unpatchifyStableDiffusion3Latents");
  const gridHeight = latentHeight / patchSize;
  const gridWidth = latentWidth / patchSize;
  const expectedLength = gridHeight * gridWidth;
  const expectedChannels = patchSize * patchSize * outChannels;
  if (shape.length !== expectedLength || shape.channels !== expectedChannels) {
    throw new Error(
      `unpatchifyStableDiffusion3Latents: expected patches [batch, ${expectedLength}, ${expectedChannels}].`,
    );
  }
  using grid = reshape(patches, [
    shape.batch,
    gridHeight,
    gridWidth,
    patchSize,
    patchSize,
    outChannels,
  ]);
  using spatial = transpose(grid, [0, 1, 3, 2, 4, 5]);
  return reshape(spatial, [shape.batch, latentHeight, latentWidth, outChannels]);
}
