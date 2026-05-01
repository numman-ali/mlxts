/**
 * Z-Image latent sampling and patch geometry helpers.
 * @module
 */

import type { DType } from "@mlxts/core";
import {
  concatenate,
  expandDims,
  formatShape,
  MxArray,
  repeat,
  reshape,
  retainArray,
  squeeze,
  stack,
  transpose,
} from "@mlxts/core";

import type { FlowMatchEulerScheduler } from "../../schedulers/flow-match-euler";
import type { ZImagePatchGeometry } from "./config";
import { assertFeature2d, freeArrays, sliceAxis } from "./tensor-utils";

export type ZImageLatentSize = {
  frames: number;
  height: number;
  width: number;
};

export type ZImageTokenGrid = {
  frames: number;
  height: number;
  width: number;
};

export type ZImagePaddedFeature = {
  features: MxArray;
  positionIds: MxArray;
  padMask: MxArray;
  totalLength: number;
  originalLength: number;
};

export type ZImagePatchifiedLatent = {
  patches: MxArray;
  size: ZImageLatentSize;
  tokenGrid: ZImageTokenGrid;
};

export type ZImageInitialLatentOptions = {
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

function resolveVaeScaleFactor(vaeScaleFactor: number | undefined): number {
  const resolved = vaeScaleFactor ?? 8;
  expectPositiveInteger(resolved, "vaeScaleFactor");
  return resolved;
}

function expectZImageLatent(
  image: MxArray,
  owner: string,
): readonly [number, number, number, number] {
  const [channels, frames, height, width] = image.shape;
  if (
    image.shape.length !== 4 ||
    channels === undefined ||
    frames === undefined ||
    height === undefined ||
    width === undefined
  ) {
    throw new Error(
      `${owner}: expected [channels, frames, height, width], got ${formatShape(image.shape)}.`,
    );
  }
  expectPositiveInteger(channels, "channels");
  expectPositiveInteger(frames, "frames");
  expectPositiveInteger(height, "height");
  expectPositiveInteger(width, "width");
  return [channels, frames, height, width];
}

function coordinateIdData(
  size: readonly [number, number, number],
  start: readonly [number, number, number],
): Int32Array {
  const [frames, height, width] = size;
  const [frameStart, heightStart, widthStart] = start;
  expectPositiveInteger(frames, "frames");
  expectPositiveInteger(height, "height");
  expectPositiveInteger(width, "width");

  const data = new Int32Array(frames * height * width * 3);
  let offset = 0;
  for (let frame = 0; frame < frames; frame += 1) {
    for (let row = 0; row < height; row += 1) {
      for (let column = 0; column < width; column += 1) {
        data[offset] = frameStart + frame;
        data[offset + 1] = heightStart + row;
        data[offset + 2] = widthStart + column;
        offset += 3;
      }
    }
  }
  return data;
}

/** Return the NCHW latent shape used by Z-Image text-to-image sampling. */
export function zImageLatentShape(
  options: Omit<ZImageInitialLatentOptions, "scheduler" | "dtype" | "rngKey">,
): readonly [number, number, number, number] {
  expectPositiveInteger(options.batchSize, "batchSize");
  expectPositiveInteger(options.height, "height");
  expectPositiveInteger(options.width, "width");
  expectPositiveInteger(options.latentChannels, "latentChannels");
  const vaeScaleFactor = resolveVaeScaleFactor(options.vaeScaleFactor);
  const imageMultiple = vaeScaleFactor * 2;
  expectDivisible(options.height, imageMultiple, "height");
  expectDivisible(options.width, imageMultiple, "width");
  return [
    options.batchSize,
    options.latentChannels,
    options.height / vaeScaleFactor,
    options.width / vaeScaleFactor,
  ];
}

/** Create Z-Image initial noise latents for text-to-image sampling. */
export function createZImageInitialLatents(options: ZImageInitialLatentOptions): MxArray {
  const shape = zImageLatentShape(options);
  return options.scheduler.samplePrior([...shape], options.dtype ?? "float32", options.rngKey);
}

/** Create flattened Z-Image coordinate ids with columns `[frame, row, column]`. */
export function createZImageCoordinateIds(
  size: readonly [number, number, number],
  start: readonly [number, number, number] = [0, 0, 0],
): MxArray {
  const data = coordinateIdData(size, start);
  return MxArray.fromData(data, [data.length / 3, 3], "int32");
}

/** Pad a feature sequence to Z-Image's sequence multiple and build position ids. */
export function padZImageFeature(
  feature: MxArray,
  sequenceMultiple: number,
  positionGridSize: readonly [number, number, number],
  positionStart: readonly [number, number, number],
): ZImagePaddedFeature {
  expectPositiveInteger(sequenceMultiple, "sequenceMultiple");
  const { length } = assertFeature2d(feature, "padZImageFeature");
  const padLength = (sequenceMultiple - (length % sequenceMultiple)) % sequenceMultiple;
  const totalLength = length + padLength;
  const positionIds = createZImageCoordinateIds(positionGridSize, positionStart);
  const mask = new Array<number>(totalLength).fill(0);
  for (let index = length; index < totalLength; index += 1) {
    mask[index] = 1;
  }
  const padMask = MxArray.fromData(mask, [totalLength], "bool");
  if (padLength === 0) {
    return {
      features: retainArray(feature),
      positionIds,
      padMask,
      totalLength,
      originalLength: length,
    };
  }

  using last = sliceAxis(feature, 0, length - 1, length);
  using repeated = repeat(last, padLength, 0);
  const zeroPositionIds = MxArray.fromData(new Int32Array(padLength * 3), [padLength, 3], "int32");
  let paddedFeatures: MxArray | null = null;
  let paddedPositionIds: MxArray | null = null;
  try {
    paddedFeatures = concatenate([feature, repeated], 0);
    paddedPositionIds = concatenate([positionIds, zeroPositionIds], 0);
    positionIds.free();
    zeroPositionIds.free();
    return {
      features: paddedFeatures,
      positionIds: paddedPositionIds,
      padMask,
      totalLength,
      originalLength: length,
    };
  } catch (error) {
    paddedFeatures?.free();
    paddedPositionIds?.free();
    positionIds.free();
    zeroPositionIds.free();
    padMask.free();
    throw error;
  }
}

/** Convert one Z-Image latent sample from `[C,F,H,W]` to patch features. */
export function patchifyZImageLatent(
  image: MxArray,
  geometry: ZImagePatchGeometry,
): ZImagePatchifiedLatent {
  const [channels, frames, height, width] = expectZImageLatent(image, "patchifyZImageLatent");
  if (
    channels !==
    geometry.packedLatentChannels /
      (geometry.patchSize * geometry.patchSize * geometry.framePatchSize)
  ) {
    throw new Error("patchifyZImageLatent: latent channels do not match geometry.");
  }
  expectDivisible(frames, geometry.framePatchSize, "frames");
  expectDivisible(height, geometry.patchSize, "height");
  expectDivisible(width, geometry.patchSize, "width");

  const frameTokens = frames / geometry.framePatchSize;
  const heightTokens = height / geometry.patchSize;
  const widthTokens = width / geometry.patchSize;
  using grid = reshape(image, [
    channels,
    frameTokens,
    geometry.framePatchSize,
    heightTokens,
    geometry.patchSize,
    widthTokens,
    geometry.patchSize,
  ]);
  using patchMajor = transpose(grid, [1, 3, 5, 2, 4, 6, 0]);
  return {
    patches: reshape(patchMajor, [
      frameTokens * heightTokens * widthTokens,
      geometry.packedLatentChannels,
    ]),
    size: { frames, height, width },
    tokenGrid: { frames: frameTokens, height: heightTokens, width: widthTokens },
  };
}

/** Convert visible Z-Image patch features back to `[C,F,H,W]` latent layout. */
export function unpatchifyZImageLatent(
  patches: MxArray,
  size: ZImageLatentSize,
  geometry: ZImagePatchGeometry,
  outChannels: number,
): MxArray {
  expectPositiveInteger(outChannels, "outChannels");
  expectDivisible(size.frames, geometry.framePatchSize, "frames");
  expectDivisible(size.height, geometry.patchSize, "height");
  expectDivisible(size.width, geometry.patchSize, "width");
  const frameTokens = size.frames / geometry.framePatchSize;
  const heightTokens = size.height / geometry.patchSize;
  const widthTokens = size.width / geometry.patchSize;
  const originalLength = frameTokens * heightTokens * widthTokens;
  const [patchLength, packedChannels] = patches.shape;
  if (
    patches.shape.length !== 2 ||
    patchLength === undefined ||
    packedChannels !== geometry.packedLatentChannels
  ) {
    throw new Error(
      `unpatchifyZImageLatent: expected patch features [length, ${geometry.packedLatentChannels}], got ${formatShape(patches.shape)}.`,
    );
  }
  if (patchLength < originalLength) {
    throw new Error(
      `unpatchifyZImageLatent: expected at least ${originalLength} patches, got ${patchLength}.`,
    );
  }

  using visible = sliceAxis(patches, 0, 0, originalLength);
  using grid = reshape(visible, [
    frameTokens,
    heightTokens,
    widthTokens,
    geometry.framePatchSize,
    geometry.patchSize,
    geometry.patchSize,
    outChannels,
  ]);
  using channelFirst = transpose(grid, [6, 0, 3, 1, 4, 2, 5]);
  return reshape(channelFirst, [outChannels, size.frames, size.height, size.width]);
}

/** Extract one `[C,1,H,W]` Z-Image latent item from an NCHW batch. */
export function sliceZImageLatentBatchItem(latents: MxArray, batchIndex: number): MxArray {
  const [batchSize] = latents.shape;
  if (latents.shape.length !== 4 || batchSize === undefined) {
    throw new Error(
      `sliceZImageLatentBatchItem: expected NCHW latents, got ${formatShape(latents.shape)}.`,
    );
  }
  if (!Number.isInteger(batchIndex) || batchIndex < 0 || batchIndex >= batchSize) {
    throw new Error(`sliceZImageLatentBatchItem: batch index ${batchIndex} is out of range.`);
  }
  using selected = sliceAxis(latents, 0, batchIndex, batchIndex + 1);
  using sample = squeeze(selected, 0);
  return expandDims(sample, 1);
}

/** Stack `[C,1,H,W]` model outputs into an NCHW latent batch. */
export function stackZImageLatentBatchItems(samples: readonly MxArray[]): MxArray {
  if (samples.length === 0) {
    throw new Error("stackZImageLatentBatchItems: at least one sample is required.");
  }
  const squeezed: MxArray[] = [];
  try {
    for (const sample of samples) {
      expectZImageLatent(sample, "stackZImageLatentBatchItems");
      if (sample.shape[1] !== 1) {
        throw new Error("stackZImageLatentBatchItems: sample frame dimension must be 1.");
      }
      squeezed.push(squeeze(sample, 1));
    }
    return stack(squeezed, 0);
  } finally {
    freeArrays(squeezed);
  }
}
