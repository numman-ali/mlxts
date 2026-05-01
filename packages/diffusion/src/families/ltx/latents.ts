import { type DType, type MxArray, reshape, transpose } from "@mlxts/core";

import type { FlowMatchEulerScheduler } from "../../schedulers/flow-match-euler";

export type LtxVideoInitialLatentOptions = {
  scheduler: FlowMatchEulerScheduler;
  batchSize: number;
  height: number;
  width: number;
  numFrames: number;
  latentChannels: number;
  vaeSpatialCompressionRatio?: number;
  vaeTemporalCompressionRatio?: number;
  patchSize?: number;
  patchSizeT?: number;
  dtype?: DType;
  rngKey?: MxArray;
};

export type Ltx2AudioInitialLatentOptions = {
  scheduler: FlowMatchEulerScheduler;
  batchSize: number;
  numFrames: number;
  frameRate: number;
  latentChannels: number;
  melBins: number;
  sampleRate?: number;
  hopLength?: number;
  temporalCompressionRatio?: number;
  melCompressionRatio?: number;
  patchSize?: number;
  patchSizeT?: number;
  dtype?: DType;
  rngKey?: MxArray;
};

function expectPositiveInteger(value: number, name: string): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
}

function expectPositiveNumber(value: number, name: string): void {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${name} must be a positive finite number.`);
  }
}

function expectDivisible(value: number, divisor: number, name: string): void {
  if (value % divisor !== 0) {
    throw new Error(`${name} must be divisible by ${divisor}.`);
  }
}

function resolvePositiveInteger(value: number | undefined, fallback: number, name: string): number {
  const resolved = value ?? fallback;
  expectPositiveInteger(resolved, name);
  return resolved;
}

function roundHalfEven(value: number): number {
  const lower = Math.floor(value);
  const fraction = value - lower;
  const tieTolerance = Number.EPSILON * Math.max(1, Math.abs(value));
  if (Math.abs(fraction - 0.5) <= tieTolerance) {
    return lower % 2 === 0 ? lower : lower + 1;
  }
  return Math.round(value);
}

function expectVideoLatents(
  latents: MxArray,
  patchSize: number,
  patchSizeT: number,
): readonly [number, number, number, number, number] {
  const [batchSize, channels, frames, height, width] = latents.shape;
  if (
    latents.shape.length !== 5 ||
    batchSize === undefined ||
    channels === undefined ||
    frames === undefined ||
    height === undefined ||
    width === undefined
  ) {
    throw new Error("LTX video latents must have BCFHW rank 5 shape.");
  }
  expectPositiveInteger(batchSize, "batchSize");
  expectPositiveInteger(channels, "channels");
  expectPositiveInteger(frames, "frames");
  expectPositiveInteger(height, "height");
  expectPositiveInteger(width, "width");
  expectDivisible(frames, patchSizeT, "frames");
  expectDivisible(height, patchSize, "height");
  expectDivisible(width, patchSize, "width");
  return [batchSize, channels, frames, height, width];
}

function expectPackedLatents(
  latents: MxArray,
  sequenceLength: number,
): readonly [number, number, number] {
  const [batchSize, length, channels] = latents.shape;
  if (
    latents.shape.length !== 3 ||
    batchSize === undefined ||
    length === undefined ||
    channels === undefined
  ) {
    throw new Error("Packed LTX latents must have rank 3 shape.");
  }
  expectPositiveInteger(batchSize, "batchSize");
  expectPositiveInteger(length, "length");
  expectPositiveInteger(channels, "channels");
  if (length !== sequenceLength) {
    throw new Error(`Packed LTX latents require ${sequenceLength} tokens, got ${length}.`);
  }
  return [batchSize, length, channels];
}

function expectAudioLatents(
  latents: MxArray,
  patchSize: number | undefined,
  patchSizeT: number | undefined,
): readonly [number, number, number, number] {
  const [batchSize, channels, length, melBins] = latents.shape;
  if (
    latents.shape.length !== 4 ||
    batchSize === undefined ||
    channels === undefined ||
    length === undefined ||
    melBins === undefined
  ) {
    throw new Error("LTX-2 audio latents must have BCLM rank 4 shape.");
  }
  expectPositiveInteger(batchSize, "batchSize");
  expectPositiveInteger(channels, "channels");
  expectPositiveInteger(length, "length");
  expectPositiveInteger(melBins, "melBins");
  if (patchSize !== undefined) {
    expectDivisible(melBins, patchSize, "melBins");
  }
  if (patchSizeT !== undefined) {
    expectDivisible(length, patchSizeT, "length");
  }
  return [batchSize, channels, length, melBins];
}

function requireAudioPatchPair(
  patchSize: number | undefined,
  patchSizeT: number | undefined,
): readonly [number, number] | null {
  if (patchSize === undefined && patchSizeT === undefined) {
    return null;
  }
  if (patchSize === undefined || patchSizeT === undefined) {
    throw new Error("LTX-2 audio patch packing requires both patchSize and patchSizeT.");
  }
  expectPositiveInteger(patchSize, "patchSize");
  expectPositiveInteger(patchSizeT, "patchSizeT");
  return [patchSize, patchSizeT];
}

/** Return the BCFHW latent shape for an LTX video request. */
export function ltxVideoLatentShape(
  options: Omit<LtxVideoInitialLatentOptions, "scheduler" | "dtype" | "rngKey">,
): [number, number, number, number, number] {
  expectPositiveInteger(options.batchSize, "batchSize");
  expectPositiveInteger(options.height, "height");
  expectPositiveInteger(options.width, "width");
  expectPositiveInteger(options.numFrames, "numFrames");
  expectPositiveInteger(options.latentChannels, "latentChannels");
  const spatialRatio = resolvePositiveInteger(
    options.vaeSpatialCompressionRatio,
    32,
    "vaeSpatialCompressionRatio",
  );
  const temporalRatio = resolvePositiveInteger(
    options.vaeTemporalCompressionRatio,
    8,
    "vaeTemporalCompressionRatio",
  );
  expectDivisible(options.height, spatialRatio, "height");
  expectDivisible(options.width, spatialRatio, "width");
  return [
    options.batchSize,
    options.latentChannels,
    Math.floor((options.numFrames - 1) / temporalRatio) + 1,
    options.height / spatialRatio,
    options.width / spatialRatio,
  ];
}

/** Packed LTX video latent sequence shape for BCFHW latent dimensions. */
export function ltxVideoPackedLatentShape(
  batchSize: number,
  latentFrames: number,
  latentHeight: number,
  latentWidth: number,
  latentChannels: number,
  patchSize = 1,
  patchSizeT = 1,
): [number, number, number] {
  expectPositiveInteger(batchSize, "batchSize");
  expectPositiveInteger(latentFrames, "latentFrames");
  expectPositiveInteger(latentHeight, "latentHeight");
  expectPositiveInteger(latentWidth, "latentWidth");
  expectPositiveInteger(latentChannels, "latentChannels");
  expectPositiveInteger(patchSize, "patchSize");
  expectPositiveInteger(patchSizeT, "patchSizeT");
  expectDivisible(latentFrames, patchSizeT, "latentFrames");
  expectDivisible(latentHeight, patchSize, "latentHeight");
  expectDivisible(latentWidth, patchSize, "latentWidth");
  return [
    batchSize,
    (latentFrames / patchSizeT) * (latentHeight / patchSize) * (latentWidth / patchSize),
    latentChannels * patchSizeT * patchSize * patchSize,
  ];
}

/** Pack BCFHW LTX video latents into Diffusers-compatible token sequences. */
export function packLtxVideoLatents(latents: MxArray, patchSize = 1, patchSizeT = 1): MxArray {
  expectPositiveInteger(patchSize, "patchSize");
  expectPositiveInteger(patchSizeT, "patchSizeT");
  const [batchSize, channels, frames, height, width] = expectVideoLatents(
    latents,
    patchSize,
    patchSizeT,
  );
  using grid = reshape(latents, [
    batchSize,
    channels,
    frames / patchSizeT,
    patchSizeT,
    height / patchSize,
    patchSize,
    width / patchSize,
    patchSize,
  ]);
  using tokenMajor = transpose(grid, [0, 2, 4, 6, 1, 3, 5, 7]);
  return reshape(
    tokenMajor,
    ltxVideoPackedLatentShape(batchSize, frames, height, width, channels, patchSize, patchSizeT),
  );
}

/** Unpack LTX video token sequences back into BCFHW latent videos. */
export function unpackLtxVideoLatents(
  latents: MxArray,
  latentFrames: number,
  latentHeight: number,
  latentWidth: number,
  patchSize = 1,
  patchSizeT = 1,
): MxArray {
  const [, , packedChannels] = expectPackedLatents(
    latents,
    ltxVideoPackedLatentShape(
      1,
      latentFrames,
      latentHeight,
      latentWidth,
      1,
      patchSize,
      patchSizeT,
    )[1],
  );
  const batchSize = latents.shape[0];
  if (batchSize === undefined) {
    throw new Error("Packed LTX latents must expose a batch dimension.");
  }
  const patchVolume = patchSizeT * patchSize * patchSize;
  expectDivisible(packedChannels, patchVolume, "packedChannels");
  using grid = reshape(latents, [
    batchSize,
    latentFrames / patchSizeT,
    latentHeight / patchSize,
    latentWidth / patchSize,
    packedChannels / patchVolume,
    patchSizeT,
    patchSize,
    patchSize,
  ]);
  using channelFirst = transpose(grid, [0, 4, 1, 5, 2, 6, 3, 7]);
  return reshape(channelFirst, [
    batchSize,
    packedChannels / patchVolume,
    latentFrames,
    latentHeight,
    latentWidth,
  ]);
}

/** Create packed LTX video initial noise latents for text-to-video sampling. */
export function createLtxVideoInitialLatents(options: LtxVideoInitialLatentOptions): MxArray {
  const shape = ltxVideoLatentShape(options);
  using latents = options.scheduler.samplePrior(shape, options.dtype ?? "float32", options.rngKey);
  return packLtxVideoLatents(latents, options.patchSize, options.patchSizeT);
}

/** Return the LTX-2 audio latent length for a video duration. */
export function ltx2AudioLatentLength(
  numFrames: number,
  frameRate: number,
  sampleRate = 16000,
  hopLength = 160,
  temporalCompressionRatio = 4,
): number {
  expectPositiveInteger(numFrames, "numFrames");
  expectPositiveNumber(frameRate, "frameRate");
  expectPositiveInteger(sampleRate, "sampleRate");
  expectPositiveInteger(hopLength, "hopLength");
  expectPositiveInteger(temporalCompressionRatio, "temporalCompressionRatio");
  return roundHalfEven(
    (numFrames / frameRate) * (sampleRate / hopLength / temporalCompressionRatio),
  );
}

/** Return the BCLM latent shape for an LTX-2 audio request. */
export function ltx2AudioLatentShape(
  options: Omit<Ltx2AudioInitialLatentOptions, "scheduler" | "dtype" | "rngKey">,
): [number, number, number, number] {
  expectPositiveInteger(options.batchSize, "batchSize");
  expectPositiveInteger(options.latentChannels, "latentChannels");
  expectPositiveInteger(options.melBins, "melBins");
  const melCompressionRatio = resolvePositiveInteger(
    options.melCompressionRatio,
    4,
    "melCompressionRatio",
  );
  expectDivisible(options.melBins, melCompressionRatio, "melBins");
  return [
    options.batchSize,
    options.latentChannels,
    ltx2AudioLatentLength(
      options.numFrames,
      options.frameRate,
      options.sampleRate,
      options.hopLength,
      options.temporalCompressionRatio,
    ),
    options.melBins / melCompressionRatio,
  ];
}

/** Packed LTX-2 audio latent sequence shape for BCLM latent dimensions. */
export function ltx2AudioPackedLatentShape(
  batchSize: number,
  latentLength: number,
  latentMelBins: number,
  latentChannels: number,
  patchSize?: number,
  patchSizeT?: number,
): [number, number, number] {
  expectPositiveInteger(batchSize, "batchSize");
  expectPositiveInteger(latentLength, "latentLength");
  expectPositiveInteger(latentMelBins, "latentMelBins");
  expectPositiveInteger(latentChannels, "latentChannels");
  const patchPair = requireAudioPatchPair(patchSize, patchSizeT);
  if (patchPair === null) {
    return [batchSize, latentLength, latentChannels * latentMelBins];
  }
  const [resolvedPatchSize, resolvedPatchSizeT] = patchPair;
  expectDivisible(latentMelBins, resolvedPatchSize, "latentMelBins");
  expectDivisible(latentLength, resolvedPatchSizeT, "latentLength");
  return [
    batchSize,
    (latentLength / resolvedPatchSizeT) * (latentMelBins / resolvedPatchSize),
    latentChannels * resolvedPatchSizeT * resolvedPatchSize,
  ];
}

/** Pack BCLM LTX-2 audio latents into Diffusers-compatible token sequences. */
export function packLtx2AudioLatents(
  latents: MxArray,
  patchSize?: number,
  patchSizeT?: number,
): MxArray {
  const patchPair = requireAudioPatchPair(patchSize, patchSizeT);
  const [batchSize, channels, length, melBins] = expectAudioLatents(
    latents,
    patchPair?.[0],
    patchPair?.[1],
  );
  if (patchPair === null) {
    using lengthMajor = transpose(latents, [0, 2, 1, 3]);
    return reshape(lengthMajor, ltx2AudioPackedLatentShape(batchSize, length, melBins, channels));
  }
  const [resolvedPatchSize, resolvedPatchSizeT] = patchPair;
  using grid = reshape(latents, [
    batchSize,
    channels,
    length / resolvedPatchSizeT,
    resolvedPatchSizeT,
    melBins / resolvedPatchSize,
    resolvedPatchSize,
  ]);
  using tokenMajor = transpose(grid, [0, 2, 4, 1, 3, 5]);
  return reshape(
    tokenMajor,
    ltx2AudioPackedLatentShape(
      batchSize,
      length,
      melBins,
      channels,
      resolvedPatchSize,
      resolvedPatchSizeT,
    ),
  );
}

/** Unpack LTX-2 audio token sequences back into BCLM audio latents. */
export function unpackLtx2AudioLatents(
  latents: MxArray,
  latentLength: number,
  latentMelBins: number,
  patchSize?: number,
  patchSizeT?: number,
): MxArray {
  const patchPair = requireAudioPatchPair(patchSize, patchSizeT);
  const [, , packedChannels] = expectPackedLatents(
    latents,
    ltx2AudioPackedLatentShape(1, latentLength, latentMelBins, 1, patchSize, patchSizeT)[1],
  );
  const batchSize = latents.shape[0];
  if (batchSize === undefined) {
    throw new Error("Packed LTX-2 audio latents must expose a batch dimension.");
  }
  if (patchPair === null) {
    expectDivisible(packedChannels, latentMelBins, "packedChannels");
    using grid = reshape(latents, [
      batchSize,
      latentLength,
      packedChannels / latentMelBins,
      latentMelBins,
    ]);
    return transpose(grid, [0, 2, 1, 3]);
  }
  const [resolvedPatchSize, resolvedPatchSizeT] = patchPair;
  const patchArea = resolvedPatchSize * resolvedPatchSizeT;
  expectDivisible(packedChannels, patchArea, "packedChannels");
  using grid = reshape(latents, [
    batchSize,
    latentLength / resolvedPatchSizeT,
    latentMelBins / resolvedPatchSize,
    packedChannels / patchArea,
    resolvedPatchSizeT,
    resolvedPatchSize,
  ]);
  using channelFirst = transpose(grid, [0, 3, 1, 4, 2, 5]);
  return reshape(channelFirst, [
    batchSize,
    packedChannels / patchArea,
    latentLength,
    latentMelBins,
  ]);
}

/** Create packed LTX-2 audio initial noise latents for audio-video sampling. */
export function createLtx2AudioInitialLatents(options: Ltx2AudioInitialLatentOptions): MxArray {
  const shape = ltx2AudioLatentShape(options);
  using latents = options.scheduler.samplePrior(shape, options.dtype ?? "float32", options.rngKey);
  return packLtx2AudioLatents(latents, options.patchSize, options.patchSizeT);
}
