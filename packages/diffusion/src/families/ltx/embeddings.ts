import { MxArray } from "@mlxts/core";

import type { LtxRotaryEmbeddings } from "./embeddings-rope";

export {
  createLtx2RotaryEmbeddings,
  type Ltx2RotaryEmbeddingOptions,
  type LtxRotaryEmbeddings,
} from "./embeddings-rope";

export type LtxVideoRopeInterpolationScale = readonly [
  temporal: number,
  height: number,
  width: number,
];

export type LtxVideoRopeCoordinateOptions = {
  batchSize: number;
  latentFrames: number;
  latentHeight: number;
  latentWidth: number;
  patchSize?: number;
  patchSizeT?: number;
  baseNumFrames?: number;
  baseHeight?: number;
  baseWidth?: number;
  ropeInterpolationScale?: LtxVideoRopeInterpolationScale;
};

export type LtxVideoRotaryEmbeddingOptions = LtxVideoRopeCoordinateOptions & {
  dim: number;
  theta?: number;
};

export type Ltx2VideoCoordinateOptions = {
  batchSize: number;
  latentFrames: number;
  latentHeight: number;
  latentWidth: number;
  patchSize?: number;
  patchSizeT?: number;
  vaeScaleFactors?: readonly [temporal: number, height: number, width: number];
  causalOffset?: number;
  frameRate?: number;
};

export type Ltx2AudioCoordinateOptions = {
  batchSize: number;
  audioLatentFrames: number;
  patchSizeT?: number;
  shift?: number;
  audioScaleFactor?: number;
  causalOffset?: number;
  hopLength?: number;
  samplingRate?: number;
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

function resolvePositiveInteger(value: number | undefined, fallback: number, name: string): number {
  const resolved = value ?? fallback;
  expectPositiveInteger(resolved, name);
  return resolved;
}

function resolvePositiveNumber(value: number | undefined, fallback: number, name: string): number {
  const resolved = value ?? fallback;
  expectPositiveNumber(resolved, name);
  return resolved;
}

function expectDivisible(value: number, divisor: number, name: string): void {
  if (value % divisor !== 0) {
    throw new Error(`${name} must be divisible by ${divisor}.`);
  }
}

function linspaceUnit(index: number, count: number): number {
  if (count === 1) {
    return 0;
  }
  return index / (count - 1);
}

function ltxVideoGridShape(options: LtxVideoRopeCoordinateOptions): {
  batchSize: number;
  latentFrames: number;
  latentHeight: number;
  latentWidth: number;
  patchSize: number;
  patchSizeT: number;
  tokens: number;
  baseNumFrames: number;
  baseHeight: number;
  baseWidth: number;
} {
  expectPositiveInteger(options.batchSize, "batchSize");
  expectPositiveInteger(options.latentFrames, "latentFrames");
  expectPositiveInteger(options.latentHeight, "latentHeight");
  expectPositiveInteger(options.latentWidth, "latentWidth");
  const patchSize = resolvePositiveInteger(options.patchSize, 1, "patchSize");
  const patchSizeT = resolvePositiveInteger(options.patchSizeT, 1, "patchSizeT");
  const baseNumFrames = resolvePositiveInteger(options.baseNumFrames, 20, "baseNumFrames");
  const baseHeight = resolvePositiveInteger(options.baseHeight, 2048, "baseHeight");
  const baseWidth = resolvePositiveInteger(options.baseWidth, 2048, "baseWidth");
  return {
    batchSize: options.batchSize,
    latentFrames: options.latentFrames,
    latentHeight: options.latentHeight,
    latentWidth: options.latentWidth,
    patchSize,
    patchSizeT,
    tokens: options.latentFrames * options.latentHeight * options.latentWidth,
    baseNumFrames,
    baseHeight,
    baseWidth,
  };
}

function ltxVideoCoordValues(
  options: LtxVideoRopeCoordinateOptions,
  shape = ltxVideoGridShape(options),
): Float32Array {
  const data = new Float32Array(shape.batchSize * shape.tokens * 3);
  let token = 0;
  for (let frame = 0; frame < shape.latentFrames; frame += 1) {
    for (let row = 0; row < shape.latentHeight; row += 1) {
      for (let column = 0; column < shape.latentWidth; column += 1) {
        const values = ltxVideoCoordTriple(frame, row, column, options, shape);
        for (let batch = 0; batch < shape.batchSize; batch += 1) {
          const offset = (batch * shape.tokens + token) * 3;
          data[offset] = values[0] ?? 0;
          data[offset + 1] = values[1] ?? 0;
          data[offset + 2] = values[2] ?? 0;
        }
        token += 1;
      }
    }
  }
  return data;
}

function ltxVideoCoordTriple(
  frame: number,
  row: number,
  column: number,
  options: LtxVideoRopeCoordinateOptions,
  shape: ReturnType<typeof ltxVideoGridShape>,
): readonly [number, number, number] {
  const scale = options.ropeInterpolationScale;
  if (scale === undefined) {
    return [frame, row, column];
  }
  return [
    (frame * scale[0] * shape.patchSizeT) / shape.baseNumFrames,
    (row * scale[1] * shape.patchSize) / shape.baseHeight,
    (column * scale[2] * shape.patchSize) / shape.baseWidth,
  ];
}

/** Create the classic LTX video RoPE coordinate grid with shape `[batch, tokens, 3]`. */
export function createLtxVideoRopeCoords(options: LtxVideoRopeCoordinateOptions): MxArray {
  const shape = ltxVideoGridShape(options);
  const data = ltxVideoCoordValues(options, shape);
  return MxArray.fromData(data, [shape.batchSize, shape.tokens, 3], "float32");
}

/** Create classic LTX interleaved cosine/sine RoPE tensors with shape `[batch, tokens, dim]`. */
export function createLtxVideoRotaryEmbeddings(
  options: LtxVideoRotaryEmbeddingOptions,
): LtxRotaryEmbeddings {
  expectPositiveInteger(options.dim, "dim");
  const shape = ltxVideoGridShape(options);
  const coordValues = ltxVideoCoordValues(options, shape);
  const frequencyCount = Math.floor(options.dim / 6);
  if (frequencyCount <= 0) {
    throw new Error("dim must be at least 6 for LTX video RoPE.");
  }
  const padding = options.dim % 6;
  const theta = resolvePositiveNumber(options.theta, 10000, "theta");
  const cosValues = new Float32Array(shape.batchSize * shape.tokens * options.dim);
  const sinValues = new Float32Array(cosValues.length);
  for (let batch = 0; batch < shape.batchSize; batch += 1) {
    for (let token = 0; token < shape.tokens; token += 1) {
      const coordOffset = (batch * shape.tokens + token) * 3;
      const outputOffset = (batch * shape.tokens + token) * options.dim;
      for (let index = 0; index < padding; index += 1) {
        cosValues[outputOffset + index] = 1;
      }
      for (let frequencyIndex = 0; frequencyIndex < frequencyCount; frequencyIndex += 1) {
        const frequency = theta ** linspaceUnit(frequencyIndex, frequencyCount) * (Math.PI / 2);
        for (let axis = 0; axis < 3; axis += 1) {
          const coord = coordValues[coordOffset + axis] ?? 0;
          const angle = frequency * (coord * 2 - 1);
          const repeatedOffset = outputOffset + padding + (frequencyIndex * 3 + axis) * 2;
          const cosValue = Math.cos(angle);
          const sinValue = Math.sin(angle);
          cosValues[repeatedOffset] = cosValue;
          cosValues[repeatedOffset + 1] = cosValue;
          sinValues[repeatedOffset] = sinValue;
          sinValues[repeatedOffset + 1] = sinValue;
        }
      }
    }
  }
  return {
    cos: MxArray.fromData(cosValues, [shape.batchSize, shape.tokens, options.dim], "float32"),
    sin: MxArray.fromData(sinValues, [shape.batchSize, shape.tokens, options.dim], "float32"),
  };
}

/** Create LTX-2 video patch-boundary coordinates with shape `[batch, 3, tokens, 2]`. */
export function createLtx2VideoCoords(options: Ltx2VideoCoordinateOptions): MxArray {
  const shape = ltx2VideoCoordShape(options);
  const data = new Float32Array(shape.batchSize * 3 * shape.tokens * 2);
  let token = 0;
  for (let frame = 0; frame < shape.latentFrames; frame += shape.patchSizeT) {
    for (let row = 0; row < shape.latentHeight; row += shape.patchSize) {
      for (let column = 0; column < shape.latentWidth; column += shape.patchSize) {
        writeLtx2VideoTokenCoords(data, token, [frame, row, column], shape);
        token += 1;
      }
    }
  }
  return MxArray.fromData(data, [shape.batchSize, 3, shape.tokens, 2], "float32");
}

function ltx2VideoCoordShape(options: Ltx2VideoCoordinateOptions): {
  batchSize: number;
  latentFrames: number;
  latentHeight: number;
  latentWidth: number;
  patchSize: number;
  patchSizeT: number;
  frameRate: number;
  causalOffset: number;
  scaleFactors: readonly [number, number, number];
  tokens: number;
} {
  expectPositiveInteger(options.batchSize, "batchSize");
  expectPositiveInteger(options.latentFrames, "latentFrames");
  expectPositiveInteger(options.latentHeight, "latentHeight");
  expectPositiveInteger(options.latentWidth, "latentWidth");
  const patchSize = resolvePositiveInteger(options.patchSize, 1, "patchSize");
  const patchSizeT = resolvePositiveInteger(options.patchSizeT, 1, "patchSizeT");
  const frameRate = resolvePositiveNumber(options.frameRate, 24, "frameRate");
  const causalOffset = options.causalOffset ?? 1;
  const scaleFactors = options.vaeScaleFactors ?? [8, 32, 32];
  expectDivisible(options.latentFrames, patchSizeT, "latentFrames");
  expectDivisible(options.latentHeight, patchSize, "latentHeight");
  expectDivisible(options.latentWidth, patchSize, "latentWidth");
  const tokens =
    (options.latentFrames / patchSizeT) *
    (options.latentHeight / patchSize) *
    (options.latentWidth / patchSize);
  return {
    batchSize: options.batchSize,
    latentFrames: options.latentFrames,
    latentHeight: options.latentHeight,
    latentWidth: options.latentWidth,
    patchSize,
    patchSizeT,
    frameRate,
    causalOffset,
    scaleFactors,
    tokens,
  };
}

function writeLtx2VideoTokenCoords(
  data: Float32Array,
  token: number,
  starts: readonly [number, number, number],
  shape: ReturnType<typeof ltx2VideoCoordShape>,
): void {
  const ends: readonly [number, number, number] = [
    starts[0] + shape.patchSizeT,
    starts[1] + shape.patchSize,
    starts[2] + shape.patchSize,
  ];
  for (let batch = 0; batch < shape.batchSize; batch += 1) {
    for (let axis = 0; axis < 3; axis += 1) {
      const [start, end] = ltx2VideoAxisBounds(axis, starts[axis] ?? 0, ends[axis] ?? 0, shape);
      const offset = ((batch * 3 + axis) * shape.tokens + token) * 2;
      data[offset] = start;
      data[offset + 1] = end;
    }
  }
}

function ltx2VideoAxisBounds(
  axis: number,
  start: number,
  end: number,
  shape: ReturnType<typeof ltx2VideoCoordShape>,
): readonly [number, number] {
  const scale = shape.scaleFactors[axis] ?? 1;
  if (axis !== 0) {
    return [start * scale, end * scale];
  }
  return [
    Math.max(0, start * scale + shape.causalOffset - scale) / shape.frameRate,
    Math.max(0, end * scale + shape.causalOffset - scale) / shape.frameRate,
  ];
}

/** Create LTX-2 audio patch-boundary coordinates with shape `[batch, 1, tokens, 2]`. */
export function createLtx2AudioCoords(options: Ltx2AudioCoordinateOptions): MxArray {
  expectPositiveInteger(options.batchSize, "batchSize");
  expectPositiveInteger(options.audioLatentFrames, "audioLatentFrames");
  const patchSizeT = resolvePositiveInteger(options.patchSizeT, 1, "patchSizeT");
  const audioScaleFactor = resolvePositiveInteger(options.audioScaleFactor, 4, "audioScaleFactor");
  const causalOffset = options.causalOffset ?? 1;
  const hopLength = resolvePositiveInteger(options.hopLength, 160, "hopLength");
  const samplingRate = resolvePositiveInteger(options.samplingRate, 16000, "samplingRate");
  const shift = options.shift ?? 0;
  const tokens = Math.ceil(options.audioLatentFrames / patchSizeT);
  const data = new Float32Array(options.batchSize * tokens * 2);
  for (let token = 0; token < tokens; token += 1) {
    const startFrame = shift + token * patchSizeT;
    const endFrame = startFrame + patchSizeT;
    const startMel = Math.max(0, startFrame * audioScaleFactor + causalOffset - audioScaleFactor);
    const endMel = Math.max(0, endFrame * audioScaleFactor + causalOffset - audioScaleFactor);
    const start = (startMel * hopLength) / samplingRate;
    const end = (endMel * hopLength) / samplingRate;
    for (let batch = 0; batch < options.batchSize; batch += 1) {
      const offset = (batch * tokens + token) * 2;
      data[offset] = start;
      data[offset + 1] = end;
    }
  }
  return MxArray.fromData(data, [options.batchSize, 1, tokens, 2], "float32");
}
