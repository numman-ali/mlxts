import type { MxArray } from "@mlxts/core";
import {
  add,
  divide,
  formatShape,
  mean,
  multiply,
  reshape,
  slice,
  sqrt,
  square,
  subtract,
} from "@mlxts/core";

export type SequenceShape3d = {
  batch: number;
  length: number;
  channels: number;
};

export type ImageShape4d = {
  batch: number;
  height: number;
  width: number;
  channels: number;
};

export type AttentionShape4d = {
  batch: number;
  heads: number;
  length: number;
  headDim: number;
};

/** Assert a `[batch, length, channels]` sequence tensor. */
export function assertSequence3d(x: MxArray, owner: string): SequenceShape3d {
  const [batch, length, channels] = x.shape;
  if (
    x.shape.length !== 3 ||
    batch === undefined ||
    length === undefined ||
    channels === undefined
  ) {
    throw new Error(`${owner}: expected rank-3 sequence input, got ${formatShape(x.shape)}.`);
  }
  return { batch, length, channels };
}

/** Assert a `[batch, height, width, channels]` image tensor. */
export function assertImage4d(x: MxArray, owner: string): ImageShape4d {
  const [batch, height, width, channels] = x.shape;
  if (
    x.shape.length !== 4 ||
    batch === undefined ||
    height === undefined ||
    width === undefined ||
    channels === undefined
  ) {
    throw new Error(`${owner}: expected rank-4 NHWC input, got ${formatShape(x.shape)}.`);
  }
  return { batch, height, width, channels };
}

/** Assert a `[batch, heads, length, headDim]` attention tensor. */
export function assertAttention4d(x: MxArray, owner: string): AttentionShape4d {
  const [batch, heads, length, headDim] = x.shape;
  if (
    x.shape.length !== 4 ||
    batch === undefined ||
    heads === undefined ||
    length === undefined ||
    headDim === undefined
  ) {
    throw new Error(`${owner}: expected rank-4 attention input, got ${formatShape(x.shape)}.`);
  }
  return { batch, heads, length, headDim };
}

/** Return a checked module-array entry. */
export function checkedModule<T>(modules: readonly T[], index: number, owner: string): T {
  const module = modules[index];
  if (module === undefined) {
    throw new Error(`${owner}: missing module at index ${index}.`);
  }
  return module;
}

/** Slice one tensor axis by start and stop indices. */
export function sliceAxis(
  x: MxArray,
  axis: number,
  startIndex: number,
  stopIndex: number,
): MxArray {
  const start = x.shape.map(() => 0);
  const stop = [...x.shape];
  start[axis] = startIndex;
  stop[axis] = stopIndex;
  return slice(x, start, stop);
}

/** Free a collection of array handles. */
export function freeArrays(arrays: readonly MxArray[]): void {
  for (const array of arrays) {
    array.free();
  }
}

/** Affine-free layer normalization over the last axis. */
export function affineFreeLayerNorm(x: MxArray, dims: number, owner: string): MxArray {
  const lastDimension = x.shape[x.shape.length - 1];
  if (lastDimension !== dims) {
    throw new Error(
      `${owner}: expected last dimension ${dims}, got ${lastDimension ?? "undefined"}.`,
    );
  }
  using center = mean(x, -1, true);
  using centered = subtract(x, center);
  using squared = square(centered);
  using variance = mean(squared, -1, true);
  using stabilized = add(variance, 1e-6);
  using denominator = sqrt(stabilized);
  return divide(centered, denominator);
}

/** Apply SD3 adaptive scale and shift tensors with shape `[batch, 1, channels]`. */
export function applyScaleShift(x: MxArray, shift: MxArray, scale: MxArray): MxArray {
  using scalePlusOne = add(scale, 1);
  using scaled = multiply(x, scalePlusOne);
  return add(scaled, shift);
}

/** Convert a `[batch, channels]` modulation vector into `[batch, 1, channels]`. */
export function reshapeModulation(part: MxArray, batch: number, hiddenSize: number): MxArray {
  return reshape(part, [batch, 1, hiddenSize]);
}
