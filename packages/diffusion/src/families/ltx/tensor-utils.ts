import type { MxArray } from "@mlxts/core";
import { add, formatShape, multiply, reshape, slice, squeeze } from "@mlxts/core";

export type SequenceShape3d = {
  batch: number;
  length: number;
  channels: number;
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

/** Select one scalar index from the last axis. */
export function selectLastAxis(x: MxArray, index: number): MxArray {
  using selected = sliceAxis(x, x.shape.length - 1, index, index + 1);
  return squeeze(selected, selected.shape.length - 1);
}

/** Free a collection of array handles. */
export function freeArrays(arrays: readonly MxArray[]): void {
  for (const array of arrays) {
    array.free();
  }
}

/** Apply adaptive scale and shift tensors with shape `[batch, 1, channels]`. */
export function applyScaleShift(x: MxArray, shift: MxArray, scale: MxArray): MxArray {
  using scalePlusOne = add(scale, 1);
  using scaled = multiply(x, scalePlusOne);
  return add(scaled, shift);
}

/** Convert a `[batch, channels]` modulation vector into `[batch, 1, channels]`. */
export function reshapeModulation(part: MxArray, batch: number, hiddenSize: number): MxArray {
  return reshape(part, [batch, 1, hiddenSize]);
}
