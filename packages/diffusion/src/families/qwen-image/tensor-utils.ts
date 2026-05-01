import type { MxArray } from "@mlxts/core";
import { formatShape, slice, squeeze } from "@mlxts/core";

export type SequenceShape3d = {
  batch: number;
  length: number;
  channels: number;
};

export type AttentionShape4d = {
  batch: number;
  heads: number;
  length: number;
  headDim: number;
};

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

export function checkedModule<T>(modules: readonly T[], index: number, owner: string): T {
  const module = modules[index];
  if (module === undefined) {
    throw new Error(`${owner}: missing module at index ${index}.`);
  }
  return module;
}

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

export function sliceLastAxis(x: MxArray, startIndex: number, stopIndex: number): MxArray {
  return sliceAxis(x, x.shape.length - 1, startIndex, stopIndex);
}

export function selectLastAxis(x: MxArray, index: number): MxArray {
  using selected = sliceLastAxis(x, index, index + 1);
  return squeeze(selected, selected.shape.length - 1);
}

export function freeArrays(arrays: readonly MxArray[]): void {
  for (const array of arrays) {
    array.free();
  }
}
