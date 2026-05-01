import type { MxArray } from "@mlxts/core";
import { formatShape, slice, squeeze } from "@mlxts/core";

export type Flux2SequenceShape3d = {
  batch: number;
  length: number;
  channels: number;
};

export type Flux2AttentionShape4d = {
  batch: number;
  heads: number;
  length: number;
  headDim: number;
};

export function assertFlux2Sequence3d(x: MxArray, owner: string): Flux2SequenceShape3d {
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

export function assertFlux2Attention4d(x: MxArray, owner: string): Flux2AttentionShape4d {
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

export function assertFlux2Ids2d(ids: MxArray, owner: string): { length: number } {
  const [length, axes] = ids.shape;
  if (ids.shape.length !== 2 || length === undefined || axes !== 4) {
    throw new Error(`${owner}: expected ids shape [length, 4], got ${formatShape(ids.shape)}.`);
  }
  return { length };
}

export function checkedFlux2Module<T>(modules: readonly T[], index: number, owner: string): T {
  const module = modules[index];
  if (module === undefined) {
    throw new Error(`${owner}: missing module at index ${index}.`);
  }
  return module;
}

export function sliceFlux2Axis(
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

export function sliceFlux2LastAxis(x: MxArray, startIndex: number, stopIndex: number): MxArray {
  return sliceFlux2Axis(x, x.shape.length - 1, startIndex, stopIndex);
}

export function selectFlux2LastAxis(x: MxArray, index: number): MxArray {
  using selected = sliceFlux2LastAxis(x, index, index + 1);
  return squeeze(selected, selected.shape.length - 1);
}

export function freeFlux2Arrays(arrays: readonly MxArray[]): void {
  for (const array of arrays) {
    array.free();
  }
}
