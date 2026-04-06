import type { DType } from "./dtype";

export type ArrayMetadata = {
  shape?: readonly number[];
  dtype?: DType;
  ndim?: number;
  size?: number;
};

export function freezeShapeMetadata(shape: readonly number[]): readonly number[] {
  return Object.freeze([...shape]);
}

export function inferElementCount(shape: readonly number[]): number {
  return shape.reduce((product, extent) => product * extent, 1);
}

export function inferArangeLength(start: number, stop: number, step: number): number {
  if (step === 0) {
    return 0;
  }
  const delta = stop - start;
  if ((delta > 0 && step < 0) || (delta < 0 && step > 0)) {
    return 0;
  }
  return Math.max(0, Math.ceil(delta / step));
}
