/**
 * Fourier transform operations.
 * @module
 */

import type { Pointer } from "bun:ffi";
import { type MxArray, readResultArrayWithMetadata } from "../array";
import { defaultStream } from "../device";
import { checkStatus } from "../error";
import { ffi } from "../ffi";

type S = Pointer | undefined;
const s = (stream?: S) => stream ?? defaultStream();

function normalizeAxis(axis: number, rank: number): number {
  const normalized = axis < 0 ? rank + axis : axis;
  if (!Number.isInteger(axis) || normalized < 0 || normalized >= rank) {
    throw new Error(`rfft: axis ${axis} is out of bounds for rank ${rank}.`);
  }
  return normalized;
}

function normalizeTransformLength(length: number | undefined, axisLength: number): number {
  const resolvedLength = length ?? axisLength;
  if (!Number.isInteger(resolvedLength) || resolvedLength <= 0) {
    throw new Error(`rfft: n must be a positive integer, got ${resolvedLength}.`);
  }
  return resolvedLength;
}

/** One-dimensional real FFT along a single axis. */
export function rfft(a: MxArray, length?: number, axis = -1, stream?: S): MxArray {
  const rank = a.shape.length;
  const normalizedAxis = normalizeAxis(axis, rank);
  const axisLength = a.shape[normalizedAxis];
  if (axisLength === undefined) {
    throw new Error(`rfft: axis ${axis} has unknown length.`);
  }
  const transformLength = normalizeTransformLength(length, axisLength);
  const outputShape = [...a.shape];
  outputShape[normalizedAxis] = Math.floor(transformLength / 2) + 1;

  return readResultArrayWithMetadata("rfft", { shape: outputShape, dtype: "complex64" }, (out) => {
    checkStatus(ffi.mlx_fft_rfft(out, a._ctx, transformLength, normalizedAxis, s(stream)), "rfft");
  });
}
