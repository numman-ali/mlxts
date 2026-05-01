/**
 * Explicit strided array views.
 * @module
 */

import type { Pointer } from "bun:ffi";
import { type MxArray, readResultArrayWithMetadata } from "../array";
import { defaultStream } from "../device";
import { checkStatus } from "../error";
import { ffi, ptr } from "../ffi";

type S = Pointer | undefined;
const s = (stream?: S) => stream ?? defaultStream();

function validateAsStridedShape(shape: readonly number[]): number[] {
  if (shape.length === 0) {
    throw new Error("asStrided: shape must contain at least one dimension.");
  }
  return shape.map((dimension, axis) => {
    if (!Number.isInteger(dimension) || dimension < 0) {
      throw new Error(`asStrided: shape[${axis}] must be a non-negative integer.`);
    }
    return dimension;
  });
}

function validateAsStridedStrides(strides: readonly number[], rank: number): bigint[] {
  if (strides.length !== rank) {
    throw new Error(`asStrided: expected ${rank} strides, got ${strides.length}.`);
  }
  return strides.map((stride, axis) => {
    if (!Number.isInteger(stride)) {
      throw new Error(`asStrided: strides[${axis}] must be an integer.`);
    }
    return BigInt(stride);
  });
}

/** Return a strided view with explicit shape and element strides. */
export function asStrided(
  a: MxArray,
  shape: readonly number[],
  strides: readonly number[],
  offset = 0,
  stream?: S,
): MxArray {
  if (!Number.isInteger(offset) || offset < 0) {
    throw new Error(`asStrided: offset must be a non-negative integer, got ${offset}.`);
  }
  const normalizedShape = validateAsStridedShape(shape);
  const normalizedStrides = validateAsStridedStrides(strides, normalizedShape.length);
  const shapeBuf = Int32Array.from(normalizedShape);
  const strideBuf = BigInt64Array.from(normalizedStrides);
  return readResultArrayWithMetadata(
    "asStrided",
    { shape: normalizedShape, dtype: a.dtype },
    (out) => {
      checkStatus(
        ffi.mlx_as_strided(
          out,
          a._ctx,
          ptr(shapeBuf),
          shapeBuf.length,
          ptr(strideBuf),
          strideBuf.length,
          offset,
          s(stream),
        ),
        "asStrided",
      );
    },
  );
}
