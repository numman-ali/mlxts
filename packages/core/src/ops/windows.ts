/**
 * Window functions for spectral preprocessing.
 * @module
 */

import type { Pointer } from "bun:ffi";
import { type MxArray, readResultArrayWithMetadata } from "../array";
import { defaultStream } from "../device";
import { checkStatus } from "../error";
import { ffi } from "../ffi";

type S = Pointer | undefined;
const s = (stream?: S) => stream ?? defaultStream();

/** Hanning window with MLX's native coefficients. */
export function hanning(length: number, stream?: S): MxArray {
  if (!Number.isInteger(length) || length < 0) {
    throw new Error(`hanning: length must be a non-negative integer, got ${length}.`);
  }
  return readResultArrayWithMetadata("hanning", { shape: [length], dtype: "float32" }, (out) => {
    checkStatus(ffi.mlx_hanning(out, length, s(stream)), "hanning");
  });
}
