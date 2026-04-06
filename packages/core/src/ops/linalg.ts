/**
 * Linear algebra operations.
 * @module
 */

import type { Pointer } from "bun:ffi";
import { type MxArray, readResultArrayWithMetadata } from "../array";
import { defaultStream } from "../device";
import { checkStatus } from "../error";
import { ffi } from "../ffi";

type S = Pointer | undefined;
const s = (stream?: S) => stream ?? defaultStream();

function inferMatmulShape(a: MxArray, b: MxArray): number[] | undefined {
  const aShape = a.shape;
  const bShape = b.shape;
  if (aShape.length >= 2 && bShape.length === 2) {
    return [...aShape.slice(0, -1), bShape[1] ?? 0];
  }
  return undefined;
}

/** Matrix multiplication. */
export function matmul(a: MxArray, b: MxArray, stream?: S): MxArray {
  const shape = inferMatmulShape(a, b);
  const metadata =
    shape === undefined
      ? a.dtype === b.dtype
        ? { dtype: a.dtype }
        : {}
      : a.dtype === b.dtype
        ? { shape, dtype: a.dtype }
        : { shape };
  return readResultArrayWithMetadata("matmul", metadata, (out) => {
    checkStatus(ffi.mlx_matmul(out, a._ctx, b._ctx, s(stream)), "matmul");
  });
}
