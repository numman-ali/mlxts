/**
 * Linear algebra operations.
 * @module
 */

import type { Pointer } from "bun:ffi";
import { type MxArray, readResultArray } from "../array";
import { defaultStream } from "../device";
import { checkStatus } from "../error";
import { ffi } from "../ffi";

type S = Pointer | undefined;
const s = (stream?: S) => stream ?? defaultStream();

/** Matrix multiplication. */
export function matmul(a: MxArray, b: MxArray, stream?: S): MxArray {
  return readResultArray("matmul", (out) => {
    checkStatus(ffi.mlx_matmul(out, a._ctx, b._ctx, s(stream)), "matmul");
  });
}
