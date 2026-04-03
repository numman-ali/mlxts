/**
 * Linear algebra operations.
 * @module
 */

import type { Pointer } from "bun:ffi";
import { MxArray, prepareOut, readOut } from "../array";
import { defaultStream } from "../device";
import { checkStatus } from "../error";
import { ffi } from "../ffi";

type S = Pointer | undefined;
const s = (stream?: S) => stream ?? defaultStream();

/** Matrix multiplication. */
export function matmul(a: MxArray, b: MxArray, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_matmul(out, a._ctx, b._ctx, s(stream)), "matmul");
  return MxArray._fromCtx(readOut());
}
