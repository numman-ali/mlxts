/**
 * Comparison operations.
 * @module
 */

import type { Pointer } from "bun:ffi";
import { MxArray, prepareOut, readOut } from "../array";
import { defaultStream } from "../device";
import { checkStatus } from "../error";
import { ffi } from "../ffi";

type S = Pointer | undefined;
const s = (stream?: S) => stream ?? defaultStream();

/** Element-wise equality. */
export function equal(a: MxArray, b: MxArray, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_equal(out, a._ctx, b._ctx, s(stream)), "equal");
  return MxArray._fromCtx(readOut());
}

/** Element-wise inequality. */
export function notEqual(a: MxArray, b: MxArray, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_not_equal(out, a._ctx, b._ctx, s(stream)), "not_equal");
  return MxArray._fromCtx(readOut());
}

/** Element-wise greater than. */
export function greater(a: MxArray, b: MxArray, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_greater(out, a._ctx, b._ctx, s(stream)), "greater");
  return MxArray._fromCtx(readOut());
}

/** Element-wise greater than or equal. */
export function greaterEqual(a: MxArray, b: MxArray, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_greater_equal(out, a._ctx, b._ctx, s(stream)), "greater_equal");
  return MxArray._fromCtx(readOut());
}

/** Element-wise less than. */
export function less(a: MxArray, b: MxArray, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_less(out, a._ctx, b._ctx, s(stream)), "less");
  return MxArray._fromCtx(readOut());
}

/** Element-wise less than or equal. */
export function lessEqual(a: MxArray, b: MxArray, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_less_equal(out, a._ctx, b._ctx, s(stream)), "less_equal");
  return MxArray._fromCtx(readOut());
}

/** Conditional select: where(condition, x, y) → condition ? x : y. */
export function where(condition: MxArray, x: MxArray, y: MxArray, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_where(out, condition._ctx, x._ctx, y._ctx, s(stream)), "where");
  return MxArray._fromCtx(readOut());
}
