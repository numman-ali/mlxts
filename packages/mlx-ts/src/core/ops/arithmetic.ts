/**
 * Element-wise arithmetic operations.
 * @module
 */

import type { Pointer } from "bun:ffi";
import { MxArray, prepareOut, readOut } from "../array";
import { defaultStream } from "../device";
import { checkStatus } from "../error";
import { ffi } from "../ffi";

type S = Pointer | undefined;
const s = (stream?: S) => stream ?? defaultStream();

// --- Binary ops ---

/** Element-wise addition. */
export function add(a: MxArray, b: MxArray, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_add(out, a._ctx, b._ctx, s(stream)), "add");
  return MxArray._fromCtx(readOut());
}

/** Element-wise subtraction. */
export function subtract(a: MxArray, b: MxArray, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_subtract(out, a._ctx, b._ctx, s(stream)), "subtract");
  return MxArray._fromCtx(readOut());
}

/** Element-wise multiplication. */
export function multiply(a: MxArray, b: MxArray, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_multiply(out, a._ctx, b._ctx, s(stream)), "multiply");
  return MxArray._fromCtx(readOut());
}

/** Element-wise division. */
export function divide(a: MxArray, b: MxArray, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_divide(out, a._ctx, b._ctx, s(stream)), "divide");
  return MxArray._fromCtx(readOut());
}

/** Element-wise power. */
export function power(a: MxArray, b: MxArray, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_power(out, a._ctx, b._ctx, s(stream)), "power");
  return MxArray._fromCtx(readOut());
}

/** Element-wise maximum of two arrays. */
export function maximum(a: MxArray, b: MxArray, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_maximum(out, a._ctx, b._ctx, s(stream)), "maximum");
  return MxArray._fromCtx(readOut());
}

/** Element-wise minimum of two arrays. */
export function minimum(a: MxArray, b: MxArray, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_minimum(out, a._ctx, b._ctx, s(stream)), "minimum");
  return MxArray._fromCtx(readOut());
}

// --- Unary ops ---

/** Element-wise negation. */
export function negative(a: MxArray, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_negative(out, a._ctx, s(stream)), "negative");
  return MxArray._fromCtx(readOut());
}

/** Element-wise absolute value. */
export function abs(a: MxArray, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_abs(out, a._ctx, s(stream)), "abs");
  return MxArray._fromCtx(readOut());
}

/** Element-wise square root. */
export function sqrt(a: MxArray, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_sqrt(out, a._ctx, s(stream)), "sqrt");
  return MxArray._fromCtx(readOut());
}

/** Element-wise square. */
export function square(a: MxArray, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_square(out, a._ctx, s(stream)), "square");
  return MxArray._fromCtx(readOut());
}

/** Element-wise exponential (e^x). */
export function exp(a: MxArray, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_exp(out, a._ctx, s(stream)), "exp");
  return MxArray._fromCtx(readOut());
}

/** Element-wise natural logarithm. */
export function log(a: MxArray, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_log(out, a._ctx, s(stream)), "log");
  return MxArray._fromCtx(readOut());
}

/** Element-wise sigmoid. */
export function sigmoid(a: MxArray, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_sigmoid(out, a._ctx, s(stream)), "sigmoid");
  return MxArray._fromCtx(readOut());
}

/** Element-wise error function (for GELU). */
export function erf(a: MxArray, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_erf(out, a._ctx, s(stream)), "erf");
  return MxArray._fromCtx(readOut());
}

/** Element-wise reciprocal (1/x). */
export function reciprocal(a: MxArray, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_reciprocal(out, a._ctx, s(stream)), "reciprocal");
  return MxArray._fromCtx(readOut());
}

/** Element-wise hyperbolic tangent. */
export function tanh(a: MxArray, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_tanh(out, a._ctx, s(stream)), "tanh");
  return MxArray._fromCtx(readOut());
}
