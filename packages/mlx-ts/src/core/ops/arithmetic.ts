/**
 * Element-wise arithmetic operations.
 *
 * Binary ops accept `MxArray | number` operands. Number operands are
 * automatically coerced to scalar (0-dim) MxArrays and freed after the op.
 *
 * @module
 */

import type { Pointer } from "bun:ffi";
import { array, MxArray, prepareOut, readOut } from "../array";
import { defaultStream } from "../device";
import { checkStatus } from "../error";
import { ffi } from "../ffi";

type S = Pointer | undefined;
const s = (stream?: S) => stream ?? defaultStream();

/** An operand that is either an MxArray or a JS number (coerced to scalar). */
export type Operand = MxArray | number;

/**
 * Run a binary FFI op with automatic scalar coercion.
 * Number operands are wrapped as 0-dim MxArrays and freed after the call.
 */
function withCoerced(
  a: Operand,
  b: Operand,
  fn: (aArr: MxArray, bArr: MxArray) => MxArray,
): MxArray {
  const aIsNum = typeof a === "number";
  const bIsNum = typeof b === "number";
  const aArr = aIsNum ? array(a) : a;
  const bArr = bIsNum ? array(b) : b;
  try {
    return fn(aArr, bArr);
  } finally {
    if (aIsNum) aArr.free();
    if (bIsNum) bArr.free();
  }
}

// --- Binary ops ---

/** Element-wise addition. Accepts MxArray or number operands. */
export function add(a: Operand, b: Operand, stream?: S): MxArray {
  return withCoerced(a, b, (aArr, bArr) => {
    const out = prepareOut();
    checkStatus(ffi.mlx_add(out, aArr._ctx, bArr._ctx, s(stream)), "add");
    return MxArray._fromCtx(readOut());
  });
}

/** Element-wise subtraction. Accepts MxArray or number operands. */
export function subtract(a: Operand, b: Operand, stream?: S): MxArray {
  return withCoerced(a, b, (aArr, bArr) => {
    const out = prepareOut();
    checkStatus(ffi.mlx_subtract(out, aArr._ctx, bArr._ctx, s(stream)), "subtract");
    return MxArray._fromCtx(readOut());
  });
}

/** Element-wise multiplication. Accepts MxArray or number operands. */
export function multiply(a: Operand, b: Operand, stream?: S): MxArray {
  return withCoerced(a, b, (aArr, bArr) => {
    const out = prepareOut();
    checkStatus(ffi.mlx_multiply(out, aArr._ctx, bArr._ctx, s(stream)), "multiply");
    return MxArray._fromCtx(readOut());
  });
}

/** Element-wise division. Accepts MxArray or number operands. */
export function divide(a: Operand, b: Operand, stream?: S): MxArray {
  return withCoerced(a, b, (aArr, bArr) => {
    const out = prepareOut();
    checkStatus(ffi.mlx_divide(out, aArr._ctx, bArr._ctx, s(stream)), "divide");
    return MxArray._fromCtx(readOut());
  });
}

/** Element-wise power. Accepts MxArray or number operands. */
export function power(a: Operand, b: Operand, stream?: S): MxArray {
  return withCoerced(a, b, (aArr, bArr) => {
    const out = prepareOut();
    checkStatus(ffi.mlx_power(out, aArr._ctx, bArr._ctx, s(stream)), "power");
    return MxArray._fromCtx(readOut());
  });
}

/** Element-wise maximum of two arrays. Accepts MxArray or number operands. */
export function maximum(a: Operand, b: Operand, stream?: S): MxArray {
  return withCoerced(a, b, (aArr, bArr) => {
    const out = prepareOut();
    checkStatus(ffi.mlx_maximum(out, aArr._ctx, bArr._ctx, s(stream)), "maximum");
    return MxArray._fromCtx(readOut());
  });
}

/** Element-wise minimum of two arrays. Accepts MxArray or number operands. */
export function minimum(a: Operand, b: Operand, stream?: S): MxArray {
  return withCoerced(a, b, (aArr, bArr) => {
    const out = prepareOut();
    checkStatus(ffi.mlx_minimum(out, aArr._ctx, bArr._ctx, s(stream)), "minimum");
    return MxArray._fromCtx(readOut());
  });
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
