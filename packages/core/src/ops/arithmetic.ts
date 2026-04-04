/**
 * Element-wise arithmetic operations.
 *
 * Binary ops accept `MxArray | number` operands. Number operands are
 * automatically coerced to scalar (0-dim) MxArrays and freed after the op.
 *
 * @module
 */

import type { Pointer } from "bun:ffi";
import { array, type MxArray, readResultArray } from "../array";
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
    return readResultArray("add", (out) => {
      checkStatus(ffi.mlx_add(out, aArr._ctx, bArr._ctx, s(stream)), "add");
    });
  });
}

/** Element-wise subtraction. Accepts MxArray or number operands. */
export function subtract(a: Operand, b: Operand, stream?: S): MxArray {
  return withCoerced(a, b, (aArr, bArr) => {
    return readResultArray("subtract", (out) => {
      checkStatus(ffi.mlx_subtract(out, aArr._ctx, bArr._ctx, s(stream)), "subtract");
    });
  });
}

/** Element-wise multiplication. Accepts MxArray or number operands. */
export function multiply(a: Operand, b: Operand, stream?: S): MxArray {
  return withCoerced(a, b, (aArr, bArr) => {
    return readResultArray("multiply", (out) => {
      checkStatus(ffi.mlx_multiply(out, aArr._ctx, bArr._ctx, s(stream)), "multiply");
    });
  });
}

/** Element-wise division. Accepts MxArray or number operands. */
export function divide(a: Operand, b: Operand, stream?: S): MxArray {
  return withCoerced(a, b, (aArr, bArr) => {
    return readResultArray("divide", (out) => {
      checkStatus(ffi.mlx_divide(out, aArr._ctx, bArr._ctx, s(stream)), "divide");
    });
  });
}

/** Element-wise power. Accepts MxArray or number operands. */
export function power(a: Operand, b: Operand, stream?: S): MxArray {
  return withCoerced(a, b, (aArr, bArr) => {
    return readResultArray("power", (out) => {
      checkStatus(ffi.mlx_power(out, aArr._ctx, bArr._ctx, s(stream)), "power");
    });
  });
}

/** Element-wise maximum of two arrays. Accepts MxArray or number operands. */
export function maximum(a: Operand, b: Operand, stream?: S): MxArray {
  return withCoerced(a, b, (aArr, bArr) => {
    return readResultArray("maximum", (out) => {
      checkStatus(ffi.mlx_maximum(out, aArr._ctx, bArr._ctx, s(stream)), "maximum");
    });
  });
}

/** Element-wise minimum of two arrays. Accepts MxArray or number operands. */
export function minimum(a: Operand, b: Operand, stream?: S): MxArray {
  return withCoerced(a, b, (aArr, bArr) => {
    return readResultArray("minimum", (out) => {
      checkStatus(ffi.mlx_minimum(out, aArr._ctx, bArr._ctx, s(stream)), "minimum");
    });
  });
}

// --- Unary ops ---

/** Element-wise negation. */
export function negative(a: MxArray, stream?: S): MxArray {
  return readResultArray("negative", (out) => {
    checkStatus(ffi.mlx_negative(out, a._ctx, s(stream)), "negative");
  });
}

/** Element-wise absolute value. */
export function abs(a: MxArray, stream?: S): MxArray {
  return readResultArray("abs", (out) => {
    checkStatus(ffi.mlx_abs(out, a._ctx, s(stream)), "abs");
  });
}

/** Element-wise square root. */
export function sqrt(a: MxArray, stream?: S): MxArray {
  return readResultArray("sqrt", (out) => {
    checkStatus(ffi.mlx_sqrt(out, a._ctx, s(stream)), "sqrt");
  });
}

/** Element-wise square. */
export function square(a: MxArray, stream?: S): MxArray {
  return readResultArray("square", (out) => {
    checkStatus(ffi.mlx_square(out, a._ctx, s(stream)), "square");
  });
}

/** Element-wise exponential (e^x). */
export function exp(a: MxArray, stream?: S): MxArray {
  return readResultArray("exp", (out) => {
    checkStatus(ffi.mlx_exp(out, a._ctx, s(stream)), "exp");
  });
}

/** Element-wise natural logarithm. */
export function log(a: MxArray, stream?: S): MxArray {
  return readResultArray("log", (out) => {
    checkStatus(ffi.mlx_log(out, a._ctx, s(stream)), "log");
  });
}

/** Element-wise sigmoid. */
export function sigmoid(a: MxArray, stream?: S): MxArray {
  return readResultArray("sigmoid", (out) => {
    checkStatus(ffi.mlx_sigmoid(out, a._ctx, s(stream)), "sigmoid");
  });
}

/** Element-wise error function (for GELU). */
export function erf(a: MxArray, stream?: S): MxArray {
  return readResultArray("erf", (out) => {
    checkStatus(ffi.mlx_erf(out, a._ctx, s(stream)), "erf");
  });
}

/** Element-wise reciprocal (1/x). */
export function reciprocal(a: MxArray, stream?: S): MxArray {
  return readResultArray("reciprocal", (out) => {
    checkStatus(ffi.mlx_reciprocal(out, a._ctx, s(stream)), "reciprocal");
  });
}

/** Element-wise hyperbolic tangent. */
export function tanh(a: MxArray, stream?: S): MxArray {
  return readResultArray("tanh", (out) => {
    checkStatus(ffi.mlx_tanh(out, a._ctx, s(stream)), "tanh");
  });
}
