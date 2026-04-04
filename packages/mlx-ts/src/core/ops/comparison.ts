/**
 * Comparison operations.
 * @module
 */

import type { Pointer } from "bun:ffi";
import { array, type MxArray, readResultArray } from "../array";
import { defaultStream } from "../device";
import { checkStatus } from "../error";
import { ffi } from "../ffi";
import type { Operand } from "./arithmetic";

type S = Pointer | undefined;
const s = (stream?: S) => stream ?? defaultStream();

function scalarOperand(value: number, reference?: MxArray): MxArray {
  return reference === undefined ? array(value) : array(value, reference.dtype);
}

function withCoercedOperands(
  a: Operand,
  b: Operand,
  fn: (left: MxArray, right: MxArray) => MxArray,
): MxArray {
  const leftIsNumber = typeof a === "number";
  const rightIsNumber = typeof b === "number";
  const left = leftIsNumber ? scalarOperand(a, rightIsNumber ? undefined : b) : a;
  const right = rightIsNumber ? scalarOperand(b, leftIsNumber ? undefined : a) : b;

  try {
    return fn(left, right);
  } finally {
    if (leftIsNumber) {
      left.free();
    }
    if (rightIsNumber) {
      right.free();
    }
  }
}

/** Element-wise equality. */
export function equal(a: Operand, b: Operand, stream?: S): MxArray {
  return withCoercedOperands(a, b, (left, right) =>
    readResultArray("equal", (out) => {
      checkStatus(ffi.mlx_equal(out, left._ctx, right._ctx, s(stream)), "equal");
    }),
  );
}

/** Element-wise inequality. */
export function notEqual(a: Operand, b: Operand, stream?: S): MxArray {
  return withCoercedOperands(a, b, (left, right) =>
    readResultArray("not_equal", (out) => {
      checkStatus(ffi.mlx_not_equal(out, left._ctx, right._ctx, s(stream)), "not_equal");
    }),
  );
}

/** Element-wise greater than. */
export function greater(a: Operand, b: Operand, stream?: S): MxArray {
  return withCoercedOperands(a, b, (left, right) =>
    readResultArray("greater", (out) => {
      checkStatus(ffi.mlx_greater(out, left._ctx, right._ctx, s(stream)), "greater");
    }),
  );
}

/** Element-wise greater than or equal. */
export function greaterEqual(a: Operand, b: Operand, stream?: S): MxArray {
  return withCoercedOperands(a, b, (left, right) =>
    readResultArray("greater_equal", (out) => {
      checkStatus(ffi.mlx_greater_equal(out, left._ctx, right._ctx, s(stream)), "greater_equal");
    }),
  );
}

/** Element-wise less than. */
export function less(a: Operand, b: Operand, stream?: S): MxArray {
  return withCoercedOperands(a, b, (left, right) =>
    readResultArray("less", (out) => {
      checkStatus(ffi.mlx_less(out, left._ctx, right._ctx, s(stream)), "less");
    }),
  );
}

/** Element-wise less than or equal. */
export function lessEqual(a: Operand, b: Operand, stream?: S): MxArray {
  return withCoercedOperands(a, b, (left, right) =>
    readResultArray("less_equal", (out) => {
      checkStatus(ffi.mlx_less_equal(out, left._ctx, right._ctx, s(stream)), "less_equal");
    }),
  );
}

/** Conditional select: where(condition, x, y) → condition ? x : y. Accepts scalar operands. */
export function where(condition: MxArray, x: Operand, y: Operand, stream?: S): MxArray {
  const xIsNum = typeof x === "number";
  const yIsNum = typeof y === "number";
  const xArr = xIsNum ? scalarOperand(x, yIsNum ? undefined : y) : x;
  const yArr = yIsNum ? scalarOperand(y, xIsNum ? undefined : x) : y;
  try {
    return readResultArray("where", (out) => {
      checkStatus(ffi.mlx_where(out, condition._ctx, xArr._ctx, yArr._ctx, s(stream)), "where");
    });
  } finally {
    if (xIsNum) xArr.free();
    if (yIsNum) yArr.free();
  }
}
