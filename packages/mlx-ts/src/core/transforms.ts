/**
 * Evaluation and transform operations.
 *
 * MLX operations are lazy — they build a computation graph.
 * Call mxEval() to force execution on the GPU/CPU.
 *
 * Autograd transforms (`valueAndGrad`, `grad`) wrap TypeScript loss functions
 * as mlx-c closures and use MLX's functional differentiation engine.
 * Phase 2 supports scalar loss, flat positional MxArray arguments only.
 * Tree flatten/unflatten for nested parameter structures comes in Phase 3.
 *
 * @module
 */

import type { MxArray } from "./array";
import { checkStatus } from "./error";
import {
  extractAllArrays,
  extractSingleArray,
  ffi,
  type GradFn,
  OutSlot,
  type Pointer,
  ptr,
  sizeToNumber,
  unwrapPointer,
  withClosure,
} from "./ffi";

/**
 * Force evaluation of one or more lazy arrays.
 * After eval, the arrays' data is computed and available for reading.
 *
 * Named `mxEval` because `eval` is a reserved word in strict-mode JavaScript.
 */
export function mxEval(...arrays: MxArray[]): void {
  if (arrays.length === 0) return;

  const vec = unwrapPointer(ffi.mlx_vector_array_new(), "mlx_vector_array_new");
  try {
    for (const arr of arrays) {
      checkStatus(ffi.mlx_vector_array_append_value(vec, arr._ctx), "eval_vec_append");
    }

    checkStatus(ffi.mlx_eval(vec), "eval");
  } finally {
    ffi.mlx_vector_array_free(vec);
  }
}

// ---------------------------------------------------------------------------
// Autograd transforms
// ---------------------------------------------------------------------------

/**
 * Normalize argnums to an array of integers for the C API.
 * Validates: integers, non-negative, unique, and in range of argCount.
 */
function normalizeArgnums(argnums: number | number[] | undefined, argCount: number): number[] {
  const indices = argnums === undefined ? [0] : typeof argnums === "number" ? [argnums] : argnums;

  for (const idx of indices) {
    if (!Number.isInteger(idx) || idx < 0) {
      throw new Error(`Invalid argnum ${idx}: must be a non-negative integer`);
    }
    if (idx >= argCount) {
      throw new Error(
        `argnum ${idx} is out of range: function was called with ${argCount} argument(s)`,
      );
    }
  }

  // Check for duplicates
  if (new Set(indices).size !== indices.length) {
    throw new Error(`Duplicate argnums: [${indices.join(", ")}]`);
  }

  return indices;
}

function formatShape(shape: readonly number[]): string {
  return shape.length === 0 ? "[]" : `[${shape.join(", ")}]`;
}

function assertScalarLoss(loss: MxArray, source: string): void {
  if (loss.ndim !== 0) {
    throw new Error(
      `${source}: expected fn to return a scalar MxArray, got shape ${formatShape(loss.shape)}. ` +
        "Reduce the loss to a scalar before calling grad/valueAndGrad.",
    );
  }
}

/**
 * Core implementation: apply value_and_grad transform to a function with arguments.
 * Returns [values vector, gradients vector] as raw MxArray results.
 */
function applyValueAndGrad(
  fn: GradFn,
  args: MxArray[],
  argnums: number[],
): { value: MxArray; grads: MxArray[] } {
  // Wrap the user's scalar-returning function as a vector-returning closure
  const wrappedFn = (...inputs: MxArray[]): MxArray[] => {
    const result = fn(...inputs);
    assertScalarLoss(result, "autograd");
    return [result];
  };

  return withClosure(wrappedFn, (closurePtr: Pointer) => {
    // Create the value_and_grad transform
    const argnumsBuf = new Int32Array(argnums);
    const vagSlot = new OutSlot();
    checkStatus(
      ffi.mlx_value_and_grad(vagSlot.prepare(), closurePtr, ptr(argnumsBuf), argnums.length),
      "value_and_grad",
    );
    const vagPtr = vagSlot.read("value_and_grad transform");

    try {
      // Pack arguments into input vector
      const inputVec = unwrapPointer(ffi.mlx_vector_array_new(), "vag_input_vec_new");
      try {
        for (const arg of args) {
          checkStatus(ffi.mlx_vector_array_append_value(inputVec, arg._ctx), "vag_input_append");
        }

        // Apply the transform — two output vectors (values, gradients)
        const valuesSlot = new OutSlot();
        const gradsSlot = new OutSlot();
        checkStatus(
          ffi.mlx_closure_value_and_grad_apply(
            valuesSlot.prepare(),
            gradsSlot.prepare(),
            vagPtr,
            inputVec,
          ),
          "closure_value_and_grad_apply",
        );

        const valuesVec = valuesSlot.read("values vector");
        const gradsVec = gradsSlot.read("grads vector");

        try {
          // Validate scalar output
          const valueCount = sizeToNumber(ffi.mlx_vector_array_size(valuesVec), "values_vec_size");
          if (valueCount !== 1) {
            throw new Error(
              `valueAndGrad: expected function to return 1 value, got ${valueCount}. ` +
                "Phase 2 supports scalar loss functions only.",
            );
          }

          const value = extractSingleArray(valuesVec, "value");
          let grads: MxArray[] = [];
          try {
            assertScalarLoss(value, "autograd");
            grads = extractAllArrays(gradsVec, "grad");

            return { value, grads };
          } catch (error) {
            value.free();
            for (const grad of grads) {
              grad.free();
            }
            throw error;
          }
        } finally {
          ffi.mlx_vector_array_free(valuesVec);
          ffi.mlx_vector_array_free(gradsVec);
        }
      } finally {
        ffi.mlx_vector_array_free(inputVec);
      }
    } finally {
      ffi.mlx_closure_value_and_grad_free(vagPtr);
    }
  });
}

// --- valueAndGrad overloads ---

/**
 * Create a function that computes both the value and gradient of `fn`.
 *
 * Returns a new function. When called, it produces `[value, grad]` where
 * `value` is the scalar loss and `grad` is the gradient with respect to
 * the argument at position `argnums`.
 *
 * @param fn - A function from MxArrays to a scalar MxArray (the loss).
 * @param argnums - Which argument index to differentiate with respect to. Defaults to 0.
 * @returns A function returning `[value, gradient]`.
 */
export function valueAndGrad(fn: GradFn): (...args: MxArray[]) => [MxArray, MxArray];
export function valueAndGrad(
  fn: GradFn,
  argnums: number,
): (...args: MxArray[]) => [MxArray, MxArray];
/**
 * Create a function that computes both the value and gradients of `fn`
 * with respect to multiple arguments.
 *
 * @param fn - A function from MxArrays to a scalar MxArray (the loss).
 * @param argnums - Array of argument indices to differentiate with respect to.
 * @returns A function returning `[value, gradients[]]`.
 */
export function valueAndGrad(
  fn: GradFn,
  argnums: number[],
): (...args: MxArray[]) => [MxArray, MxArray[]];
export function valueAndGrad(
  fn: GradFn,
  argnums?: number | number[],
): (...args: MxArray[]) => [MxArray, MxArray] | [MxArray, MxArray[]] {
  const isMulti = Array.isArray(argnums);

  return (...args: MxArray[]) => {
    const indices = normalizeArgnums(argnums, args.length);
    const { value, grads } = applyValueAndGrad(fn, args, indices);

    if (isMulti) {
      return [value, grads];
    }
    const singleGrad = grads[0];
    if (singleGrad === undefined) {
      throw new Error("valueAndGrad: gradient vector was empty");
    }
    // Free remaining grads that the caller won't use (shouldn't happen for single argnum)
    for (let i = 1; i < grads.length; i++) {
      grads[i]?.free();
    }
    return [value, singleGrad];
  };
}

// --- grad overloads ---

/**
 * Create a function that computes the gradient of `fn`.
 *
 * Returns a new function. When called, it produces the gradient of `fn`
 * with respect to the argument at position `argnums`.
 *
 * @param fn - A function from MxArrays to a scalar MxArray (the loss).
 * @param argnums - Which argument index to differentiate with respect to. Defaults to 0.
 * @returns A function returning the gradient MxArray.
 */
export function grad(fn: GradFn): (...args: MxArray[]) => MxArray;
export function grad(fn: GradFn, argnums: number): (...args: MxArray[]) => MxArray;
/**
 * Create a function that computes the gradients of `fn`
 * with respect to multiple arguments.
 *
 * @param fn - A function from MxArrays to a scalar MxArray (the loss).
 * @param argnums - Array of argument indices to differentiate with respect to.
 * @returns A function returning an array of gradient MxArrays.
 */
export function grad(fn: GradFn, argnums: number[]): (...args: MxArray[]) => MxArray[];
export function grad(
  fn: GradFn,
  argnums?: number | number[],
): (...args: MxArray[]) => MxArray | MxArray[] {
  const isMulti = Array.isArray(argnums);

  return (...args: MxArray[]) => {
    const indices = normalizeArgnums(argnums, args.length);
    const { value, grads } = applyValueAndGrad(fn, args, indices);

    // Caller asked for gradients only — free the value handle
    value.free();

    if (isMulti) {
      return grads;
    }
    const singleGrad = grads[0];
    if (singleGrad === undefined) {
      throw new Error("grad: gradient vector was empty");
    }
    for (let i = 1; i < grads.length; i++) {
      grads[i]?.free();
    }
    return singleGrad;
  };
}
