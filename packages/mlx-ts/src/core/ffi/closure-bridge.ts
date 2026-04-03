/**
 * Closure bridge — wraps TypeScript functions as mlx-c closures.
 *
 * This module sits at the FFI boundary. It translates between TypeScript
 * functions and the `mlx_closure` type that mlx-c uses for autograd.
 * JSCallback handles are kept alive for the duration of each closure call.
 *
 * The callback signature matches mlx-c's `mlx_closure_new_func`:
 *   int (*fun)(mlx_vector_array* out, const mlx_vector_array in)
 *
 * MLX calls this synchronously on the main thread during graph construction,
 * so `threadsafe: false` (the default) is correct.
 *
 * @module
 */

import { FFIType, JSCallback } from "bun:ffi";
import { MxArray } from "../array";
import { checkStatus } from "../error";
import { ffi } from "./lib";
import type { Pointer } from "./pointer";
import { OutSlot, sizeToNumber, unwrapPointer } from "./pointer";

/** A function from MxArrays to a single MxArray (e.g., a loss function). */
export type GradFn = (...args: MxArray[]) => MxArray;

/**
 * Extract a single MxArray from position 0 of a native vector.
 * Used for reading the scalar loss value from value_and_grad output.
 */
export function extractSingleArray(vec: Pointer, label: string): MxArray {
  const slot = new OutSlot();
  checkStatus(ffi.mlx_vector_array_get(slot.prepare(), vec, 0), `${label}_vec_get`);
  return MxArray._fromCtx(slot.read(`${label} array`));
}

/**
 * Extract all MxArrays from a native vector.
 * Used for reading gradient arrays from value_and_grad output.
 */
export function extractAllArrays(vec: Pointer, label: string): MxArray[] {
  const count = sizeToNumber(ffi.mlx_vector_array_size(vec), `${label}_vec_size`);
  const results: MxArray[] = [];
  const slot = new OutSlot();

  for (let i = 0; i < count; i++) {
    checkStatus(ffi.mlx_vector_array_get(slot.prepare(), vec, i), `${label}_vec_get_${i}`);
    results.push(MxArray._fromCtx(slot.read(`${label} array ${i}`)));
  }
  return results;
}

/**
 * Execute `body` with a live mlx_closure wrapping the given TypeScript function.
 *
 * Creates a JSCallback + mlx_closure, calls body(closurePtr), then frees both.
 * If the TypeScript function throws during a callback invocation, the original
 * exception is captured and rethrown after cleanup.
 *
 * @param fn - The TypeScript function to wrap. Receives MxArrays, returns MxArray[].
 * @param body - Receives the live mlx_closure pointer. Do transform work here.
 * @returns Whatever body returns.
 */
export function withClosure<T>(
  fn: (...args: MxArray[]) => MxArray[],
  body: (closurePtr: Pointer) => T,
): T {
  let capturedError: unknown = null;

  // The JSCallback bridges the C function pointer to our TypeScript function.
  // It matches: int (*fun)(mlx_vector_array* out, const mlx_vector_array in)
  const callback = new JSCallback(
    (outVecPtr: Pointer | null, inVec: Pointer | null): number => {
      if (outVecPtr === null || inVec === null) return 1;

      let inputs: MxArray[] = [];
      let results: MxArray[] = [];
      try {
        // Unpack input vector into MxArray arguments
        inputs = extractAllArrays(inVec, "closure_input");

        // Call the user's function
        results = fn(...inputs);
        if (results.length !== 1) {
          throw new Error(
            `Closure function returned ${results.length} output(s); expected exactly 1`,
          );
        }
        const firstResult = results[0];

        if (firstResult === undefined) {
          throw new Error(
            "Closure function returned an empty array — expected at least one MxArray",
          );
        }

        // Write the result into the pre-allocated output vector.
        // Phase 2 supports single-output (scalar loss) only.
        checkStatus(
          ffi.mlx_vector_array_set_value(outVecPtr, firstResult._ctx),
          "closure_output_set",
        );
        firstResult.free();

        return 0;
      } catch (error) {
        capturedError = error;
        return 1;
      } finally {
        for (const result of results) {
          result.free();
        }
        for (const input of inputs) {
          input.free();
        }
      }
    },
    { args: [FFIType.ptr, FFIType.ptr], returns: FFIType.i32 },
  );

  // Create mlx_closure from the native callback function pointer
  const callbackPtr = unwrapPointer(callback.ptr, "closure callback pointer");
  const closurePtr = unwrapPointer(ffi.mlx_closure_new_func(callbackPtr), "mlx_closure_new_func");

  try {
    let result: T;
    try {
      result = body(closurePtr);
    } catch (bodyError) {
      // If we captured a JS error from the callback, that's the real cause.
      // The body typically fails because checkStatus sees error code 1 from
      // our callback returning failure — the MxError is just the symptom.
      if (capturedError !== null) {
        throw capturedError;
      }
      throw bodyError;
    }

    // Also check after successful body return (callback error may not have
    // caused a status failure in all code paths).
    if (capturedError !== null) {
      throw capturedError;
    }

    return result;
  } finally {
    ffi.mlx_closure_free(closurePtr);
    callback.close();
  }
}
