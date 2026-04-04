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

import { formatShape } from "../utils/format-shape";
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
  ReusableClosure,
  sizeToNumber,
  unwrapPointer,
} from "./ffi";

function withArrayVector<T>(arrays: MxArray[], label: string, body: (vec: Pointer) => T): T {
  const vec = unwrapPointer(ffi.mlx_vector_array_new(), "mlx_vector_array_new");
  try {
    for (const arr of arrays) {
      checkStatus(ffi.mlx_vector_array_append_value(vec, arr._ctx), `${label}_vec_append`);
    }
    return body(vec);
  } finally {
    ffi.mlx_vector_array_free(vec);
  }
}

/**
 * Force evaluation of one or more lazy arrays.
 * After eval, the arrays' data is computed and available for reading.
 *
 * Named `mxEval` because `eval` is a reserved word in strict-mode JavaScript.
 */
export function mxEval(...arrays: MxArray[]): void {
  if (arrays.length === 0) return;

  withArrayVector(arrays, "eval", (vec) => {
    checkStatus(ffi.mlx_eval(vec), "eval");
  });
}

/**
 * Schedule evaluation of one or more lazy arrays without waiting for completion.
 *
 * Use this when you want to overlap host-side bookkeeping with MLX execution
 * and synchronize explicitly later.
 */
export function mxAsyncEval(...arrays: MxArray[]): void {
  if (arrays.length === 0) return;

  withArrayVector(arrays, "async_eval", (vec) => {
    checkStatus(ffi.mlx_async_eval(vec), "async_eval");
  });
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

function assertScalarLoss(loss: MxArray, source: string): void {
  if (loss.ndim !== 0) {
    throw new Error(
      `${source}: expected fn to return a scalar MxArray, got shape ${formatShape(loss.shape)}. ` +
        "Reduce the loss to a scalar before calling grad/valueAndGrad.",
    );
  }
}

type CachedValueAndGrad = {
  argnums: readonly number[];
  closure: ReusableClosure;
  transform: Pointer;
};

type CachedClosureTransform = {
  closure: ReusableClosure;
  transform: Pointer;
  freeTransform: (transform: Pointer) => void;
};

export type DisposableTransform<T extends (...args: MxArray[]) => unknown> = T & Disposable;

export type CompileMode = "disabled" | "no_simplify" | "no_fuse" | "enabled";

const COMPILE_MODE_TO_NATIVE: Record<CompileMode, number> = {
  disabled: 0,
  no_simplify: 1,
  no_fuse: 2,
  enabled: 3,
};

const valueAndGradRegistry = new FinalizationRegistry<CachedValueAndGrad>((cached) => {
  ffi.mlx_closure_value_and_grad_free(cached.transform);
  cached.closure[Symbol.dispose]();
});

const closureTransformRegistry = new FinalizationRegistry<CachedClosureTransform>((cached) => {
  cached.freeTransform(cached.transform);
  cached.closure[Symbol.dispose]();
});

function disposeCachedValueAndGrad(cached: CachedValueAndGrad): void {
  ffi.mlx_closure_value_and_grad_free(cached.transform);
  cached.closure[Symbol.dispose]();
}

function disposeCachedClosureTransform(cached: CachedClosureTransform): void {
  cached.freeTransform(cached.transform);
  cached.closure[Symbol.dispose]();
}

function attachDisposableTransform<T extends (...args: MxArray[]) => unknown>(
  fn: T,
  dispose: () => void,
): DisposableTransform<T> {
  return Object.assign(fn, {
    [Symbol.dispose]: dispose,
  });
}

function sameIndices(left: readonly number[], right: readonly number[]): boolean {
  if (left.length !== right.length) {
    return false;
  }
  for (let index = 0; index < left.length; index++) {
    if (left[index] !== right[index]) {
      return false;
    }
  }
  return true;
}

function createCachedValueAndGrad(fn: GradFn, argnums: number[]): CachedValueAndGrad {
  const wrappedFn = (...inputs: MxArray[]): MxArray[] => {
    const result = fn(...inputs);
    assertScalarLoss(result, "autograd");
    return [result];
  };

  const closure = new ReusableClosure(wrappedFn);
  try {
    const argnumsBuf = new Int32Array(argnums);
    const transformSlot = new OutSlot();
    checkStatus(
      ffi.mlx_value_and_grad(
        transformSlot.prepare(),
        closure.pointer,
        ptr(argnumsBuf),
        argnums.length,
      ),
      "value_and_grad",
    );

    return {
      argnums: [...argnums],
      closure,
      transform: transformSlot.read("value_and_grad transform"),
    };
  } catch (error) {
    closure[Symbol.dispose]();
    throw error;
  }
}

function createCachedClosureTransform(
  fn: GradFn,
  label: string,
  buildTransform: (out: Pointer, closure: Pointer) => void,
  freeTransform: (transform: Pointer) => void,
): CachedClosureTransform {
  const wrappedFn = (...inputs: MxArray[]): MxArray[] => [fn(...inputs)];
  const closure = new ReusableClosure(wrappedFn);
  try {
    const transformSlot = new OutSlot();
    buildTransform(transformSlot.prepare(), closure.pointer);
    return {
      closure,
      transform: transformSlot.read(`${label} transform`),
      freeTransform,
    };
  } catch (error) {
    closure[Symbol.dispose]();
    throw error;
  }
}

/**
 * Core implementation: apply value_and_grad transform to a function with arguments.
 * Returns [values vector, gradients vector] as raw MxArray results.
 */
function applyValueAndGrad(
  cached: CachedValueAndGrad,
  args: MxArray[],
): { value: MxArray; grads: MxArray[] } {
  cached.closure.consumeCapturedError();

  const inputVec = unwrapPointer(ffi.mlx_vector_array_new(), "vag_input_vec_new");
  try {
    for (const arg of args) {
      checkStatus(ffi.mlx_vector_array_append_value(inputVec, arg._ctx), "vag_input_append");
    }

    const valuesSlot = new OutSlot();
    const gradsSlot = new OutSlot();
    try {
      try {
        checkStatus(
          ffi.mlx_closure_value_and_grad_apply(
            valuesSlot.prepare(),
            gradsSlot.prepare(),
            cached.transform,
            inputVec,
          ),
          "closure_value_and_grad_apply",
        );
      } catch (error) {
        const capturedError = cached.closure.consumeCapturedError();
        if (capturedError !== null) {
          throw capturedError;
        }
        throw error;
      }

      const capturedError = cached.closure.consumeCapturedError();
      if (capturedError !== null) {
        throw capturedError;
      }

      const valuesVec = valuesSlot.read("values vector");
      const gradsVec = gradsSlot.read("grads vector");

      try {
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
  } catch (error) {
    const capturedError = cached.closure.consumeCapturedError();
    if (capturedError !== null) {
      throw capturedError;
    }
    throw error;
  }
}

function applyClosureTransform(
  cached: CachedClosureTransform,
  args: MxArray[],
  label: string,
): MxArray {
  cached.closure.consumeCapturedError();

  const inputVec = unwrapPointer(ffi.mlx_vector_array_new(), `${label}_input_vec_new`);
  try {
    for (const arg of args) {
      checkStatus(ffi.mlx_vector_array_append_value(inputVec, arg._ctx), `${label}_input_append`);
    }

    const outputSlot = new OutSlot();
    let outputsVec: Pointer | null = null;
    try {
      try {
        checkStatus(ffi.mlx_closure_apply(outputSlot.prepare(), cached.transform, inputVec), label);
      } catch (error) {
        const capturedError = cached.closure.consumeCapturedError();
        if (capturedError !== null) {
          throw capturedError;
        }
        throw error;
      }

      const capturedError = cached.closure.consumeCapturedError();
      if (capturedError !== null) {
        throw capturedError;
      }

      outputsVec = outputSlot.read(`${label} outputs`);
      const outputCount = sizeToNumber(ffi.mlx_vector_array_size(outputsVec), `${label}_vec_size`);
      if (outputCount !== 1) {
        throw new Error(`${label}: expected exactly 1 output, got ${outputCount}`);
      }

      return extractSingleArray(outputsVec, label);
    } finally {
      if (outputsVec !== null) {
        ffi.mlx_vector_array_free(outputsVec);
      }
    }
  } finally {
    ffi.mlx_vector_array_free(inputVec);
  }
}

/**
 * Transform `fn` into a compiled function.
 *
 * This is a flat-array closure transform intended for performance-sensitive
 * code that already works with explicit `MxArray` arguments.
 */
export function compile(
  fn: GradFn,
  options?: { shapeless?: boolean },
): DisposableTransform<(...args: MxArray[]) => MxArray> {
  let cached: CachedClosureTransform | null = null;
  let disposed = false;

  const transformed = attachDisposableTransform(
    (...args: MxArray[]): MxArray => {
      if (disposed) {
        throw new Error("compile: transform has already been disposed");
      }
      if (cached === null) {
        cached = createCachedClosureTransform(
          fn,
          "compile",
          (out, closure) => {
            checkStatus(ffi.mlx_compile(out, closure, options?.shapeless ?? false), "compile");
          },
          (transform) => {
            ffi.mlx_closure_free(transform);
          },
        );
        closureTransformRegistry.register(transformed, cached, transformed);
      }
      return applyClosureTransform(cached, args, "compile_apply");
    },
    () => {
      if (disposed) {
        return;
      }
      disposed = true;
      closureTransformRegistry.unregister(transformed);
      if (cached !== null) {
        disposeCachedClosureTransform(cached);
        cached = null;
      }
    },
  );

  return transformed;
}

/**
 * Transform `fn` into a gradient-checkpointed function.
 *
 * MLX will save inputs/outputs and recompute intermediates during the
 * backward pass, reducing activation memory at the cost of extra compute.
 */
export function checkpoint(fn: GradFn): DisposableTransform<(...args: MxArray[]) => MxArray> {
  let cached: CachedClosureTransform | null = null;
  let disposed = false;

  const transformed = attachDisposableTransform(
    (...args: MxArray[]): MxArray => {
      if (disposed) {
        throw new Error("checkpoint: transform has already been disposed");
      }
      if (cached === null) {
        cached = createCachedClosureTransform(
          fn,
          "checkpoint",
          (out, closure) => {
            checkStatus(ffi.mlx_checkpoint(out, closure), "checkpoint");
          },
          (transform) => {
            ffi.mlx_closure_free(transform);
          },
        );
        closureTransformRegistry.register(transformed, cached, transformed);
      }
      return applyClosureTransform(cached, args, "checkpoint_apply");
    },
    () => {
      if (disposed) {
        return;
      }
      disposed = true;
      closureTransformRegistry.unregister(transformed);
      if (cached !== null) {
        disposeCachedClosureTransform(cached);
        cached = null;
      }
    },
  );

  return transformed;
}

/** Clear MLX's internal compile cache. */
export function clearCompileCache(): void {
  checkStatus(ffi.mlx_detail_compile_clear_cache(), "detail_compile_clear_cache");
}

/** Enable MLX compile transforms globally. */
export function enableCompile(): void {
  checkStatus(ffi.mlx_enable_compile(), "enable_compile");
}

/** Disable MLX compile transforms globally. */
export function disableCompile(): void {
  checkStatus(ffi.mlx_disable_compile(), "disable_compile");
}

/** Set MLX's global compile mode. */
export function setCompileMode(mode: CompileMode): void {
  checkStatus(ffi.mlx_set_compile_mode(COMPILE_MODE_TO_NATIVE[mode]), "set_compile_mode");
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
export function valueAndGrad(
  fn: GradFn,
): DisposableTransform<(...args: MxArray[]) => [MxArray, MxArray]>;
export function valueAndGrad(
  fn: GradFn,
  argnums: number,
): DisposableTransform<(...args: MxArray[]) => [MxArray, MxArray]>;
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
): DisposableTransform<(...args: MxArray[]) => [MxArray, MxArray[]]>;
export function valueAndGrad(
  fn: GradFn,
  argnums?: number | number[],
): DisposableTransform<(...args: MxArray[]) => [MxArray, MxArray] | [MxArray, MxArray[]]> {
  const isMulti = Array.isArray(argnums);
  let cached: CachedValueAndGrad | null = null;
  let disposed = false;

  const transformed = attachDisposableTransform(
    (...args: MxArray[]): [MxArray, MxArray] | [MxArray, MxArray[]] => {
      if (disposed) {
        throw new Error("valueAndGrad: transform has already been disposed");
      }
      const indices = normalizeArgnums(argnums, args.length);
      if (cached === null) {
        cached = createCachedValueAndGrad(fn, indices);
        valueAndGradRegistry.register(transformed, cached, transformed);
      } else if (!sameIndices(cached.argnums, indices)) {
        throw new Error(
          `valueAndGrad: argnums changed between calls. Expected [${cached.argnums.join(", ")}], got [${indices.join(", ")}].`,
        );
      }
      const { value, grads } = applyValueAndGrad(cached, args);

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
    },
    () => {
      if (disposed) {
        return;
      }
      disposed = true;
      valueAndGradRegistry.unregister(transformed);
      if (cached !== null) {
        disposeCachedValueAndGrad(cached);
        cached = null;
      }
    },
  );

  return transformed;
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
export function grad(fn: GradFn): DisposableTransform<(...args: MxArray[]) => MxArray>;
export function grad(
  fn: GradFn,
  argnums: number,
): DisposableTransform<(...args: MxArray[]) => MxArray>;
/**
 * Create a function that computes the gradients of `fn`
 * with respect to multiple arguments.
 *
 * @param fn - A function from MxArrays to a scalar MxArray (the loss).
 * @param argnums - Array of argument indices to differentiate with respect to.
 * @returns A function returning an array of gradient MxArrays.
 */
export function grad(
  fn: GradFn,
  argnums: number[],
): DisposableTransform<(...args: MxArray[]) => MxArray[]>;
export function grad(
  fn: GradFn,
  argnums?: number | number[],
): DisposableTransform<(...args: MxArray[]) => MxArray | MxArray[]> {
  const isMulti = Array.isArray(argnums);
  let cached: CachedValueAndGrad | null = null;
  let disposed = false;

  const transformed = attachDisposableTransform(
    (...args: MxArray[]): MxArray | MxArray[] => {
      if (disposed) {
        throw new Error("grad: transform has already been disposed");
      }
      const indices = normalizeArgnums(argnums, args.length);
      if (cached === null) {
        cached = createCachedValueAndGrad(fn, indices);
        valueAndGradRegistry.register(transformed, cached, transformed);
      } else if (!sameIndices(cached.argnums, indices)) {
        throw new Error(
          `grad: argnums changed between calls. Expected [${cached.argnums.join(", ")}], got [${indices.join(", ")}].`,
        );
      }
      const { value, grads } = applyValueAndGrad(cached, args);

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
    },
    () => {
      if (disposed) {
        return;
      }
      disposed = true;
      valueAndGradRegistry.unregister(transformed);
      if (cached !== null) {
        disposeCachedValueAndGrad(cached);
        cached = null;
      }
    },
  );

  return transformed;
}
