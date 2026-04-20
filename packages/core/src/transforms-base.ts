import type { MxArray } from "./array";
import { checkStatus } from "./error";
import {
  extractAllArrays,
  extractSingleArray,
  type GradFn,
  type MultiOutputFn,
  ReusableClosure,
  ReusableMultiClosure,
} from "./ffi/closure-bridge";
import { ffi } from "./ffi/lib";
import { OutSlot, type Pointer, ptr, sizeToNumber, unwrapPointer } from "./ffi/pointer";
import { formatShape } from "./format-shape";

export type DisposableTransform<T extends (...args: MxArray[]) => unknown> = T & Disposable;

export type CachedValueAndGrad = {
  argnums: readonly number[];
  closure: ReusableClosure;
  transform: Pointer;
};

export type CachedClosureTransform = {
  closure: ReusableClosure;
  transform: Pointer;
  freeTransform: (transform: Pointer) => void;
};

export function withArrayVector<T>(arrays: MxArray[], label: string, body: (vec: Pointer) => T): T {
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

function assertScalarLoss(loss: MxArray, source: string): void {
  if (loss.ndim !== 0) {
    throw new Error(
      `${source}: expected fn to return a scalar MxArray, got shape ${formatShape(loss.shape)}. ` +
        "Reduce the loss to a scalar before calling grad/valueAndGrad.",
    );
  }
}

export function normalizeArgnums(
  argnums: number | number[] | undefined,
  argCount: number,
): number[] {
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

  if (new Set(indices).size !== indices.length) {
    throw new Error(`Duplicate argnums: [${indices.join(", ")}]`);
  }

  return indices;
}

export const valueAndGradRegistry = new FinalizationRegistry<CachedValueAndGrad>((cached) => {
  ffi.mlx_closure_value_and_grad_free(cached.transform);
  cached.closure[Symbol.dispose]();
});

export const closureTransformRegistry = new FinalizationRegistry<CachedClosureTransform>(
  (cached) => {
    cached.freeTransform(cached.transform);
    cached.closure[Symbol.dispose]();
  },
);

export function disposeCachedValueAndGrad(cached: CachedValueAndGrad): void {
  ffi.mlx_closure_value_and_grad_free(cached.transform);
  cached.closure[Symbol.dispose]();
}

export function disposeCachedClosureTransform(cached: CachedClosureTransform): void {
  cached.freeTransform(cached.transform);
  cached.closure[Symbol.dispose]();
}

export function attachDisposableTransform<T extends (...args: MxArray[]) => unknown>(
  fn: T,
  dispose: () => void,
): DisposableTransform<T> {
  return Object.assign(fn, {
    [Symbol.dispose]: dispose,
  });
}

export function sameIndices(left: readonly number[], right: readonly number[]): boolean {
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

export function createCachedValueAndGrad(fn: GradFn, argnums: number[]): CachedValueAndGrad {
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

export function createCachedClosureTransform(
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

export function applyValueAndGrad(
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

export function applyClosureTransform(
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

// ---------------------------------------------------------------------------
// Multi-output closure transform infrastructure
// ---------------------------------------------------------------------------

export type CachedClosureMultiTransform = {
  closure: ReusableMultiClosure;
  transform: Pointer;
  freeTransform: (transform: Pointer) => void;
};

export const closureMultiTransformRegistry = new FinalizationRegistry<CachedClosureMultiTransform>(
  (cached) => {
    cached.freeTransform(cached.transform);
    cached.closure[Symbol.dispose]();
  },
);

export function disposeCachedClosureMultiTransform(cached: CachedClosureMultiTransform): void {
  cached.freeTransform(cached.transform);
  cached.closure[Symbol.dispose]();
}

export function createCachedClosureMultiTransform(
  fn: MultiOutputFn,
  label: string,
  buildTransform: (out: Pointer, closure: Pointer) => void,
  freeTransform: (transform: Pointer) => void,
): CachedClosureMultiTransform {
  const closure = new ReusableMultiClosure(fn);
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

export function applyClosureMultiTransform(
  cached: CachedClosureMultiTransform,
  args: MxArray[],
  label: string,
): MxArray[] {
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
      return extractAllArrays(outputsVec, label);
    } finally {
      if (outputsVec !== null) {
        ffi.mlx_vector_array_free(outputsVec);
      }
    }
  } finally {
    ffi.mlx_vector_array_free(inputVec);
  }
}
