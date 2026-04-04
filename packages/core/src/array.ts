/**
 * MxArray — the core array type wrapping an mlx-c opaque pointer.
 *
 * Every MxArray owns an `mlx_array` handle (a C++ object behind `void* ctx`).
 * Memory is managed via:
 * 1. Explicit disposal with `using` / `[Symbol.dispose]()` (preferred in hot paths)
 * 2. FinalizationRegistry as a GC safety net (prevents leaks if dispose is missed)
 *
 * @module
 */

import type { Pointer } from "bun:ffi";
import {
  assertElementCount,
  createDataBuffer,
  type DirectTypedArray,
  expectPresent,
  getDirectStorageDtype,
  type NestedArray,
  type NumericArrayData,
  normalizeInput,
  reshapeFlat,
  type SupportedTypedArray,
} from "./array-data";
import { getDataPointer } from "./array-ffi-data";
import { defaultStream } from "./device";
import { DTYPE_BYTE_SIZE, DTYPE_TO_MLX, type DType, MLX_TO_DTYPE } from "./dtype";
import { checkStatus, initializeErrorHandler } from "./error";
import { ffi } from "./ffi/lib";
import { nativeSlice, OutSlot, ptr, readI32, sizeToNumber, unwrapPointer } from "./ffi/pointer";

export type { NestedArray } from "./array-data";

// Ensure error handler is registered before any array operations.
initializeErrorHandler();

/** Read an FFI pointer result written into a temporary output slot. */
export function readResultPointer(label: string, invoke: (out: Pointer) => void): Pointer {
  const slot = new OutSlot();
  invoke(slot.prepare());
  return slot.read(label);
}

/** Read an FFI array result written into a temporary output slot. */
export function readResultArray(label: string, invoke: (out: Pointer) => void): MxArray {
  return MxArray._fromCtx(readResultPointer(label, invoke));
}

// FinalizationRegistry is a safety net for missed explicit disposal.
const registry = new FinalizationRegistry<Pointer>((ctx: Pointer) => {
  ffi.mlx_array_free(ctx);
});

/**
 * N-dimensional array backed by MLX.
 *
 * Operations on MxArray are lazy — they build a computation graph.
 * Call `eval()` or the module-level `mxEval()` to force execution.
 */
export class MxArray implements Disposable {
  /** The opaque native context pointer (`mlx_array.ctx`). */
  readonly _ctx: Pointer;

  private _disposed = false;
  private _shape: readonly number[] | null = null;
  private _dtype: DType | null = null;
  private _ndim: number | null = null;
  private _size: number | null = null;

  private constructor(ctx: Pointer) {
    this._ctx = ctx;
    registry.register(this, ctx, this);
  }

  /** Wrap an existing mlx_array context pointer and take ownership of it. */
  static _fromCtx(ctx: Pointer): MxArray {
    return new MxArray(ctx);
  }

  /** Create an array from JavaScript numbers or a supported TypedArray. */
  static fromData(
    data: NumericArrayData | SupportedTypedArray,
    shape?: number[],
    dtype?: DType,
  ): MxArray {
    const { values, inferredShape, inferredDtype } = normalizeInput(data, shape, dtype);
    const storageDtype = getDirectStorageDtype(inferredDtype);

    assertElementCount(values.length, inferredShape);

    const dataBuffer = createDataBuffer(values, storageDtype);
    const shapeBuffer =
      inferredShape.length === 0 ? new Int32Array(1) : new Int32Array(inferredShape);
    const baseArray = new MxArray(
      unwrapPointer(
        ffi.mlx_array_new_data(
          ptr(dataBuffer),
          ptr(shapeBuffer),
          inferredShape.length,
          DTYPE_TO_MLX[storageDtype],
        ),
        "mlx_array_new_data",
      ),
    );

    if (storageDtype === inferredDtype) {
      return baseArray;
    }

    try {
      return baseArray.asType(inferredDtype);
    } finally {
      baseArray.free();
    }
  }

  /** Number of dimensions. */
  get ndim(): number {
    if (this._ndim === null) {
      this._ndim = sizeToNumber(ffi.mlx_array_ndim(this._ctx), "mlx_array_ndim");
    }
    return this._ndim;
  }

  /** Shape of the array. */
  get shape(): readonly number[] {
    if (this._shape === null) {
      const ndim = this.ndim;
      if (ndim === 0) {
        this._shape = [];
      } else {
        const shapePtr = unwrapPointer(ffi.mlx_array_shape(this._ctx), "mlx_array_shape");
        const shape: number[] = [];
        for (let index = 0; index < ndim; index++) {
          shape.push(readI32(shapePtr, index * 4));
        }
        this._shape = Object.freeze(shape);
      }
    }
    return this._shape;
  }

  /** Element data type. */
  get dtype(): DType {
    if (this._dtype !== null) {
      return this._dtype;
    }

    const rawDType = ffi.mlx_array_dtype(this._ctx);
    const dtype = MLX_TO_DTYPE[rawDType];
    if (dtype === undefined) {
      throw new Error(`Unknown mlx dtype: ${rawDType}`);
    }

    this._dtype = dtype;
    return dtype;
  }

  /** Total number of elements. */
  get size(): number {
    if (this._size === null) {
      this._size = sizeToNumber(ffi.mlx_array_size(this._ctx), "mlx_array_size");
    }
    return this._size;
  }

  /** Bytes per element. */
  get itemsize(): number {
    return DTYPE_BYTE_SIZE[this.dtype];
  }

  /** Total bytes in the evaluated array. */
  get nbytes(): number {
    return this.size * this.itemsize;
  }

  /** Force evaluation of this array. */
  eval(): void {
    checkStatus(ffi.mlx_array_eval(this._ctx), "array_eval");
  }

  /**
   * Copy the evaluated array into a JavaScript TypedArray.
   *
   * Returned data is always contiguous and row-major.
   */
  toTypedArray(): DirectTypedArray {
    const contiguous = this._makeContiguous();
    try {
      contiguous.eval();

      const dtype = contiguous.dtype;
      const size = contiguous.size;

      switch (dtype) {
        case "float32":
          return new Float32Array(
            nativeSlice(getDataPointer(contiguous, dtype), 0, size * 4).slice(0),
          );
        case "float64":
          return new Float64Array(
            nativeSlice(getDataPointer(contiguous, dtype), 0, size * 8).slice(0),
          );
        case "int8":
          return new Int8Array(nativeSlice(getDataPointer(contiguous, dtype), 0, size).slice(0));
        case "int16":
          return new Int16Array(
            nativeSlice(getDataPointer(contiguous, dtype), 0, size * 2).slice(0),
          );
        case "int32":
          return new Int32Array(
            nativeSlice(getDataPointer(contiguous, dtype), 0, size * 4).slice(0),
          );
        case "uint8":
        case "bool":
          return new Uint8Array(
            nativeSlice(getDataPointer(contiguous, dtype), 0, size * DTYPE_BYTE_SIZE[dtype]).slice(
              0,
            ),
          );
        case "uint16":
          return new Uint16Array(
            nativeSlice(getDataPointer(contiguous, dtype), 0, size * 2).slice(0),
          );
        case "uint32":
          return new Uint32Array(
            nativeSlice(getDataPointer(contiguous, dtype), 0, size * 4).slice(0),
          );
        default: {
          const asFloat32 = contiguous.asType("float32");
          try {
            return asFloat32.toTypedArray();
          } finally {
            asFloat32.free();
          }
        }
      }
    } finally {
      contiguous.free();
    }
  }

  /** Convert the array to a nested JavaScript structure matching its shape. */
  toList(): NestedArray<number> {
    const flat = this.toTypedArray();
    if (this.ndim === 0) {
      return Number(expectPresent(flat[0], "scalar array value"));
    }
    return reshapeFlat(Array.from(flat, Number), this.shape);
  }

  /** Read a single scalar value from a scalar or single-element array. */
  item(): number {
    this.eval();

    switch (this.dtype) {
      case "bool": {
        const buffer = new Uint8Array(1);
        checkStatus(ffi.mlx_array_item_bool(ptr(buffer), this._ctx), "array_item_bool");
        return expectPresent(buffer[0], "bool scalar value");
      }
      case "uint8": {
        const buffer = new Uint8Array(1);
        checkStatus(ffi.mlx_array_item_uint8(ptr(buffer), this._ctx), "array_item_uint8");
        return expectPresent(buffer[0], "uint8 scalar value");
      }
      case "uint16": {
        const buffer = new Uint16Array(1);
        checkStatus(ffi.mlx_array_item_uint16(ptr(buffer), this._ctx), "array_item_uint16");
        return expectPresent(buffer[0], "uint16 scalar value");
      }
      case "uint32": {
        const buffer = new Uint32Array(1);
        checkStatus(ffi.mlx_array_item_uint32(ptr(buffer), this._ctx), "array_item_uint32");
        return expectPresent(buffer[0], "uint32 scalar value");
      }
      case "int8": {
        const buffer = new Int8Array(1);
        checkStatus(ffi.mlx_array_item_int8(ptr(buffer), this._ctx), "array_item_int8");
        return expectPresent(buffer[0], "int8 scalar value");
      }
      case "int16": {
        const buffer = new Int16Array(1);
        checkStatus(ffi.mlx_array_item_int16(ptr(buffer), this._ctx), "array_item_int16");
        return expectPresent(buffer[0], "int16 scalar value");
      }
      case "int32": {
        const buffer = new Int32Array(1);
        checkStatus(ffi.mlx_array_item_int32(ptr(buffer), this._ctx), "array_item_int32");
        return expectPresent(buffer[0], "int32 scalar value");
      }
      case "float32": {
        const buffer = new Float32Array(1);
        checkStatus(ffi.mlx_array_item_float32(ptr(buffer), this._ctx), "array_item_float32");
        return expectPresent(buffer[0], "float32 scalar value");
      }
      case "float64": {
        const buffer = new Float64Array(1);
        checkStatus(ffi.mlx_array_item_float64(ptr(buffer), this._ctx), "array_item_float64");
        return expectPresent(buffer[0], "float64 scalar value");
      }
      default: {
        const asFloat32 = this.asType("float32");
        try {
          return asFloat32.item();
        } finally {
          asFloat32.free();
        }
      }
    }
  }

  /** Cast the array to a different dtype. */
  asType(dtype: DType): MxArray {
    return readResultArray("astype", (out) => {
      checkStatus(ffi.mlx_astype(out, this._ctx, DTYPE_TO_MLX[dtype], defaultStream()), "astype");
    });
  }

  /** Whether this array has already been disposed. */
  get isDisposed(): boolean {
    return this._disposed;
  }

  /** Explicitly free the underlying native array handle. */
  free(): void {
    if (this._disposed) {
      return;
    }

    this._disposed = true;
    registry.unregister(this);
    ffi.mlx_array_free(this._ctx);
  }

  /** Support `using` declarations for explicit resource management. */
  [Symbol.dispose](): void {
    this.free();
  }

  /** Make the array contiguous in row-major order. */
  private _makeContiguous(): MxArray {
    return readResultArray("contiguous", (out) => {
      checkStatus(ffi.mlx_contiguous(out, this._ctx, false, defaultStream()), "contiguous");
    });
  }
}

/** Create an array filled with zeros. */
export function zeros(shape: number[], dtype: DType = "float32"): MxArray {
  const shapeArray = new Int32Array(shape);
  return readResultArray("zeros", (out) => {
    checkStatus(
      ffi.mlx_zeros(out, ptr(shapeArray), shape.length, DTYPE_TO_MLX[dtype], defaultStream()),
      "zeros",
    );
  });
}

/** Create an array filled with ones. */
export function ones(shape: number[], dtype: DType = "float32"): MxArray {
  const shapeArray = new Int32Array(shape);
  return readResultArray("ones", (out) => {
    checkStatus(
      ffi.mlx_ones(out, ptr(shapeArray), shape.length, DTYPE_TO_MLX[dtype], defaultStream()),
      "ones",
    );
  });
}

/** Create an array filled with a constant value. */
export function full(shape: number[], value: number, dtype: DType = "float32"): MxArray {
  const shapeArray = new Int32Array(shape);
  const valueHandle = createScalarHandle(value, dtype);

  try {
    return readResultArray("full", (out) => {
      checkStatus(
        ffi.mlx_full(
          out,
          ptr(shapeArray),
          shape.length,
          valueHandle,
          DTYPE_TO_MLX[dtype],
          defaultStream(),
        ),
        "full",
      );
    });
  } finally {
    ffi.mlx_array_free(valueHandle);
  }
}

/** Create an array with evenly spaced values. */
export function arange(start: number, stop: number, step = 1, dtype: DType = "float32"): MxArray {
  return readResultArray("arange", (out) => {
    checkStatus(
      ffi.mlx_arange(out, start, stop, step, DTYPE_TO_MLX[dtype], defaultStream()),
      "arange",
    );
  });
}

/** Create an MxArray from a JavaScript number or array data. */
export function array(
  data: number | NumericArrayData | SupportedTypedArray,
  dtype?: DType,
): MxArray {
  if (typeof data === "number") {
    return MxArray._fromCtx(createScalarHandle(data, dtype ?? "float32"));
  }
  return MxArray.fromData(data, undefined, dtype);
}

/**
 * Create a second owned handle to the same MLX array value.
 *
 * This preserves the array's value without aliasing the exact same `MxArray`
 * object, which is useful for no-op layer paths that should not share manual
 * disposal ownership with their inputs.
 */
export function retainArray(source: MxArray): MxArray {
  return readResultArray("retained array", (out) => {
    checkStatus(ffi.mlx_array_set(out, source._ctx), "array_set");
  });
}

function createScalarHandle(value: number, dtype: DType): Pointer {
  switch (dtype) {
    case "bool":
      return unwrapPointer(ffi.mlx_array_new_bool(value !== 0), "mlx_array_new_bool");
    case "float32":
      return unwrapPointer(ffi.mlx_array_new_float32(value), "mlx_array_new_float32");
    case "float64":
      return unwrapPointer(ffi.mlx_array_new_float64(value), "mlx_array_new_float64");
    default: {
      const baseHandle = isIntegerLike(dtype)
        ? unwrapPointer(ffi.mlx_array_new_int(Math.trunc(value)), "mlx_array_new_int")
        : unwrapPointer(ffi.mlx_array_new_float(value), "mlx_array_new_float");
      const baseArray = MxArray._fromCtx(baseHandle);

      if (baseArray.dtype === dtype) {
        registry.unregister(baseArray);
        return baseHandle;
      }

      try {
        const converted = baseArray.asType(dtype);
        registry.unregister(converted);
        return converted._ctx;
      } finally {
        baseArray.free();
      }
    }
  }
}

function isIntegerLike(dtype: DType): boolean {
  return (
    dtype === "uint8" ||
    dtype === "uint16" ||
    dtype === "uint32" ||
    dtype === "uint64" ||
    dtype === "int8" ||
    dtype === "int16" ||
    dtype === "int32" ||
    dtype === "int64"
  );
}
