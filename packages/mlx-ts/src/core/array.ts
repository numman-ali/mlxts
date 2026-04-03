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
import { defaultStream } from "./device";
import { DTYPE_BYTE_SIZE, DTYPE_TO_MLX, type DType, MLX_TO_DTYPE } from "./dtype";
import { checkStatus, initializeErrorHandler } from "./error";
import { ffi, nativeSlice, ptr, readI32, readPtr, sizeToNumber, unwrapPointer } from "./ffi";

// Ensure error handler is registered before any array operations.
initializeErrorHandler();

/** Recursive array type for toList() output. */
export type NestedArray<T> = T | NestedArray<T>[];

type NumericArrayData = readonly (number | NumericArrayData)[];
type DirectDataDType =
  | "bool"
  | "uint8"
  | "uint16"
  | "uint32"
  | "int8"
  | "int16"
  | "int32"
  | "float32"
  | "float64";
type DirectTypedArray =
  | Float32Array
  | Float64Array
  | Int8Array
  | Int16Array
  | Int32Array
  | Uint8Array
  | Uint16Array
  | Uint32Array;
type NormalizedInput = {
  inferredDtype: DType;
  inferredShape: number[];
  values: number[];
};

// Operations write their results into an `mlx_array*`, which is a pointer to an
// 8-byte struct containing the native context pointer.
const _outBuf = new Uint8Array(8);
const _outPtr = ptr(_outBuf);

/** Prepare the shared output buffer for an mlx-c call. */
export function prepareOut(): Pointer {
  _outBuf.fill(0);
  return _outPtr;
}

/** Read the result pointer from the shared output buffer. */
export function readOut(): Pointer {
  return unwrapPointer(readPtr(_outPtr, 0), "FFI operation result");
}

/** Read a pointer result from a local 8-byte mlx_array output buffer. */
function readLocalOut(out: Pointer, label: string): Pointer {
  return unwrapPointer(readPtr(out, 0), label);
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
    data: NumericArrayData | Float32Array | Int32Array | Uint8Array,
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
    const out = prepareOut();
    checkStatus(ffi.mlx_astype(out, this._ctx, DTYPE_TO_MLX[dtype], defaultStream()), "astype");
    return MxArray._fromCtx(readOut());
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
    const out = prepareOut();
    checkStatus(ffi.mlx_contiguous(out, this._ctx, false, defaultStream()), "contiguous");
    return MxArray._fromCtx(readOut());
  }
}

/** Create an array filled with zeros. */
export function zeros(shape: number[], dtype: DType = "float32"): MxArray {
  const shapeArray = new Int32Array(shape);
  const out = prepareOut();
  checkStatus(
    ffi.mlx_zeros(out, ptr(shapeArray), shape.length, DTYPE_TO_MLX[dtype], defaultStream()),
    "zeros",
  );
  return MxArray._fromCtx(readOut());
}

/** Create an array filled with ones. */
export function ones(shape: number[], dtype: DType = "float32"): MxArray {
  const shapeArray = new Int32Array(shape);
  const out = prepareOut();
  checkStatus(
    ffi.mlx_ones(out, ptr(shapeArray), shape.length, DTYPE_TO_MLX[dtype], defaultStream()),
    "ones",
  );
  return MxArray._fromCtx(readOut());
}

/** Create an array filled with a constant value. */
export function full(shape: number[], value: number, dtype: DType = "float32"): MxArray {
  const shapeArray = new Int32Array(shape);
  const valueHandle = createScalarHandle(value, dtype);
  const out = prepareOut();

  try {
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
    return MxArray._fromCtx(readOut());
  } finally {
    ffi.mlx_array_free(valueHandle);
  }
}

/** Create an array with evenly spaced values. */
export function arange(start: number, stop: number, step = 1, dtype: DType = "float32"): MxArray {
  const out = prepareOut();
  checkStatus(
    ffi.mlx_arange(out, start, stop, step, DTYPE_TO_MLX[dtype], defaultStream()),
    "arange",
  );
  return MxArray._fromCtx(readOut());
}

/** Create an MxArray from a JavaScript number or array data. */
export function array(
  data: number | NumericArrayData | Float32Array | Int32Array | Uint8Array,
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
  const outBuf = new Uint8Array(8);
  const out = ptr(outBuf);
  checkStatus(ffi.mlx_array_set(out, source._ctx), "array_set");
  return MxArray._fromCtx(readLocalOut(out, "retained array"));
}

function normalizeInput(
  data: NumericArrayData | Float32Array | Int32Array | Uint8Array,
  shape?: number[],
  dtype?: DType,
): NormalizedInput {
  if (data instanceof Float32Array) {
    return {
      values: Array.from(data),
      inferredShape: shape ?? [data.length],
      inferredDtype: dtype ?? "float32",
    };
  }

  if (data instanceof Int32Array) {
    return {
      values: Array.from(data),
      inferredShape: shape ?? [data.length],
      inferredDtype: dtype ?? "int32",
    };
  }

  if (data instanceof Uint8Array) {
    return {
      values: Array.from(data),
      inferredShape: shape ?? [data.length],
      inferredDtype: dtype ?? "uint8",
    };
  }

  const { flat, inferredShape } = flattenArray(data);
  return {
    values: flat,
    inferredShape: shape ?? inferredShape,
    inferredDtype: dtype ?? "float32",
  };
}

function getDirectStorageDtype(dtype: DType): DirectDataDType {
  switch (dtype) {
    case "bool":
    case "uint8":
    case "uint16":
    case "uint32":
    case "int8":
    case "int16":
    case "int32":
    case "float32":
    case "float64":
      return dtype;
    case "uint64":
    case "int64":
      return "float64";
    case "float16":
    case "bfloat16":
    case "complex64":
      return "float32";
  }
}

function assertElementCount(elementCount: number, shape: readonly number[]): void {
  const expectedCount =
    shape.length === 0 ? 1 : shape.reduce((product, dimension) => product * dimension, 1);

  if (elementCount !== expectedCount) {
    throw new Error(
      `Data length ${elementCount} does not match shape [${shape.join(", ")}] (${expectedCount} elements)`,
    );
  }
}

function createDataBuffer(values: readonly number[], dtype: DirectDataDType): DirectTypedArray {
  const source = values.length === 0 ? [0] : values;

  switch (dtype) {
    case "bool":
      return new Uint8Array(source.map((value) => (value === 0 ? 0 : 1)));
    case "uint8":
      return new Uint8Array(source);
    case "uint16":
      return new Uint16Array(source);
    case "uint32":
      return new Uint32Array(source);
    case "int8":
      return new Int8Array(source);
    case "int16":
      return new Int16Array(source);
    case "int32":
      return new Int32Array(source);
    case "float32":
      return new Float32Array(source);
    case "float64":
      return new Float64Array(source);
  }
}

function getDataPointer(arr: MxArray, dtype: DirectDataDType): Pointer {
  let result: Pointer | null = null;

  switch (dtype) {
    case "float32":
      result = ffi.mlx_array_data_float32(arr._ctx);
      break;
    case "float64":
      result = ffi.mlx_array_data_float64(arr._ctx);
      break;
    case "int8":
      result = ffi.mlx_array_data_int8(arr._ctx);
      break;
    case "int16":
      result = ffi.mlx_array_data_int16(arr._ctx);
      break;
    case "int32":
      result = ffi.mlx_array_data_int32(arr._ctx);
      break;
    case "uint16":
      result = ffi.mlx_array_data_uint16(arr._ctx);
      break;
    case "uint32":
      result = ffi.mlx_array_data_uint32(arr._ctx);
      break;
    case "uint8":
    case "bool":
      result = ffi.mlx_array_data_uint8(arr._ctx);
      break;
  }

  return unwrapPointer(result, `mlx_array_data_${dtype}`);
}

function flattenArray(data: NumericArrayData): { flat: number[]; inferredShape: number[] } {
  const flat: number[] = [];
  return { flat, inferredShape: flattenInto(data, flat) };
}

function flattenInto(data: NumericArrayData | number, flat: number[]): number[] {
  if (typeof data === "number") {
    flat.push(data);
    return [];
  }

  if (data.length === 0) {
    return [0];
  }

  const firstShape = flattenInto(expectPresent(data[0], "nested array item"), flat);
  for (let index = 1; index < data.length; index++) {
    const currentShape = flattenInto(expectPresent(data[index], "nested array item"), flat);
    if (!shapesEqual(firstShape, currentShape)) {
      throw new Error("Nested array data must have a uniform shape");
    }
  }

  return [data.length, ...firstShape];
}

function shapesEqual(left: readonly number[], right: readonly number[]): boolean {
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

function reshapeFlat(flat: number[], shape: readonly number[]): NestedArray<number> {
  if (shape.length === 0) {
    return expectPresent(flat[0], "scalar reshape value");
  }

  if (shape.length === 1) {
    return flat;
  }

  const outerSize = expectPresent(shape[0], "outer shape dimension");
  const innerShape = shape.slice(1);
  const innerSize = innerShape.reduce((product, dimension) => product * dimension, 1);

  const result: NestedArray<number>[] = [];
  for (let index = 0; index < outerSize; index++) {
    result.push(reshapeFlat(flat.slice(index * innerSize, (index + 1) * innerSize), innerShape));
  }
  return result;
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

function expectPresent<T>(value: T | undefined, label: string): T {
  if (value === undefined) {
    throw new Error(`${label} was unexpectedly undefined`);
  }

  return value;
}
