/**
 * FFI bindings to libmlxc.dylib.
 *
 * This is the sole point of contact with the native library. Every mlx-c
 * function used by mlx-ts is declared here and loaded via a single dlopen call.
 *
 * Conventions (from mlx-c headers):
 * - Creation functions return `mlx_array` by value → FFIType.ptr
 * - Operations write results via `mlx_array*` (first arg) → FFIType.ptr to an 8-byte buffer
 * - All operations return int (0 = success) → FFIType.i32
 * - Property getters return values directly (size_t, const int*, etc.)
 * - `mlx_array` by value is a struct { void* ctx } — on ARM64, just a pointer
 *
 * @module
 */

import type { Pointer } from "bun:ffi";
import {
  ptr as bunPtr,
  read as bunRead,
  toArrayBuffer as bunToArrayBuffer,
  CString,
  dlopen,
  FFIType,
} from "bun:ffi";
import { resolve } from "node:path";

const DYLIB_PATH = resolve(import.meta.dirname, "../../native/lib/libmlxc.dylib");

// --- Type shorthands for readability ---
const {
  ptr: P,
  i32: I32,
  f32: F32,
  f64: F64,
  bool: BOOL,
  void: VOID,
  cstring: CSTRING,
  u64_fast: U64_FAST,
} = FFIType;

// --- Symbol groups ---

const ERROR_SYMBOLS = {
  // void mlx_set_error_handler(handler, data, dtor)
  mlx_set_error_handler: { args: [P, P, P], returns: VOID },
} as const;

const ARRAY_LIFECYCLE_SYMBOLS = {
  // Returns mlx_array by value (ctx pointer)
  mlx_array_new: { args: [], returns: P },
  mlx_array_new_data: { args: [P, P, I32, I32], returns: P },
  mlx_array_new_bool: { args: [BOOL], returns: P },
  mlx_array_new_int: { args: [I32], returns: P },
  mlx_array_new_float32: { args: [F32], returns: P },
  mlx_array_new_float: { args: [F32], returns: P },
  mlx_array_new_float64: { args: [F64], returns: P },

  // int mlx_array_set(mlx_array* arr, const mlx_array src)
  mlx_array_set: { args: [P, P], returns: I32 },

  // int mlx_array_free(mlx_array arr)
  mlx_array_free: { args: [P], returns: I32 },

  // int mlx_array_eval(mlx_array arr)
  mlx_array_eval: { args: [P], returns: I32 },

  // Property getters — return directly, no output pointer
  mlx_array_ndim: { args: [P], returns: U64_FAST },
  mlx_array_shape: { args: [P], returns: P },
  mlx_array_dtype: { args: [P], returns: I32 },
  mlx_array_size: { args: [P], returns: U64_FAST },
  mlx_array_itemsize: { args: [P], returns: U64_FAST },
  mlx_array_nbytes: { args: [P], returns: U64_FAST },

  // Scalar item extraction (output pointer for the value)
  mlx_array_item_bool: { args: [P, P], returns: I32 },
  mlx_array_item_uint8: { args: [P, P], returns: I32 },
  mlx_array_item_uint16: { args: [P, P], returns: I32 },
  mlx_array_item_uint32: { args: [P, P], returns: I32 },
  mlx_array_item_int8: { args: [P, P], returns: I32 },
  mlx_array_item_int16: { args: [P, P], returns: I32 },
  mlx_array_item_int32: { args: [P, P], returns: I32 },
  mlx_array_item_float32: { args: [P, P], returns: I32 },
  mlx_array_item_float64: { args: [P, P], returns: I32 },

  // Data pointer access (returns pointer to raw data, array must be evaluated)
  mlx_array_data_bool: { args: [P], returns: P },
  mlx_array_data_uint8: { args: [P], returns: P },
  mlx_array_data_uint16: { args: [P], returns: P },
  mlx_array_data_uint32: { args: [P], returns: P },
  mlx_array_data_int8: { args: [P], returns: P },
  mlx_array_data_int16: { args: [P], returns: P },
  mlx_array_data_int32: { args: [P], returns: P },
  mlx_array_data_int64: { args: [P], returns: P },
  mlx_array_data_float16: { args: [P], returns: P },
  mlx_array_data_float32: { args: [P], returns: P },
  mlx_array_data_float64: { args: [P], returns: P },

  // String representation
  mlx_array_tostring: { args: [P, P], returns: I32 },
} as const;

const STRING_SYMBOLS = {
  mlx_string_new: { args: [], returns: P },
  mlx_string_free: { args: [P], returns: I32 },
  mlx_string_data: { args: [P], returns: CSTRING },
} as const;

const DEVICE_SYMBOLS = {
  mlx_device_new: { args: [], returns: P },
  mlx_device_new_type: { args: [I32, I32], returns: P },
  mlx_device_free: { args: [P], returns: I32 },
  mlx_get_default_device: { args: [P], returns: I32 },
  mlx_set_default_device: { args: [P], returns: I32 },
} as const;

const STREAM_SYMBOLS = {
  mlx_stream_new: { args: [], returns: P },
  mlx_stream_free: { args: [P], returns: I32 },
  mlx_default_cpu_stream_new: { args: [], returns: P },
  mlx_default_gpu_stream_new: { args: [], returns: P },
  mlx_get_default_stream: { args: [P, P], returns: I32 },
  mlx_set_default_stream: { args: [P], returns: I32 },
  mlx_synchronize: { args: [P], returns: I32 },
} as const;

const VECTOR_SYMBOLS = {
  mlx_vector_array_new: { args: [], returns: P },
  mlx_vector_array_new_data: { args: [P, U64_FAST], returns: P },
  mlx_vector_array_new_value: { args: [P], returns: P },
  mlx_vector_array_free: { args: [P], returns: I32 },
  mlx_vector_array_append_value: { args: [P, P], returns: I32 },
  mlx_vector_array_size: { args: [P], returns: U64_FAST },
  mlx_vector_array_get: { args: [P, P, U64_FAST], returns: I32 },
} as const;

const ARITHMETIC_SYMBOLS = {
  // int mlx_op(mlx_array* res, const mlx_array a, [const mlx_array b,] const mlx_stream s)
  mlx_add: { args: [P, P, P, P], returns: I32 },
  mlx_subtract: { args: [P, P, P, P], returns: I32 },
  mlx_multiply: { args: [P, P, P, P], returns: I32 },
  mlx_divide: { args: [P, P, P, P], returns: I32 },
  mlx_power: { args: [P, P, P, P], returns: I32 },
  mlx_maximum: { args: [P, P, P, P], returns: I32 },
  mlx_minimum: { args: [P, P, P, P], returns: I32 },
  mlx_negative: { args: [P, P, P], returns: I32 },
  mlx_abs: { args: [P, P, P], returns: I32 },
  mlx_sqrt: { args: [P, P, P], returns: I32 },
  mlx_square: { args: [P, P, P], returns: I32 },
  mlx_exp: { args: [P, P, P], returns: I32 },
  mlx_log: { args: [P, P, P], returns: I32 },
  mlx_sigmoid: { args: [P, P, P], returns: I32 },
  mlx_erf: { args: [P, P, P], returns: I32 },
  mlx_reciprocal: { args: [P, P, P], returns: I32 },
  mlx_floor: { args: [P, P, P], returns: I32 },
  mlx_ceil: { args: [P, P, P], returns: I32 },
  mlx_tanh: { args: [P, P, P], returns: I32 },
  mlx_sin: { args: [P, P, P], returns: I32 },
  mlx_cos: { args: [P, P, P], returns: I32 },
} as const;

const REDUCTION_SYMBOLS = {
  // Full reduction: int mlx_sum(res, a, keepdims, stream)
  mlx_sum: { args: [P, P, BOOL, P], returns: I32 },
  mlx_sum_axis: { args: [P, P, I32, BOOL, P], returns: I32 },
  mlx_sum_axes: { args: [P, P, P, U64_FAST, BOOL, P], returns: I32 },
  mlx_mean: { args: [P, P, BOOL, P], returns: I32 },
  mlx_mean_axis: { args: [P, P, I32, BOOL, P], returns: I32 },
  mlx_mean_axes: { args: [P, P, P, U64_FAST, BOOL, P], returns: I32 },
  mlx_max: { args: [P, P, BOOL, P], returns: I32 },
  mlx_max_axis: { args: [P, P, I32, BOOL, P], returns: I32 },
  mlx_max_axes: { args: [P, P, P, U64_FAST, BOOL, P], returns: I32 },
  mlx_min: { args: [P, P, BOOL, P], returns: I32 },
  mlx_min_axis: { args: [P, P, I32, BOOL, P], returns: I32 },
  mlx_min_axes: { args: [P, P, P, U64_FAST, BOOL, P], returns: I32 },
  mlx_argmax: { args: [P, P, BOOL, P], returns: I32 },
  mlx_argmax_axis: { args: [P, P, I32, BOOL, P], returns: I32 },
  mlx_argmin: { args: [P, P, BOOL, P], returns: I32 },
  mlx_argmin_axis: { args: [P, P, I32, BOOL, P], returns: I32 },
  mlx_logsumexp: { args: [P, P, BOOL, P], returns: I32 },
  mlx_logsumexp_axis: { args: [P, P, I32, BOOL, P], returns: I32 },
  mlx_logsumexp_axes: { args: [P, P, P, U64_FAST, BOOL, P], returns: I32 },
} as const;

const SHAPE_SYMBOLS = {
  // int mlx_reshape(res, a, shape, shape_num, stream)
  mlx_reshape: { args: [P, P, P, U64_FAST, P], returns: I32 },
  mlx_transpose: { args: [P, P, P], returns: I32 },
  mlx_transpose_axes: { args: [P, P, P, U64_FAST, P], returns: I32 },
  mlx_squeeze: { args: [P, P, P], returns: I32 },
  mlx_squeeze_axis: { args: [P, P, I32, P], returns: I32 },
  mlx_expand_dims: { args: [P, P, I32, P], returns: I32 },
  mlx_broadcast_to: { args: [P, P, P, U64_FAST, P], returns: I32 },
  mlx_astype: { args: [P, P, I32, P], returns: I32 },
  mlx_concatenate_axis: { args: [P, P, I32, P], returns: I32 },
  mlx_concatenate: { args: [P, P, P], returns: I32 },
  mlx_stack_axis: { args: [P, P, I32, P], returns: I32 },
  mlx_stack: { args: [P, P, P], returns: I32 },
  mlx_split: { args: [P, P, I32, I32, P], returns: I32 },
  mlx_flatten: { args: [P, P, I32, I32, P], returns: I32 },
  mlx_contiguous: { args: [P, P, BOOL, P], returns: I32 },
  mlx_stop_gradient: { args: [P, P, P], returns: I32 },
} as const;

const LINALG_SYMBOLS = {
  mlx_matmul: { args: [P, P, P, P], returns: I32 },
} as const;

const COMPARISON_SYMBOLS = {
  mlx_equal: { args: [P, P, P, P], returns: I32 },
  mlx_not_equal: { args: [P, P, P, P], returns: I32 },
  mlx_greater: { args: [P, P, P, P], returns: I32 },
  mlx_greater_equal: { args: [P, P, P, P], returns: I32 },
  mlx_less: { args: [P, P, P, P], returns: I32 },
  mlx_less_equal: { args: [P, P, P, P], returns: I32 },
  mlx_where: { args: [P, P, P, P, P], returns: I32 },
} as const;

const CREATION_SYMBOLS = {
  // int mlx_zeros(res, shape, shape_num, dtype, stream)
  mlx_zeros: { args: [P, P, U64_FAST, I32, P], returns: I32 },
  mlx_ones: { args: [P, P, U64_FAST, I32, P], returns: I32 },
  mlx_full: { args: [P, P, U64_FAST, P, I32, P], returns: I32 },
  mlx_arange: { args: [P, F64, F64, F64, I32, P], returns: I32 },
  mlx_softmax_axis: { args: [P, P, I32, BOOL, P], returns: I32 },
  mlx_softmax: { args: [P, P, BOOL, P], returns: I32 },
  mlx_take_along_axis: { args: [P, P, P, I32, P], returns: I32 },
} as const;

const RANDOM_SYMBOLS = {
  mlx_random_seed: { args: [U64_FAST], returns: I32 },
  mlx_random_key: { args: [P, U64_FAST], returns: I32 },
  // mlx_random_split(res_0, res_1, key, stream)
  mlx_random_split: { args: [P, P, P, P], returns: I32 },
  mlx_random_split_num: { args: [P, P, I32, P], returns: I32 },
  // mlx_random_normal(res, shape, shape_num, dtype, loc, scale, key, stream)
  mlx_random_normal: { args: [P, P, U64_FAST, I32, F32, F32, P, P], returns: I32 },
  // mlx_random_uniform(res, low, high, shape, shape_num, dtype, key, stream)
  mlx_random_uniform: { args: [P, P, P, P, U64_FAST, I32, P, P], returns: I32 },
  // mlx_random_bernoulli(res, p, shape, shape_num, key, stream)
  mlx_random_bernoulli: { args: [P, P, P, U64_FAST, P, P], returns: I32 },
} as const;

const TRANSFORM_SYMBOLS = {
  // int mlx_eval(const mlx_vector_array outputs)
  mlx_eval: { args: [P], returns: I32 },
  mlx_async_eval: { args: [P], returns: I32 },
} as const;

// --- Load library ---

const lib = dlopen(DYLIB_PATH, {
  ...ERROR_SYMBOLS,
  ...ARRAY_LIFECYCLE_SYMBOLS,
  ...STRING_SYMBOLS,
  ...DEVICE_SYMBOLS,
  ...STREAM_SYMBOLS,
  ...VECTOR_SYMBOLS,
  ...ARITHMETIC_SYMBOLS,
  ...REDUCTION_SYMBOLS,
  ...SHAPE_SYMBOLS,
  ...LINALG_SYMBOLS,
  ...COMPARISON_SYMBOLS,
  ...CREATION_SYMBOLS,
  ...RANDOM_SYMBOLS,
  ...TRANSFORM_SYMBOLS,
});

/** All FFI symbols from libmlxc.dylib. */
export const ffi = lib.symbols;

// ---------------------------------------------------------------------------
// Pointer utilities — re-exports with proper types
// ---------------------------------------------------------------------------
// We re-export Bun's FFI utilities using their native Pointer type throughout.
// Numeric addresses returned by Bun are narrowed back to Pointer via a local
// type guard so the brand does not leak through the rest of the codebase.
// ---------------------------------------------------------------------------

export type { Pointer };

type NativeBufferView = NodeJS.TypedArray | ArrayBufferLike | DataView;

/** Get a native pointer from a TypedArray. */
export function ptr(view: NativeBufferView, byteOffset?: number): Pointer {
  return bunPtr(view, byteOffset);
}

/** Read a pointer-sized value from native memory. */
export function readPtr(address: Pointer, byteOffset: number): Pointer | null {
  const rawAddress = bunRead.ptr(address, byteOffset);
  return isPointer(rawAddress) ? rawAddress : null;
}

/** Read an i32 from native memory. */
export function readI32(address: Pointer, byteOffset: number): number {
  return bunRead.i32(address, byteOffset);
}

/** Create an ArrayBuffer viewing native memory. */
export function nativeSlice(address: Pointer, byteOffset: number, byteLength: number): ArrayBuffer {
  return bunToArrayBuffer(address, byteOffset, byteLength);
}

/** Convert a C string pointer to a JS string. */
export function readCString(pointer: Pointer): string {
  return new CString(pointer).toString();
}

/** Require a non-null native pointer from an FFI call. */
export function unwrapPointer(pointer: Pointer | null, source: string): Pointer {
  if (pointer === null) {
    throw new Error(`${source} returned a null pointer`);
  }
  return pointer;
}

/** Convert a size_t-like return value to a safe JavaScript number. */
export function sizeToNumber(value: number | bigint, source: string): number {
  if (typeof value === "number") {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new RangeError(`${source} returned ${value}, which is not a safe non-negative integer`);
    }
    return value;
  }

  if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new RangeError(`${source} returned ${value}, which exceeds JS safe integer range`);
  }

  return Number(value);
}

function isPointer(address: number): address is Pointer {
  return address !== 0;
}
