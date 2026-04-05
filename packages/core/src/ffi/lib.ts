/**
 * Native library loading.
 *
 * Performs the single dlopen call that loads libmlxc.dylib and binds
 * all FFI symbols. Other modules import the `ffi` object from here.
 *
 * @module
 */

import { dlopen } from "bun:ffi";
import { resolve } from "path";

import {
  ARITHMETIC_SYMBOLS,
  ARRAY_LIFECYCLE_SYMBOLS,
  CLOSURE_SYMBOLS,
  COMPARISON_SYMBOLS,
  CREATION_SYMBOLS,
  DEVICE_SYMBOLS,
  ERROR_SYMBOLS,
  FAST_SYMBOLS,
  GRAD_TRANSFORM_SYMBOLS,
  LINALG_SYMBOLS,
  MAP_SYMBOLS,
  MEMORY_SYMBOLS,
  METAL_SYMBOLS,
  MLXTS_NATIVE_SYMBOLS,
  QUANTIZATION_SYMBOLS,
  RANDOM_SYMBOLS,
  REDUCTION_SYMBOLS,
  SHAPE_SYMBOLS,
  STREAM_SYMBOLS,
  STRING_SYMBOLS,
  TAKE_SYMBOLS,
  TRANSFORM_SYMBOLS,
  VECTOR_SYMBOLS,
} from "./symbols";

export const DYLIB_PATH = resolve(import.meta.dirname, "../../native/lib/libmlxc.dylib");
export const MLXTS_CORE_NATIVE_DYLIB_PATH = resolve(
  import.meta.dirname,
  "../../native/lib/libmlxts_core_native.dylib",
);

const lib = dlopen(DYLIB_PATH, {
  ...ERROR_SYMBOLS,
  ...ARRAY_LIFECYCLE_SYMBOLS,
  ...STRING_SYMBOLS,
  ...MAP_SYMBOLS,
  ...DEVICE_SYMBOLS,
  ...STREAM_SYMBOLS,
  ...VECTOR_SYMBOLS,
  ...ARITHMETIC_SYMBOLS,
  ...REDUCTION_SYMBOLS,
  ...SHAPE_SYMBOLS,
  ...LINALG_SYMBOLS,
  ...COMPARISON_SYMBOLS,
  ...CREATION_SYMBOLS,
  ...TAKE_SYMBOLS,
  ...QUANTIZATION_SYMBOLS,
  ...RANDOM_SYMBOLS,
  ...MEMORY_SYMBOLS,
  ...METAL_SYMBOLS,
  ...FAST_SYMBOLS,
  ...TRANSFORM_SYMBOLS,
  ...CLOSURE_SYMBOLS,
  ...GRAD_TRANSFORM_SYMBOLS,
});

const mlxtsCoreNativeLib = dlopen(MLXTS_CORE_NATIVE_DYLIB_PATH, {
  ...MLXTS_NATIVE_SYMBOLS,
});

/** All FFI symbols from libmlxc.dylib. */
export const ffi = {
  ...lib.symbols,
  ...mlxtsCoreNativeLib.symbols,
};
