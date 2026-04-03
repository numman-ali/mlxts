/**
 * Native library loading.
 *
 * Performs the single dlopen call that loads libmlxc.dylib and binds
 * all FFI symbols. Other modules import the `ffi` object from here.
 *
 * @module
 */

import { dlopen } from "bun:ffi";
import { resolve } from "node:path";

import {
  ARITHMETIC_SYMBOLS,
  ARRAY_LIFECYCLE_SYMBOLS,
  CLOSURE_SYMBOLS,
  COMPARISON_SYMBOLS,
  CREATION_SYMBOLS,
  DEVICE_SYMBOLS,
  ERROR_SYMBOLS,
  GRAD_TRANSFORM_SYMBOLS,
  LINALG_SYMBOLS,
  RANDOM_SYMBOLS,
  REDUCTION_SYMBOLS,
  SHAPE_SYMBOLS,
  STREAM_SYMBOLS,
  STRING_SYMBOLS,
  TAKE_SYMBOLS,
  TRANSFORM_SYMBOLS,
  VECTOR_SYMBOLS,
} from "./symbols";

export const DYLIB_PATH = resolve(import.meta.dirname, "../../../native/lib/libmlxc.dylib");

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
  ...TAKE_SYMBOLS,
  ...RANDOM_SYMBOLS,
  ...TRANSFORM_SYMBOLS,
  ...CLOSURE_SYMBOLS,
  ...GRAD_TRANSFORM_SYMBOLS,
});

/** All FFI symbols from libmlxc.dylib. */
export const ffi = lib.symbols;
