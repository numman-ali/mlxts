/**
 * Random number generation.
 *
 * MLX uses explicit key-based RNG (JAX-style), not implicit global state.
 * Each random function takes an optional key. If not provided, a module-level
 * default key is used and auto-split after each call.
 * @module
 */

import type { Pointer } from "bun:ffi";
import { MxArray, prepareOut, readOut } from "./array";
import { defaultStream } from "./device";
import { DTYPE_TO_MLX, type DType } from "./dtype";
import { checkStatus } from "./error";
import { ffi, ptr, readPtr, unwrapPointer } from "./ffi";

// Module-level default RNG key (auto-split on each use)
let _defaultKey: Pointer | null = null;

/** Initialize or get the module-level default RNG key. */
function getDefaultKey(): Pointer {
  if (_defaultKey === null) {
    const out = prepareOut();
    checkStatus(ffi.mlx_random_key(out, 0), "random_key_default");
    _defaultKey = readOut();
  }
  return _defaultKey;
}

/** Auto-split the default key and return one half for use. */
function useDefaultKey(): Pointer {
  const currentKey = getDefaultKey();

  // Split into two keys — each output buffer holds an mlx_array struct (8 bytes)
  const out0 = new Uint8Array(8);
  const out1 = new Uint8Array(8);
  checkStatus(
    ffi.mlx_random_split(ptr(out0), ptr(out1), currentKey, defaultStream()),
    "random_split_default",
  );

  const newDefault = unwrapPointer(readPtr(ptr(out0), 0), "random split default key");
  const forUse = unwrapPointer(readPtr(ptr(out1), 0), "random split result key");

  // Free old default key, keep the new one
  ffi.mlx_array_free(currentKey);
  _defaultKey = newDefault;

  return forUse;
}

/** Set the global random seed. */
export function seed(value: number): void {
  if (_defaultKey !== null) {
    ffi.mlx_array_free(_defaultKey);
    _defaultKey = null;
  }
  const out = prepareOut();
  checkStatus(ffi.mlx_random_key(out, value), "random_key");
  _defaultKey = readOut();
}

/** Create an explicit RNG key from a seed. */
export function key(seedValue: number): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_random_key(out, seedValue), "random_key");
  return MxArray._fromCtx(readOut());
}

/** Split an RNG key into two new keys. */
export function split(rngKey: MxArray): [MxArray, MxArray] {
  const out0 = new Uint8Array(8);
  const out1 = new Uint8Array(8);
  checkStatus(
    ffi.mlx_random_split(ptr(out0), ptr(out1), rngKey._ctx, defaultStream()),
    "random_split",
  );
  const k0Ptr = unwrapPointer(readPtr(ptr(out0), 0), "random split first key");
  const k1Ptr = unwrapPointer(readPtr(ptr(out1), 0), "random split second key");
  return [MxArray._fromCtx(k0Ptr), MxArray._fromCtx(k1Ptr)];
}

/** Generate random values from a normal distribution. */
export function normal(
  shape: number[],
  dtype: DType = "float32",
  loc = 0.0,
  scale = 1.0,
  rngKey?: MxArray,
): MxArray {
  const shapeBuf = new Int32Array(shape);
  const ownKey = rngKey === undefined;
  const keyCtx = rngKey?._ctx ?? useDefaultKey();
  const out = prepareOut();
  try {
    checkStatus(
      ffi.mlx_random_normal(
        out,
        ptr(shapeBuf),
        shape.length,
        DTYPE_TO_MLX[dtype],
        loc,
        scale,
        keyCtx,
        defaultStream(),
      ),
      "random_normal",
    );
    return MxArray._fromCtx(readOut());
  } finally {
    if (ownKey) {
      ffi.mlx_array_free(keyCtx);
    }
  }
}

/** Generate random values from a uniform distribution. */
export function uniform(
  low: number,
  high: number,
  shape: number[],
  dtype: DType = "float32",
  rngKey?: MxArray,
): MxArray {
  const shapeBuf = new Int32Array(shape);
  const ownKey = rngKey === undefined;
  const keyCtx = rngKey?._ctx ?? useDefaultKey();
  const lowArr = unwrapPointer(ffi.mlx_array_new_float(low), "mlx_array_new_float(low)");
  const highArr = unwrapPointer(ffi.mlx_array_new_float(high), "mlx_array_new_float(high)");
  const out = prepareOut();
  try {
    checkStatus(
      ffi.mlx_random_uniform(
        out,
        lowArr,
        highArr,
        ptr(shapeBuf),
        shape.length,
        DTYPE_TO_MLX[dtype],
        keyCtx,
        defaultStream(),
      ),
      "random_uniform",
    );
    return MxArray._fromCtx(readOut());
  } finally {
    ffi.mlx_array_free(lowArr);
    ffi.mlx_array_free(highArr);
    if (ownKey) {
      ffi.mlx_array_free(keyCtx);
    }
  }
}

/** Generate random Bernoulli samples. */
export function bernoulli(p: number, shape: number[], rngKey?: MxArray): MxArray {
  const shapeBuf = new Int32Array(shape);
  const ownKey = rngKey === undefined;
  const keyCtx = rngKey?._ctx ?? useDefaultKey();
  const pArr = unwrapPointer(ffi.mlx_array_new_float(p), "mlx_array_new_float(p)");
  const out = prepareOut();
  try {
    checkStatus(
      ffi.mlx_random_bernoulli(out, pArr, ptr(shapeBuf), shape.length, keyCtx, defaultStream()),
      "random_bernoulli",
    );
    return MxArray._fromCtx(readOut());
  } finally {
    ffi.mlx_array_free(pArr);
    if (ownKey) {
      ffi.mlx_array_free(keyCtx);
    }
  }
}
