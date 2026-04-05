/**
 * ABI helpers for mlx-c optional scalar structs passed by value.
 *
 * mlx_optional_* values are tiny POD structs `{ value, has_value }`.
 * On Apple Silicon they fit in a single 64-bit general-purpose register,
 * so we pack them into a little-endian u64 payload for Bun FFI.
 *
 * @module
 */

import { DTYPE_TO_MLX, type DType } from "../dtype";

function packOptionalValue(write: (view: DataView) => void): bigint {
  const buffer = new ArrayBuffer(8);
  const view = new DataView(buffer);
  write(view);
  return view.getBigUint64(0, true);
}

/** Pack an `mlx_optional_int` value. */
export function optionalInt(value: number | undefined): bigint {
  return packOptionalValue((view) => {
    if (value === undefined) {
      return;
    }
    view.setInt32(0, value, true);
    view.setUint8(4, 1);
  });
}

/** Pack an `mlx_optional_float` value. */
export function optionalFloat(value: number | undefined): bigint {
  return packOptionalValue((view) => {
    if (value === undefined) {
      return;
    }
    view.setFloat32(0, value, true);
    view.setUint8(4, 1);
  });
}

/** Pack an `mlx_optional_dtype` value. */
export function optionalDType(value: DType | undefined): bigint {
  return packOptionalValue((view) => {
    if (value === undefined) {
      return;
    }
    view.setInt32(0, DTYPE_TO_MLX[value], true);
    view.setUint8(4, 1);
  });
}
