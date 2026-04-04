import type { Pointer } from "bun:ffi";
import type { MxArray } from "./array";
import type { DirectDataDType } from "./array-data";
import { ffi } from "./ffi/lib";
import { unwrapPointer } from "./ffi/pointer";

export function getDataPointer(arr: MxArray, dtype: DirectDataDType): Pointer {
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
