import type { MxArray } from "./array";
import { type DirectTypedArray, expectPresent } from "./array-data";
import { getDataPointer } from "./array-ffi-data";
import { DTYPE_BYTE_SIZE } from "./dtype";
import { nativeSlice } from "./ffi/pointer";

export function copyTypedArrayFromContiguous(contiguous: MxArray): DirectTypedArray {
  contiguous.eval();

  const dtype = contiguous.dtype;
  const size = contiguous.size;

  switch (dtype) {
    case "float32":
      return new Float32Array(nativeSlice(getDataPointer(contiguous, dtype), 0, size * 4).slice(0));
    case "float64":
      return new Float64Array(nativeSlice(getDataPointer(contiguous, dtype), 0, size * 8).slice(0));
    case "int8":
      return new Int8Array(nativeSlice(getDataPointer(contiguous, dtype), 0, size).slice(0));
    case "int16":
      return new Int16Array(nativeSlice(getDataPointer(contiguous, dtype), 0, size * 2).slice(0));
    case "int32":
      return new Int32Array(nativeSlice(getDataPointer(contiguous, dtype), 0, size * 4).slice(0));
    case "uint8":
    case "bool":
      return new Uint8Array(
        nativeSlice(getDataPointer(contiguous, dtype), 0, size * DTYPE_BYTE_SIZE[dtype]).slice(0),
      );
    case "uint16":
      return new Uint16Array(nativeSlice(getDataPointer(contiguous, dtype), 0, size * 2).slice(0));
    case "uint32":
      return new Uint32Array(nativeSlice(getDataPointer(contiguous, dtype), 0, size * 4).slice(0));
    default: {
      const asFloat32 = contiguous.asType("float32");
      try {
        return copyTypedArrayFromContiguous(asFloat32);
      } finally {
        asFloat32.free();
      }
    }
  }
}

export function scalarFromTypedArrayCopy(array: DirectTypedArray): number {
  return Number(expectPresent(array[0], "scalar array value"));
}
