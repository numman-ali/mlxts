/**
 * Shape manipulation operations.
 * @module
 */

import type { Pointer } from "bun:ffi";
import { MxArray, readResultArray, readResultPointer } from "../array";
import { defaultStream } from "../device";
import { DTYPE_TO_MLX, type DType } from "../dtype";
import { checkStatus } from "../error";
import { ffi, OutSlot, ptr, sizeToNumber, unwrapPointer } from "../ffi";

type S = Pointer | undefined;
const s = (stream?: S) => stream ?? defaultStream();

/** Reshape an array to a new shape. */
export function reshape(a: MxArray, shape: number[], stream?: S): MxArray {
  const shapeBuf = new Int32Array(shape);
  return readResultArray("reshape", (out) => {
    checkStatus(ffi.mlx_reshape(out, a._ctx, ptr(shapeBuf), shape.length, s(stream)), "reshape");
  });
}

/** Transpose an array. Without axes: reverses dimensions. With axes: permutes. */
export function transpose(a: MxArray, axes?: number[], stream?: S): MxArray {
  return readResultArray("transpose", (out) => {
    if (axes === undefined) {
      checkStatus(ffi.mlx_transpose(out, a._ctx, s(stream)), "transpose");
    } else {
      const axesBuf = new Int32Array(axes);
      checkStatus(
        ffi.mlx_transpose_axes(out, a._ctx, ptr(axesBuf), axes.length, s(stream)),
        "transpose",
      );
    }
  });
}

/** Remove size-1 dimensions. */
export function squeeze(a: MxArray, axis?: number, stream?: S): MxArray {
  return readResultArray("squeeze", (out) => {
    if (axis === undefined) {
      checkStatus(ffi.mlx_squeeze(out, a._ctx, s(stream)), "squeeze");
    } else {
      checkStatus(ffi.mlx_squeeze_axis(out, a._ctx, axis, s(stream)), "squeeze");
    }
  });
}

/** Add a size-1 dimension at the given axis. */
export function expandDims(a: MxArray, axis: number, stream?: S): MxArray {
  return readResultArray("expand_dims", (out) => {
    checkStatus(ffi.mlx_expand_dims(out, a._ctx, axis, s(stream)), "expand_dims");
  });
}

/** Broadcast an array to a target shape. */
export function broadcastTo(a: MxArray, shape: number[], stream?: S): MxArray {
  const shapeBuf = new Int32Array(shape);
  return readResultArray("broadcast_to", (out) => {
    checkStatus(
      ffi.mlx_broadcast_to(out, a._ctx, ptr(shapeBuf), shape.length, s(stream)),
      "broadcast_to",
    );
  });
}

/** Cast an array to a different dtype. */
export function asType(a: MxArray, dtype: DType, stream?: S): MxArray {
  return readResultArray("astype", (out) => {
    checkStatus(ffi.mlx_astype(out, a._ctx, DTYPE_TO_MLX[dtype], s(stream)), "astype");
  });
}

/** Flatten dimensions of an array. */
export function flatten(a: MxArray, startAxis = 0, endAxis = -1, stream?: S): MxArray {
  return readResultArray("flatten", (out) => {
    checkStatus(ffi.mlx_flatten(out, a._ctx, startAxis, endAxis, s(stream)), "flatten");
  });
}

/** Concatenate arrays along an axis. */
export function concatenate(arrays: MxArray[], axis = 0, stream?: S): MxArray {
  const vec = unwrapPointer(ffi.mlx_vector_array_new(), "mlx_vector_array_new");
  try {
    for (const arr of arrays) {
      checkStatus(ffi.mlx_vector_array_append_value(vec, arr._ctx), "concatenate_vec_append");
    }

    return readResultArray("concatenate", (out) => {
      checkStatus(ffi.mlx_concatenate_axis(out, vec, axis, s(stream)), "concatenate");
    });
  } finally {
    ffi.mlx_vector_array_free(vec);
  }
}

/** Stack arrays along a new axis. */
export function stack(arrays: MxArray[], axis = 0, stream?: S): MxArray {
  const vec = unwrapPointer(ffi.mlx_vector_array_new(), "mlx_vector_array_new");
  try {
    for (const arr of arrays) {
      checkStatus(ffi.mlx_vector_array_append_value(vec, arr._ctx), "stack_vec_append");
    }

    return readResultArray("stack", (out) => {
      checkStatus(ffi.mlx_stack_axis(out, vec, axis, s(stream)), "stack");
    });
  } finally {
    ffi.mlx_vector_array_free(vec);
  }
}

/** Gather elements along an axis. */
export function takeAlongAxis(a: MxArray, indices: MxArray, axis: number, stream?: S): MxArray {
  return readResultArray("take_along_axis", (out) => {
    checkStatus(
      ffi.mlx_take_along_axis(out, a._ctx, indices._ctx, axis, s(stream)),
      "take_along_axis",
    );
  });
}

/** Select elements from an array along a specific axis. */
export function takeAxis(a: MxArray, indices: MxArray, axis: number, stream?: S): MxArray {
  return readResultArray("take_axis", (out) => {
    checkStatus(ffi.mlx_take_axis(out, a._ctx, indices._ctx, axis, s(stream)), "take_axis");
  });
}

/** Stop gradient propagation. */
export function stopGradient(a: MxArray, stream?: S): MxArray {
  return readResultArray("stop_gradient", (out) => {
    checkStatus(ffi.mlx_stop_gradient(out, a._ctx, s(stream)), "stop_gradient");
  });
}

/** Extract lower triangle of a 2D array (or last two dims of higher-rank). */
export function tril(a: MxArray, k = 0, stream?: S): MxArray {
  return readResultArray("tril", (out) => {
    checkStatus(ffi.mlx_tril(out, a._ctx, k, s(stream)), "tril");
  });
}

/** Extract upper triangle of a 2D array (or last two dims of higher-rank). */
export function triu(a: MxArray, k = 0, stream?: S): MxArray {
  return readResultArray("triu", (out) => {
    checkStatus(ffi.mlx_triu(out, a._ctx, k, s(stream)), "triu");
  });
}

/** Split an array into equal parts along an axis. */
export function split(a: MxArray, numSplits: number, axis = 0, stream?: S): MxArray[] {
  const vec = readResultPointer("split outputs", (out) => {
    checkStatus(ffi.mlx_split(out, a._ctx, numSplits, axis, s(stream)), "split");
  });

  try {
    const count = sizeToNumber(ffi.mlx_vector_array_size(vec), "split_vec_size");
    const results: MxArray[] = [];
    const slot = new OutSlot();
    for (let i = 0; i < count; i++) {
      checkStatus(ffi.mlx_vector_array_get(slot.prepare(), vec, i), `split_get_${i}`);
      results.push(MxArray._fromCtx(slot.read(`split result ${i}`)));
    }
    return results;
  } finally {
    ffi.mlx_vector_array_free(vec);
  }
}
