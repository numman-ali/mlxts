/**
 * Shape manipulation operations.
 * @module
 */

import type { Pointer } from "bun:ffi";
import { MxArray, prepareOut, readOut } from "../array";
import { defaultStream } from "../device";
import { DTYPE_TO_MLX, type DType } from "../dtype";
import { checkStatus } from "../error";
import { ffi, ptr, unwrapPointer } from "../ffi";

type S = Pointer | undefined;
const s = (stream?: S) => stream ?? defaultStream();

/** Reshape an array to a new shape. */
export function reshape(a: MxArray, shape: number[], stream?: S): MxArray {
  const shapeBuf = new Int32Array(shape);
  const out = prepareOut();
  checkStatus(ffi.mlx_reshape(out, a._ctx, ptr(shapeBuf), shape.length, s(stream)), "reshape");
  return MxArray._fromCtx(readOut());
}

/** Transpose an array. Without axes: reverses dimensions. With axes: permutes. */
export function transpose(a: MxArray, axes?: number[], stream?: S): MxArray {
  const out = prepareOut();
  if (axes === undefined) {
    checkStatus(ffi.mlx_transpose(out, a._ctx, s(stream)), "transpose");
  } else {
    const axesBuf = new Int32Array(axes);
    checkStatus(
      ffi.mlx_transpose_axes(out, a._ctx, ptr(axesBuf), axes.length, s(stream)),
      "transpose",
    );
  }
  return MxArray._fromCtx(readOut());
}

/** Remove size-1 dimensions. */
export function squeeze(a: MxArray, axis?: number, stream?: S): MxArray {
  const out = prepareOut();
  if (axis === undefined) {
    checkStatus(ffi.mlx_squeeze(out, a._ctx, s(stream)), "squeeze");
  } else {
    checkStatus(ffi.mlx_squeeze_axis(out, a._ctx, axis, s(stream)), "squeeze");
  }
  return MxArray._fromCtx(readOut());
}

/** Add a size-1 dimension at the given axis. */
export function expandDims(a: MxArray, axis: number, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_expand_dims(out, a._ctx, axis, s(stream)), "expand_dims");
  return MxArray._fromCtx(readOut());
}

/** Broadcast an array to a target shape. */
export function broadcastTo(a: MxArray, shape: number[], stream?: S): MxArray {
  const shapeBuf = new Int32Array(shape);
  const out = prepareOut();
  checkStatus(
    ffi.mlx_broadcast_to(out, a._ctx, ptr(shapeBuf), shape.length, s(stream)),
    "broadcast_to",
  );
  return MxArray._fromCtx(readOut());
}

/** Cast an array to a different dtype. */
export function astype(a: MxArray, dtype: DType, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_astype(out, a._ctx, DTYPE_TO_MLX[dtype], s(stream)), "astype");
  return MxArray._fromCtx(readOut());
}

/** Flatten dimensions of an array. */
export function flatten(a: MxArray, startAxis = 0, endAxis = -1, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_flatten(out, a._ctx, startAxis, endAxis, s(stream)), "flatten");
  return MxArray._fromCtx(readOut());
}

/** Concatenate arrays along an axis. */
export function concatenate(arrays: MxArray[], axis = 0, stream?: S): MxArray {
  const vec = unwrapPointer(ffi.mlx_vector_array_new(), "mlx_vector_array_new");
  try {
    for (const arr of arrays) {
      checkStatus(ffi.mlx_vector_array_append_value(vec, arr._ctx), "concatenate_vec_append");
    }

    const out = prepareOut();
    checkStatus(ffi.mlx_concatenate_axis(out, vec, axis, s(stream)), "concatenate");
    return MxArray._fromCtx(readOut());
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

    const out = prepareOut();
    checkStatus(ffi.mlx_stack_axis(out, vec, axis, s(stream)), "stack");
    return MxArray._fromCtx(readOut());
  } finally {
    ffi.mlx_vector_array_free(vec);
  }
}

/** Gather elements along an axis. */
export function takeAlongAxis(a: MxArray, indices: MxArray, axis: number, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(
    ffi.mlx_take_along_axis(out, a._ctx, indices._ctx, axis, s(stream)),
    "take_along_axis",
  );
  return MxArray._fromCtx(readOut());
}

/** Select elements from an array along a specific axis. */
export function takeAxis(a: MxArray, indices: MxArray, axis: number, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_take_axis(out, a._ctx, indices._ctx, axis, s(stream)), "take_axis");
  return MxArray._fromCtx(readOut());
}

/** Stop gradient propagation. */
export function stopGradient(a: MxArray, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_stop_gradient(out, a._ctx, s(stream)), "stop_gradient");
  return MxArray._fromCtx(readOut());
}
