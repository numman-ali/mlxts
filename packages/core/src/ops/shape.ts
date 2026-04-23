/**
 * Shape manipulation operations.
 * @module
 */

import type { Pointer } from "bun:ffi";
import { MxArray, readResultArray, readResultArrayWithMetadata, readResultPointer } from "../array";
import { defaultStream } from "../device";
import { DTYPE_TO_MLX, type DType } from "../dtype";
import { checkStatus } from "../error";
import { ffi, OutSlot, ptr, sizeToNumber, unwrapPointer } from "../ffi";
import { coreRuntimeProfileTimestamp, recordFfiInvokeDuration } from "../runtime-profile";

type S = Pointer | undefined;
const s = (stream?: S) => stream ?? defaultStream();

function sliceExtent(start: number, stop: number, stride: number): number {
  if (stride <= 0) {
    throw new Error(`slice: strides must be positive integers, got ${stride}.`);
  }
  if (stop <= start) {
    return 0;
  }
  return Math.ceil((stop - start) / stride);
}

/** Reshape an array to a new shape. */
export function reshape(a: MxArray, shape: number[], stream?: S): MxArray {
  const shapeBuf = new Int32Array(shape);
  return readResultArrayWithMetadata("reshape", { shape, dtype: a.dtype }, (out) => {
    checkStatus(ffi.mlx_reshape(out, a._ctx, ptr(shapeBuf), shape.length, s(stream)), "reshape");
  });
}

/** Transpose an array. Without axes: reverses dimensions. With axes: permutes. */
export function transpose(a: MxArray, axes?: number[], stream?: S): MxArray {
  const sourceShape = a.shape;
  const shape =
    axes === undefined ? [...sourceShape].reverse() : axes.map((axis) => sourceShape[axis] ?? 0);
  return readResultArrayWithMetadata("transpose", { shape, dtype: a.dtype }, (out) => {
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

/** Repeat elements of an array either globally or along a specific axis. */
export function repeat(a: MxArray, repeats: number, axis?: number, stream?: S): MxArray {
  return readResultArray("repeat", (out) => {
    if (axis === undefined) {
      checkStatus(ffi.mlx_repeat(out, a._ctx, repeats, s(stream)), "repeat");
      return;
    }

    checkStatus(ffi.mlx_repeat_axis(out, a._ctx, repeats, axis, s(stream)), "repeat");
  });
}

/** Tile an array by repeating it according to the provided reps. */
export function tile(a: MxArray, reps: number | number[], stream?: S): MxArray {
  const normalizedReps = typeof reps === "number" ? [reps] : reps;
  const repsBuf = new Int32Array(normalizedReps);
  return readResultArray("tile", (out) => {
    checkStatus(ffi.mlx_tile(out, a._ctx, ptr(repsBuf), normalizedReps.length, s(stream)), "tile");
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

/** Scatter replacement values along an axis using explicit indices. */
export function putAlongAxis(
  a: MxArray,
  indices: MxArray,
  values: MxArray,
  axis: number,
  stream?: S,
): MxArray {
  return readResultArray("put_along_axis", (out) => {
    checkStatus(
      ffi.mlx_put_along_axis(out, a._ctx, indices._ctx, values._ctx, axis, s(stream)),
      "put_along_axis",
    );
  });
}

/** Replace values where a boolean mask is true using a flat source array. */
export function maskedScatter(a: MxArray, mask: MxArray, src: MxArray, stream?: S): MxArray {
  return readResultArray("masked_scatter", (out) => {
    checkStatus(
      ffi.mlx_masked_scatter(out, a._ctx, mask._ctx, src._ctx, s(stream)),
      "masked_scatter",
    );
  });
}

/** Select elements from an array along a specific axis. */
export function takeAxis(a: MxArray, indices: MxArray, axis: number, stream?: S): MxArray {
  return readResultArray("take_axis", (out) => {
    checkStatus(ffi.mlx_take_axis(out, a._ctx, indices._ctx, axis, s(stream)), "take_axis");
  });
}

/** Extract a strided slice view using explicit start/stop/stride triples. */
export function slice(
  a: MxArray,
  start: number[],
  stop: number[],
  strides?: number[],
  stream?: S,
): MxArray {
  const normalizedStrides = strides ?? Array.from({ length: start.length }, () => 1);
  if (start.length !== stop.length || start.length !== normalizedStrides.length) {
    throw new Error(
      `slice: start, stop, and strides must have the same rank, got ${start.length}, ${stop.length}, and ${normalizedStrides.length}.`,
    );
  }

  const startBuf = new Int32Array(start);
  const stopBuf = new Int32Array(stop);
  const strideBuf = new Int32Array(normalizedStrides);
  const shape = start.map((startIndex, axis) =>
    sliceExtent(startIndex, stop[axis] ?? startIndex, normalizedStrides[axis] ?? 1),
  );
  return readResultArrayWithMetadata("slice", { shape, dtype: a.dtype }, (out) => {
    checkStatus(
      ffi.mlx_slice(
        out,
        a._ctx,
        ptr(startBuf),
        start.length,
        ptr(stopBuf),
        stop.length,
        ptr(strideBuf),
        normalizedStrides.length,
        s(stream),
      ),
      "slice",
    );
  });
}

/** Extract a slice view whose start positions are provided dynamically. */
export function sliceDynamic(
  a: MxArray,
  start: MxArray,
  axes: number[],
  sliceSize: number[],
  stream?: S,
): MxArray {
  const normalizedSliceSize =
    sliceSize.length === a.shape.length
      ? sliceSize
      : (() => {
          if (sliceSize.length !== axes.length) {
            throw new Error(
              `sliceDynamic: sliceSize must describe either every axis or just the dynamic axes, got ${sliceSize.length} sizes for ${axes.length} axes on rank ${a.shape.length}.`,
            );
          }
          return a.shape.map((extent, index) => {
            const dynamicAxisIndex = axes.indexOf(index);
            return dynamicAxisIndex === -1 ? extent : (sliceSize[dynamicAxisIndex] ?? extent);
          });
        })();
  const axesBuf = new Int32Array(axes);
  const sliceSizeBuf = new Int32Array(normalizedSliceSize);
  return readResultArray("slice_dynamic", (out) => {
    checkStatus(
      ffi.mlx_slice_dynamic(
        out,
        a._ctx,
        start._ctx,
        ptr(axesBuf),
        axes.length,
        ptr(sliceSizeBuf),
        normalizedSliceSize.length,
        s(stream),
      ),
      "slice_dynamic",
    );
  });
}

/** Update a slice with a dynamic start index along one or more axes. */
export function sliceUpdate(
  src: MxArray,
  update: MxArray,
  start: number[],
  stop: number[],
  strides?: number[],
  stream?: S,
): MxArray {
  const normalizedStrides = strides ?? Array.from({ length: start.length }, () => 1);
  if (start.length !== stop.length || start.length !== normalizedStrides.length) {
    throw new Error(
      `sliceUpdate: start, stop, and strides must have the same rank, got ${start.length}, ${stop.length}, and ${normalizedStrides.length}.`,
    );
  }

  const startBuf = new Int32Array(start);
  const stopBuf = new Int32Array(stop);
  const strideBuf = new Int32Array(normalizedStrides);
  return readResultArrayWithMetadata(
    "slice_update",
    { shape: src.shape, dtype: src.dtype },
    (out) => {
      checkStatus(
        ffi.mlx_slice_update(
          out,
          src._ctx,
          update._ctx,
          ptr(startBuf),
          start.length,
          ptr(stopBuf),
          stop.length,
          ptr(strideBuf),
          normalizedStrides.length,
          s(stream),
        ),
        "slice_update",
      );
    },
  );
}

/** Mutate an existing array handle in place using slice-update semantics. */
export function sliceUpdateInPlace(
  src: MxArray,
  update: MxArray,
  start: number[],
  stop: number[],
  strides?: number[],
  stream?: S,
): void {
  const normalizedStrides = strides ?? Array.from({ length: start.length }, () => 1);
  if (start.length !== stop.length || start.length !== normalizedStrides.length) {
    throw new Error(
      `sliceUpdateInPlace: start, stop, and strides must have the same rank, got ${start.length}, ${stop.length}, and ${normalizedStrides.length}.`,
    );
  }

  const startBuf = new Int32Array(start);
  const stopBuf = new Int32Array(stop);
  const strideBuf = new Int32Array(normalizedStrides);
  const started = coreRuntimeProfileTimestamp();
  checkStatus(
    ffi.mlxts_slice_update_inplace(
      src._ctx,
      update._ctx,
      ptr(startBuf),
      start.length,
      ptr(stopBuf),
      stop.length,
      ptr(strideBuf),
      normalizedStrides.length,
      s(stream),
    ),
    "slice_update_inplace",
  );
  recordFfiInvokeDuration("slice_update_inplace", coreRuntimeProfileTimestamp() - started);
}

/** Update a slice with a dynamic start index along one or more axes. */
export function sliceUpdateDynamic(
  src: MxArray,
  update: MxArray,
  start: MxArray,
  axes: number[],
  stream?: S,
): MxArray {
  const axesBuf = new Int32Array(axes);
  return readResultArray("slice_update_dynamic", (out) => {
    checkStatus(
      ffi.mlx_slice_update_dynamic(
        out,
        src._ctx,
        update._ctx,
        start._ctx,
        ptr(axesBuf),
        axes.length,
        s(stream),
      ),
      "slice_update_dynamic",
    );
  });
}

/** Retarget an existing array handle to another array value in place. */
export function arrayAssignInPlace(target: MxArray, source: MxArray): void {
  const started = coreRuntimeProfileTimestamp();
  checkStatus(ffi.mlxts_array_assign_inplace(target._ctx, source._ctx), "array_assign_inplace");
  recordFfiInvokeDuration("array_assign_inplace", coreRuntimeProfileTimestamp() - started);
  target._replaceMetadata({
    shape: source.shape,
    dtype: source.dtype,
    ndim: source.ndim,
    size: source.size,
  });
}

/** Retarget an existing array handle to a slice view of another array in place. */
export function sliceViewInPlace(
  target: MxArray,
  source: MxArray,
  start: number[],
  stop: number[],
  strides?: number[],
  stream?: S,
): void {
  const normalizedStrides = strides ?? Array.from({ length: start.length }, () => 1);
  if (start.length !== stop.length || start.length !== normalizedStrides.length) {
    throw new Error(
      `sliceViewInPlace: start, stop, and strides must have the same rank, got ${start.length}, ${stop.length}, and ${normalizedStrides.length}.`,
    );
  }

  const startBuf = new Int32Array(start);
  const stopBuf = new Int32Array(stop);
  const strideBuf = new Int32Array(normalizedStrides);
  const shape = start.map((startIndex, axis) =>
    sliceExtent(startIndex, stop[axis] ?? startIndex, normalizedStrides[axis] ?? 1),
  );
  const started = coreRuntimeProfileTimestamp();
  checkStatus(
    ffi.mlxts_slice_view_inplace(
      target._ctx,
      source._ctx,
      ptr(startBuf),
      start.length,
      ptr(stopBuf),
      stop.length,
      ptr(strideBuf),
      normalizedStrides.length,
      s(stream),
    ),
    "slice_view_inplace",
  );
  recordFfiInvokeDuration("slice_view_inplace", coreRuntimeProfileTimestamp() - started);
  target._replaceMetadata({
    shape,
    dtype: source.dtype,
    ndim: shape.length,
    size: shape.reduce((product, dimension) => product * dimension, 1),
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
