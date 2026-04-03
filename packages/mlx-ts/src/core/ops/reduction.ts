/**
 * Reduction operations (sum, mean, max, min, argmax, argmin, logsumexp).
 * @module
 */

import type { Pointer } from "bun:ffi";
import { MxArray, prepareOut, readOut } from "../array";
import { defaultStream } from "../device";
import { checkStatus } from "../error";
import { ffi, ptr } from "../ffi";

type S = Pointer | undefined;
const s = (stream?: S) => stream ?? defaultStream();

/**
 * Dispatch a reduction to the right mlx-c variant based on axis type.
 * mlx-c has 3 variants per reduction: _all, _axis, _axes.
 */
function reduceDispatch(
  fnAll: (out: Pointer, a: Pointer, keepdims: boolean, stream: Pointer) => number,
  fnAxis: (out: Pointer, a: Pointer, axis: number, keepdims: boolean, stream: Pointer) => number,
  fnAxes: (
    out: Pointer,
    a: Pointer,
    axes: Pointer,
    axesNum: number,
    keepdims: boolean,
    stream: Pointer,
  ) => number,
  a: MxArray,
  axis: number | number[] | undefined,
  keepdims: boolean,
  stream: S,
  name: string,
): MxArray {
  const out = prepareOut();

  if (axis === undefined) {
    checkStatus(fnAll(out, a._ctx, keepdims, s(stream)), name);
  } else if (typeof axis === "number") {
    checkStatus(fnAxis(out, a._ctx, axis, keepdims, s(stream)), name);
  } else {
    const axesBuf = new Int32Array(axis);
    checkStatus(fnAxes(out, a._ctx, ptr(axesBuf), axis.length, keepdims, s(stream)), name);
  }

  return MxArray._fromCtx(readOut());
}

/** Sum reduction. */
export function sum(a: MxArray, axis?: number | number[], keepdims = false, stream?: S): MxArray {
  return reduceDispatch(
    (o, arr, kd, st) => ffi.mlx_sum(o, arr, kd, st),
    (o, arr, ax, kd, st) => ffi.mlx_sum_axis(o, arr, ax, kd, st),
    (o, arr, ax, n, kd, st) => ffi.mlx_sum_axes(o, arr, ax, n, kd, st),
    a,
    axis,
    keepdims,
    stream,
    "sum",
  );
}

/** Mean reduction. */
export function mean(a: MxArray, axis?: number | number[], keepdims = false, stream?: S): MxArray {
  return reduceDispatch(
    (o, arr, kd, st) => ffi.mlx_mean(o, arr, kd, st),
    (o, arr, ax, kd, st) => ffi.mlx_mean_axis(o, arr, ax, kd, st),
    (o, arr, ax, n, kd, st) => ffi.mlx_mean_axes(o, arr, ax, n, kd, st),
    a,
    axis,
    keepdims,
    stream,
    "mean",
  );
}

/** Max reduction. */
export function max(a: MxArray, axis?: number | number[], keepdims = false, stream?: S): MxArray {
  return reduceDispatch(
    (o, arr, kd, st) => ffi.mlx_max(o, arr, kd, st),
    (o, arr, ax, kd, st) => ffi.mlx_max_axis(o, arr, ax, kd, st),
    (o, arr, ax, n, kd, st) => ffi.mlx_max_axes(o, arr, ax, n, kd, st),
    a,
    axis,
    keepdims,
    stream,
    "max",
  );
}

/** Min reduction. */
export function min(a: MxArray, axis?: number | number[], keepdims = false, stream?: S): MxArray {
  return reduceDispatch(
    (o, arr, kd, st) => ffi.mlx_min(o, arr, kd, st),
    (o, arr, ax, kd, st) => ffi.mlx_min_axis(o, arr, ax, kd, st),
    (o, arr, ax, n, kd, st) => ffi.mlx_min_axes(o, arr, ax, n, kd, st),
    a,
    axis,
    keepdims,
    stream,
    "min",
  );
}

/** Index of maximum value. */
export function argmax(a: MxArray, axis?: number, keepdims = false, stream?: S): MxArray {
  const out = prepareOut();
  if (axis === undefined) {
    checkStatus(ffi.mlx_argmax(out, a._ctx, keepdims, s(stream)), "argmax");
  } else {
    checkStatus(ffi.mlx_argmax_axis(out, a._ctx, axis, keepdims, s(stream)), "argmax");
  }
  return MxArray._fromCtx(readOut());
}

/** Index of minimum value. */
export function argmin(a: MxArray, axis?: number, keepdims = false, stream?: S): MxArray {
  const out = prepareOut();
  if (axis === undefined) {
    checkStatus(ffi.mlx_argmin(out, a._ctx, keepdims, s(stream)), "argmin");
  } else {
    checkStatus(ffi.mlx_argmin_axis(out, a._ctx, axis, keepdims, s(stream)), "argmin");
  }
  return MxArray._fromCtx(readOut());
}

/** Log-sum-exp reduction (numerically stable). */
export function logsumexp(
  a: MxArray,
  axis?: number | number[],
  keepdims = false,
  stream?: S,
): MxArray {
  return reduceDispatch(
    (o, arr, kd, st) => ffi.mlx_logsumexp(o, arr, kd, st),
    (o, arr, ax, kd, st) => ffi.mlx_logsumexp_axis(o, arr, ax, kd, st),
    (o, arr, ax, n, kd, st) => ffi.mlx_logsumexp_axes(o, arr, ax, n, kd, st),
    a,
    axis,
    keepdims,
    stream,
    "logsumexp",
  );
}

/** Softmax along an axis. */
export function softmax(a: MxArray, axis = -1, stream?: S): MxArray {
  const out = prepareOut();
  checkStatus(ffi.mlx_softmax_axis(out, a._ctx, axis, false, s(stream)), "softmax");
  return MxArray._fromCtx(readOut());
}
