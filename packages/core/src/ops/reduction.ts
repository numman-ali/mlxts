/**
 * Reduction operations (sum, mean, max, min, argmax, argmin, logsumexp).
 * @module
 */

import type { Pointer } from "bun:ffi";
import { type MxArray, readResultArray } from "../array";
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
  if (axis === undefined) {
    return readResultArray(name, (out) => {
      checkStatus(fnAll(out, a._ctx, keepdims, s(stream)), name);
    });
  } else if (typeof axis === "number") {
    return readResultArray(name, (out) => {
      checkStatus(fnAxis(out, a._ctx, axis, keepdims, s(stream)), name);
    });
  } else {
    const axesBuf = new Int32Array(axis);
    return readResultArray(name, (out) => {
      checkStatus(fnAxes(out, a._ctx, ptr(axesBuf), axis.length, keepdims, s(stream)), name);
    });
  }
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
  return readResultArray("argmax", (out) => {
    if (axis === undefined) {
      checkStatus(ffi.mlx_argmax(out, a._ctx, keepdims, s(stream)), "argmax");
    } else {
      checkStatus(ffi.mlx_argmax_axis(out, a._ctx, axis, keepdims, s(stream)), "argmax");
    }
  });
}

/** Index of minimum value. */
export function argmin(a: MxArray, axis?: number, keepdims = false, stream?: S): MxArray {
  return readResultArray("argmin", (out) => {
    if (axis === undefined) {
      checkStatus(ffi.mlx_argmin(out, a._ctx, keepdims, s(stream)), "argmin");
    } else {
      checkStatus(ffi.mlx_argmin_axis(out, a._ctx, axis, keepdims, s(stream)), "argmin");
    }
  });
}

/** Indices that would sort values along an axis. Defaults to the last axis. */
export function argsort(a: MxArray, axis = -1, stream?: S): MxArray {
  return readResultArray("argsort", (out) => {
    checkStatus(ffi.mlx_argsort_axis(out, a._ctx, axis, s(stream)), "argsort");
  });
}

/** Indices that partition values around `kth` along an axis. Defaults to the last axis. */
export function argpartition(a: MxArray, kth: number, axis = -1, stream?: S): MxArray {
  return readResultArray("argpartition", (out) => {
    checkStatus(ffi.mlx_argpartition_axis(out, a._ctx, kth, axis, s(stream)), "argpartition");
  });
}

/** Cumulative sum along an axis. */
export function cumsum(
  a: MxArray,
  axis = -1,
  options?: { reverse?: boolean; inclusive?: boolean; stream?: S },
): MxArray {
  return readResultArray("cumsum", (out) => {
    checkStatus(
      ffi.mlx_cumsum(
        out,
        a._ctx,
        axis,
        options?.reverse ?? false,
        options?.inclusive ?? true,
        s(options?.stream),
      ),
      "cumsum",
    );
  });
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

export type SoftmaxOptions = {
  precise?: boolean;
  stream?: S;
};

/** Softmax along an axis. */
export function softmax(a: MxArray, axis = -1, options?: SoftmaxOptions): MxArray {
  return readResultArray("softmax", (out) => {
    checkStatus(
      ffi.mlx_softmax_axis(out, a._ctx, axis, options?.precise ?? false, s(options?.stream)),
      "softmax",
    );
  });
}

/** Sort values along an axis. Defaults to the last axis. */
export function sort(a: MxArray, axis = -1, stream?: S): MxArray {
  return readResultArray("sort", (out) => {
    checkStatus(ffi.mlx_sort_axis(out, a._ctx, axis, s(stream)), "sort");
  });
}

/** Return the largest `k` values along an axis. Defaults to the last axis. */
export function topk(a: MxArray, k: number, axis = -1, stream?: S): MxArray {
  return readResultArray("topk", (out) => {
    checkStatus(ffi.mlx_topk_axis(out, a._ctx, k, axis, s(stream)), "topk");
  });
}
