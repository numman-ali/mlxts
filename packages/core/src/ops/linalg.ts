/**
 * Linear algebra operations.
 * @module
 */

import type { Pointer } from "bun:ffi";
import { type MxArray, readResultArrayWithMetadata } from "../array";
import { defaultStream } from "../device";
import { checkStatus } from "../error";
import { ffi } from "../ffi";

type S = Pointer | undefined;
const s = (stream?: S) => stream ?? defaultStream();

function inferMatmulShape(a: MxArray, b: MxArray): number[] | undefined {
  const aShape = a.shape;
  const bShape = b.shape;
  if (aShape.length >= 2 && bShape.length === 2) {
    return [...aShape.slice(0, -1), bShape[1] ?? 0];
  }
  return undefined;
}

function conv1dOutputLength(
  inputLength: number,
  kernelSize: number,
  padding: number,
  dilation: number,
  stride: number,
): number {
  const paddedLength = inputLength + padding * 2;
  const effectiveKernelSize = dilation * (kernelSize - 1) + 1;
  if (paddedLength < effectiveKernelSize) {
    return 0;
  }
  return Math.floor((paddedLength - effectiveKernelSize) / stride) + 1;
}

function inferConv1dShape(input: MxArray, weight: MxArray): number[] | undefined {
  const [batch, inputLength] = input.shape;
  const [outputChannels, kernelSize] = weight.shape;
  if (
    batch === undefined ||
    inputLength === undefined ||
    outputChannels === undefined ||
    kernelSize === undefined ||
    input.shape.length !== 3 ||
    weight.shape.length !== 3
  ) {
    return undefined;
  }
  return [batch, inputLength, outputChannels];
}

/** Matrix multiplication. */
export function matmul(a: MxArray, b: MxArray, stream?: S): MxArray {
  const shape = inferMatmulShape(a, b);
  const metadata =
    shape === undefined
      ? a.dtype === b.dtype
        ? { dtype: a.dtype }
        : {}
      : a.dtype === b.dtype
        ? { shape, dtype: a.dtype }
        : { shape };
  return readResultArrayWithMetadata("matmul", metadata, (out) => {
    checkStatus(ffi.mlx_matmul(out, a._ctx, b._ctx, s(stream)), "matmul");
  });
}

/** 1D convolution over channel-last sequence inputs. */
export function conv1d(
  input: MxArray,
  weight: MxArray,
  stride = 1,
  padding = 0,
  dilation = 1,
  groups = 1,
  stream?: S,
): MxArray {
  const shape = inferConv1dShape(input, weight);
  const outputLength =
    shape === undefined
      ? undefined
      : conv1dOutputLength(shape[1] ?? 0, weight.shape[1] ?? 0, padding, dilation, stride);
  const metadata =
    shape === undefined
      ? input.dtype === weight.dtype
        ? { dtype: input.dtype }
        : {}
      : input.dtype === weight.dtype
        ? { shape: [shape[0] ?? 0, outputLength ?? 0, shape[2] ?? 0], dtype: input.dtype }
        : { shape: [shape[0] ?? 0, outputLength ?? 0, shape[2] ?? 0] };
  return readResultArrayWithMetadata("conv1d", metadata, (out) => {
    checkStatus(
      ffi.mlx_conv1d(out, input._ctx, weight._ctx, stride, padding, dilation, groups, s(stream)),
      "conv1d",
    );
  });
}
