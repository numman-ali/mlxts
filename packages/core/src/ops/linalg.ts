/**
 * Linear algebra operations.
 * @module
 */

import type { Pointer } from "bun:ffi";
import { type MxArray, readResultArray, readResultArrayWithMetadata } from "../array";
import { defaultStream } from "../device";
import { checkStatus } from "../error";
import { ffi, ptr } from "../ffi";

type S = Pointer | undefined;
const s = (stream?: S) => stream ?? defaultStream();

/** Options for MLX matrix multiplication with per-matrix gather indices. */
export type GatherMmOptions = {
  lhsIndices?: MxArray;
  rhsIndices?: MxArray;
  sortedIndices?: boolean;
  stream?: S;
};

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

function convTranspose1dOutputLength(
  inputLength: number,
  kernelSize: number,
  padding: number,
  dilation: number,
  stride: number,
  outputPadding: number,
): number {
  return (inputLength - 1) * stride - 2 * padding + dilation * (kernelSize - 1) + outputPadding + 1;
}

function normalizeSpatialPair(
  value: number | readonly [number, number],
  name: string,
): [number, number] {
  if (typeof value === "number") {
    if (!Number.isInteger(value) || value < 0) {
      throw new Error(`conv2d: ${name} must be a non-negative integer.`);
    }
    return [value, value];
  }

  const height = value[0];
  const width = value[1];
  if (!Number.isInteger(height) || !Number.isInteger(width) || height < 0 || width < 0) {
    throw new Error(`conv2d: ${name} must be a non-negative integer pair.`);
  }
  return [height, width];
}

function normalizeSpatialTriple(
  value: number | readonly [number, number, number],
  name: string,
): [number, number, number] {
  if (typeof value === "number") {
    if (!Number.isInteger(value) || value < 0) {
      throw new Error(`conv3d: ${name} must be a non-negative integer.`);
    }
    return [value, value, value];
  }

  const depth = value[0];
  const height = value[1];
  const width = value[2];
  if (
    !Number.isInteger(depth) ||
    !Number.isInteger(height) ||
    !Number.isInteger(width) ||
    depth < 0 ||
    height < 0 ||
    width < 0
  ) {
    throw new Error(`conv3d: ${name} must be a non-negative integer triple.`);
  }
  return [depth, height, width];
}

function conv2dOutputLength(
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

function conv3dOutputLength(
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

function inferConv2dShape(input: MxArray, weight: MxArray): number[] | undefined {
  const [batch, inputHeight, inputWidth] = input.shape;
  const [outputChannels, kernelHeight, kernelWidth] = weight.shape;
  if (
    batch === undefined ||
    inputHeight === undefined ||
    inputWidth === undefined ||
    outputChannels === undefined ||
    kernelHeight === undefined ||
    kernelWidth === undefined ||
    input.shape.length !== 4 ||
    weight.shape.length !== 4
  ) {
    return undefined;
  }
  return [batch, inputHeight, inputWidth, outputChannels];
}

function inferConv3dShape(input: MxArray, weight: MxArray): number[] | undefined {
  const [batch, inputDepth, inputHeight, inputWidth] = input.shape;
  const [outputChannels, kernelDepth, kernelHeight, kernelWidth] = weight.shape;
  if (
    batch === undefined ||
    inputDepth === undefined ||
    inputHeight === undefined ||
    inputWidth === undefined ||
    outputChannels === undefined ||
    kernelDepth === undefined ||
    kernelHeight === undefined ||
    kernelWidth === undefined ||
    input.shape.length !== 5 ||
    weight.shape.length !== 5
  ) {
    return undefined;
  }
  return [batch, inputDepth, inputHeight, inputWidth, outputChannels];
}

function inferConv2dOutputShape(
  input: MxArray,
  weight: MxArray,
  stride: readonly [number, number],
  padding: readonly [number, number],
  dilation: readonly [number, number],
): number[] | undefined {
  const shape = inferConv2dShape(input, weight);
  if (shape === undefined) {
    return undefined;
  }
  const [strideHeight, strideWidth] = stride;
  const [paddingHeight, paddingWidth] = padding;
  const [dilationHeight, dilationWidth] = dilation;
  const outputHeight = conv2dOutputLength(
    shape[1] ?? 0,
    weight.shape[1] ?? 0,
    paddingHeight,
    dilationHeight,
    strideHeight,
  );
  const outputWidth = conv2dOutputLength(
    shape[2] ?? 0,
    weight.shape[2] ?? 0,
    paddingWidth,
    dilationWidth,
    strideWidth,
  );
  return [shape[0] ?? 0, outputHeight, outputWidth, shape[3] ?? 0];
}

function inferConv3dOutputShape(
  input: MxArray,
  weight: MxArray,
  stride: readonly [number, number, number],
  padding: readonly [number, number, number],
  dilation: readonly [number, number, number],
): number[] | undefined {
  const shape = inferConv3dShape(input, weight);
  if (shape === undefined) {
    return undefined;
  }
  const [strideDepth, strideHeight, strideWidth] = stride;
  const [paddingDepth, paddingHeight, paddingWidth] = padding;
  const [dilationDepth, dilationHeight, dilationWidth] = dilation;
  const outputDepth = conv3dOutputLength(
    shape[1] ?? 0,
    weight.shape[1] ?? 0,
    paddingDepth,
    dilationDepth,
    strideDepth,
  );
  const outputHeight = conv3dOutputLength(
    shape[2] ?? 0,
    weight.shape[2] ?? 0,
    paddingHeight,
    dilationHeight,
    strideHeight,
  );
  const outputWidth = conv3dOutputLength(
    shape[3] ?? 0,
    weight.shape[3] ?? 0,
    paddingWidth,
    dilationWidth,
    strideWidth,
  );
  return [shape[0] ?? 0, outputDepth, outputHeight, outputWidth, shape[4] ?? 0];
}

function convMetadata(
  input: MxArray,
  weight: MxArray,
  shape: number[] | undefined,
): {
  shape?: number[];
  dtype?: MxArray["dtype"];
} {
  if (shape === undefined) {
    return input.dtype === weight.dtype ? { dtype: input.dtype } : {};
  }
  return input.dtype === weight.dtype ? { shape, dtype: input.dtype } : { shape };
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

/** Matrix multiplication that gathers batch matrices before contraction. */
export function gatherMm(a: MxArray, b: MxArray, options: GatherMmOptions = {}): MxArray {
  return readResultArray("gather_mm", (out) => {
    checkStatus(
      ffi.mlx_gather_mm(
        out,
        a._ctx,
        b._ctx,
        options.lhsIndices?._ctx ?? null,
        options.rhsIndices?._ctx ?? null,
        options.sortedIndices ?? false,
        s(options.stream),
      ),
      "gather_mm",
    );
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
  const outputShape =
    shape === undefined ? undefined : [shape[0] ?? 0, outputLength ?? 0, shape[2] ?? 0];
  const metadata = convMetadata(input, weight, outputShape);
  return readResultArrayWithMetadata("conv1d", metadata, (out) => {
    checkStatus(
      ffi.mlx_conv1d(out, input._ctx, weight._ctx, stride, padding, dilation, groups, s(stream)),
      "conv1d",
    );
  });
}

/** 1D transposed convolution over channel-last sequence inputs. */
export function convTranspose1d(
  input: MxArray,
  weight: MxArray,
  stride = 1,
  padding = 0,
  dilation = 1,
  outputPadding = 0,
  groups = 1,
  stream?: S,
): MxArray {
  const shape = inferConv1dShape(input, weight);
  const outputLength =
    shape === undefined
      ? undefined
      : convTranspose1dOutputLength(
          shape[1] ?? 0,
          weight.shape[1] ?? 0,
          padding,
          dilation,
          stride,
          outputPadding,
        );
  const outputShape =
    shape === undefined ? undefined : [shape[0] ?? 0, outputLength ?? 0, shape[2] ?? 0];
  const metadata = convMetadata(input, weight, outputShape);
  return readResultArrayWithMetadata("conv_transpose1d", metadata, (out) => {
    checkStatus(
      ffi.mlx_conv_transpose1d(
        out,
        input._ctx,
        weight._ctx,
        stride,
        padding,
        dilation,
        outputPadding,
        groups,
        s(stream),
      ),
      "conv_transpose1d",
    );
  });
}

/** 2D convolution over channel-last image inputs. */
export function conv2d(
  input: MxArray,
  weight: MxArray,
  stride: number | readonly [number, number] = 1,
  padding: number | readonly [number, number] = 0,
  dilation: number | readonly [number, number] = 1,
  groups = 1,
  stream?: S,
): MxArray {
  const [strideHeight, strideWidth] = normalizeSpatialPair(stride, "stride");
  const [paddingHeight, paddingWidth] = normalizeSpatialPair(padding, "padding");
  const [dilationHeight, dilationWidth] = normalizeSpatialPair(dilation, "dilation");
  if (strideHeight === 0 || strideWidth === 0) {
    throw new Error("conv2d: stride must be positive.");
  }
  if (dilationHeight === 0 || dilationWidth === 0) {
    throw new Error("conv2d: dilation must be positive.");
  }

  const outputShape = inferConv2dOutputShape(
    input,
    weight,
    [strideHeight, strideWidth],
    [paddingHeight, paddingWidth],
    [dilationHeight, dilationWidth],
  );
  const metadata = convMetadata(input, weight, outputShape);
  const params = Int32Array.from([
    strideHeight,
    strideWidth,
    paddingHeight,
    paddingWidth,
    dilationHeight,
    dilationWidth,
    groups,
  ]);
  return readResultArrayWithMetadata("conv2d", metadata, (out) => {
    checkStatus(
      ffi.mlxts_conv2d(out, input._ctx, weight._ctx, ptr(params), params.length, s(stream)),
      "conv2d",
    );
  });
}

/** 3D convolution over channel-last volume inputs. */
export function conv3d(
  input: MxArray,
  weight: MxArray,
  stride: number | readonly [number, number, number] = 1,
  padding: number | readonly [number, number, number] = 0,
  dilation: number | readonly [number, number, number] = 1,
  groups = 1,
  stream?: S,
): MxArray {
  const [strideDepth, strideHeight, strideWidth] = normalizeSpatialTriple(stride, "stride");
  const [paddingDepth, paddingHeight, paddingWidth] = normalizeSpatialTriple(padding, "padding");
  const [dilationDepth, dilationHeight, dilationWidth] = normalizeSpatialTriple(
    dilation,
    "dilation",
  );
  if (strideDepth === 0 || strideHeight === 0 || strideWidth === 0) {
    throw new Error("conv3d: stride must be positive.");
  }
  if (dilationDepth === 0 || dilationHeight === 0 || dilationWidth === 0) {
    throw new Error("conv3d: dilation must be positive.");
  }
  if (groups !== 1) {
    throw new Error("conv3d: groups other than 1 are not supported by MLX.");
  }

  const outputShape = inferConv3dOutputShape(
    input,
    weight,
    [strideDepth, strideHeight, strideWidth],
    [paddingDepth, paddingHeight, paddingWidth],
    [dilationDepth, dilationHeight, dilationWidth],
  );
  const metadata = convMetadata(input, weight, outputShape);
  const params = Int32Array.from([
    strideDepth,
    strideHeight,
    strideWidth,
    paddingDepth,
    paddingHeight,
    paddingWidth,
    dilationDepth,
    dilationHeight,
    dilationWidth,
    groups,
  ]);
  return readResultArrayWithMetadata("conv3d", metadata, (out) => {
    checkStatus(
      ffi.mlxts_conv3d(out, input._ctx, weight._ctx, ptr(params), params.length, s(stream)),
      "conv3d",
    );
  });
}
