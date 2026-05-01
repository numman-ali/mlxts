import {
  add,
  concatenate,
  MxArray,
  pad,
  reshape,
  retainArray,
  slice,
  takeAxis,
  tile,
  transpose,
} from "@mlxts/core";
import { Conv3d, type Conv3dSpatialTriple, Dropout, LayerNorm, Module, silu } from "@mlxts/nn";

import { LtxVideoVaeRMSNorm } from "./autoencoder-blocks";
import { expectLtxVideoVaeVolume } from "./autoencoder-volume";
import type { Ltx2SpatialPaddingMode, Ltx2VideoAutoencoderConfig } from "./config";
import { checkedModule } from "./tensor-utils";

type VideoStride = readonly [number, number, number];

function assertChannels(x: MxArray, channels: number, owner: string): void {
  const [, , , , actualChannels] = expectLtxVideoVaeVolume(x, owner);
  if (actualChannels !== channels) {
    throw new Error(`${owner}: expected ${channels} channels, got ${actualChannels}.`);
  }
}

function normalizeTriple(value: Conv3dSpatialTriple, name: string): [number, number, number] {
  const triple = typeof value === "number" ? [value, value, value] : value;
  const depth = triple[0];
  const height = triple[1];
  const width = triple[2];
  if (
    !Number.isInteger(depth) ||
    !Number.isInteger(height) ||
    !Number.isInteger(width) ||
    depth <= 0 ||
    height <= 0 ||
    width <= 0
  ) {
    throw new Error(`Ltx2VideoCausalConv3d: ${name} must be a positive integer or triple.`);
  }
  return [depth, height, width];
}

function reverse<T>(values: readonly T[]): T[] {
  return [...values].reverse();
}

function reflectIndices(
  length: number,
  lowPad: number,
  highPad: number,
  owner: string,
): Int32Array {
  if (lowPad === 0 && highPad === 0) {
    return Int32Array.from({ length }, (_, index) => index);
  }
  if (length <= 1 || lowPad >= length || highPad >= length) {
    throw new Error(`${owner}: reflect padding requires pad sizes smaller than the axis length.`);
  }
  return Int32Array.from({ length: length + lowPad + highPad }, (_, outputIndex) => {
    let sourceIndex = outputIndex - lowPad;
    while (sourceIndex < 0 || sourceIndex >= length) {
      sourceIndex = sourceIndex < 0 ? -sourceIndex : 2 * length - 2 - sourceIndex;
    }
    return sourceIndex;
  });
}

function reflectPadAxis(
  x: MxArray,
  axis: number,
  lowPad: number,
  highPad: number,
  owner: string,
): MxArray {
  const length = x.shape[axis];
  if (length === undefined) {
    throw new Error(`${owner}: missing axis ${axis} length.`);
  }
  using indices = MxArray.fromData(
    reflectIndices(length, lowPad, highPad, owner),
    [length + lowPad + highPad],
    "int32",
  );
  return takeAxis(x, indices, axis);
}

function reflectPadSpatial(x: MxArray, heightPad: number, widthPad: number): MxArray {
  let hidden = retainArray(x);
  try {
    if (heightPad > 0) {
      const heightPadded = reflectPadAxis(hidden, 2, heightPad, heightPad, "reflectPadSpatial");
      hidden.free();
      hidden = heightPadded;
    }
    if (widthPad > 0) {
      const widthPadded = reflectPadAxis(hidden, 3, widthPad, widthPad, "reflectPadSpatial");
      hidden.free();
      hidden = widthPadded;
    }
    return retainArray(hidden);
  } finally {
    hidden.free();
  }
}

function spatialPad(
  x: MxArray,
  heightPad: number,
  widthPad: number,
  mode: Ltx2SpatialPaddingMode,
): MxArray {
  if (heightPad === 0 && widthPad === 0) {
    return retainArray(x);
  }
  if (mode === "zeros") {
    return pad(x, [
      [0, 0],
      [0, 0],
      [heightPad, heightPad],
      [widthPad, widthPad],
      [0, 0],
    ]);
  }
  return reflectPadSpatial(x, heightPad, widthPad);
}

function sliceFrames(x: MxArray, start: number, stop: number): MxArray {
  return slice(
    x,
    [0, start, 0, 0, 0],
    [x.shape[0] ?? 0, stop, x.shape[2] ?? 0, x.shape[3] ?? 0, x.shape[4] ?? 0],
  );
}

function edgePadFrames(x: MxArray, kernelFrames: number, isCausal: boolean): MxArray {
  if (kernelFrames <= 1) {
    return retainArray(x);
  }
  const [, frames] = expectLtxVideoVaeVolume(x, "Ltx2VideoCausalConv3d.forward");
  const leftCount = isCausal ? kernelFrames - 1 : Math.floor((kernelFrames - 1) / 2);
  const rightCount = isCausal ? 0 : Math.floor((kernelFrames - 1) / 2);
  using first = leftCount > 0 ? sliceFrames(x, 0, 1) : null;
  using left = first === null ? null : tile(first, [1, leftCount, 1, 1, 1]);
  using last = rightCount > 0 ? sliceFrames(x, frames - 1, frames) : null;
  using right = last === null ? null : tile(last, [1, rightCount, 1, 1, 1]);
  const parts = [left, x, right].filter((part): part is MxArray => part !== null);
  return concatenate(parts, 1);
}

function pixelShuffleVideo(x: MxArray, stride: VideoStride): MxArray {
  const [batch, frames, height, width, channels] = expectLtxVideoVaeVolume(x, "pixelShuffleVideo");
  const [frameStride, heightStride, widthStride] = stride;
  const strideProduct = frameStride * heightStride * widthStride;
  if (channels % strideProduct !== 0) {
    throw new Error("pixelShuffleVideo: channels must divide by stride product.");
  }
  const outputChannels = channels / strideProduct;
  using grid = reshape(x, [
    batch,
    frames,
    height,
    width,
    outputChannels,
    frameStride,
    heightStride,
    widthStride,
  ]);
  using shuffled = transpose(grid, [0, 1, 5, 2, 6, 3, 7, 4]);
  using upsampled = reshape(shuffled, [
    batch,
    frames * frameStride,
    height * heightStride,
    width * widthStride,
    outputChannels,
  ]);
  return sliceFrames(upsampled, frameStride - 1, frames * frameStride);
}

function validateConfig(config: Ltx2VideoAutoencoderConfig): void {
  if (config.decoderBlockOutChannels.length === 0) {
    throw new Error("Ltx2VideoDecoder3d: decoderBlockOutChannels must not be empty.");
  }
  if (config.decoderLayersPerBlock.length !== config.decoderBlockOutChannels.length + 1) {
    throw new Error("Ltx2VideoDecoder3d: decoderLayersPerBlock length is invalid.");
  }
  if (config.decoderSpatioTemporalScaling.length !== config.decoderBlockOutChannels.length) {
    throw new Error("Ltx2VideoDecoder3d: decoderSpatioTemporalScaling length is invalid.");
  }
  if (config.upsampleResidual.length !== config.decoderBlockOutChannels.length) {
    throw new Error("Ltx2VideoDecoder3d: upsampleResidual length is invalid.");
  }
  if (config.upsampleFactors.length !== config.decoderBlockOutChannels.length) {
    throw new Error("Ltx2VideoDecoder3d: upsampleFactors length is invalid.");
  }
  if (config.patchSizeT !== 1) {
    throw new Error("Ltx2VideoDecoder3d: temporal VAE patch sizes other than 1 are not supported.");
  }
  if (config.timestepConditioning) {
    throw new Error("Ltx2VideoDecoder3d: timestep-conditioned decode is not supported.");
  }
  if (config.decoderInjectNoise.some((enabled) => enabled)) {
    throw new Error("Ltx2VideoDecoder3d: decoder noise injection is not supported.");
  }
}

/** LTX-2 causal 3D convolution over package-internal BFHWC volumes. */
export class Ltx2VideoCausalConv3d extends Module {
  conv: Conv3d;
  #kernelSize: readonly [number, number, number];
  #defaultCausal: boolean;
  #spatialPaddingMode: Ltx2SpatialPaddingMode;

  constructor(
    inputChannels: number,
    outputChannels: number,
    kernelSize: Conv3dSpatialTriple = 3,
    stride: Conv3dSpatialTriple = 1,
    dilation: Conv3dSpatialTriple = 1,
    defaultCausal = true,
    spatialPaddingMode: Ltx2SpatialPaddingMode = "zeros",
  ) {
    super();
    this.#kernelSize = normalizeTriple(kernelSize, "kernelSize");
    this.#defaultCausal = defaultCausal;
    this.#spatialPaddingMode = spatialPaddingMode;
    this.conv = new Conv3d(inputChannels, outputChannels, this.#kernelSize, stride, 0, dilation);
  }

  forward(x: MxArray): MxArray {
    return this.run(x);
  }

  run(x: MxArray, causal = this.#defaultCausal): MxArray {
    assertChannels(x, this.conv.inputChannels, "Ltx2VideoCausalConv3d.forward");
    const [, heightPadding, widthPadding] = [
      this.#kernelSize[0] - 1,
      Math.floor(this.#kernelSize[1] / 2),
      Math.floor(this.#kernelSize[2] / 2),
    ];
    using temporal = edgePadFrames(x, this.#kernelSize[0], causal);
    using padded = spatialPad(temporal, heightPadding, widthPadding, this.#spatialPaddingMode);
    return this.conv.forward(padded);
  }
}

/** LTX-2 video VAE residual block. */
export class Ltx2VideoResnetBlock3d extends Module {
  norm1: LtxVideoVaeRMSNorm;
  conv1: Ltx2VideoCausalConv3d;
  norm2: LtxVideoVaeRMSNorm;
  dropout: Dropout;
  conv2: Ltx2VideoCausalConv3d;
  norm3: LayerNorm | null;
  convShortcut: Conv3d | null;
  #inputChannels: number;
  #outputChannels: number;

  constructor(
    inputChannels: number,
    outputChannels = inputChannels,
    spatialPaddingMode: Ltx2SpatialPaddingMode = "zeros",
  ) {
    super();
    this.norm1 = new LtxVideoVaeRMSNorm(inputChannels);
    this.conv1 = new Ltx2VideoCausalConv3d(
      inputChannels,
      outputChannels,
      3,
      1,
      1,
      true,
      spatialPaddingMode,
    );
    this.norm2 = new LtxVideoVaeRMSNorm(outputChannels);
    this.dropout = new Dropout(0);
    this.conv2 = new Ltx2VideoCausalConv3d(
      outputChannels,
      outputChannels,
      3,
      1,
      1,
      true,
      spatialPaddingMode,
    );
    this.norm3 = inputChannels === outputChannels ? null : new LayerNorm(inputChannels, 1e-6);
    this.convShortcut =
      inputChannels === outputChannels ? null : new Conv3d(inputChannels, outputChannels, 1);
    this.#inputChannels = inputChannels;
    this.#outputChannels = outputChannels;
  }

  forward(x: MxArray): MxArray {
    return this.run(x);
  }

  run(x: MxArray, causal = true): MxArray {
    assertChannels(x, this.#inputChannels, "Ltx2VideoResnetBlock3d.forward");
    using shortcutNorm = this.norm3 === null ? retainArray(x) : this.norm3.forward(x);
    using shortcut =
      this.convShortcut === null
        ? retainArray(shortcutNorm)
        : this.convShortcut.forward(shortcutNorm);
    using normed = this.norm1.forward(x);
    using activated = silu(normed);
    using first = this.conv1.run(activated, causal);
    using normedFirst = this.norm2.forward(first);
    using activatedFirst = silu(normedFirst);
    using dropped = this.dropout.forward(activatedFirst);
    using residual = this.conv2.run(dropped, causal);
    assertChannels(residual, this.#outputChannels, "Ltx2VideoResnetBlock3d.forward residual");
    return add(shortcut, residual);
  }
}

/** LTX-2 decoder upsampler with residual pixel shuffle. */
export class Ltx2VideoUpsampler3d extends Module {
  conv: Ltx2VideoCausalConv3d;
  #stride: VideoStride;
  #inputChannels: number;
  #residual: boolean;
  #upscaleFactor: number;

  constructor(
    inputChannels: number,
    stride: VideoStride,
    residual: boolean,
    upscaleFactor: number,
    spatialPaddingMode: Ltx2SpatialPaddingMode,
  ) {
    super();
    const strideProduct = stride[0] * stride[1] * stride[2];
    if (inputChannels % upscaleFactor !== 0) {
      throw new Error("Ltx2VideoUpsampler3d: input channels must divide by upscale factor.");
    }
    this.#stride = stride;
    this.#inputChannels = inputChannels;
    this.#residual = residual;
    this.#upscaleFactor = upscaleFactor;
    this.conv = new Ltx2VideoCausalConv3d(
      inputChannels,
      (inputChannels * strideProduct) / upscaleFactor,
      3,
      1,
      1,
      true,
      spatialPaddingMode,
    );
  }

  forward(x: MxArray): MxArray {
    return this.run(x);
  }

  run(x: MxArray, causal = true): MxArray {
    assertChannels(x, this.#inputChannels, "Ltx2VideoUpsampler3d.forward");
    using residual = this.#residual ? this.residual(x) : null;
    using projected = this.conv.run(x, causal);
    using shuffled = pixelShuffleVideo(projected, this.#stride);
    if (residual === null) {
      return retainArray(shuffled);
    }
    return add(shuffled, residual);
  }

  private residual(x: MxArray): MxArray {
    const strideProduct = this.#stride[0] * this.#stride[1] * this.#stride[2];
    const repeats = strideProduct / this.#upscaleFactor;
    if (!Number.isInteger(repeats) || repeats <= 0) {
      throw new Error("Ltx2VideoUpsampler3d: stride product must divide by upscale factor.");
    }
    using shuffled = pixelShuffleVideo(x, this.#stride);
    return tile(shuffled, [1, 1, 1, 1, repeats]);
  }
}

/** LTX-2 decoder mid block. */
export class Ltx2VideoMidBlock3d extends Module {
  resnets: Ltx2VideoResnetBlock3d[];

  constructor(channels: number, numLayers: number, spatialPaddingMode: Ltx2SpatialPaddingMode) {
    super();
    this.resnets = Array.from(
      { length: numLayers },
      () => new Ltx2VideoResnetBlock3d(channels, channels, spatialPaddingMode),
    );
  }

  forward(x: MxArray): MxArray {
    return this.run(x);
  }

  run(x: MxArray, causal = true): MxArray {
    let hidden = retainArray(x);
    try {
      for (let index = 0; index < this.resnets.length; index += 1) {
        const block = checkedModule(this.resnets, index, "Ltx2VideoMidBlock3d.forward");
        const next = block.run(hidden, causal);
        hidden.free();
        hidden = next;
      }
      return retainArray(hidden);
    } finally {
      hidden.free();
    }
  }
}

/** LTX-2 decoder up block. */
export class Ltx2VideoUpBlock3d extends Module {
  convIn: Ltx2VideoResnetBlock3d | null;
  upsamplers: Ltx2VideoUpsampler3d[];
  resnets: Ltx2VideoResnetBlock3d[];

  constructor(
    inputChannels: number,
    outputChannels: number,
    numLayers: number,
    spatioTemporalScale: boolean,
    upsampleResidual: boolean,
    upsampleFactor: number,
    spatialPaddingMode: Ltx2SpatialPaddingMode,
  ) {
    super();
    this.convIn =
      inputChannels === outputChannels
        ? null
        : new Ltx2VideoResnetBlock3d(inputChannels, outputChannels, spatialPaddingMode);
    this.upsamplers = spatioTemporalScale
      ? [
          new Ltx2VideoUpsampler3d(
            outputChannels * upsampleFactor,
            [2, 2, 2],
            upsampleResidual,
            upsampleFactor,
            spatialPaddingMode,
          ),
        ]
      : [];
    this.resnets = Array.from(
      { length: numLayers },
      () => new Ltx2VideoResnetBlock3d(outputChannels, outputChannels, spatialPaddingMode),
    );
  }

  forward(x: MxArray): MxArray {
    return this.run(x);
  }

  run(x: MxArray, causal = true): MxArray {
    let hidden = this.convIn === null ? retainArray(x) : this.convIn.run(x, causal);
    try {
      for (let index = 0; index < this.upsamplers.length; index += 1) {
        const upsampler = checkedModule(this.upsamplers, index, "Ltx2VideoUpBlock3d.forward");
        const upsampled = upsampler.run(hidden, causal);
        hidden.free();
        hidden = upsampled;
      }
      for (let index = 0; index < this.resnets.length; index += 1) {
        const block = checkedModule(this.resnets, index, "Ltx2VideoUpBlock3d.forward");
        const next = block.run(hidden, causal);
        hidden.free();
        hidden = next;
      }
      return retainArray(hidden);
    } finally {
      hidden.free();
    }
  }
}

/** Create decoder block parameters in Diffusers LTX-2 reversed order. */
export function ltx2VideoDecoderLayout(config: Ltx2VideoAutoencoderConfig): {
  channels: number[];
  layers: number[];
  scaling: boolean[];
  residual: boolean[];
  factors: number[];
} {
  validateConfig(config);
  return {
    channels: reverse(config.decoderBlockOutChannels),
    layers: reverse(config.decoderLayersPerBlock),
    scaling: reverse(config.decoderSpatioTemporalScaling),
    residual: reverse(config.upsampleResidual),
    factors: reverse(config.upsampleFactors),
  };
}
