/**
 * 3D convolution over channel-last volumes.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { add, conv3d as conv3dOp, formatShape, random, zeros } from "@mlxts/core";
import { Module } from "../module";

export type Conv3dSpatialTriple = number | readonly [number, number, number];

function normalizePositiveSpatialTriple(
  value: Conv3dSpatialTriple,
  name: string,
): [number, number, number] {
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
    throw new Error(`Conv3d: ${name} must be a positive integer or triple.`);
  }
  return [depth, height, width];
}

function normalizePadding(value: Conv3dSpatialTriple): [number, number, number] {
  const triple = typeof value === "number" ? [value, value, value] : value;
  const depth = triple[0];
  const height = triple[1];
  const width = triple[2];
  if (
    !Number.isInteger(depth) ||
    !Number.isInteger(height) ||
    !Number.isInteger(width) ||
    depth < 0 ||
    height < 0 ||
    width < 0
  ) {
    throw new Error("Conv3d: padding must be a non-negative integer or triple.");
  }
  return [depth, height, width];
}

/** 3D convolution layer for channel-last volume inputs. */
export class Conv3d extends Module {
  weight: MxArray;
  bias: MxArray | null;
  #inputChannels: number;
  #outputChannels: number;
  #kernelSize: [number, number, number];
  #stride: [number, number, number];
  #padding: [number, number, number];
  #dilation: [number, number, number];
  #groups: number;

  /**
   * @param inputChannels - Number of input channels. Must be > 0.
   * @param outputChannels - Number of output channels. Must be > 0.
   * @param kernelSize - Spatial convolution kernel size. Each dimension must be > 0.
   * @param stride - Step between output positions. Each dimension must be > 0.
   * @param padding - Symmetric zero padding applied to spatial axes. Each dimension must be >= 0.
   * @param dilation - Spacing between kernel taps. Each dimension must be > 0.
   * @param groups - MLX-backed Conv3d currently supports only 1.
   * @param hasBias - Whether to include a learnable output bias. Defaults to true.
   */
  constructor(
    inputChannels: number,
    outputChannels: number,
    kernelSize: Conv3dSpatialTriple,
    stride: Conv3dSpatialTriple = 1,
    padding: Conv3dSpatialTriple = 0,
    dilation: Conv3dSpatialTriple = 1,
    groups = 1,
    hasBias = true,
  ) {
    super();
    if (inputChannels <= 0) {
      throw new Error(`Conv3d: inputChannels must be > 0, got ${inputChannels}`);
    }
    if (outputChannels <= 0) {
      throw new Error(`Conv3d: outputChannels must be > 0, got ${outputChannels}`);
    }
    if (groups !== 1) {
      throw new Error("Conv3d: groups other than 1 are not supported by MLX.");
    }

    const [kernelDepth, kernelHeight, kernelWidth] = normalizePositiveSpatialTriple(
      kernelSize,
      "kernelSize",
    );
    this.#inputChannels = inputChannels;
    this.#outputChannels = outputChannels;
    this.#kernelSize = [kernelDepth, kernelHeight, kernelWidth];
    this.#stride = normalizePositiveSpatialTriple(stride, "stride");
    this.#padding = normalizePadding(padding);
    this.#dilation = normalizePositiveSpatialTriple(dilation, "dilation");
    this.#groups = groups;

    const inputChannelsPerGroup = inputChannels / groups;
    const scale = Math.sqrt(1 / (inputChannelsPerGroup * kernelDepth * kernelHeight * kernelWidth));
    this.weight = random.uniform(-scale, scale, [
      outputChannels,
      kernelDepth,
      kernelHeight,
      kernelWidth,
      inputChannelsPerGroup,
    ]);
    this.bias = hasBias ? zeros([outputChannels]) : null;
  }

  get inputChannels(): number {
    return this.#inputChannels;
  }

  get outputChannels(): number {
    return this.#outputChannels;
  }

  get kernelSize(): readonly [number, number, number] {
    return this.#kernelSize;
  }

  get stride(): readonly [number, number, number] {
    return this.#stride;
  }

  get padding(): readonly [number, number, number] {
    return this.#padding;
  }

  get dilation(): readonly [number, number, number] {
    return this.#dilation;
  }

  get groups(): number {
    return this.#groups;
  }

  forward(x: MxArray): MxArray {
    const [batch, inputDepth, inputHeight, inputWidth, inputChannels] = x.shape;
    if (
      batch === undefined ||
      inputDepth === undefined ||
      inputHeight === undefined ||
      inputWidth === undefined ||
      inputChannels === undefined
    ) {
      throw new Error(`Conv3d.forward: expected rank-5 input, got shape ${formatShape(x.shape)}.`);
    }
    if (inputChannels !== this.#inputChannels) {
      throw new Error(
        `Conv3d.forward: expected last dimension ${this.#inputChannels}, got ${inputChannels}.`,
      );
    }

    this.#validateWeightShape(inputChannels);
    const output = conv3dOp(
      x,
      this.weight,
      this.#stride,
      this.#padding,
      this.#dilation,
      this.#groups,
    );
    if (this.bias === null) {
      return output;
    }

    if (this.bias.shape.length !== 1 || this.bias.shape[0] !== this.#outputChannels) {
      output.free();
      throw new Error(
        `Conv3d.forward: expected bias shape [${this.#outputChannels}], got ${formatShape(this.bias.shape)}.`,
      );
    }

    using unbiased = output;
    return add(unbiased, this.bias);
  }

  #validateWeightShape(inputChannels: number): void {
    const [outputChannels, kernelDepth, kernelHeight, kernelWidth, inputChannelsPerGroup] =
      this.weight.shape;
    if (
      outputChannels !== this.#outputChannels ||
      kernelDepth !== this.#kernelSize[0] ||
      kernelHeight !== this.#kernelSize[1] ||
      kernelWidth !== this.#kernelSize[2] ||
      inputChannelsPerGroup !== inputChannels / this.#groups
    ) {
      throw new Error(
        `Conv3d.forward: expected weight shape [${this.#outputChannels}, ${this.#kernelSize[0]}, ${this.#kernelSize[1]}, ${this.#kernelSize[2]}, ${inputChannels / this.#groups}], got ${formatShape(this.weight.shape)}.`,
      );
    }
  }
}
