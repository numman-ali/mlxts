/**
 * 2D convolution over channel-last images.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { add, conv2d as conv2dOp, formatShape, random, zeros } from "@mlxts/core";
import { Module } from "../module";

export type Conv2dSpatialPair = number | readonly [number, number];

function normalizePositiveSpatialPair(value: Conv2dSpatialPair, name: string): [number, number] {
  const pair = typeof value === "number" ? [value, value] : value;
  const height = pair[0];
  const width = pair[1];
  if (!Number.isInteger(height) || !Number.isInteger(width) || height <= 0 || width <= 0) {
    throw new Error(`Conv2d: ${name} must be a positive integer or pair.`);
  }
  return [height, width];
}

function normalizePadding(value: Conv2dSpatialPair): [number, number] {
  const pair = typeof value === "number" ? [value, value] : value;
  const height = pair[0];
  const width = pair[1];
  if (!Number.isInteger(height) || !Number.isInteger(width) || height < 0 || width < 0) {
    throw new Error("Conv2d: padding must be a non-negative integer or pair.");
  }
  return [height, width];
}

/** 2D convolution layer for channel-last image inputs. */
export class Conv2d extends Module {
  weight: MxArray;
  bias: MxArray | null;
  #inputChannels: number;
  #outputChannels: number;
  #kernelSize: [number, number];
  #stride: [number, number];
  #padding: [number, number];
  #dilation: [number, number];
  #groups: number;

  /**
   * @param inputChannels - Number of input channels. Must be > 0.
   * @param outputChannels - Number of output channels. Must be > 0.
   * @param kernelSize - Spatial convolution kernel size. Each dimension must be > 0.
   * @param stride - Step between output positions. Each dimension must be > 0.
   * @param padding - Symmetric zero padding applied to spatial axes. Each dimension must be >= 0.
   * @param dilation - Spacing between kernel taps. Each dimension must be > 0.
   * @param groups - Number of independent channel groups. Must divide input and output channels.
   * @param hasBias - Whether to include a learnable output bias. Defaults to true.
   */
  constructor(
    inputChannels: number,
    outputChannels: number,
    kernelSize: Conv2dSpatialPair,
    stride: Conv2dSpatialPair = 1,
    padding: Conv2dSpatialPair = 0,
    dilation: Conv2dSpatialPair = 1,
    groups = 1,
    hasBias = true,
  ) {
    super();
    if (inputChannels <= 0) {
      throw new Error(`Conv2d: inputChannels must be > 0, got ${inputChannels}`);
    }
    if (outputChannels <= 0) {
      throw new Error(`Conv2d: outputChannels must be > 0, got ${outputChannels}`);
    }
    if (groups <= 0 || inputChannels % groups !== 0 || outputChannels % groups !== 0) {
      throw new Error("Conv2d: groups must divide inputChannels and outputChannels.");
    }

    const [kernelHeight, kernelWidth] = normalizePositiveSpatialPair(kernelSize, "kernelSize");
    this.#inputChannels = inputChannels;
    this.#outputChannels = outputChannels;
    this.#kernelSize = [kernelHeight, kernelWidth];
    this.#stride = normalizePositiveSpatialPair(stride, "stride");
    this.#padding = normalizePadding(padding);
    this.#dilation = normalizePositiveSpatialPair(dilation, "dilation");
    this.#groups = groups;

    const inputChannelsPerGroup = inputChannels / groups;
    const scale = Math.sqrt(1 / (inputChannelsPerGroup * kernelHeight * kernelWidth));
    this.weight = random.uniform(-scale, scale, [
      outputChannels,
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

  get kernelSize(): readonly [number, number] {
    return this.#kernelSize;
  }

  get stride(): readonly [number, number] {
    return this.#stride;
  }

  get padding(): readonly [number, number] {
    return this.#padding;
  }

  get dilation(): readonly [number, number] {
    return this.#dilation;
  }

  get groups(): number {
    return this.#groups;
  }

  forward(x: MxArray): MxArray {
    const [batch, inputHeight, inputWidth, inputChannels] = x.shape;
    if (
      batch === undefined ||
      inputHeight === undefined ||
      inputWidth === undefined ||
      inputChannels === undefined
    ) {
      throw new Error(`Conv2d.forward: expected rank-4 input, got shape ${formatShape(x.shape)}.`);
    }
    if (inputChannels !== this.#inputChannels) {
      throw new Error(
        `Conv2d.forward: expected last dimension ${this.#inputChannels}, got ${inputChannels}.`,
      );
    }

    this.#validateWeightShape(inputChannels);
    const output = conv2dOp(
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
        `Conv2d.forward: expected bias shape [${this.#outputChannels}], got ${formatShape(this.bias.shape)}.`,
      );
    }

    using unbiased = output;
    return add(unbiased, this.bias);
  }

  #validateWeightShape(inputChannels: number): void {
    const [outputChannels, kernelHeight, kernelWidth, inputChannelsPerGroup] = this.weight.shape;
    if (
      outputChannels !== this.#outputChannels ||
      kernelHeight !== this.#kernelSize[0] ||
      kernelWidth !== this.#kernelSize[1] ||
      inputChannelsPerGroup !== inputChannels / this.#groups
    ) {
      throw new Error(
        `Conv2d.forward: expected weight shape [${this.#outputChannels}, ${this.#kernelSize[0]}, ${this.#kernelSize[1]}, ${inputChannels / this.#groups}], got ${formatShape(this.weight.shape)}.`,
      );
    }
  }
}
