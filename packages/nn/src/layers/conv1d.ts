/**
 * 1D convolution over channel-last sequences.
 *
 * Expects input shape `[batch, length, channels]` and stores weights in MLX
 * layout `[outputChannels, kernelSize, inputChannels / groups]`.
 *
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { add, conv1d as conv1dOp, formatShape, random, zeros } from "@mlxts/core";
import { Module } from "../module";

/** 1D convolution layer for channel-last sequence inputs. */
export class Conv1d extends Module {
  weight: MxArray;
  bias: MxArray | null;
  #inputChannels: number;
  #outputChannels: number;
  #kernelSize: number;
  #stride: number;
  #padding: number;
  #dilation: number;
  #groups: number;

  /**
   * @param inputChannels - Number of input channels. Must be > 0.
   * @param outputChannels - Number of output channels. Must be > 0.
   * @param kernelSize - Convolution kernel width. Must be > 0.
   * @param stride - Step between output positions. Must be > 0.
   * @param padding - Zero padding applied to both sequence ends. Must be >= 0.
   * @param dilation - Spacing between kernel taps. Must be > 0.
   * @param groups - Number of independent channel groups. Must divide input and output channels.
   * @param hasBias - Whether to include a learnable output bias. Defaults to true.
   */
  constructor(
    inputChannels: number,
    outputChannels: number,
    kernelSize: number,
    stride = 1,
    padding = 0,
    dilation = 1,
    groups = 1,
    hasBias = true,
  ) {
    super();

    if (inputChannels <= 0) {
      throw new Error(`Conv1d: inputChannels must be > 0, got ${inputChannels}`);
    }
    if (outputChannels <= 0) {
      throw new Error(`Conv1d: outputChannels must be > 0, got ${outputChannels}`);
    }
    if (kernelSize <= 0) {
      throw new Error(`Conv1d: kernelSize must be > 0, got ${kernelSize}`);
    }
    if (stride <= 0) {
      throw new Error(`Conv1d: stride must be > 0, got ${stride}`);
    }
    if (padding < 0) {
      throw new Error(`Conv1d: padding must be >= 0, got ${padding}`);
    }
    if (dilation <= 0) {
      throw new Error(`Conv1d: dilation must be > 0, got ${dilation}`);
    }
    if (groups <= 0) {
      throw new Error(`Conv1d: groups must be > 0, got ${groups}`);
    }
    if (inputChannels % groups !== 0) {
      throw new Error(
        `Conv1d: inputChannels (${inputChannels}) must be divisible by groups (${groups})`,
      );
    }
    if (outputChannels % groups !== 0) {
      throw new Error(
        `Conv1d: outputChannels (${outputChannels}) must be divisible by groups (${groups})`,
      );
    }

    const inputChannelsPerGroup = inputChannels / groups;
    const scale = Math.sqrt(1 / (inputChannelsPerGroup * kernelSize));
    this.#inputChannels = inputChannels;
    this.#outputChannels = outputChannels;
    this.#kernelSize = kernelSize;
    this.#stride = stride;
    this.#padding = padding;
    this.#dilation = dilation;
    this.#groups = groups;
    this.weight = random.uniform(-scale, scale, [
      outputChannels,
      kernelSize,
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

  get kernelSize(): number {
    return this.#kernelSize;
  }

  get stride(): number {
    return this.#stride;
  }

  get padding(): number {
    return this.#padding;
  }

  get dilation(): number {
    return this.#dilation;
  }

  get groups(): number {
    return this.#groups;
  }

  forward(x: MxArray): MxArray {
    const [batch, inputLength, inputChannels] = x.shape;
    if (batch === undefined || inputLength === undefined || inputChannels === undefined) {
      throw new Error(`Conv1d.forward: expected rank-3 input, got shape ${formatShape(x.shape)}.`);
    }
    if (inputChannels !== this.#inputChannels) {
      throw new Error(
        `Conv1d.forward: expected last dimension ${this.#inputChannels}, got ${inputChannels}.`,
      );
    }

    const [outputChannels, kernelSize, inputChannelsPerGroup] = this.weight.shape;
    if (
      outputChannels !== this.#outputChannels ||
      kernelSize !== this.#kernelSize ||
      inputChannelsPerGroup !== inputChannels / this.#groups
    ) {
      throw new Error(
        `Conv1d.forward: expected weight shape [${this.#outputChannels}, ${this.#kernelSize}, ${inputChannels / this.#groups}], got ${formatShape(this.weight.shape)}.`,
      );
    }

    const output = conv1dOp(
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
        `Conv1d.forward: expected bias shape [${this.#outputChannels}], got ${formatShape(this.bias.shape)}.`,
      );
    }

    using unbiased = output;
    return add(unbiased, this.bias);
  }
}
