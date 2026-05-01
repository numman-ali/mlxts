import type { MxArray } from "@mlxts/core";
import {
  add,
  concatenate,
  conv3d as conv3dOp,
  formatShape,
  pad,
  random,
  transpose,
  zeros,
} from "@mlxts/core";
import { Module } from "@mlxts/nn";

export type QwenImageSpatialTriple = number | readonly [number, number, number];

function normalizePositiveSpatialTriple(
  value: QwenImageSpatialTriple,
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
    throw new Error(`QwenImageCausalConv3d: ${name} must be a positive integer or triple.`);
  }
  return [depth, height, width];
}

function normalizePadding(value: QwenImageSpatialTriple): [number, number, number] {
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
    throw new Error("QwenImageCausalConv3d: padding must be a non-negative integer or triple.");
  }
  return [depth, height, width];
}

export function expectQwenImageVaeVolume(
  x: MxArray,
  owner: string,
): readonly [number, number, number, number, number] {
  const [batch, depth, height, width, channels] = x.shape;
  if (
    x.shape.length !== 5 ||
    batch === undefined ||
    depth === undefined ||
    height === undefined ||
    width === undefined ||
    channels === undefined
  ) {
    throw new Error(`${owner}: expected rank-5 volume, got ${formatShape(x.shape)}.`);
  }
  return [batch, depth, height, width, channels];
}

function validateCausalPrefix(x: MxArray, prefix: MxArray): number {
  const [batch, , height, width, channels] = expectQwenImageVaeVolume(
    x,
    "QwenImageCausalConv3d.forward",
  );
  const [prefixBatch, prefixDepth, prefixHeight, prefixWidth, prefixChannels] =
    expectQwenImageVaeVolume(prefix, "QwenImageCausalConv3d.forward prefix");
  if (
    prefixBatch !== batch ||
    prefixHeight !== height ||
    prefixWidth !== width ||
    prefixChannels !== channels
  ) {
    throw new Error(
      `QwenImageCausalConv3d.forward: prefix shape ${formatShape(
        prefix.shape,
      )} is not compatible with input shape ${formatShape(x.shape)}.`,
    );
  }
  return prefixDepth;
}

function sameShape(left: readonly number[], right: readonly number[]): boolean {
  return (
    left.length === right.length && left.every((dimension, index) => dimension === right[index])
  );
}

/** Convert Diffusers-style channel-first Qwen-Image VAE volumes from NCFHW to NDHWC. */
export function qwenImageNcfhwToNdhwc(x: MxArray): MxArray {
  expectQwenImageVaeVolume(x, "qwenImageNcfhwToNdhwc");
  return transpose(x, [0, 2, 3, 4, 1]);
}

/** Convert package-internal channel-last Qwen-Image VAE volumes from NDHWC to NCFHW. */
export function qwenImageNdhwcToNcfhw(x: MxArray): MxArray {
  expectQwenImageVaeVolume(x, "qwenImageNdhwcToNcfhw");
  return transpose(x, [0, 4, 1, 2, 3]);
}

/** Qwen-Image causal 3D convolution over package-internal NDHWC VAE volumes. */
export class QwenImageCausalConv3d extends Module {
  weight: MxArray;
  bias: MxArray | null;
  #inputChannels: number;
  #outputChannels: number;
  #kernelSize: [number, number, number];
  #stride: [number, number, number];
  #padding: [number, number, number];
  #dilation: [number, number, number];

  constructor(
    inputChannels: number,
    outputChannels: number,
    kernelSize: QwenImageSpatialTriple,
    stride: QwenImageSpatialTriple = 1,
    padding: QwenImageSpatialTriple = 0,
    dilation: QwenImageSpatialTriple = 1,
    hasBias = true,
  ) {
    super();
    if (inputChannels <= 0) {
      throw new Error(`QwenImageCausalConv3d: inputChannels must be > 0, got ${inputChannels}`);
    }
    if (outputChannels <= 0) {
      throw new Error(`QwenImageCausalConv3d: outputChannels must be > 0, got ${outputChannels}`);
    }

    this.#inputChannels = inputChannels;
    this.#outputChannels = outputChannels;
    this.#kernelSize = normalizePositiveSpatialTriple(kernelSize, "kernelSize");
    this.#stride = normalizePositiveSpatialTriple(stride, "stride");
    this.#padding = normalizePadding(padding);
    this.#dilation = normalizePositiveSpatialTriple(dilation, "dilation");

    const [kernelDepth, kernelHeight, kernelWidth] = this.#kernelSize;
    const scale = Math.sqrt(1 / (inputChannels * kernelDepth * kernelHeight * kernelWidth));
    this.weight = random.uniform(-scale, scale, [
      outputChannels,
      kernelDepth,
      kernelHeight,
      kernelWidth,
      inputChannels,
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

  forward(x: MxArray, causalPrefix?: MxArray): MxArray {
    const [, , , , channels] = expectQwenImageVaeVolume(x, "QwenImageCausalConv3d.forward");
    if (channels !== this.#inputChannels) {
      throw new Error(
        `QwenImageCausalConv3d.forward: expected last dimension ${this.#inputChannels}, got ${channels}.`,
      );
    }
    this.#validateWeightShape();

    const [temporalPadding, heightPadding, widthPadding] = this.#padding;
    const prefixDepth = causalPrefix === undefined ? 0 : validateCausalPrefix(x, causalPrefix);
    const prefixed = causalPrefix === undefined ? null : concatenate([causalPrefix, x], 1);
    let input = prefixed ?? x;
    let lowTemporalPad = temporalPadding * 2;
    if (causalPrefix !== undefined) {
      lowTemporalPad = Math.max(0, lowTemporalPad - prefixDepth);
    }

    let padded: MxArray | null = null;
    try {
      if (lowTemporalPad > 0 || heightPadding > 0 || widthPadding > 0) {
        padded = pad(input, [
          [0, 0],
          [lowTemporalPad, 0],
          [heightPadding, heightPadding],
          [widthPadding, widthPadding],
          [0, 0],
        ]);
        input = padded;
      }

      const output = conv3dOp(input, this.weight, this.#stride, 0, this.#dilation);
      if (this.bias === null) {
        return output;
      }

      this.#validateBiasShape(output);
      using unbiased = output;
      return add(unbiased, this.bias);
    } finally {
      padded?.free();
      prefixed?.free();
    }
  }

  #validateWeightShape(): void {
    const expectedShape = [
      this.#outputChannels,
      this.#kernelSize[0],
      this.#kernelSize[1],
      this.#kernelSize[2],
      this.#inputChannels,
    ];
    if (!sameShape(this.weight.shape, expectedShape)) {
      throw new Error(
        `QwenImageCausalConv3d.forward: expected weight shape ${formatShape(
          expectedShape,
        )}, got ${formatShape(this.weight.shape)}.`,
      );
    }
  }

  #validateBiasShape(output: MxArray): void {
    if (this.bias === null) {
      return;
    }
    if (this.bias.shape.length !== 1 || this.bias.shape[0] !== this.#outputChannels) {
      output.free();
      throw new Error(
        `QwenImageCausalConv3d.forward: expected bias shape [${this.#outputChannels}], got ${formatShape(
          this.bias.shape,
        )}.`,
      );
    }
  }
}
