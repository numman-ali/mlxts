import {
  add,
  divide,
  formatShape,
  type MxArray,
  matmul,
  mean,
  pad,
  repeat,
  reshape,
  retainArray,
  slice,
  softmax,
  sqrt,
  square,
  transpose,
} from "@mlxts/core";
import { Conv2d, Dropout, Module, silu } from "@mlxts/nn";

import type { Ltx2AudioCausalityAxis } from "./config";
import { checkedModule } from "./tensor-utils";

export type Ltx2AudioSpatialPair = number | readonly [number, number];
type Ltx2AudioCausalConvAxis = Exclude<Ltx2AudioCausalityAxis, null>;
type Ltx2AudioConv2d = Ltx2AudioCausalConv2d | Conv2d;

function normalizePositivePair(value: Ltx2AudioSpatialPair, name: string): [number, number] {
  const pair = typeof value === "number" ? [value, value] : value;
  const height = pair[0];
  const width = pair[1];
  if (!Number.isInteger(height) || !Number.isInteger(width) || height <= 0 || width <= 0) {
    throw new Error(`Ltx2AudioCausalConv2d: ${name} must be a positive integer or pair.`);
  }
  return [height, width];
}

function causalPadding(
  kernelSize: readonly [number, number],
  dilation: readonly [number, number],
  causalityAxis: Ltx2AudioCausalConvAxis,
): readonly [left: number, right: number, top: number, bottom: number] {
  const padHeight = (kernelSize[0] - 1) * dilation[0];
  const padWidth = (kernelSize[1] - 1) * dilation[1];
  if (causalityAxis === "width" || causalityAxis === "width-compatibility") {
    return [padWidth, 0, Math.floor(padHeight / 2), padHeight - Math.floor(padHeight / 2)];
  }
  if (causalityAxis === "height") {
    return [Math.floor(padWidth / 2), padWidth - Math.floor(padWidth / 2), padHeight, 0];
  }
  return [
    Math.floor(padWidth / 2),
    padWidth - Math.floor(padWidth / 2),
    Math.floor(padHeight / 2),
    padHeight - Math.floor(padHeight / 2),
  ];
}

function padBlmc(
  x: MxArray,
  padding: readonly [left: number, right: number, top: number, bottom: number],
): MxArray {
  const [left, right, top, bottom] = padding;
  if (left === 0 && right === 0 && top === 0 && bottom === 0) {
    return retainArray(x);
  }
  return pad(x, [
    [0, 0],
    [top, bottom],
    [left, right],
    [0, 0],
  ]);
}

function symmetricPadding(
  kernelSize: Ltx2AudioSpatialPair,
  dilation: Ltx2AudioSpatialPair,
): readonly [number, number] {
  const [kernelHeight, kernelWidth] = normalizePositivePair(kernelSize, "kernelSize");
  const [dilationHeight, dilationWidth] = normalizePositivePair(dilation, "dilation");
  return [
    Math.floor(((kernelHeight - 1) * dilationHeight) / 2),
    Math.floor(((kernelWidth - 1) * dilationWidth) / 2),
  ];
}

function audioConv2d(
  inputChannels: number,
  outputChannels: number,
  kernelSize: Ltx2AudioSpatialPair,
  stride: Ltx2AudioSpatialPair,
  dilation: Ltx2AudioSpatialPair,
  groups: number,
  hasBias: boolean,
  causalityAxis: Ltx2AudioCausalityAxis,
): Ltx2AudioConv2d {
  if (causalityAxis === null) {
    return new Conv2d(
      inputChannels,
      outputChannels,
      kernelSize,
      stride,
      symmetricPadding(kernelSize, dilation),
      dilation,
      groups,
      hasBias,
    );
  }
  return new Ltx2AudioCausalConv2d(
    inputChannels,
    outputChannels,
    kernelSize,
    stride,
    dilation,
    groups,
    hasBias,
    causalityAxis,
  );
}

function expectBlmc(x: MxArray, owner: string): readonly [number, number, number, number] {
  const [batch, length, melBins, channels] = x.shape;
  if (
    x.shape.length !== 4 ||
    batch === undefined ||
    length === undefined ||
    melBins === undefined ||
    channels === undefined
  ) {
    throw new Error(`${owner}: expected BLMC rank-4 input, got ${formatShape(x.shape)}.`);
  }
  return [batch, length, melBins, channels];
}

function assertChannels(x: MxArray, channels: number, owner: string): void {
  const [, , , actualChannels] = expectBlmc(x, owner);
  if (actualChannels !== channels) {
    throw new Error(`${owner}: expected ${channels} channels, got ${actualChannels}.`);
  }
}

function nearestUpsample2d(x: MxArray): MxArray {
  using lengthRepeated = repeat(x, 2, 1);
  return repeat(lengthRepeated, 2, 2);
}

function cropAfterUpsample(x: MxArray, causalityAxis: Ltx2AudioCausalityAxis): MxArray {
  const [batch, length, melBins, channels] = expectBlmc(x, "Ltx2AudioUpsample.forward");
  if (causalityAxis === "height") {
    return slice(x, [0, 1, 0, 0], [batch, length, melBins, channels]);
  }
  if (causalityAxis === "width") {
    return slice(x, [0, 0, 1, 0], [batch, length, melBins, channels]);
  }
  return retainArray(x);
}

/** LTX-2 causal 2D convolution over package-internal BLMC audio spectrograms. */
export class Ltx2AudioCausalConv2d extends Module {
  conv: Conv2d;
  #kernelSize: readonly [number, number];
  #dilation: readonly [number, number];
  #causalityAxis: Ltx2AudioCausalConvAxis;

  constructor(
    inputChannels: number,
    outputChannels: number,
    kernelSize: Ltx2AudioSpatialPair,
    stride: Ltx2AudioSpatialPair = 1,
    dilation: Ltx2AudioSpatialPair = 1,
    groups = 1,
    hasBias = true,
    causalityAxis: Ltx2AudioCausalConvAxis = "height",
  ) {
    super();
    this.#kernelSize = normalizePositivePair(kernelSize, "kernelSize");
    this.#dilation = normalizePositivePair(dilation, "dilation");
    this.#causalityAxis = causalityAxis;
    this.conv = new Conv2d(
      inputChannels,
      outputChannels,
      this.#kernelSize,
      stride,
      0,
      dilation,
      groups,
      hasBias,
    );
  }

  forward(x: MxArray): MxArray {
    assertChannels(x, this.conv.inputChannels, "Ltx2AudioCausalConv2d.forward");
    using padded = padBlmc(x, causalPadding(this.#kernelSize, this.#dilation, this.#causalityAxis));
    return this.conv.forward(padded);
  }
}

/** Per-pixel RMS normalization used by the current LTX-2 audio VAE. */
export class Ltx2AudioPixelNorm extends Module {
  #eps: number;

  constructor(eps = 1e-6) {
    super();
    this.#eps = eps;
  }

  forward(x: MxArray): MxArray {
    using squared = square(x);
    using meanSquared = mean(squared, 3, true);
    using shifted = add(meanSquared, this.#eps);
    using rms = sqrt(shifted);
    return divide(x, rms);
  }
}

/** LTX-2 audio self-attention block over flattened spectrogram positions. */
export class Ltx2AudioAttnBlock extends Module {
  norm: Ltx2AudioPixelNorm;
  q: Conv2d;
  k: Conv2d;
  v: Conv2d;
  projOut: Conv2d;
  #channels: number;

  constructor(channels: number) {
    super();
    this.norm = new Ltx2AudioPixelNorm();
    this.q = new Conv2d(channels, channels, 1);
    this.k = new Conv2d(channels, channels, 1);
    this.v = new Conv2d(channels, channels, 1);
    this.projOut = new Conv2d(channels, channels, 1);
    this.#channels = channels;
  }

  forward(x: MxArray): MxArray {
    const [batch, length, melBins, channels] = expectBlmc(x, "Ltx2AudioAttnBlock.forward");
    if (channels !== this.#channels) {
      throw new Error(
        `Ltx2AudioAttnBlock.forward: expected ${this.#channels} channels, got ${channels}.`,
      );
    }
    using normed = this.norm.forward(x);
    using q = this.q.forward(normed);
    using k = this.k.forward(normed);
    using v = this.v.forward(normed);
    using qSeq = reshape(q, [batch, length * melBins, channels]);
    using kSeq = reshape(k, [batch, length * melBins, channels]);
    using kT = transpose(kSeq, [0, 2, 1]);
    using scores = matmul(qSeq, kT);
    using scaled = divide(scores, Math.sqrt(channels));
    using probs = softmax(scaled, 2);
    using vSeq = reshape(v, [batch, length * melBins, channels]);
    using attendedSeq = matmul(probs, vSeq);
    using attended = reshape(attendedSeq, [batch, length, melBins, channels]);
    using projected = this.projOut.forward(attended);
    return add(x, projected);
  }
}

/** LTX-2 audio VAE residual block. */
export class Ltx2AudioResnetBlock extends Module {
  norm1: Ltx2AudioPixelNorm;
  conv1: Ltx2AudioConv2d;
  norm2: Ltx2AudioPixelNorm;
  dropout: Dropout;
  conv2: Ltx2AudioConv2d;
  ninShortcut: Ltx2AudioConv2d | null;
  #inputChannels: number;
  #outputChannels: number;

  constructor(
    inputChannels: number,
    outputChannels = inputChannels,
    dropout = 0,
    causalityAxis: Ltx2AudioCausalityAxis = "height",
  ) {
    super();
    this.norm1 = new Ltx2AudioPixelNorm();
    this.conv1 = audioConv2d(inputChannels, outputChannels, 3, 1, 1, 1, true, causalityAxis);
    this.norm2 = new Ltx2AudioPixelNorm();
    this.dropout = new Dropout(dropout);
    this.conv2 = audioConv2d(outputChannels, outputChannels, 3, 1, 1, 1, true, causalityAxis);
    this.ninShortcut =
      inputChannels === outputChannels
        ? null
        : audioConv2d(inputChannels, outputChannels, 1, 1, 1, 1, true, causalityAxis);
    this.#inputChannels = inputChannels;
    this.#outputChannels = outputChannels;
  }

  forward(x: MxArray): MxArray {
    assertChannels(x, this.#inputChannels, "Ltx2AudioResnetBlock.forward");
    using shortcut = this.ninShortcut === null ? retainArray(x) : this.ninShortcut.forward(x);
    using normed = this.norm1.forward(x);
    using activated = silu(normed);
    using first = this.conv1.forward(activated);
    using normedFirst = this.norm2.forward(first);
    using activatedFirst = silu(normedFirst);
    using dropped = this.dropout.forward(activatedFirst);
    using residual = this.conv2.forward(dropped);
    assertChannels(residual, this.#outputChannels, "Ltx2AudioResnetBlock.forward residual");
    return add(shortcut, residual);
  }
}

/** LTX-2 audio decoder upsample block. */
export class Ltx2AudioUpsample extends Module {
  conv: Ltx2AudioConv2d;
  #channels: number;
  #causalityAxis: Ltx2AudioCausalityAxis;

  constructor(channels: number, causalityAxis: Ltx2AudioCausalityAxis = "height") {
    super();
    this.conv = audioConv2d(channels, channels, 3, 1, 1, 1, true, causalityAxis);
    this.#channels = channels;
    this.#causalityAxis = causalityAxis;
  }

  forward(x: MxArray): MxArray {
    assertChannels(x, this.#channels, "Ltx2AudioUpsample.forward");
    using upsampled = nearestUpsample2d(x);
    using convolved = this.conv.forward(upsampled);
    return cropAfterUpsample(convolved, this.#causalityAxis);
  }
}

/** LTX-2 audio decoder middle block. */
export class Ltx2AudioMidBlock extends Module {
  block1: Ltx2AudioResnetBlock;
  attn1: Ltx2AudioAttnBlock | null;
  block2: Ltx2AudioResnetBlock;

  constructor(
    channels: number,
    dropout: number,
    causalityAxis: Ltx2AudioCausalityAxis,
    addAttention: boolean,
  ) {
    super();
    this.block1 = new Ltx2AudioResnetBlock(channels, channels, dropout, causalityAxis);
    this.attn1 = addAttention ? new Ltx2AudioAttnBlock(channels) : null;
    this.block2 = new Ltx2AudioResnetBlock(channels, channels, dropout, causalityAxis);
  }

  forward(x: MxArray): MxArray {
    using first = this.block1.forward(x);
    using attended = this.attn1 === null ? retainArray(first) : this.attn1.forward(first);
    return this.block2.forward(attended);
  }
}

/** One reversed-resolution stage in the LTX-2 audio decoder. */
export class Ltx2AudioUpStage extends Module {
  block: Ltx2AudioResnetBlock[];
  attn: Ltx2AudioAttnBlock[];
  upsample: Ltx2AudioUpsample | null;

  constructor(options: {
    inputChannels: number;
    outputChannels: number;
    numBlocks: number;
    useAttention: boolean;
    hasUpsample: boolean;
    dropout: number;
    causalityAxis: Ltx2AudioCausalityAxis;
  }) {
    super();
    this.block = [];
    this.attn = [];
    let channels = options.inputChannels;
    for (let index = 0; index < options.numBlocks; index += 1) {
      const block = new Ltx2AudioResnetBlock(
        channels,
        options.outputChannels,
        options.dropout,
        options.causalityAxis,
      );
      this.block.push(block);
      channels = options.outputChannels;
      if (options.useAttention) {
        this.attn.push(new Ltx2AudioAttnBlock(channels));
      }
    }
    this.upsample = options.hasUpsample
      ? new Ltx2AudioUpsample(channels, options.causalityAxis)
      : null;
  }

  forward(x: MxArray): MxArray {
    let hidden = retainArray(x);
    try {
      for (let index = 0; index < this.block.length; index += 1) {
        const block = checkedModule(this.block, index, "Ltx2AudioUpStage.forward");
        const next = block.forward(hidden);
        hidden.free();
        hidden = next;
        const attn = this.attn[index];
        if (attn !== undefined) {
          const attended = attn.forward(hidden);
          hidden.free();
          hidden = attended;
        }
      }
      if (this.upsample !== null) {
        const upsampled = this.upsample.forward(hidden);
        hidden.free();
        hidden = upsampled;
      }
      return retainArray(hidden);
    } finally {
      hidden.free();
    }
  }
}
