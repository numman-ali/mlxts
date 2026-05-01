import {
  add,
  formatShape,
  type MxArray,
  maximum,
  mean,
  minimum,
  multiply,
  reshape,
  retainArray,
  stack,
  tanh,
  transpose,
} from "@mlxts/core";
import { Conv1d, ConvTranspose1d, Module } from "@mlxts/nn";

import type { Ltx2VocoderConfig, Ltx2VocoderFinalActivation } from "./config";
import { checkedModule } from "./tensor-utils";

function expectBclm(x: MxArray, owner: string): readonly [number, number, number, number] {
  const [batch, channels, length, melBins] = x.shape;
  if (
    x.shape.length !== 4 ||
    batch === undefined ||
    channels === undefined ||
    length === undefined ||
    melBins === undefined
  ) {
    throw new Error(`${owner}: expected BCLM rank-4 mel spectrogram, got ${formatShape(x.shape)}.`);
  }
  return [batch, channels, length, melBins];
}

function expectBlc(x: MxArray, channels: number, owner: string): readonly [number, number, number] {
  const [batch, length, actualChannels] = x.shape;
  if (
    x.shape.length !== 3 ||
    batch === undefined ||
    length === undefined ||
    actualChannels === undefined
  ) {
    throw new Error(
      `${owner}: expected BLC rank-3 waveform features, got ${formatShape(x.shape)}.`,
    );
  }
  if (actualChannels !== channels) {
    throw new Error(`${owner}: expected ${channels} channels, got ${actualChannels}.`);
  }
  return [batch, length, actualChannels];
}

function leakyRelu(x: MxArray, negativeSlope: number): MxArray {
  using scaled = multiply(x, negativeSlope);
  return maximum(x, scaled);
}

function applyFinalActivation(x: MxArray, activation: Ltx2VocoderFinalActivation): MxArray {
  if (activation === "tanh") {
    return tanh(x);
  }
  if (activation === "clamp") {
    using lower = maximum(x, -1);
    return minimum(lower, 1);
  }
  return retainArray(x);
}

function bclmToFlattenedBlc(x: MxArray): MxArray {
  const [batch, channels, length, melBins] = expectBclm(x, "Ltx2Vocoder.forward");
  using timeMajor = transpose(x, [0, 2, 1, 3]);
  return reshape(timeMajor, [batch, length, channels * melBins]);
}

function samePadding(kernelSize: number, dilation: number): number {
  const effectiveKernelSize = dilation * (kernelSize - 1) + 1;
  if (effectiveKernelSize % 2 !== 1) {
    throw new Error("Ltx2Vocoder: same padding requires an odd effective kernel size.");
  }
  return (effectiveKernelSize - 1) / 2;
}

/** Residual block used by the default LTX-2 vocoder. */
export class Ltx2VocoderResBlock extends Module {
  convs1: Conv1d[];
  convs2: Conv1d[];
  #channels: number;
  #negativeSlope: number;

  constructor(
    channels: number,
    kernelSize: number,
    dilations: readonly number[],
    negativeSlope: number,
  ) {
    super();
    this.convs1 = dilations.map(
      (dilation) =>
        new Conv1d(channels, channels, kernelSize, 1, samePadding(kernelSize, dilation), dilation),
    );
    this.convs2 = dilations.map(
      () => new Conv1d(channels, channels, kernelSize, 1, samePadding(kernelSize, 1)),
    );
    this.#channels = channels;
    this.#negativeSlope = negativeSlope;
  }

  forward(x: MxArray): MxArray {
    let hidden = retainArray(x);
    try {
      for (let index = 0; index < this.convs1.length; index += 1) {
        const conv1 = checkedModule(this.convs1, index, "Ltx2VocoderResBlock.forward convs1");
        const conv2 = checkedModule(this.convs2, index, "Ltx2VocoderResBlock.forward convs2");
        using activated = leakyRelu(hidden, this.#negativeSlope);
        using first = conv1.forward(activated);
        using activatedFirst = leakyRelu(first, this.#negativeSlope);
        using residual = conv2.forward(activatedFirst);
        expectBlc(residual, this.#channels, "Ltx2VocoderResBlock.forward residual");
        const next = add(residual, hidden);
        hidden.free();
        hidden = next;
      }
      return retainArray(hidden);
    } finally {
      hidden.free();
    }
  }
}

/** Diffusers-compatible LTX-2.0 vocoder for BCLM mel spectrograms. */
export class Ltx2Vocoder extends Module {
  convIn: Conv1d;
  upsamplers: ConvTranspose1d[];
  resnets: Ltx2VocoderResBlock[];
  convOut: Conv1d;
  #inChannels: number;
  #outChannels: number;
  #totalUpsampleFactor: number;
  #negativeSlope: number;
  #finalActFn: Ltx2VocoderFinalActivation;
  #resnetsPerUpsample: number;

  constructor(config: Ltx2VocoderConfig) {
    super();
    if (config.actFn !== "leaky_relu") {
      throw new Error("Ltx2Vocoder: only leaky_relu activation is supported.");
    }
    if (config.antialias) {
      throw new Error("Ltx2Vocoder: antialiasing activations are not supported.");
    }
    this.convIn = new Conv1d(config.inChannels, config.hiddenChannels, 7, 1, 3);
    this.upsamplers = [];
    this.resnets = [];
    let inputChannels = config.hiddenChannels;
    for (let index = 0; index < config.upsampleFactors.length; index += 1) {
      const stride = checkedNumber(config.upsampleFactors, index, "Ltx2Vocoder.upsampleFactors");
      const kernelSize = checkedNumber(
        config.upsampleKernelSizes,
        index,
        "Ltx2Vocoder.upsampleKernelSizes",
      );
      const outputChannels = inputChannels / 2;
      if (!Number.isInteger(outputChannels) || outputChannels <= 0) {
        throw new Error("Ltx2Vocoder: hidden channels must halve cleanly at each upsample.");
      }
      this.upsamplers.push(
        new ConvTranspose1d(
          inputChannels,
          outputChannels,
          kernelSize,
          stride,
          Math.floor((kernelSize - stride) / 2),
        ),
      );
      for (let resnetIndex = 0; resnetIndex < config.resnetKernelSizes.length; resnetIndex += 1) {
        const resnetKernelSize = checkedNumber(
          config.resnetKernelSizes,
          resnetIndex,
          "Ltx2Vocoder.resnetKernelSizes",
        );
        const dilations = checkedDilationRow(
          config.resnetDilations,
          resnetIndex,
          "Ltx2Vocoder.resnetDilations",
        );
        this.resnets.push(
          new Ltx2VocoderResBlock(
            outputChannels,
            resnetKernelSize,
            dilations,
            config.leakyReluNegativeSlope,
          ),
        );
      }
      inputChannels = outputChannels;
    }
    this.convOut = new Conv1d(inputChannels, config.outChannels, 7, 1, 3, 1, 1, config.finalBias);
    this.#inChannels = config.inChannels;
    this.#outChannels = config.outChannels;
    this.#totalUpsampleFactor = config.totalUpsampleFactor;
    this.#negativeSlope = config.leakyReluNegativeSlope;
    this.#finalActFn = config.finalActFn;
    this.#resnetsPerUpsample = config.resnetKernelSizes.length;
  }

  get inChannels(): number {
    return this.#inChannels;
  }

  get outChannels(): number {
    return this.#outChannels;
  }

  get totalUpsampleFactor(): number {
    return this.#totalUpsampleFactor;
  }

  forward(melSpectrograms: MxArray): MxArray {
    using flattened = bclmToFlattenedBlc(melSpectrograms);
    expectBlc(flattened, this.#inChannels, "Ltx2Vocoder.forward input");
    let hidden = this.convIn.forward(flattened);
    try {
      for (let index = 0; index < this.upsamplers.length; index += 1) {
        using activated = leakyRelu(hidden, this.#negativeSlope);
        const upsampler = checkedModule(this.upsamplers, index, "Ltx2Vocoder.forward upsamplers");
        const upsampled = upsampler.forward(activated);
        hidden.free();
        hidden = upsampled;
        const averaged = this.forwardResnetGroup(hidden, index);
        hidden.free();
        hidden = averaged;
      }
      using actOut = leakyRelu(hidden, 0.01);
      using projected = this.convOut.forward(actOut);
      using activated = applyFinalActivation(projected, this.#finalActFn);
      return transpose(activated, [0, 2, 1]);
    } finally {
      hidden.free();
    }
  }

  private forwardResnetGroup(hidden: MxArray, upsampleIndex: number): MxArray {
    const outputs: MxArray[] = [];
    try {
      const start = upsampleIndex * this.#resnetsPerUpsample;
      const end = start + this.#resnetsPerUpsample;
      for (let index = start; index < end; index += 1) {
        const resnet = checkedModule(this.resnets, index, "Ltx2Vocoder.forward resnets");
        outputs.push(resnet.forward(hidden));
      }
      using stacked = stack(outputs, 0);
      return mean(stacked, 0);
    } finally {
      for (const output of outputs) {
        output.free();
      }
    }
  }
}

function checkedNumber(values: readonly number[], index: number, owner: string): number {
  const value = values[index];
  if (value === undefined) {
    throw new Error(`${owner}: missing numeric value at index ${index}.`);
  }
  return value;
}

function checkedDilationRow(
  values: readonly (readonly number[])[],
  index: number,
  owner: string,
): readonly number[] {
  const value = values[index];
  if (value === undefined) {
    throw new Error(`${owner}: missing dilation row at index ${index}.`);
  }
  return value;
}
