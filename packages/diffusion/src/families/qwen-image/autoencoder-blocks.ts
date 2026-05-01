import type { MxArray } from "@mlxts/core";
import {
  add,
  fastRmsNorm,
  formatShape,
  ones,
  pad,
  repeat,
  reshape,
  retainArray,
  scaledDotProductAttention,
  split,
  zeros,
} from "@mlxts/core";
import { Conv2d, Dropout, Module, silu } from "@mlxts/nn";
import { expectQwenImageVaeVolume, QwenImageCausalConv3d } from "./autoencoder-volume";

export {
  QwenImageCausalConv3d,
  type QwenImageSpatialTriple,
  qwenImageNcfhwToNdhwc,
  qwenImageNdhwcToNcfhw,
} from "./autoencoder-volume";

/** Qwen-Image VAE RMS normalization over channel-last feature axes. */
export class QwenImageRMSNorm extends Module {
  weight: MxArray;
  bias: MxArray | null;
  #channels: number;

  constructor(channels: number, hasBias = false) {
    super();
    if (!Number.isInteger(channels) || channels <= 0) {
      throw new Error(`QwenImageRMSNorm: channels must be a positive integer, got ${channels}.`);
    }
    this.#channels = channels;
    this.weight = ones([channels]);
    this.bias = hasBias ? zeros([channels]) : null;
  }

  forward(x: MxArray): MxArray {
    const channels = x.shape[x.shape.length - 1];
    if (channels !== this.#channels) {
      throw new Error(
        `QwenImageRMSNorm.forward: expected last dimension ${this.#channels}, got ${
          channels ?? "undefined"
        } for shape ${formatShape(x.shape)}.`,
      );
    }

    const normalized = fastRmsNorm(x, this.weight, { eps: 1e-12 });
    if (this.bias === null) {
      return normalized;
    }

    using unbiased = normalized;
    return add(unbiased, this.bias);
  }
}

export type QwenImageResampleMode =
  | "none"
  | "upsample2d"
  | "upsample3d"
  | "downsample2d"
  | "downsample3d";

function upsampleNearest2d(x: MxArray): MxArray {
  using heightRepeated = repeat(x, 2, 1);
  return repeat(heightRepeated, 2, 2);
}

function padBottomRight2d(x: MxArray): MxArray {
  return pad(x, [
    [0, 0],
    [0, 1],
    [0, 1],
    [0, 0],
  ]);
}

/** Qwen-Image spatial resampling block used by the 3D VAE encoder and decoder. */
export class QwenImageResample extends Module {
  resample: Conv2d | null;
  timeConv: QwenImageCausalConv3d | null;
  #mode: QwenImageResampleMode;
  #channels: number;

  constructor(channels: number, mode: QwenImageResampleMode) {
    super();
    if (!Number.isInteger(channels) || channels <= 0) {
      throw new Error(`QwenImageResample: channels must be a positive integer, got ${channels}.`);
    }
    if ((mode === "upsample2d" || mode === "upsample3d") && channels % 2 !== 0) {
      throw new Error("QwenImageResample: upsample modes require an even channel count.");
    }
    this.#mode = mode;
    this.#channels = channels;

    if (mode === "upsample2d" || mode === "upsample3d") {
      this.resample = new Conv2d(channels, channels / 2, 3, 1, 1);
    } else if (mode === "downsample2d" || mode === "downsample3d") {
      this.resample = new Conv2d(channels, channels, 3, [2, 2], 0);
    } else {
      this.resample = null;
    }

    this.timeConv =
      mode === "upsample3d"
        ? new QwenImageCausalConv3d(channels, channels * 2, [3, 1, 1], 1, [1, 0, 0])
        : mode === "downsample3d"
          ? new QwenImageCausalConv3d(channels, channels, [3, 1, 1], [2, 1, 1], 0)
          : null;
  }

  get mode(): QwenImageResampleMode {
    return this.#mode;
  }

  forward(x: MxArray): MxArray {
    const [batch, frames, height, width, channels] = expectQwenImageVaeVolume(
      x,
      "QwenImageResample.forward",
    );
    if (channels !== this.#channels) {
      throw new Error(
        `QwenImageResample.forward: expected last dimension ${this.#channels}, got ${channels}.`,
      );
    }
    if (this.resample === null) {
      return retainArray(x);
    }

    using flattened = reshape(x, [batch * frames, height, width, channels]);
    let spatial: MxArray;
    if (this.#mode === "upsample2d" || this.#mode === "upsample3d") {
      using upsampled = upsampleNearest2d(flattened);
      spatial = this.resample.forward(upsampled);
    } else {
      using padded = padBottomRight2d(flattened);
      spatial = this.resample.forward(padded);
    }

    using spatialOutput = spatial;
    const [, outputHeight, outputWidth, outputChannels] = spatialOutput.shape;
    if (outputHeight === undefined || outputWidth === undefined || outputChannels === undefined) {
      throw new Error(
        `QwenImageResample.forward: resample returned malformed shape ${formatShape(
          spatialOutput.shape,
        )}.`,
      );
    }
    return reshape(spatialOutput, [batch, frames, outputHeight, outputWidth, outputChannels]);
  }
}

/** Residual block used by the Qwen-Image VAE encoder and decoder. */
export class QwenImageResidualBlock extends Module {
  norm1: QwenImageRMSNorm;
  conv1: QwenImageCausalConv3d;
  norm2: QwenImageRMSNorm;
  dropout: Dropout;
  conv2: QwenImageCausalConv3d;
  convShortcut: QwenImageCausalConv3d | null;

  constructor(inputChannels: number, outputChannels: number, dropout = 0) {
    super();
    this.norm1 = new QwenImageRMSNorm(inputChannels);
    this.conv1 = new QwenImageCausalConv3d(inputChannels, outputChannels, 3, 1, 1);
    this.norm2 = new QwenImageRMSNorm(outputChannels);
    this.dropout = new Dropout(dropout);
    this.conv2 = new QwenImageCausalConv3d(outputChannels, outputChannels, 3, 1, 1);
    this.convShortcut =
      inputChannels === outputChannels
        ? null
        : new QwenImageCausalConv3d(inputChannels, outputChannels, 1);
  }

  forward(x: MxArray): MxArray {
    using shortcut = this.convShortcut === null ? retainArray(x) : this.convShortcut.forward(x);
    using normalized = this.norm1.forward(x);
    using activated = silu(normalized);
    using first = this.conv1.forward(activated);
    using normalizedFirst = this.norm2.forward(first);
    using activatedFirst = silu(normalizedFirst);
    using dropped = this.dropout.forward(activatedFirst);
    using residual = this.conv2.forward(dropped);
    return add(shortcut, residual);
  }
}

/** Single-head spatial attention block used inside the Qwen-Image VAE. */
export class QwenImageAttentionBlock extends Module {
  norm: QwenImageRMSNorm;
  toQkv: Conv2d;
  proj: Conv2d;
  #channels: number;

  constructor(channels: number) {
    super();
    if (!Number.isInteger(channels) || channels <= 0) {
      throw new Error(
        `QwenImageAttentionBlock: channels must be a positive integer, got ${channels}.`,
      );
    }
    this.#channels = channels;
    this.norm = new QwenImageRMSNorm(channels);
    this.toQkv = new Conv2d(channels, channels * 3, 1);
    this.proj = new Conv2d(channels, channels, 1);
  }

  forward(x: MxArray): MxArray {
    const [batch, frames, height, width, channels] = expectQwenImageVaeVolume(
      x,
      "QwenImageAttentionBlock.forward",
    );
    if (channels !== this.#channels) {
      throw new Error(
        `QwenImageAttentionBlock.forward: expected last dimension ${this.#channels}, got ${channels}.`,
      );
    }
    const batchFrames = batch * frames;
    const sequenceLength = height * width;

    using flat = reshape(x, [batchFrames, height, width, channels]);
    using normalized = this.norm.forward(flat);
    using qkvImage = this.toQkv.forward(normalized);
    using qkv = reshape(qkvImage, [batchFrames, sequenceLength, channels * 3]);
    const parts = split(qkv, 3, -1);
    const queriesFlat = parts[0];
    const keysFlat = parts[1];
    const valuesFlat = parts[2];
    if (queriesFlat === undefined || keysFlat === undefined || valuesFlat === undefined) {
      throw new Error("QwenImageAttentionBlock.forward: expected q/k/v split outputs.");
    }

    try {
      using queries = reshape(queriesFlat, [batchFrames, 1, sequenceLength, channels]);
      using keys = reshape(keysFlat, [batchFrames, 1, sequenceLength, channels]);
      using values = reshape(valuesFlat, [batchFrames, 1, sequenceLength, channels]);
      using attended = scaledDotProductAttention(queries, keys, values, {
        scale: 1 / Math.sqrt(channels),
      });
      using attendedImage = reshape(attended, [batchFrames, height, width, channels]);
      using projected = this.proj.forward(attendedImage);
      using volume = reshape(projected, [batch, frames, height, width, channels]);
      return add(x, volume);
    } finally {
      queriesFlat.free();
      keysFlat.free();
      valuesFlat.free();
    }
  }
}

/** Middle block used by Qwen-Image VAE encoder and decoder modules. */
export class QwenImageMidBlock extends Module {
  resnets: QwenImageResidualBlock[];
  attentions: QwenImageAttentionBlock[];

  constructor(channels: number, dropout = 0, numLayers = 1) {
    super();
    if (!Number.isInteger(numLayers) || numLayers < 0) {
      throw new Error(
        `QwenImageMidBlock: numLayers must be a non-negative integer, got ${numLayers}.`,
      );
    }
    this.resnets = [new QwenImageResidualBlock(channels, channels, dropout)];
    this.attentions = [];
    for (let index = 0; index < numLayers; index += 1) {
      this.attentions.push(new QwenImageAttentionBlock(channels));
      this.resnets.push(new QwenImageResidualBlock(channels, channels, dropout));
    }
  }

  forward(x: MxArray): MxArray {
    const first = this.resnets[0];
    if (first === undefined) {
      throw new Error("QwenImageMidBlock.forward: missing first residual block.");
    }
    let hidden = first.forward(x);
    try {
      for (let index = 0; index < this.attentions.length; index += 1) {
        const attention = this.attentions[index];
        const resnet = this.resnets[index + 1];
        if (attention === undefined || resnet === undefined) {
          throw new Error(`QwenImageMidBlock.forward: missing block at index ${index}.`);
        }

        const attended = attention.forward(hidden);
        hidden.free();
        hidden = attended;
        const residual = resnet.forward(hidden);
        hidden.free();
        hidden = residual;
      }
      return hidden;
    } catch (error) {
      hidden.free();
      throw error;
    }
  }
}

/** Decoder up block used by the Qwen-Image VAE. */
export class QwenImageUpBlock extends Module {
  resnets: QwenImageResidualBlock[];
  upsampler: QwenImageResample | null;

  constructor(
    inputChannels: number,
    outputChannels: number,
    numResBlocks: number,
    dropout = 0,
    upsampleMode: Extract<QwenImageResampleMode, "upsample2d" | "upsample3d"> | null = null,
  ) {
    super();
    if (!Number.isInteger(numResBlocks) || numResBlocks < 0) {
      throw new Error(
        `QwenImageUpBlock: numResBlocks must be a non-negative integer, got ${numResBlocks}.`,
      );
    }
    this.resnets = Array.from({ length: numResBlocks + 1 }, (_, index) => {
      const currentInputChannels = index === 0 ? inputChannels : outputChannels;
      return new QwenImageResidualBlock(currentInputChannels, outputChannels, dropout);
    });
    this.upsampler =
      upsampleMode === null ? null : new QwenImageResample(outputChannels, upsampleMode);
  }

  forward(x: MxArray): MxArray {
    let hidden = retainArray(x);
    try {
      for (let index = 0; index < this.resnets.length; index += 1) {
        const resnet = this.resnets[index];
        if (resnet === undefined) {
          throw new Error(`QwenImageUpBlock.forward: missing residual block at index ${index}.`);
        }
        const residual = resnet.forward(hidden);
        hidden.free();
        hidden = residual;
      }

      if (this.upsampler === null) {
        return hidden;
      }

      const upsampled = this.upsampler.forward(hidden);
      hidden.free();
      return upsampled;
    } catch (error) {
      hidden.free();
      throw error;
    }
  }
}
