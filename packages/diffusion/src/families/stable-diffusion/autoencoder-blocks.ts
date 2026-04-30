/**
 * Stable Diffusion AutoencoderKL block modules.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { add, formatShape, matmul, multiply, reshape, softmax, transpose } from "@mlxts/core";
import { Conv2d, GroupNorm, Linear, Module, silu } from "@mlxts/nn";
import { padBottomRight2d, upsampleNearest2d } from "./spatial";

function assertImage4d(x: MxArray, owner: string): readonly [number, number, number, number] {
  const [batch, height, width, channels] = x.shape;
  if (
    x.shape.length !== 4 ||
    batch === undefined ||
    height === undefined ||
    width === undefined ||
    channels === undefined
  ) {
    throw new Error(`${owner}: expected rank-4 NHWC input, got ${formatShape(x.shape)}.`);
  }
  return [batch, height, width, channels];
}

function checkedResnet(
  modules: readonly StableDiffusionVaeResnetBlock2d[],
  index: number,
  owner: string,
): StableDiffusionVaeResnetBlock2d {
  const module = modules[index];
  if (module === undefined) {
    throw new Error(`${owner}: missing module at index ${index}.`);
  }
  return module;
}

/** Residual block used by the Stable Diffusion VAE encoder and decoder. */
export class StableDiffusionVaeResnetBlock2d extends Module {
  norm1: GroupNorm;
  conv1: Conv2d;
  norm2: GroupNorm;
  conv2: Conv2d;
  convShortcut: Conv2d | null;

  constructor(inChannels: number, outChannels: number, normGroups: number) {
    super();
    this.norm1 = new GroupNorm(normGroups, inChannels, 1e-6);
    this.conv1 = new Conv2d(inChannels, outChannels, 3, 1, 1);
    this.norm2 = new GroupNorm(normGroups, outChannels, 1e-6);
    this.conv2 = new Conv2d(outChannels, outChannels, 3, 1, 1);
    this.convShortcut =
      inChannels === outChannels ? null : new Conv2d(inChannels, outChannels, 1, 1, 0);
  }

  /** Run the residual block over an NHWC feature map. */
  forward(x: MxArray): MxArray {
    using normalized = this.norm1.forward(x);
    using activated = silu(normalized);
    using first = this.conv1.forward(activated);
    using normalizedFirst = this.norm2.forward(first);
    using activatedFirst = silu(normalizedFirst);
    using residual = this.conv2.forward(activatedFirst);

    if (this.convShortcut === null) {
      return add(x, residual);
    }

    using shortcut = this.convShortcut.forward(x);
    return add(shortcut, residual);
  }
}

/** Single-head spatial self-attention used by Stable Diffusion VAE mid blocks. */
export class StableDiffusionVaeAttentionBlock2d extends Module {
  groupNorm: GroupNorm;
  queryProjection: Linear;
  keyProjection: Linear;
  valueProjection: Linear;
  outputProjection: Linear;

  constructor(channels: number, normGroups: number) {
    super();
    this.groupNorm = new GroupNorm(normGroups, channels, 1e-6);
    this.queryProjection = new Linear(channels, channels);
    this.keyProjection = new Linear(channels, channels);
    this.valueProjection = new Linear(channels, channels);
    this.outputProjection = new Linear(channels, channels);
  }

  /** Run unmasked spatial self-attention over an NHWC feature map. */
  forward(x: MxArray): MxArray {
    const [batch, height, width, channels] = assertImage4d(
      x,
      "StableDiffusionVaeAttentionBlock2d.forward",
    );
    const sequenceLength = height * width;

    using normalized = this.groupNorm.forward(x);
    using rawQueries = this.queryProjection.forward(normalized);
    using queries = reshape(rawQueries, [batch, sequenceLength, channels]);
    using rawKeys = this.keyProjection.forward(normalized);
    using keys = reshape(rawKeys, [batch, sequenceLength, channels]);
    using rawValues = this.valueProjection.forward(normalized);
    using values = reshape(rawValues, [batch, sequenceLength, channels]);

    using scaledQueries = multiply(queries, 1 / Math.sqrt(channels));
    using transposedKeys = transpose(keys, [0, 2, 1]);
    using scores = matmul(scaledQueries, transposedKeys);
    using attention = softmax(scores, -1, { precise: true });
    using contextFlat = matmul(attention, values);
    using context = reshape(contextFlat, [batch, height, width, channels]);
    using projected = this.outputProjection.forward(context);
    return add(x, projected);
  }
}

/** Stable Diffusion VAE downsampling convolution with bottom/right pre-padding. */
export class StableDiffusionVaeDownsample2d extends Module {
  conv: Conv2d;

  constructor(channels: number) {
    super();
    this.conv = new Conv2d(channels, channels, 3, 2, 0);
  }

  /** Downsample an NHWC feature map by 2. */
  forward(x: MxArray): MxArray {
    using padded = padBottomRight2d(x);
    return this.conv.forward(padded);
  }
}

/** Stable Diffusion VAE nearest-neighbor upsample followed by a 3x3 convolution. */
export class StableDiffusionVaeUpsample2d extends Module {
  conv: Conv2d;

  constructor(channels: number) {
    super();
    this.conv = new Conv2d(channels, channels, 3, 1, 1);
  }

  /** Upsample an NHWC feature map by 2. */
  forward(x: MxArray): MxArray {
    using upsampled = upsampleNearest2d(x);
    return this.conv.forward(upsampled);
  }
}

/** Down encoder block used by Stable Diffusion AutoencoderKL. */
export class StableDiffusionVaeDownEncoderBlock2d extends Module {
  resnets: StableDiffusionVaeResnetBlock2d[];
  downsample: StableDiffusionVaeDownsample2d | null;

  constructor(
    inChannels: number,
    outChannels: number,
    layers: number,
    normGroups: number,
    addDownsample: boolean,
  ) {
    super();
    this.resnets = Array.from({ length: layers }, (_, index) => {
      const currentInputChannels = index === 0 ? inChannels : outChannels;
      return new StableDiffusionVaeResnetBlock2d(currentInputChannels, outChannels, normGroups);
    });
    this.downsample = addDownsample ? new StableDiffusionVaeDownsample2d(outChannels) : null;
  }

  /** Run the encoder block over an NHWC feature map. */
  forward(x: MxArray): MxArray {
    let hidden = x;
    let ownsHidden = false;
    try {
      for (let index = 0; index < this.resnets.length; index += 1) {
        const resnet = checkedResnet(
          this.resnets,
          index,
          "StableDiffusionVaeDownEncoderBlock2d.forward",
        );
        const nextHidden = resnet.forward(hidden);
        if (ownsHidden) {
          hidden.free();
        }
        hidden = nextHidden;
        ownsHidden = true;
      }

      if (this.downsample === null) {
        return hidden;
      }

      const downsampled = this.downsample.forward(hidden);
      if (ownsHidden) {
        hidden.free();
      }
      return downsampled;
    } catch (error) {
      if (ownsHidden) {
        hidden.free();
      }
      throw error;
    }
  }
}

/** Up decoder block used by Stable Diffusion AutoencoderKL. */
export class StableDiffusionVaeUpDecoderBlock2d extends Module {
  resnets: StableDiffusionVaeResnetBlock2d[];
  upsample: StableDiffusionVaeUpsample2d | null;

  constructor(
    inChannels: number,
    outChannels: number,
    layers: number,
    normGroups: number,
    addUpsample: boolean,
  ) {
    super();
    this.resnets = Array.from({ length: layers }, (_, index) => {
      const currentInputChannels = index === 0 ? inChannels : outChannels;
      return new StableDiffusionVaeResnetBlock2d(currentInputChannels, outChannels, normGroups);
    });
    this.upsample = addUpsample ? new StableDiffusionVaeUpsample2d(outChannels) : null;
  }

  /** Run the decoder block over an NHWC feature map. */
  forward(x: MxArray): MxArray {
    let hidden = x;
    let ownsHidden = false;
    try {
      for (let index = 0; index < this.resnets.length; index += 1) {
        const resnet = checkedResnet(
          this.resnets,
          index,
          "StableDiffusionVaeUpDecoderBlock2d.forward",
        );
        const nextHidden = resnet.forward(hidden);
        if (ownsHidden) {
          hidden.free();
        }
        hidden = nextHidden;
        ownsHidden = true;
      }

      if (this.upsample === null) {
        return hidden;
      }

      const upsampled = this.upsample.forward(hidden);
      if (ownsHidden) {
        hidden.free();
      }
      return upsampled;
    } catch (error) {
      if (ownsHidden) {
        hidden.free();
      }
      throw error;
    }
  }
}

/** Mid block shared by the Stable Diffusion VAE encoder and decoder. */
export class StableDiffusionVaeMidBlock2d extends Module {
  resnetIn: StableDiffusionVaeResnetBlock2d;
  attention: StableDiffusionVaeAttentionBlock2d;
  resnetOut: StableDiffusionVaeResnetBlock2d;

  constructor(channels: number, normGroups: number) {
    super();
    this.resnetIn = new StableDiffusionVaeResnetBlock2d(channels, channels, normGroups);
    this.attention = new StableDiffusionVaeAttentionBlock2d(channels, normGroups);
    this.resnetOut = new StableDiffusionVaeResnetBlock2d(channels, channels, normGroups);
  }

  /** Run the VAE mid block over an NHWC feature map. */
  forward(x: MxArray): MxArray {
    using first = this.resnetIn.forward(x);
    using attended = this.attention.forward(first);
    return this.resnetOut.forward(attended);
  }
}
