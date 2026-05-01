import type { MxArray } from "@mlxts/core";
import { add, fastRmsNorm, reshape, retainArray, slice, transpose } from "@mlxts/core";
import { Dropout, LayerNorm, Module, silu } from "@mlxts/nn";
import { expectLtxVideoVaeVolume, LtxVideoCausalConv3d } from "./autoencoder-volume";
import type { LtxVideoAutoencoderConfig } from "./config";
import { checkedModule } from "./tensor-utils";

function assertChannels(x: MxArray, channels: number, owner: string): void {
  const [, , , , actualChannels] = expectLtxVideoVaeVolume(x, owner);
  if (actualChannels !== channels) {
    throw new Error(`${owner}: expected ${channels} channels, got ${actualChannels}.`);
  }
}

function reversed<T>(values: readonly T[]): T[] {
  return [...values].reverse();
}

function validateDecoderConfig(config: LtxVideoAutoencoderConfig): void {
  if (config.decoderBlockOutChannels.length === 0) {
    throw new Error("LtxVideoDecoder3d: decoderBlockOutChannels must not be empty.");
  }
  if (config.decoderLayersPerBlock.length !== config.decoderBlockOutChannels.length + 1) {
    throw new Error("LtxVideoDecoder3d: decoderLayersPerBlock length is invalid.");
  }
  if (config.decoderSpatioTemporalScaling.length !== config.decoderBlockOutChannels.length) {
    throw new Error("LtxVideoDecoder3d: decoderSpatioTemporalScaling length is invalid.");
  }
  if (config.timestepConditioning) {
    throw new Error("LtxVideoDecoder3d: timestep-conditioned LTX VAE decode is not supported yet.");
  }
  if (config.patchSizeT !== 1) {
    throw new Error(
      "LtxVideoDecoder3d: temporal VAE patch sizes other than 1 are not supported yet.",
    );
  }
  if (config.decoderInjectNoise.some((enabled) => enabled)) {
    throw new Error("LtxVideoDecoder3d: decoder noise injection is not supported yet.");
  }
  if (config.upsampleFactors.some((factor) => factor !== 1)) {
    throw new Error(
      "LtxVideoDecoder3d: decoder upsample factors other than 1 are not supported yet.",
    );
  }
  if (config.upsampleResidual.some((enabled) => enabled)) {
    throw new Error("LtxVideoDecoder3d: residual decoder upsampling is not supported yet.");
  }
}

/** Affine-free RMSNorm used by the classic LTX VAE. */
export class LtxVideoVaeRMSNorm extends Module {
  #channels: number;

  constructor(channels: number) {
    super();
    this.#channels = channels;
  }

  forward(x: MxArray): MxArray {
    assertChannels(x, this.#channels, "LtxVideoVaeRMSNorm.forward");
    return fastRmsNorm(x, undefined, { eps: 1e-8 });
  }
}

/** Classic LTX 3D VAE residual block. */
export class LtxVideoResnetBlock3d extends Module {
  norm1: LtxVideoVaeRMSNorm;
  conv1: LtxVideoCausalConv3d;
  norm2: LtxVideoVaeRMSNorm;
  dropout: Dropout;
  conv2: LtxVideoCausalConv3d;
  norm3: LayerNorm | null;
  convShortcut: LtxVideoCausalConv3d | null;
  #inputChannels: number;
  #outputChannels: number;

  constructor(
    inputChannels: number,
    outputChannels = inputChannels,
    dropout = 0,
    isCausal = false,
  ) {
    super();
    this.norm1 = new LtxVideoVaeRMSNorm(inputChannels);
    this.conv1 = new LtxVideoCausalConv3d(inputChannels, outputChannels, 3, 1, 1, isCausal);
    this.norm2 = new LtxVideoVaeRMSNorm(outputChannels);
    this.dropout = new Dropout(dropout);
    this.conv2 = new LtxVideoCausalConv3d(outputChannels, outputChannels, 3, 1, 1, isCausal);
    this.norm3 = inputChannels === outputChannels ? null : new LayerNorm(inputChannels);
    this.convShortcut =
      inputChannels === outputChannels
        ? null
        : new LtxVideoCausalConv3d(inputChannels, outputChannels, 1, 1, 1, isCausal);
    this.#inputChannels = inputChannels;
    this.#outputChannels = outputChannels;
  }

  forward(x: MxArray): MxArray {
    assertChannels(x, this.#inputChannels, "LtxVideoResnetBlock3d.forward");
    using shortcutNorm = this.norm3 === null ? retainArray(x) : this.norm3.forward(x);
    using shortcut =
      this.convShortcut === null
        ? retainArray(shortcutNorm)
        : this.convShortcut.forward(shortcutNorm);
    using normed = this.norm1.forward(x);
    using activated = silu(normed);
    using first = this.conv1.forward(activated);
    using normedFirst = this.norm2.forward(first);
    using activatedFirst = silu(normedFirst);
    using dropped = this.dropout.forward(activatedFirst);
    using residual = this.conv2.forward(dropped);
    assertChannels(residual, this.#outputChannels, "LtxVideoResnetBlock3d.forward residual");
    return add(shortcut, residual);
  }
}

/** Diffusers LTX decoder upsampler with spatiotemporal pixel shuffle. */
export class LtxVideoUpsampler3d extends Module {
  conv: LtxVideoCausalConv3d;
  #stride: readonly [number, number, number];
  #channels: number;

  constructor(channels: number, stride: readonly [number, number, number], isCausal = false) {
    super();
    this.#channels = channels;
    this.#stride = stride;
    this.conv = new LtxVideoCausalConv3d(
      channels,
      channels * stride[0] * stride[1] * stride[2],
      3,
      1,
      1,
      isCausal,
    );
  }

  forward(x: MxArray): MxArray {
    const [batch, frames, height, width, channels] = expectLtxVideoVaeVolume(
      x,
      "LtxVideoUpsampler3d.forward",
    );
    if (channels !== this.#channels) {
      throw new Error(
        `LtxVideoUpsampler3d.forward: expected ${this.#channels} channels, got ${channels}.`,
      );
    }
    const [frameStride, heightStride, widthStride] = this.#stride;
    using projected = this.conv.forward(x);
    const expectedChannels = channels * frameStride * heightStride * widthStride;
    if (projected.shape[4] !== expectedChannels) {
      throw new Error(
        `LtxVideoUpsampler3d.forward: expected projected channels ${expectedChannels}, got ${
          projected.shape[4] ?? "undefined"
        }.`,
      );
    }
    using grid = reshape(projected, [
      batch,
      frames,
      height,
      width,
      channels,
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
      channels,
    ]);
    return sliceLeadingFrames(upsampled, frameStride - 1);
  }
}

function sliceLeadingFrames(x: MxArray, startFrame: number): MxArray {
  if (startFrame === 0) {
    return retainArray(x);
  }
  const [batch, frames, height, width, channels] = expectLtxVideoVaeVolume(x, "sliceLeadingFrames");
  return slice(x, [0, startFrame, 0, 0, 0], [batch, frames, height, width, channels]);
}

/** LTX VAE mid block. */
export class LtxVideoMidBlock3d extends Module {
  resnets: LtxVideoResnetBlock3d[];

  constructor(channels: number, numLayers: number, isCausal = false) {
    super();
    this.resnets = Array.from(
      { length: numLayers },
      () => new LtxVideoResnetBlock3d(channels, channels, 0, isCausal),
    );
  }

  forward(x: MxArray): MxArray {
    let hidden = retainArray(x);
    try {
      for (let index = 0; index < this.resnets.length; index += 1) {
        const block = checkedModule(this.resnets, index, "LtxVideoMidBlock3d.forward");
        const next = block.forward(hidden);
        hidden.free();
        hidden = next;
      }
      return retainArray(hidden);
    } finally {
      hidden.free();
    }
  }
}

/** LTX VAE decoder up block. */
export class LtxVideoUpBlock3d extends Module {
  convIn: LtxVideoResnetBlock3d | null;
  upsampler: LtxVideoUpsampler3d | null;
  resnets: LtxVideoResnetBlock3d[];

  constructor(
    inputChannels: number,
    outputChannels: number,
    numLayers: number,
    spatioTemporalScale: boolean,
    isCausal = false,
  ) {
    super();
    this.convIn =
      inputChannels === outputChannels
        ? null
        : new LtxVideoResnetBlock3d(inputChannels, outputChannels, 0, isCausal);
    this.upsampler = spatioTemporalScale
      ? new LtxVideoUpsampler3d(outputChannels, [2, 2, 2], isCausal)
      : null;
    this.resnets = Array.from(
      { length: numLayers },
      () => new LtxVideoResnetBlock3d(outputChannels, outputChannels, 0, isCausal),
    );
  }

  forward(x: MxArray): MxArray {
    let hidden = this.convIn === null ? retainArray(x) : this.convIn.forward(x);
    try {
      if (this.upsampler !== null) {
        const upsampled = this.upsampler.forward(hidden);
        hidden.free();
        hidden = upsampled;
      }
      for (let index = 0; index < this.resnets.length; index += 1) {
        const block = checkedModule(this.resnets, index, "LtxVideoUpBlock3d.forward");
        const next = block.forward(hidden);
        hidden.free();
        hidden = next;
      }
      return retainArray(hidden);
    } finally {
      hidden.free();
    }
  }
}

/** Create decoder block parameters in Diffusers reversed order. */
export function ltxVideoDecoderLayout(config: LtxVideoAutoencoderConfig): {
  channels: number[];
  layers: number[];
  scaling: boolean[];
} {
  validateDecoderConfig(config);
  return {
    channels: reversed(config.decoderBlockOutChannels),
    layers: reversed(config.decoderLayersPerBlock),
    scaling: reversed(config.decoderSpatioTemporalScaling),
  };
}
