import type { MxArray } from "@mlxts/core";
import { concatenate, formatShape, pad, repeat, retainArray, slice, transpose } from "@mlxts/core";
import { Conv3d, type Conv3dSpatialTriple, Module } from "@mlxts/nn";

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
    throw new Error(`LtxVideoCausalConv3d: ${name} must be a positive integer or triple.`);
  }
  return [depth, height, width];
}

/** Assert an internal LTX VAE `[batch, frames, height, width, channels]` volume. */
export function expectLtxVideoVaeVolume(
  x: MxArray,
  owner: string,
): readonly [number, number, number, number, number] {
  const [batch, frames, height, width, channels] = x.shape;
  if (
    x.shape.length !== 5 ||
    batch === undefined ||
    frames === undefined ||
    height === undefined ||
    width === undefined ||
    channels === undefined
  ) {
    throw new Error(`${owner}: expected rank-5 BFHWC volume, got ${formatShape(x.shape)}.`);
  }
  return [batch, frames, height, width, channels];
}

/** Convert Diffusers-style channel-first LTX VAE volumes from BCFHW to BFHWC. */
export function ltxVideoBcfhwToBfhwc(x: MxArray): MxArray {
  expectLtxVideoVaeVolume(x, "ltxVideoBcfhwToBfhwc input");
  return transpose(x, [0, 2, 3, 4, 1]);
}

/** Convert package-internal channel-last LTX VAE volumes from BFHWC to BCFHW. */
export function ltxVideoBfhwcToBcfhw(x: MxArray): MxArray {
  expectLtxVideoVaeVolume(x, "ltxVideoBfhwcToBcfhw");
  return transpose(x, [0, 4, 1, 2, 3]);
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
  const [, frames] = expectLtxVideoVaeVolume(x, "LtxVideoCausalConv3d.forward");
  const leftCount = isCausal ? kernelFrames - 1 : Math.floor((kernelFrames - 1) / 2);
  const rightCount = isCausal ? 0 : Math.floor((kernelFrames - 1) / 2);
  using first = leftCount > 0 ? sliceFrames(x, 0, 1) : null;
  using left = first === null ? null : repeat(first, leftCount, 1);
  using last = rightCount > 0 ? sliceFrames(x, frames - 1, frames) : null;
  using right = last === null ? null : repeat(last, rightCount, 1);
  const parts = [left, x, right].filter((part): part is MxArray => part !== null);
  return concatenate(parts, 1);
}

/** Diffusers LTX causal 3D convolution over package-internal BFHWC volumes. */
export class LtxVideoCausalConv3d extends Module {
  conv: Conv3d;
  #kernelSize: readonly [number, number, number];
  #isCausal: boolean;

  constructor(
    inputChannels: number,
    outputChannels: number,
    kernelSize: Conv3dSpatialTriple = 3,
    stride: Conv3dSpatialTriple = 1,
    dilation: Conv3dSpatialTriple = 1,
    isCausal = true,
  ) {
    super();
    const normalizedKernel = normalizeTriple(kernelSize, "kernelSize");
    const normalizedStride = normalizeTriple(stride, "stride");
    const normalizedDilation = normalizeTriple(dilation, "dilation");
    this.#kernelSize = normalizedKernel;
    this.#isCausal = isCausal;
    this.conv = new Conv3d(
      inputChannels,
      outputChannels,
      normalizedKernel,
      normalizedStride,
      0,
      normalizedDilation,
    );
  }

  forward(x: MxArray): MxArray {
    const [, , , , channels] = expectLtxVideoVaeVolume(x, "LtxVideoCausalConv3d.forward");
    if (channels !== this.conv.inputChannels) {
      throw new Error(
        `LtxVideoCausalConv3d.forward: expected ${this.conv.inputChannels} channels, got ${channels}.`,
      );
    }
    const [, heightPadding, widthPadding] = [
      this.#kernelSize[0] - 1,
      Math.floor(this.#kernelSize[1] / 2),
      Math.floor(this.#kernelSize[2] / 2),
    ];
    using temporal = edgePadFrames(x, this.#kernelSize[0], this.#isCausal);
    using padded =
      heightPadding > 0 || widthPadding > 0
        ? pad(temporal, [
            [0, 0],
            [0, 0],
            [heightPadding, heightPadding],
            [widthPadding, widthPadding],
            [0, 0],
          ])
        : retainArray(temporal);
    return this.conv.forward(padded);
  }
}
