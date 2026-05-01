/**
 * Z-Image single-stream attention.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import {
  add,
  expandDims,
  formatShape,
  multiply,
  reshape,
  retainArray,
  scaledDotProductAttention,
  transpose,
} from "@mlxts/core";
import { Linear, Module, RMSNorm } from "@mlxts/nn";

import { assertAttention4d, assertSequence3d, selectLastAxis } from "./tensor-utils";

/** Apply Z-Image RoPE matrices to `[batch, heads, length, headDim]` q/k tensors. */
export function applyZImageRotary(x: MxArray, rope: MxArray): MxArray {
  const { batch, heads, length, headDim } = assertAttention4d(x, "applyZImageRotary");
  if (headDim % 2 !== 0) {
    throw new Error("applyZImageRotary: headDim must be even.");
  }
  const expectedRope = [1, 1, length, headDim / 2, 2, 2];
  if (
    rope.shape.length !== 6 ||
    rope.shape[0] !== expectedRope[0] ||
    rope.shape[1] !== expectedRope[1] ||
    rope.shape[2] !== expectedRope[2] ||
    rope.shape[3] !== expectedRope[3] ||
    rope.shape[4] !== expectedRope[4] ||
    rope.shape[5] !== expectedRope[5]
  ) {
    throw new Error(
      `applyZImageRotary: rope shape must be ${formatShape(expectedRope)}, got ${formatShape(rope.shape)}.`,
    );
  }

  using pairs = reshape(x, [batch, heads, length, headDim / 2, 1, 2]);
  using x0 = selectLastAxis(pairs, 0);
  using x1 = selectLastAxis(pairs, 1);
  using rope0 = selectLastAxis(rope, 0);
  using rope1 = selectLastAxis(rope, 1);
  using rotated0 = multiply(x0, rope0);
  using rotated1 = multiply(x1, rope1);
  using rotated = add(rotated0, rotated1);
  return reshape(rotated, [batch, heads, length, headDim]);
}

function attentionMaskForKernel(mask: MxArray | null): MxArray | null {
  if (mask === null) {
    return null;
  }
  if (mask.shape.length !== 2) {
    return retainArray(mask);
  }
  using expandedBatch = expandDims(mask, 1);
  return expandDims(expandedBatch, 1);
}

/** Z-Image q/k/v attention with optional q/k RMS normalization. */
export class ZImageSelfAttention extends Module {
  query: Linear;
  key: Linear;
  value: Linear;
  queryNorm: RMSNorm | null;
  keyNorm: RMSNorm | null;
  projection: Linear;
  #numHeads: number;
  #headDim: number;
  #hiddenSize: number;

  constructor(hiddenSize: number, numHeads: number, qkNorm: boolean, normEps: number) {
    super();
    if (hiddenSize % numHeads !== 0) {
      throw new Error("ZImageSelfAttention: hiddenSize must divide numHeads evenly.");
    }
    this.#numHeads = numHeads;
    this.#headDim = hiddenSize / numHeads;
    this.#hiddenSize = hiddenSize;
    this.query = new Linear(hiddenSize, hiddenSize, false);
    this.key = new Linear(hiddenSize, hiddenSize, false);
    this.value = new Linear(hiddenSize, hiddenSize, false);
    this.queryNorm = qkNorm ? new RMSNorm(this.#headDim, normEps) : null;
    this.keyNorm = qkNorm ? new RMSNorm(this.#headDim, normEps) : null;
    this.projection = new Linear(hiddenSize, hiddenSize, false);
  }

  /** Run non-causal Z-Image self-attention over one hidden stream. */
  forward(hiddenStates: MxArray, rope: MxArray, attentionMask: MxArray | null = null): MxArray {
    const { batch, length, channels } = assertSequence3d(
      hiddenStates,
      "ZImageSelfAttention.forward",
    );
    if (channels !== this.#hiddenSize) {
      throw new Error("ZImageSelfAttention.forward: hidden size mismatch.");
    }

    using rawQueries = this.query.forward(hiddenStates);
    using rawKeys = this.key.forward(hiddenStates);
    using rawValues = this.value.forward(hiddenStates);
    using queries = this.#project(rawQueries, this.queryNorm);
    using keys = this.#project(rawKeys, this.keyNorm);
    using values = this.#project(rawValues, null);
    using rotaryQueries = applyZImageRotary(queries, rope);
    using rotaryKeys = applyZImageRotary(keys, rope);
    const kernelMask = attentionMaskForKernel(attentionMask);
    try {
      using attended = scaledDotProductAttention(rotaryQueries, rotaryKeys, values, {
        scale: this.#headDim ** -0.5,
        ...(kernelMask === null ? {} : { maskArray: kernelMask }),
      });
      using sequenceFirst = transpose(attended, [0, 2, 1, 3]);
      using flattened = reshape(sequenceFirst, [batch, length, this.#hiddenSize]);
      return this.projection.forward(flattened);
    } finally {
      kernelMask?.free();
    }
  }

  #project(projected: MxArray, norm: RMSNorm | null): MxArray {
    const { batch, length } = assertSequence3d(projected, "ZImageSelfAttention.project");
    using heads = reshape(projected, [batch, length, this.#numHeads, this.#headDim]);
    using normalized = norm === null ? retainArray(heads) : norm.forward(heads);
    return transpose(normalized, [0, 2, 1, 3]);
  }

  get hiddenSize(): number {
    return this.#hiddenSize;
  }
}
