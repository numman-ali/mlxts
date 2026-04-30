/**
 * FLUX.1 attention projections and RoPE application.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import {
  add,
  multiply,
  reshape,
  retainArray,
  scaledDotProductAttention,
  split,
  transpose,
} from "@mlxts/core";
import { Linear, Module, RMSNorm } from "@mlxts/nn";

import { assertAttention4d, assertSequence3d, freeArrays, selectLastAxis } from "./tensor-utils";

export type FluxAttentionProjection = {
  queries: MxArray;
  keys: MxArray;
  values: MxArray;
};

function partAt(parts: readonly MxArray[], index: number, owner: string): MxArray {
  const part = parts[index];
  if (part === undefined) {
    throw new Error(`${owner}: projection split failed.`);
  }
  return part;
}

/** Apply FLUX RoPE matrices to `[batch, heads, length, headDim]` q/k tensors. */
export function applyFluxRotary(x: MxArray, rope: MxArray): MxArray {
  const { batch, heads, length, headDim } = assertAttention4d(x, "applyFluxRotary");
  if (headDim % 2 !== 0) {
    throw new Error("applyFluxRotary: headDim must be even.");
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
    throw new Error("applyFluxRotary: rope shape must match attention length and headDim.");
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

/** Shared FLUX attention kernel over already-projected q/k/v tensors. */
export function fluxAttention(
  queries: MxArray,
  keys: MxArray,
  values: MxArray,
  rope: MxArray,
): MxArray {
  const { batch, heads, length, headDim } = assertAttention4d(queries, "fluxAttention");
  const keyShape = assertAttention4d(keys, "fluxAttention");
  const valueShape = assertAttention4d(values, "fluxAttention");
  if (
    keyShape.batch !== batch ||
    keyShape.heads !== heads ||
    keyShape.length !== length ||
    keyShape.headDim !== headDim ||
    valueShape.batch !== batch ||
    valueShape.heads !== heads ||
    valueShape.length !== length ||
    valueShape.headDim !== headDim
  ) {
    throw new Error("fluxAttention: q, k, and v shapes must match.");
  }
  using rotaryQueries = applyFluxRotary(queries, rope);
  using rotaryKeys = applyFluxRotary(keys, rope);
  using attended = scaledDotProductAttention(rotaryQueries, rotaryKeys, values, {
    scale: headDim ** -0.5,
  });
  using sequenceFirst = transpose(attended, [0, 2, 1, 3]);
  return reshape(sequenceFirst, [batch, length, heads * headDim]);
}

/** FLUX Q/K RMS normalization pair. */
export class FluxQKNorm extends Module {
  queryNorm: RMSNorm;
  keyNorm: RMSNorm;

  constructor(headDim: number) {
    super();
    this.queryNorm = new RMSNorm(headDim, 1e-6);
    this.keyNorm = new RMSNorm(headDim, 1e-6);
  }

  normalize(queries: MxArray, keys: MxArray): { queries: MxArray; keys: MxArray } {
    return {
      queries: this.queryNorm.forward(queries),
      keys: this.keyNorm.forward(keys),
    };
  }

  forward(queries: MxArray): MxArray {
    return this.queryNorm.forward(queries);
  }
}

/** FLUX self-attention projection surface used by double and single stream blocks. */
export class FluxSelfAttention extends Module {
  qkv: Linear;
  norm: FluxQKNorm;
  projection: Linear;
  #numHeads: number;
  #headDim: number;
  #hiddenSize: number;

  constructor(hiddenSize: number, numHeads: number, headDim: number, qkvBias: boolean) {
    super();
    if (hiddenSize !== numHeads * headDim) {
      throw new Error("FluxSelfAttention: hiddenSize must equal numHeads * headDim.");
    }
    this.qkv = new Linear(hiddenSize, hiddenSize * 3, qkvBias);
    this.norm = new FluxQKNorm(headDim);
    this.projection = new Linear(hiddenSize, hiddenSize);
    this.#numHeads = numHeads;
    this.#headDim = headDim;
    this.#hiddenSize = hiddenSize;
  }

  forward(_input: MxArray): MxArray {
    throw new Error("FluxSelfAttention.forward: use project() inside a FLUX block.");
  }

  project(input: MxArray): FluxAttentionProjection {
    const { batch, length } = assertSequence3d(input, "FluxSelfAttention.project");
    using projected = this.qkv.forward(input);
    const parts = split(projected, 3, -1);
    try {
      const rawQueries = partAt(parts, 0, "FluxSelfAttention.project");
      const rawKeys = partAt(parts, 1, "FluxSelfAttention.project");
      const rawValues = partAt(parts, 2, "FluxSelfAttention.project");
      using queries = reshape(rawQueries, [batch, length, this.#numHeads, this.#headDim]);
      using keys = reshape(rawKeys, [batch, length, this.#numHeads, this.#headDim]);
      using values = reshape(rawValues, [batch, length, this.#numHeads, this.#headDim]);
      using transposedQueries = transpose(queries, [0, 2, 1, 3]);
      using transposedKeys = transpose(keys, [0, 2, 1, 3]);
      using transposedValues = transpose(values, [0, 2, 1, 3]);
      const normalized = this.norm.normalize(transposedQueries, transposedKeys);
      return {
        queries: normalized.queries,
        keys: normalized.keys,
        values: retainArray(transposedValues),
      };
    } finally {
      freeArrays(parts);
    }
  }

  projectOutput(input: MxArray): MxArray {
    return this.projection.forward(input);
  }

  get hiddenSize(): number {
    return this.#hiddenSize;
  }
}
