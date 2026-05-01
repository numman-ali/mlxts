/**
 * FLUX.2 Klein attention projections and RoPE application.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import {
  add,
  concatenate,
  multiply,
  reshape,
  retainArray,
  scaledDotProductAttention,
  transpose,
} from "@mlxts/core";
import { Linear, Module, RMSNorm, swiglu } from "@mlxts/nn";

import {
  assertFlux2Attention4d,
  assertFlux2Sequence3d,
  selectFlux2LastAxis,
  sliceFlux2Axis,
  sliceFlux2LastAxis,
} from "./tensor-utils";

export type Flux2AttentionPair = {
  image: MxArray;
  text: MxArray;
};

export type Flux2AttentionProjection = {
  queries: MxArray;
  keys: MxArray;
  values: MxArray;
};

function disposeProjection(projection: Flux2AttentionProjection): void {
  projection.queries.free();
  projection.keys.free();
  projection.values.free();
}

/** Apply FLUX.2 RoPE matrices to `[batch, heads, length, headDim]` q/k tensors. */
export function applyFlux2Rotary(x: MxArray, rope: MxArray): MxArray {
  const { batch, heads, length, headDim } = assertFlux2Attention4d(x, "applyFlux2Rotary");
  if (headDim % 2 !== 0) {
    throw new Error("applyFlux2Rotary: headDim must be even.");
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
    throw new Error("applyFlux2Rotary: rope shape must match attention length and headDim.");
  }

  using pairs = reshape(x, [batch, heads, length, headDim / 2, 1, 2]);
  using x0 = selectFlux2LastAxis(pairs, 0);
  using x1 = selectFlux2LastAxis(pairs, 1);
  using rope0 = selectFlux2LastAxis(rope, 0);
  using rope1 = selectFlux2LastAxis(rope, 1);
  using rotated0 = multiply(x0, rope0);
  using rotated1 = multiply(x1, rope1);
  using rotated = add(rotated0, rotated1);
  return reshape(rotated, [batch, heads, length, headDim]);
}

/** Shared FLUX.2 attention kernel over already-projected q/k/v tensors. */
export function flux2Attention(
  queries: MxArray,
  keys: MxArray,
  values: MxArray,
  rope: MxArray,
): MxArray {
  const { batch, heads, length, headDim } = assertFlux2Attention4d(queries, "flux2Attention");
  const keyShape = assertFlux2Attention4d(keys, "flux2Attention");
  const valueShape = assertFlux2Attention4d(values, "flux2Attention");
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
    throw new Error("flux2Attention: q, k, and v shapes must match.");
  }
  using rotaryQueries = applyFlux2Rotary(queries, rope);
  using rotaryKeys = applyFlux2Rotary(keys, rope);
  using attended = scaledDotProductAttention(rotaryQueries, rotaryKeys, values, {
    scale: headDim ** -0.5,
  });
  using sequenceFirst = transpose(attended, [0, 2, 1, 3]);
  return reshape(sequenceFirst, [batch, length, heads * headDim]);
}

/** FLUX.2 Q/K RMS normalization pair. */
export class Flux2QKNorm extends Module {
  queryNorm: RMSNorm;
  keyNorm: RMSNorm;

  constructor(headDim: number, eps: number) {
    super();
    this.queryNorm = new RMSNorm(headDim, eps);
    this.keyNorm = new RMSNorm(headDim, eps);
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

/** Joint FLUX.2 attention over separate image and text streams. */
export class Flux2JointAttention extends Module {
  toQ: Linear;
  toK: Linear;
  toV: Linear;
  addQProj: Linear;
  addKProj: Linear;
  addVProj: Linear;
  norm: Flux2QKNorm;
  addedNorm: Flux2QKNorm;
  toOut: Linear;
  toAddOut: Linear;
  #numHeads: number;
  #headDim: number;

  constructor(hiddenSize: number, numHeads: number, headDim: number, eps: number) {
    super();
    if (hiddenSize !== numHeads * headDim) {
      throw new Error("Flux2JointAttention: hiddenSize must equal numHeads * headDim.");
    }
    this.toQ = new Linear(hiddenSize, hiddenSize, false);
    this.toK = new Linear(hiddenSize, hiddenSize, false);
    this.toV = new Linear(hiddenSize, hiddenSize, false);
    this.addQProj = new Linear(hiddenSize, hiddenSize, false);
    this.addKProj = new Linear(hiddenSize, hiddenSize, false);
    this.addVProj = new Linear(hiddenSize, hiddenSize, false);
    this.norm = new Flux2QKNorm(headDim, eps);
    this.addedNorm = new Flux2QKNorm(headDim, eps);
    this.toOut = new Linear(hiddenSize, hiddenSize, false);
    this.toAddOut = new Linear(hiddenSize, hiddenSize, false);
    this.#numHeads = numHeads;
    this.#headDim = headDim;
  }

  forward(_image: MxArray): MxArray {
    throw new Error("Flux2JointAttention.forward: use run() inside a FLUX.2 block.");
  }

  /** Run joint `[text, image]` attention and return separated projected streams. */
  run(image: MxArray, text: MxArray, rope: MxArray): Flux2AttentionPair {
    const imageProjection = this.#projectImage(image);
    const textProjection = this.#projectText(text);
    try {
      using queries = concatenate([textProjection.queries, imageProjection.queries], 2);
      using keys = concatenate([textProjection.keys, imageProjection.keys], 2);
      using values = concatenate([textProjection.values, imageProjection.values], 2);
      using attended = flux2Attention(queries, keys, values, rope);
      using textAttention = sliceFlux2Axis(attended, 1, 0, text.shape[1] ?? 0);
      using imageAttention = sliceFlux2Axis(
        attended,
        1,
        text.shape[1] ?? 0,
        attended.shape[1] ?? 0,
      );
      using projectedText = this.toAddOut.forward(textAttention);
      using projectedImage = this.toOut.forward(imageAttention);
      return { text: retainArray(projectedText), image: retainArray(projectedImage) };
    } finally {
      disposeProjection(imageProjection);
      disposeProjection(textProjection);
    }
  }

  #projectImage(input: MxArray): Flux2AttentionProjection {
    return this.#project(input, this.toQ, this.toK, this.toV, this.norm, "image");
  }

  #projectText(input: MxArray): Flux2AttentionProjection {
    return this.#project(
      input,
      this.addQProj,
      this.addKProj,
      this.addVProj,
      this.addedNorm,
      "text",
    );
  }

  #project(
    input: MxArray,
    queryProjection: Linear,
    keyProjection: Linear,
    valueProjection: Linear,
    norm: Flux2QKNorm,
    streamName: string,
  ): Flux2AttentionProjection {
    const { batch, length } = assertFlux2Sequence3d(
      input,
      `Flux2JointAttention.project ${streamName}`,
    );
    using queriesProjection = queryProjection.forward(input);
    using keysProjection = keyProjection.forward(input);
    using valuesProjection = valueProjection.forward(input);
    using queries = reshape(queriesProjection, [batch, length, this.#numHeads, this.#headDim]);
    using keys = reshape(keysProjection, [batch, length, this.#numHeads, this.#headDim]);
    using values = reshape(valuesProjection, [batch, length, this.#numHeads, this.#headDim]);
    using transposedQueries = transpose(queries, [0, 2, 1, 3]);
    using transposedKeys = transpose(keys, [0, 2, 1, 3]);
    using transposedValues = transpose(values, [0, 2, 1, 3]);
    const normalized = norm.normalize(transposedQueries, transposedKeys);
    return {
      queries: normalized.queries,
      keys: normalized.keys,
      values: retainArray(transposedValues),
    };
  }
}

/** FLUX.2 parallel self-attention with fused QKV and SwiGLU input projection. */
export class Flux2ParallelSelfAttention extends Module {
  toQkvMlpProj: Linear;
  norm: Flux2QKNorm;
  toOut: Linear;
  #numHeads: number;
  #headDim: number;
  #hiddenSize: number;
  #mlpHiddenSize: number;

  constructor(
    hiddenSize: number,
    numHeads: number,
    headDim: number,
    mlpHiddenSize: number,
    eps: number,
  ) {
    super();
    if (hiddenSize !== numHeads * headDim) {
      throw new Error("Flux2ParallelSelfAttention: hiddenSize must equal numHeads * headDim.");
    }
    this.toQkvMlpProj = new Linear(hiddenSize, hiddenSize * 3 + mlpHiddenSize * 2, false);
    this.norm = new Flux2QKNorm(headDim, eps);
    this.toOut = new Linear(hiddenSize + mlpHiddenSize, hiddenSize, false);
    this.#numHeads = numHeads;
    this.#headDim = headDim;
    this.#hiddenSize = hiddenSize;
    this.#mlpHiddenSize = mlpHiddenSize;
  }

  /** Run fused self-attention and parallel SwiGLU feed-forward projection. */
  forward(hiddenStates: MxArray, rope: MxArray): MxArray {
    const { batch, length } = assertFlux2Sequence3d(
      hiddenStates,
      "Flux2ParallelSelfAttention.forward",
    );
    using projected = this.toQkvMlpProj.forward(hiddenStates);
    using rawQueries = sliceFlux2LastAxis(projected, 0, this.#hiddenSize);
    using rawKeys = sliceFlux2LastAxis(projected, this.#hiddenSize, this.#hiddenSize * 2);
    using rawValues = sliceFlux2LastAxis(projected, this.#hiddenSize * 2, this.#hiddenSize * 3);
    using rawMlp = sliceFlux2LastAxis(
      projected,
      this.#hiddenSize * 3,
      this.#hiddenSize * 3 + this.#mlpHiddenSize * 2,
    );
    using queries = this.#reshapeAttention(rawQueries, batch, length);
    using keys = this.#reshapeAttention(rawKeys, batch, length);
    using values = this.#reshapeAttention(rawValues, batch, length);
    const normalized = this.norm.normalize(queries, keys);
    try {
      using attention = flux2Attention(normalized.queries, normalized.keys, values, rope);
      using mlpGate = sliceFlux2LastAxis(rawMlp, 0, this.#mlpHiddenSize);
      using mlpValue = sliceFlux2LastAxis(rawMlp, this.#mlpHiddenSize, this.#mlpHiddenSize * 2);
      using mlp = swiglu(mlpGate, mlpValue);
      using combined = concatenate([attention, mlp], -1);
      return this.toOut.forward(combined);
    } finally {
      normalized.queries.free();
      normalized.keys.free();
    }
  }

  #reshapeAttention(x: MxArray, batch: number, length: number): MxArray {
    using reshaped = reshape(x, [batch, length, this.#numHeads, this.#headDim]);
    return transpose(reshaped, [0, 2, 1, 3]);
  }
}
