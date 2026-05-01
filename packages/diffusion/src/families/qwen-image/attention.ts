import type { MxArray } from "@mlxts/core";
import {
  add,
  asType,
  concatenate,
  expandDims,
  multiply,
  ones,
  reshape,
  retainArray,
  scaledDotProductAttention,
  transpose,
} from "@mlxts/core";
import { Linear, Module, RMSNorm } from "@mlxts/nn";

import { assertAttention4d, assertSequence3d, selectLastAxis, sliceAxis } from "./tensor-utils";

export type QwenImageAttentionPair = {
  image: MxArray;
  text: MxArray;
};

export type QwenImageAttentionProjection = {
  queries: MxArray;
  keys: MxArray;
  values: MxArray;
};

function disposeProjection(projection: QwenImageAttentionProjection): void {
  projection.queries.free();
  projection.keys.free();
  projection.values.free();
}

/** Apply Qwen-Image RoPE matrices to `[batch, heads, length, headDim]` q/k tensors. */
export function applyQwenImageRotary(x: MxArray, rope: MxArray): MxArray {
  const { batch, heads, length, headDim } = assertAttention4d(x, "applyQwenImageRotary");
  if (headDim % 2 !== 0) {
    throw new Error("applyQwenImageRotary: headDim must be even.");
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
    throw new Error("applyQwenImageRotary: rope shape must match attention length and headDim.");
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

/** Qwen-Image Q/K RMS normalization pair. */
export class QwenImageQKNorm extends Module {
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

/** Joint Qwen-Image attention over separate image and text streams. */
export class QwenImageJointAttention extends Module {
  toQ: Linear;
  toK: Linear;
  toV: Linear;
  addQProj: Linear;
  addKProj: Linear;
  addVProj: Linear;
  norm: QwenImageQKNorm;
  addedNorm: QwenImageQKNorm;
  toOut: Linear;
  toAddOut: Linear;
  #numHeads: number;
  #headDim: number;
  #hiddenSize: number;

  constructor(hiddenSize: number, numHeads: number, headDim: number) {
    super();
    if (hiddenSize !== numHeads * headDim) {
      throw new Error("QwenImageJointAttention: hiddenSize must equal numHeads * headDim.");
    }
    this.toQ = new Linear(hiddenSize, hiddenSize);
    this.toK = new Linear(hiddenSize, hiddenSize);
    this.toV = new Linear(hiddenSize, hiddenSize);
    this.addQProj = new Linear(hiddenSize, hiddenSize);
    this.addKProj = new Linear(hiddenSize, hiddenSize);
    this.addVProj = new Linear(hiddenSize, hiddenSize);
    this.norm = new QwenImageQKNorm(headDim);
    this.addedNorm = new QwenImageQKNorm(headDim);
    this.toOut = new Linear(hiddenSize, hiddenSize);
    this.toAddOut = new Linear(hiddenSize, hiddenSize);
    this.#numHeads = numHeads;
    this.#headDim = headDim;
    this.#hiddenSize = hiddenSize;
  }

  forward(_image: MxArray): MxArray {
    throw new Error("QwenImageJointAttention.forward: use run() inside a Qwen-Image block.");
  }

  /** Run joint `[text, image]` attention and return separated projected streams. */
  run(image: MxArray, text: MxArray, rope: MxArray, textMask?: MxArray): QwenImageAttentionPair {
    const imageProjection = this.#projectImage(image);
    const textProjection = this.#projectText(text);
    try {
      using queries = concatenate([textProjection.queries, imageProjection.queries], 2);
      using keys = concatenate([textProjection.keys, imageProjection.keys], 2);
      using values = concatenate([textProjection.values, imageProjection.values], 2);
      using rotaryQueries = applyQwenImageRotary(queries, rope);
      using rotaryKeys = applyQwenImageRotary(keys, rope);
      const mask = textMask === undefined ? null : this.#jointAttentionMask(textMask, image);
      try {
        using attended =
          mask === null
            ? scaledDotProductAttention(rotaryQueries, rotaryKeys, values, {
                scale: this.#headDim ** -0.5,
              })
            : scaledDotProductAttention(rotaryQueries, rotaryKeys, values, {
                scale: this.#headDim ** -0.5,
                maskMode: "array",
                maskArray: mask,
              });
        using sequenceFirst = transpose(attended, [0, 2, 1, 3]);
        using joint = reshape(sequenceFirst, [
          image.shape[0] ?? 0,
          (text.shape[1] ?? 0) + (image.shape[1] ?? 0),
          this.#hiddenSize,
        ]);
        using textAttention = sliceAxis(joint, 1, 0, text.shape[1] ?? 0);
        using imageAttention = sliceAxis(joint, 1, text.shape[1] ?? 0, joint.shape[1] ?? 0);
        using projectedText = this.toAddOut.forward(textAttention);
        using projectedImage = this.toOut.forward(imageAttention);
        return { text: retainArray(projectedText), image: retainArray(projectedImage) };
      } finally {
        mask?.free();
      }
    } finally {
      disposeProjection(imageProjection);
      disposeProjection(textProjection);
    }
  }

  #projectImage(input: MxArray): QwenImageAttentionProjection {
    return this.#project(input, this.toQ, this.toK, this.toV, this.norm, "image");
  }

  #projectText(input: MxArray): QwenImageAttentionProjection {
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
    norm: QwenImageQKNorm,
    streamName: string,
  ): QwenImageAttentionProjection {
    const { batch, length } = assertSequence3d(
      input,
      `QwenImageJointAttention.project ${streamName}`,
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

  #jointAttentionMask(textMask: MxArray, image: MxArray): MxArray {
    const { batch, length: imageLength } = assertSequence3d(
      image,
      "QwenImageJointAttention.jointAttentionMask image",
    );
    const [maskBatch, textLength] = textMask.shape;
    if (textMask.shape.length !== 2 || maskBatch !== batch || textLength === undefined) {
      throw new Error("QwenImageJointAttention.run: textMask must have shape [batch, textLength].");
    }
    using booleanTextMask =
      textMask.dtype === "bool" ? retainArray(textMask) : asType(textMask, "bool");
    using imageMask = ones([batch, imageLength], "bool");
    using jointMask = concatenate([booleanTextMask, imageMask], 1);
    using queryBroadcast = expandDims(jointMask, 1);
    return expandDims(queryBroadcast, 1);
  }
}
