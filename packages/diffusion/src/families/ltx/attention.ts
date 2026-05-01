import type { MxArray } from "@mlxts/core";
import {
  add,
  asType,
  formatShape,
  multiply,
  reshape,
  retainArray,
  scaledDotProductAttention,
  stack,
  transpose,
} from "@mlxts/core";
import { Linear, Module, RMSNorm } from "@mlxts/nn";

import type { LtxRotaryEmbeddings } from "./embeddings";
import { assertSequence3d, selectLastAxis } from "./tensor-utils";

export type LtxVideoAttentionOptions = {
  encoderHiddenStates?: MxArray;
  encoderAttentionMask?: MxArray;
  imageRotaryEmbeddings?: LtxRotaryEmbeddings;
};

type LtxVideoAttentionProjection = {
  queries: MxArray;
  keys: MxArray;
  values: MxArray;
};

function disposeProjection(projection: LtxVideoAttentionProjection): void {
  projection.queries.free();
  projection.keys.free();
  projection.values.free();
}

function assertRotaryShape(
  x: MxArray,
  embeddings: LtxRotaryEmbeddings,
  owner: string,
): { batch: number; length: number; channels: number } {
  const shape = assertSequence3d(x, owner);
  const expected = [shape.batch, shape.length, shape.channels];
  const cosShape = embeddings.cos.shape;
  const sinShape = embeddings.sin.shape;
  const matches =
    cosShape.length === 3 &&
    sinShape.length === 3 &&
    expected.every(
      (dimension, index) => cosShape[index] === dimension && sinShape[index] === dimension,
    );
  if (!matches) {
    throw new Error(`${owner}: RoPE tensors must have shape ${formatShape(expected)}.`);
  }
  if (shape.channels % 2 !== 0) {
    throw new Error(`${owner}: hidden size must be even for pairwise rotation.`);
  }
  return shape;
}

/** Apply classic LTX interleaved RoPE over `[batch, tokens, hiddenSize]` q/k tensors. */
export function applyLtxVideoRotary(x: MxArray, embeddings: LtxRotaryEmbeddings): MxArray {
  const { batch, length, channels } = assertRotaryShape(x, embeddings, "applyLtxVideoRotary");
  using xFloat = x.dtype === "float32" ? retainArray(x) : asType(x, "float32");
  using pairs = reshape(xFloat, [batch, length, channels / 2, 2]);
  using first = selectLastAxis(pairs, 0);
  using second = selectLastAxis(pairs, 1);
  using negSecond = multiply(second, -1);
  using rotatedPairs = stack([negSecond, first], -1);
  using rotated = reshape(rotatedPairs, [batch, length, channels]);
  using direct = multiply(xFloat, embeddings.cos);
  using shifted = multiply(rotated, embeddings.sin);
  using output = add(direct, shifted);
  if (output.dtype === x.dtype) {
    return retainArray(output);
  }
  return asType(output, x.dtype);
}

/** Diffusers LTX attention with RMS-normalized Q/K projections. */
export class LtxVideoAttention extends Module {
  normQ: RMSNorm;
  normK: RMSNorm;
  toQ: Linear;
  toK: Linear;
  toV: Linear;
  toOut: Linear;
  #numHeads: number;
  #headDim: number;
  #hiddenSize: number;
  #crossAttentionDim: number;

  constructor(options: {
    hiddenSize: number;
    numHeads: number;
    headDim: number;
    crossAttentionDim?: number;
    attentionBias: boolean;
    attentionOutBias: boolean;
  }) {
    super();
    if (options.hiddenSize !== options.numHeads * options.headDim) {
      throw new Error("LtxVideoAttention: hiddenSize must equal numHeads * headDim.");
    }
    this.normQ = new RMSNorm(options.hiddenSize, 1e-5);
    this.normK = new RMSNorm(options.hiddenSize, 1e-5);
    this.toQ = new Linear(options.hiddenSize, options.hiddenSize, options.attentionBias);
    this.#crossAttentionDim = options.crossAttentionDim ?? options.hiddenSize;
    this.toK = new Linear(this.#crossAttentionDim, options.hiddenSize, options.attentionBias);
    this.toV = new Linear(this.#crossAttentionDim, options.hiddenSize, options.attentionBias);
    this.toOut = new Linear(options.hiddenSize, options.hiddenSize, options.attentionOutBias);
    this.#numHeads = options.numHeads;
    this.#headDim = options.headDim;
    this.#hiddenSize = options.hiddenSize;
  }

  forward(hiddenStates: MxArray): MxArray {
    return this.run(hiddenStates);
  }

  /** Run self-attention or cross-attention over LTX hidden states. */
  run(hiddenStates: MxArray, options: LtxVideoAttentionOptions = {}): MxArray {
    const hiddenShape = assertSequence3d(hiddenStates, "LtxVideoAttention.run hiddenStates");
    if (hiddenShape.channels !== this.#hiddenSize) {
      throw new Error("LtxVideoAttention.run: hiddenStates hidden size mismatch.");
    }
    const context = options.encoderHiddenStates ?? hiddenStates;
    const contextShape = assertSequence3d(context, "LtxVideoAttention.run encoderHiddenStates");
    if (
      contextShape.batch !== hiddenShape.batch ||
      contextShape.channels !== this.#crossAttentionDim
    ) {
      throw new Error(
        `LtxVideoAttention.run: encoderHiddenStates must have shape [${hiddenShape.batch}, length, ${this.#crossAttentionDim}], got ${formatShape(
          context.shape,
        )}.`,
      );
    }
    const projection = this.#project(hiddenStates, context, options.imageRotaryEmbeddings);
    try {
      using attended =
        options.encoderAttentionMask === undefined
          ? scaledDotProductAttention(projection.queries, projection.keys, projection.values, {
              scale: this.#headDim ** -0.5,
            })
          : scaledDotProductAttention(projection.queries, projection.keys, projection.values, {
              scale: this.#headDim ** -0.5,
              maskMode: "array",
              maskArray: options.encoderAttentionMask,
            });
      using sequenceFirst = transpose(attended, [0, 2, 1, 3]);
      using merged = reshape(sequenceFirst, [
        hiddenShape.batch,
        hiddenShape.length,
        this.#hiddenSize,
      ]);
      return this.toOut.forward(merged);
    } finally {
      disposeProjection(projection);
    }
  }

  #project(
    querySource: MxArray,
    keyValueSource: MxArray,
    imageRotaryEmbeddings?: LtxRotaryEmbeddings,
  ): LtxVideoAttentionProjection {
    const queryShape = assertSequence3d(querySource, "LtxVideoAttention.project querySource");
    const keyValueShape = assertSequence3d(
      keyValueSource,
      "LtxVideoAttention.project keyValueSource",
    );
    using queryProjection = this.toQ.forward(querySource);
    using keyProjection = this.toK.forward(keyValueSource);
    using valueProjection = this.toV.forward(keyValueSource);
    using normalizedQueries = this.normQ.forward(queryProjection);
    using normalizedKeys = this.normK.forward(keyProjection);
    const rotaryEmbeddings =
      querySource === keyValueSource && imageRotaryEmbeddings !== undefined
        ? imageRotaryEmbeddings
        : null;
    const queries =
      rotaryEmbeddings === null
        ? retainArray(normalizedQueries)
        : applyLtxVideoRotary(normalizedQueries, rotaryEmbeddings);
    const keys =
      rotaryEmbeddings === null
        ? retainArray(normalizedKeys)
        : applyLtxVideoRotary(normalizedKeys, rotaryEmbeddings);
    try {
      using queryHeads = reshape(queries, [
        queryShape.batch,
        queryShape.length,
        this.#numHeads,
        this.#headDim,
      ]);
      using keyHeads = reshape(keys, [
        keyValueShape.batch,
        keyValueShape.length,
        this.#numHeads,
        this.#headDim,
      ]);
      using valueHeads = reshape(valueProjection, [
        keyValueShape.batch,
        keyValueShape.length,
        this.#numHeads,
        this.#headDim,
      ]);
      return {
        queries: transpose(queryHeads, [0, 2, 1, 3]),
        keys: transpose(keyHeads, [0, 2, 1, 3]),
        values: transpose(valueHeads, [0, 2, 1, 3]),
      };
    } finally {
      queries.free();
      keys.free();
    }
  }
}
