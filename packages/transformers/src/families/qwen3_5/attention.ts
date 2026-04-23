/**
 * Qwen 3.5 full-attention block.
 * @module
 */

import {
  type DType,
  fastRoPE,
  formatShape,
  MxArray,
  multiply,
  reshape,
  retainArray,
  type ScaledDotProductAttentionOptions,
  scaledDotProductAttention,
  sigmoid,
  slice,
  transpose,
} from "@mlxts/core";
import { Linear, Module, RMSNorm } from "@mlxts/nn";

import {
  retainTransformerCacheView,
  updateAndFetchTransformerCacheView,
} from "../../infrastructure/cache/view";
import { type AttentionMask, createCausalMask } from "../../infrastructure/masks";
import type { TransformerCache } from "../../types";
import { Qwen3_5TextRotaryEmbedding } from "./rotary";
import type { Qwen3_5TextConfig } from "./types";

type ProjectedAttentionInputs = {
  queries: MxArray;
  gates: MxArray;
  keys: MxArray;
  values: MxArray;
};

type AttentionInputShape = {
  batch: number;
  sequenceLength: number;
};

type RotatedAttentionHeads = {
  queries: MxArray;
  keys: MxArray;
};

function takeLastAxisRange(x: MxArray, start: number, end: number): MxArray {
  const rank = x.shape.length;
  const lastDimension = x.shape[rank - 1];
  if (lastDimension === undefined || rank < 3) {
    throw new Error(
      `Qwen3_5TextAttention: expected rank >= 3 projection tensor, got ${formatShape(x.shape)}.`,
    );
  }
  const startIndices = Array(rank).fill(0);
  const stopIndices = [...x.shape];
  startIndices[rank - 1] = start;
  stopIndices[rank - 1] = end;
  return slice(x, startIndices, stopIndices);
}

export function splitPackedQueryGateHeads(
  packedQueries: MxArray,
  numHeads: number,
  headDim: number,
): { queries: MxArray; gates: MxArray } {
  const [batch, sequenceLength, totalWidth] = packedQueries.shape;
  const expectedWidth = numHeads * headDim * 2;
  if (
    batch === undefined ||
    sequenceLength === undefined ||
    totalWidth === undefined ||
    packedQueries.shape.length !== 3
  ) {
    throw new Error(
      `splitPackedQueryGateHeads: expected rank-3 packed queries, got ${formatShape(packedQueries.shape)}.`,
    );
  }
  if (totalWidth !== expectedWidth) {
    throw new Error(
      `splitPackedQueryGateHeads: expected last dimension ${expectedWidth}, got ${totalWidth}.`,
    );
  }

  using packedHeads = reshape(packedQueries, [batch, sequenceLength, numHeads, headDim * 2]);
  using queryHeads = takeLastAxisRange(packedHeads, 0, headDim);
  using gateHeads = takeLastAxisRange(packedHeads, headDim, headDim * 2);
  return {
    queries: reshape(queryHeads, [batch, sequenceLength, numHeads * headDim]),
    gates: reshape(gateHeads, [batch, sequenceLength, numHeads * headDim]),
  };
}

function attentionOptionsForMask(
  mask: AttentionMask,
  headDim: number,
): ScaledDotProductAttentionOptions {
  if (mask === null) {
    return { scale: headDim ** -0.5 };
  }
  if (mask === "causal") {
    return { scale: headDim ** -0.5, maskMode: "causal" };
  }
  return { scale: headDim ** -0.5, maskArray: mask };
}

/** Cache-aware full attention used on Qwen 3.5 full-attention layers. */
export class Qwen3_5TextAttention extends Module {
  qProjection: Linear;
  kProjection: Linear;
  vProjection: Linear;
  outputProjection: Linear;
  qNorm: RMSNorm;
  kNorm: RMSNorm;
  #hiddenSize: number;
  #numHeads: number;
  #numKeyValueHeads: number;
  #headDim: number;
  #ropeBase: number;
  #rotaryEmbedding: Qwen3_5TextRotaryEmbedding;

  constructor(config: Qwen3_5TextConfig) {
    super();
    if (config.numAttentionHeads % config.numKeyValueHeads !== 0) {
      throw new Error(
        `Qwen3_5TextAttention: numAttentionHeads ${config.numAttentionHeads} must be divisible by numKeyValueHeads ${config.numKeyValueHeads}.`,
      );
    }

    this.#hiddenSize = config.hiddenSize;
    this.#numHeads = config.numAttentionHeads;
    this.#numKeyValueHeads = config.numKeyValueHeads;
    this.#headDim = config.headDim;
    this.#ropeBase = config.ropeParameters.ropeTheta;
    this.qProjection = new Linear(
      config.hiddenSize,
      config.numAttentionHeads * config.headDim * 2,
      config.attentionBias,
    );
    this.kProjection = new Linear(
      config.hiddenSize,
      config.numKeyValueHeads * config.headDim,
      config.attentionBias,
    );
    this.vProjection = new Linear(
      config.hiddenSize,
      config.numKeyValueHeads * config.headDim,
      config.attentionBias,
    );
    this.outputProjection = new Linear(
      config.numAttentionHeads * config.headDim,
      config.hiddenSize,
      config.attentionBias,
    );
    this.qNorm = new RMSNorm(config.headDim, config.rmsNormEps);
    this.kNorm = new RMSNorm(config.headDim, config.rmsNormEps);
    this.#rotaryEmbedding = new Qwen3_5TextRotaryEmbedding(config);
  }

  forward(x: MxArray): MxArray {
    return this.run(x, 0);
  }

  run(
    x: MxArray,
    layerIndex: number,
    cache?: TransformerCache,
    attentionMask?: AttentionMask,
    positionIds?: MxArray,
  ): MxArray {
    const { batch, sequenceLength } = this.validateInputShape(x);

    const projected = this.projectAttentionInputs(x);
    try {
      using queryInputs = reshape(projected.queries, [
        batch,
        sequenceLength,
        this.#numHeads,
        this.#headDim,
      ]);
      using keyInputs = reshape(projected.keys, [
        batch,
        sequenceLength,
        this.#numKeyValueHeads,
        this.#headDim,
      ]);
      using valueInputs = reshape(projected.values, [
        batch,
        sequenceLength,
        this.#numKeyValueHeads,
        this.#headDim,
      ]);
      using normalizedQueries = this.qNorm.forward(queryInputs);
      using normalizedKeys = this.kNorm.forward(keyInputs);
      using queryHeads = transpose(normalizedQueries, [0, 2, 1, 3]);
      using keyHeads = transpose(normalizedKeys, [0, 2, 1, 3]);
      using valueHeads = transpose(valueInputs, [0, 2, 1, 3]);

      const rotated = this.rotateAttentionHeads(queryHeads, keyHeads, cache, positionIds);
      try {
        using activeKeyValues =
          cache === undefined
            ? retainTransformerCacheView(rotated.keys, valueHeads)
            : updateAndFetchTransformerCacheView(cache, layerIndex, rotated.keys, valueHeads);
        const retainedMask = this.retainAttentionMask(
          attentionMask,
          activeKeyValues.keys,
          sequenceLength,
          rotated.queries.dtype,
        );

        try {
          using attentionOutput = scaledDotProductAttention(
            rotated.queries,
            activeKeyValues.keys,
            activeKeyValues.values,
            attentionOptionsForMask(retainedMask, this.#headDim),
          );
          using transposedOutput = transpose(attentionOutput, [0, 2, 1, 3]);
          using mergedOutput = reshape(transposedOutput, [
            batch,
            sequenceLength,
            this.#numHeads * this.#headDim,
          ]);
          using gatedOutput = sigmoid(projected.gates);
          using scaledOutput = multiply(mergedOutput, gatedOutput);
          return this.outputProjection.forward(scaledOutput);
        } finally {
          if (retainedMask instanceof MxArray) {
            retainedMask.free();
          }
        }
      } finally {
        rotated.queries.free();
        rotated.keys.free();
      }
    } finally {
      projected.queries.free();
      projected.gates.free();
      projected.keys.free();
      projected.values.free();
    }
  }

  private validateInputShape(x: MxArray): AttentionInputShape {
    const [batch, sequenceLength, hiddenSize] = x.shape;
    if (batch === undefined || sequenceLength === undefined || hiddenSize === undefined) {
      throw new Error(
        `Qwen3_5TextAttention.forward: expected rank-3 input, got ${formatShape(x.shape)}.`,
      );
    }
    if (hiddenSize !== this.#hiddenSize) {
      throw new Error(
        `Qwen3_5TextAttention.forward: expected hidden size ${this.#hiddenSize}, got ${hiddenSize}.`,
      );
    }
    return { batch, sequenceLength };
  }

  private rotateAttentionHeads(
    queryHeads: MxArray,
    keyHeads: MxArray,
    cache: TransformerCache | undefined,
    positionIds: MxArray | undefined,
  ): RotatedAttentionHeads {
    let queries: MxArray | null = null;
    let keys: MxArray | null = null;
    try {
      if (positionIds === undefined) {
        queries = fastRoPE(queryHeads, this.#rotaryEmbedding.rotaryDimensions, {
          traditional: false,
          base: this.#ropeBase,
          offset: cache?.offset ?? 0,
        });
        keys = fastRoPE(keyHeads, this.#rotaryEmbedding.rotaryDimensions, {
          traditional: false,
          base: this.#ropeBase,
          offset: cache?.offset ?? 0,
        });
        return { queries, keys };
      }

      const rotated = this.#rotaryEmbedding.apply(queryHeads, keyHeads, positionIds);
      queries = rotated.queries;
      keys = rotated.keys;
      return { queries, keys };
    } catch (error) {
      queries?.free();
      keys?.free();
      throw error;
    }
  }

  private retainAttentionMask(
    attentionMask: AttentionMask | undefined,
    activeKeys: MxArray,
    sequenceLength: number,
    dtype: DType,
  ): AttentionMask {
    if (attentionMask !== undefined) {
      return attentionMask === null || attentionMask === "causal"
        ? attentionMask
        : retainArray(attentionMask);
    }

    const totalKeyLength = activeKeys.shape[2];
    if (totalKeyLength === undefined) {
      throw new Error(
        "Qwen3_5TextAttention.forward: attention key cache is missing a sequence axis.",
      );
    }

    const visiblePastLength = Math.max(0, totalKeyLength - sequenceLength);
    return createCausalMask(sequenceLength, totalKeyLength, visiblePastLength, dtype);
  }

  private projectAttentionInputs(x: MxArray): ProjectedAttentionInputs {
    using packedQueries = this.qProjection.forward(x);
    const split = splitPackedQueryGateHeads(packedQueries, this.#numHeads, this.#headDim);
    return {
      queries: split.queries,
      gates: split.gates,
      keys: this.kProjection.forward(x),
      values: this.vProjection.forward(x),
    };
  }

  override [Symbol.dispose](): void {
    this.#rotaryEmbedding[Symbol.dispose]();
    super[Symbol.dispose]();
  }
}
