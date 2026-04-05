/**
 * Grouped-query self-attention.
 *
 * This module keeps the control flow explicit: projection, optional RoPE,
 * key/value head repetition, attention, and output projection are all visible.
 *
 * @module
 */

import {
  add,
  asType,
  expandDims,
  formatShape,
  MxArray,
  matmul,
  multiply,
  repeat,
  reshape,
  retainArray,
  softmax,
  transpose,
} from "@mlxts/core";
import { Dropout } from "./dropout";
import { Linear } from "./linear";
import { Module } from "./module";
import type { RoPE } from "./rope";

export type GroupedQueryAttentionOptions = {
  bias?: boolean;
  dropout?: number;
  rope?: RoPE;
};

export type GroupedQueryAttentionForwardOptions = {
  attentionMask?: MxArray;
  offset?: number | MxArray;
};

function castAttentionWeights(weights: MxArray, dtype: MxArray["dtype"]): MxArray {
  if (dtype === "float32") {
    return retainArray(weights);
  }

  return asType(weights, dtype);
}

/** Decoder-style self-attention with grouped key/value heads. */
export class GroupedQueryAttention extends Module {
  qProjection: Linear;
  kProjection: Linear;
  vProjection: Linear;
  outputProjection: Linear;
  attentionDropout: Dropout;
  rope: RoPE | null;
  #embedDim: number;
  #numHeads: number;
  #numKeyValueHeads: number;
  #headDim: number;
  #numKeyValueGroups: number;

  /**
   * @param embedDim - Input and output embedding dimension.
   * @param numHeads - Number of query heads.
   * @param numKeyValueHeads - Number of key/value heads. Must divide numHeads.
   * @param options - Optional projection bias, attention dropout, and RoPE module.
   */
  constructor(
    embedDim: number,
    numHeads: number,
    numKeyValueHeads: number,
    options: GroupedQueryAttentionOptions = {},
  ) {
    super();

    if (embedDim <= 0) {
      throw new Error(`GroupedQueryAttention: embedDim must be > 0, got ${embedDim}`);
    }
    if (numHeads <= 0) {
      throw new Error(`GroupedQueryAttention: numHeads must be > 0, got ${numHeads}`);
    }
    if (numKeyValueHeads <= 0) {
      throw new Error(
        `GroupedQueryAttention: numKeyValueHeads must be > 0, got ${numKeyValueHeads}`,
      );
    }
    if (embedDim % numHeads !== 0) {
      throw new Error(
        `GroupedQueryAttention: embedDim (${embedDim}) must be divisible by numHeads (${numHeads})`,
      );
    }
    if (numHeads % numKeyValueHeads !== 0) {
      throw new Error(
        `GroupedQueryAttention: numHeads (${numHeads}) must be divisible by numKeyValueHeads (${numKeyValueHeads})`,
      );
    }

    this.#embedDim = embedDim;
    this.#numHeads = numHeads;
    this.#numKeyValueHeads = numKeyValueHeads;
    this.#headDim = embedDim / numHeads;
    this.#numKeyValueGroups = numHeads / numKeyValueHeads;

    this.qProjection = new Linear(embedDim, numHeads * this.#headDim, options.bias ?? false);
    this.kProjection = new Linear(
      embedDim,
      numKeyValueHeads * this.#headDim,
      options.bias ?? false,
    );
    this.vProjection = new Linear(
      embedDim,
      numKeyValueHeads * this.#headDim,
      options.bias ?? false,
    );
    this.outputProjection = new Linear(numHeads * this.#headDim, embedDim, options.bias ?? false);
    this.attentionDropout = new Dropout(options.dropout ?? 0.0);
    this.rope = options.rope ?? null;
  }

  forward(x: MxArray, attentionMask?: MxArray, offset?: MxArray): MxArray;
  forward(x: MxArray, options?: GroupedQueryAttentionForwardOptions): MxArray;
  forward(
    x: MxArray,
    attentionMaskOrOptions?: MxArray | GroupedQueryAttentionForwardOptions,
    offset?: MxArray,
  ): MxArray {
    const [batch, sequenceLength, embedDim] = x.shape;
    if (batch === undefined || sequenceLength === undefined || embedDim === undefined) {
      throw new Error(
        `GroupedQueryAttention.forward: expected rank-3 input, got shape ${formatShape(x.shape)}.`,
      );
    }
    if (embedDim !== this.#embedDim) {
      throw new Error(
        `GroupedQueryAttention.forward: expected last dimension ${this.#embedDim}, got ${embedDim}.`,
      );
    }

    using projectedQueries = this.qProjection.forward(x);
    using projectedKeys = this.kProjection.forward(x);
    using projectedValues = this.vProjection.forward(x);

    using queryInputs = reshape(projectedQueries, [
      batch,
      sequenceLength,
      this.#numHeads,
      this.#headDim,
    ]);
    using keyInputs = reshape(projectedKeys, [
      batch,
      sequenceLength,
      this.#numKeyValueHeads,
      this.#headDim,
    ]);
    using valueInputs = reshape(projectedValues, [
      batch,
      sequenceLength,
      this.#numKeyValueHeads,
      this.#headDim,
    ]);

    using queryHeads = transpose(queryInputs, [0, 2, 1, 3]);
    using keyHeads = transpose(keyInputs, [0, 2, 1, 3]);
    using valueHeads = transpose(valueInputs, [0, 2, 1, 3]);

    const resolvedOptions =
      attentionMaskOrOptions instanceof MxArray || attentionMaskOrOptions === undefined
        ? {
            attentionMask: attentionMaskOrOptions,
            offset,
          }
        : attentionMaskOrOptions;
    const positionOffset = resolvedOptions.offset ?? 0;
    using rotatedQueries =
      this.rope === null ? retainArray(queryHeads) : this.rope.forward(queryHeads, positionOffset);
    using rotatedKeys =
      this.rope === null ? retainArray(keyHeads) : this.rope.forward(keyHeads, positionOffset);
    using repeatedKeys = this.repeatKeyValueHeads(rotatedKeys, batch, sequenceLength);
    using repeatedValues = this.repeatKeyValueHeads(valueHeads, batch, sequenceLength);
    using attentionOutput = this.computeAttention(
      rotatedQueries,
      repeatedKeys,
      repeatedValues,
      resolvedOptions.attentionMask,
    );
    using transposedOutput = transpose(attentionOutput, [0, 2, 1, 3]);
    using mergedOutput = reshape(transposedOutput, [
      batch,
      sequenceLength,
      this.#numHeads * this.#headDim,
    ]);

    return this.outputProjection.forward(mergedOutput);
  }

  private repeatKeyValueHeads(x: MxArray, batch: number, sequenceLength: number): MxArray {
    if (this.#numKeyValueGroups === 1) {
      return retainArray(x);
    }

    using expanded = expandDims(x, 2);
    using repeated = repeat(expanded, this.#numKeyValueGroups, 2);
    return reshape(repeated, [batch, this.#numHeads, sequenceLength, this.#headDim]);
  }

  private computeAttention(
    queries: MxArray,
    keys: MxArray,
    values: MxArray,
    attentionMask?: MxArray,
  ): MxArray {
    using keyTranspose = transpose(keys, [0, 1, 3, 2]);
    using scores = matmul(queries, keyTranspose);
    using scaledScores = multiply(scores, this.#headDim ** -0.5);
    using maskedScores =
      attentionMask === undefined ? retainArray(scaledScores) : add(scaledScores, attentionMask);
    using stableScores = asType(maskedScores, "float32");
    using stableWeights = softmax(stableScores, -1);
    using weights = castAttentionWeights(stableWeights, queries.dtype);
    using droppedWeights = this.attentionDropout.forward(weights);
    return matmul(droppedWeights, values);
  }
}
