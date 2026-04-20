/**
 * Cache-aware grouped-query attention for pretrained decoder inference.
 * @module
 */

import {
  arange,
  formatShape,
  MxArray,
  reshape,
  retainArray,
  scaledDotProductAttention,
  takeAxis,
  transpose,
} from "@mlxts/core";
import { Linear, Module, RoPE } from "@mlxts/nn";

import {
  retainTransformerCacheView,
  updateAndFetchTransformerCacheView,
} from "../../infrastructure/cache/view";
import { type AttentionMask, createCausalMask } from "../../infrastructure/masks";
import type { TransformerCache } from "../../types";
import type { LlamaLikeConfig } from "./types";

function takeLastAxisRange(x: MxArray, start: number, end: number): MxArray {
  using indices = arange(start, end, 1, "int32");
  return takeAxis(x, indices, x.shape.length - 1);
}

/** Decoder self-attention with cache integration and offset-aware RoPE. */
export class LlamaLikeAttention extends Module {
  qProjection: Linear | null;
  kProjection: Linear | null;
  vProjection: Linear | null;
  qkvProjection: Linear | null;
  outputProjection: Linear;
  rope: RoPE;
  #hiddenSize: number;
  #numHeads: number;
  #numKeyValueHeads: number;
  #headDim: number;

  constructor(config: LlamaLikeConfig) {
    super();
    if (config.hiddenSize % config.numAttentionHeads !== 0) {
      throw new Error(
        `LlamaLikeAttention: hiddenSize ${config.hiddenSize} must be divisible by numAttentionHeads ${config.numAttentionHeads}.`,
      );
    }
    if (config.numAttentionHeads % config.numKeyValueHeads !== 0) {
      throw new Error(
        `LlamaLikeAttention: numAttentionHeads ${config.numAttentionHeads} must be divisible by numKeyValueHeads ${config.numKeyValueHeads}.`,
      );
    }

    this.#hiddenSize = config.hiddenSize;
    this.#numHeads = config.numAttentionHeads;
    this.#numKeyValueHeads = config.numKeyValueHeads;
    this.#headDim = config.headDim;
    const queryWidth = config.numAttentionHeads * config.headDim;
    const keyValueWidth = config.numKeyValueHeads * config.headDim;
    if (config.attentionProjectionLayout === "packed_qkv") {
      this.qProjection = null;
      this.kProjection = null;
      this.vProjection = null;
      this.qkvProjection = new Linear(
        config.hiddenSize,
        queryWidth + keyValueWidth + keyValueWidth,
        config.attentionBias,
      );
    } else {
      this.qProjection = new Linear(config.hiddenSize, queryWidth, config.attentionBias);
      this.kProjection = new Linear(config.hiddenSize, keyValueWidth, config.attentionBias);
      this.vProjection = new Linear(config.hiddenSize, keyValueWidth, config.attentionBias);
      this.qkvProjection = null;
    }
    this.outputProjection = new Linear(
      config.numAttentionHeads * config.headDim,
      config.hiddenSize,
      config.attentionBias,
    );
    this.rope = new RoPE(config.rotaryDimensions ?? config.headDim, false, config.ropeTheta);
  }

  forward(x: MxArray): MxArray {
    return this.run(x, 0);
  }

  run(
    x: MxArray,
    layerIndex: number,
    cache?: TransformerCache,
    attentionMask?: AttentionMask,
  ): MxArray {
    const [batch, sequenceLength, hiddenSize] = x.shape;
    if (batch === undefined || sequenceLength === undefined || hiddenSize === undefined) {
      throw new Error(
        `LlamaLikeAttention.forward: expected rank-3 input, got ${formatShape(x.shape)}.`,
      );
    }
    if (hiddenSize !== this.#hiddenSize) {
      throw new Error(
        `LlamaLikeAttention.forward: expected hidden size ${this.#hiddenSize}, got ${hiddenSize}.`,
      );
    }

    const projected = this.projectQueryKeyValue(x);

    try {
      using projectedQueries = projected.queries;
      using projectedKeys = projected.keys;
      using projectedValues = projected.values;
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
      using rotatedQueries = this.rope.forward(queryHeads, cache?.offset ?? 0);
      using rotatedKeys = this.rope.forward(keyHeads, cache?.offset ?? 0);
      using activeKeyValues =
        cache === undefined
          ? retainTransformerCacheView(rotatedKeys, valueHeads)
          : updateAndFetchTransformerCacheView(cache, layerIndex, rotatedKeys, valueHeads);
      const retainedMask = (() => {
        const totalKeyLength = activeKeyValues.keys.shape[2];
        if (totalKeyLength === undefined) {
          throw new Error(
            "LlamaLikeAttention.forward: attention key cache is missing a sequence axis.",
          );
        }

        if (attentionMask !== undefined) {
          return attentionMask === null || attentionMask === "causal"
            ? attentionMask
            : retainArray(attentionMask);
        }

        const visiblePastLength = Math.max(0, totalKeyLength - sequenceLength);
        return createCausalMask(
          sequenceLength,
          totalKeyLength,
          visiblePastLength,
          rotatedQueries.dtype,
        );
      })();

      try {
        const totalKeyLength = activeKeyValues.keys.shape[2];
        if (totalKeyLength === undefined) {
          throw new Error(
            "LlamaLikeAttention.forward: attention key cache is missing a sequence axis.",
          );
        }

        const attentionOptions =
          retainedMask === null
            ? { scale: this.#headDim ** -0.5 }
            : retainedMask === "causal"
              ? { scale: this.#headDim ** -0.5, maskMode: "causal" as const }
              : { scale: this.#headDim ** -0.5, maskArray: retainedMask };
        using attentionOutput = scaledDotProductAttention(
          rotatedQueries,
          activeKeyValues.keys,
          activeKeyValues.values,
          attentionOptions,
        );
        using transposedOutput = transpose(attentionOutput, [0, 2, 1, 3]);
        using mergedOutput = reshape(transposedOutput, [
          batch,
          sequenceLength,
          this.#numHeads * this.#headDim,
        ]);
        return this.outputProjection.forward(mergedOutput);
      } finally {
        if (retainedMask instanceof MxArray) {
          retainedMask.free();
        }
      }
    } finally {
      projected.queries.free();
      projected.keys.free();
      projected.values.free();
    }
  }

  private projectQueryKeyValue(x: MxArray): { queries: MxArray; keys: MxArray; values: MxArray } {
    if (
      this.qProjection !== null &&
      this.kProjection !== null &&
      this.vProjection !== null &&
      this.qkvProjection === null
    ) {
      return {
        queries: this.qProjection.forward(x),
        keys: this.kProjection.forward(x),
        values: this.vProjection.forward(x),
      };
    }

    if (this.qkvProjection === null) {
      throw new Error(
        "LlamaLikeAttention: expected either split or packed query/key/value projections.",
      );
    }

    using packed = this.qkvProjection.forward(x);
    const queryWidth = this.#numHeads * this.#headDim;
    const keyValueWidth = this.#numKeyValueHeads * this.#headDim;
    return {
      queries: takeLastAxisRange(packed, 0, queryWidth),
      keys: takeLastAxisRange(packed, queryWidth, queryWidth + keyValueWidth),
      values: takeLastAxisRange(
        packed,
        queryWidth + keyValueWidth,
        queryWidth + keyValueWidth + keyValueWidth,
      ),
    };
  }
}
