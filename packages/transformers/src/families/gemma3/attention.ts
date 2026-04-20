/**
 * Gemma 3 text attention with q/k norm and mixed local/global masking.
 * @module
 */

import {
  formatShape,
  MxArray,
  reshape,
  retainArray,
  scaledDotProductAttention,
  transpose,
} from "@mlxts/core";
import { Linear, Module, RoPE } from "@mlxts/nn";

import {
  retainTransformerCacheView,
  updateAndFetchTransformerCacheView,
} from "../../infrastructure/cache/view";
import { type AttentionMask, createCausalMask } from "../../infrastructure/masks";
import type { TransformerCache } from "../../types";
import { Gemma3RMSNorm } from "./norm";
import type { Gemma3LayerType, Gemma3TextConfig } from "./types";

/** Self-attention module used by Gemma 3 text decoder blocks. */
export class Gemma3Attention extends Module {
  qProjection: Linear;
  kProjection: Linear;
  vProjection: Linear;
  outputProjection: Linear;
  qNorm: Gemma3RMSNorm;
  kNorm: Gemma3RMSNorm;
  rope: RoPE;
  #layerType: Gemma3LayerType;
  #layerIndex: number;
  #hiddenSize: number;
  #numHeads: number;
  #numKeyValueHeads: number;
  #headDim: number;
  #scale: number;
  #windowSize: number | undefined;

  constructor(config: Gemma3TextConfig, layerIndex: number) {
    super();
    if (config.hiddenSize % config.numAttentionHeads !== 0) {
      throw new Error(
        `Gemma3Attention: hiddenSize ${config.hiddenSize} must be divisible by numAttentionHeads ${config.numAttentionHeads}.`,
      );
    }
    if (config.numAttentionHeads % config.numKeyValueHeads !== 0) {
      throw new Error(
        `Gemma3Attention: numAttentionHeads ${config.numAttentionHeads} must be divisible by numKeyValueHeads ${config.numKeyValueHeads}.`,
      );
    }

    this.#layerType = config.layerTypes[layerIndex] ?? "full_attention";
    this.#layerIndex = layerIndex;
    this.#hiddenSize = config.hiddenSize;
    this.#numHeads = config.numAttentionHeads;
    this.#numKeyValueHeads = config.numKeyValueHeads;
    this.#headDim = config.headDim;
    this.#scale = config.queryPreAttentionScalar ** -0.5;
    this.#windowSize = this.#layerType === "sliding_attention" ? config.slidingWindow : undefined;

    this.qProjection = new Linear(
      config.hiddenSize,
      config.numAttentionHeads * config.headDim,
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
    this.qNorm = new Gemma3RMSNorm(config.headDim, config.rmsNormEps);
    this.kNorm = new Gemma3RMSNorm(config.headDim, config.rmsNormEps);
    this.rope = new RoPE(
      config.headDim,
      false,
      this.#layerType === "sliding_attention" ? config.ropeLocalBaseFreq : config.ropeTheta,
    );
  }

  get layerType(): Gemma3LayerType {
    return this.#layerType;
  }

  forward(x: MxArray): MxArray {
    return this.run(x);
  }

  run(x: MxArray, cache?: TransformerCache, attentionMask?: AttentionMask): MxArray {
    const [batch, sequenceLength, hiddenSize] = x.shape;
    if (batch === undefined || sequenceLength === undefined || hiddenSize === undefined) {
      throw new Error(
        `Gemma3Attention.forward: expected rank-3 input, got ${formatShape(x.shape)}.`,
      );
    }
    if (hiddenSize !== this.#hiddenSize) {
      throw new Error(
        `Gemma3Attention.forward: expected hidden size ${this.#hiddenSize}, got ${hiddenSize}.`,
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
    using normalizedQueries = this.qNorm.forward(queryHeads);
    using normalizedKeys = this.kNorm.forward(keyHeads);
    using rotatedQueries = this.rope.forward(normalizedQueries, cache?.offset ?? 0);
    using rotatedKeys = this.rope.forward(normalizedKeys, cache?.offset ?? 0);
    using activeKeyValues =
      cache === undefined
        ? retainTransformerCacheView(rotatedKeys, valueHeads)
        : updateAndFetchTransformerCacheView(cache, this.#layerIndex, rotatedKeys, valueHeads);
    const retainedMask = (() => {
      const totalKeyLength = activeKeyValues.keys.shape[2];
      if (totalKeyLength === undefined) {
        throw new Error("Gemma3Attention.forward: attention key cache is missing a sequence axis.");
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
        this.#windowSize,
      );
    })();

    try {
      const totalKeyLength = activeKeyValues.keys.shape[2];
      if (totalKeyLength === undefined) {
        throw new Error("Gemma3Attention.forward: attention key cache is missing a sequence axis.");
      }

      const attentionOptions =
        retainedMask === null
          ? { scale: this.#scale }
          : retainedMask === "causal"
            ? { scale: this.#scale, maskMode: "causal" as const }
            : { scale: this.#scale, maskArray: retainedMask };
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
  }
}
