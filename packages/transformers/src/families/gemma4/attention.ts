/**
 * Gemma 4 dense text attention.
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
import { Linear, Module } from "@mlxts/nn";

import { type AttentionMask, createCausalMask } from "../../infrastructure/masks";
import type { TransformerCache } from "../../types";
import { Gemma4RMSNorm } from "./norm";
import { createGemma4RoPE, type Gemma4RotaryEmbedding } from "./rope";
import {
  type Gemma4SharedKeyValues,
  type Gemma4TextConfig,
  gemma4UsesAlternativeAttention,
} from "./types";

type Gemma4AttentionResult = {
  output: MxArray;
  keyValues: Gemma4SharedKeyValues | null;
};

type Gemma4FreshKeyValues = {
  keys: MxArray;
  values: MxArray;
};

type Gemma4ActiveKeyValues = {
  keys: MxArray;
  values: MxArray;
  ownsBuffers: boolean;
};

/** Self-attention module used by Gemma 4 dense text decoder blocks. */
export class Gemma4TextAttention extends Module {
  qProjection: Linear;
  kProjection: Linear;
  vProjection: Linear | null;
  outputProjection: Linear;
  qNorm: Gemma4RMSNorm;
  kNorm: Gemma4RMSNorm;
  vNorm: Gemma4RMSNorm;
  rope: Gemma4RotaryEmbedding;
  readonly layerType: Gemma4TextConfig["layerTypes"][number];
  #layerIndex: number;
  #hiddenSize: number;
  #numHeads: number;
  #numKeyValueHeads: number;
  #headDim: number;
  #windowSize: number | undefined;
  #usesAlternativeAttention: boolean;

  constructor(config: Gemma4TextConfig, layerIndex: number) {
    super();
    this.layerType = config.layerTypes[layerIndex] ?? "sliding_attention";
    this.#layerIndex = layerIndex;
    this.#hiddenSize = config.hiddenSize;
    this.#numHeads = config.numAttentionHeads;
    this.#headDim = this.layerType === "full_attention" ? config.globalHeadDim : config.headDim;
    this.#usesAlternativeAttention = gemma4UsesAlternativeAttention(config, layerIndex);
    this.#numKeyValueHeads =
      this.#usesAlternativeAttention && config.numGlobalKeyValueHeads !== null
        ? config.numGlobalKeyValueHeads
        : config.numKeyValueHeads;
    if (config.numAttentionHeads % this.#numKeyValueHeads !== 0) {
      throw new Error(
        `Gemma4TextAttention: numAttentionHeads ${config.numAttentionHeads} must be divisible by numKeyValueHeads ${this.#numKeyValueHeads}.`,
      );
    }
    this.#windowSize = this.layerType === "sliding_attention" ? config.slidingWindow : undefined;

    this.qProjection = new Linear(
      config.hiddenSize,
      config.numAttentionHeads * this.#headDim,
      config.attentionBias,
    );
    this.kProjection = new Linear(
      config.hiddenSize,
      this.#numKeyValueHeads * this.#headDim,
      config.attentionBias,
    );
    this.vProjection = this.#usesAlternativeAttention
      ? null
      : new Linear(config.hiddenSize, this.#numKeyValueHeads * this.#headDim, config.attentionBias);
    this.outputProjection = new Linear(
      config.numAttentionHeads * this.#headDim,
      config.hiddenSize,
      config.attentionBias,
    );
    this.qNorm = new Gemma4RMSNorm(this.#headDim, config.rmsNormEps);
    this.kNorm = new Gemma4RMSNorm(this.#headDim, config.rmsNormEps);
    this.vNorm = new Gemma4RMSNorm(this.#headDim, config.rmsNormEps, false);
    this.rope = createGemma4RoPE(
      this.#headDim,
      this.layerType === "full_attention" ? config.fullRopeTheta : config.slidingRopeTheta,
      this.layerType === "full_attention" ? config.fullRotaryDimensions : undefined,
    );
  }

  forward(x: MxArray): MxArray {
    const result = this.run(x);
    result.keyValues?.keys.free();
    result.keyValues?.values.free();
    return result.output;
  }

  run(
    x: MxArray,
    cache?: TransformerCache,
    sharedKeyValues?: Gemma4SharedKeyValues,
    attentionMask?: AttentionMask,
  ): Gemma4AttentionResult {
    const { batch, sequenceLength } = this.assertInputShape(x);

    using projectedQueries = this.qProjection.forward(x);
    using queryInputs = reshape(projectedQueries, [
      batch,
      sequenceLength,
      this.#numHeads,
      this.#headDim,
    ]);
    using normalizedQueries = this.qNorm.forward(queryInputs);
    using queryHeads = transpose(normalizedQueries, [0, 2, 1, 3]);
    using rotatedQueries = this.rope.forward(queryHeads, cache?.offset ?? 0);
    const activeKeyValues = this.resolveActiveKeyValues(
      x,
      batch,
      sequenceLength,
      cache,
      sharedKeyValues,
    );
    const retainedMask = this.resolveAttentionMask(
      attentionMask,
      sequenceLength,
      activeKeyValues.keys,
      rotatedQueries.dtype,
    );
    let returnedKeyValues: Gemma4SharedKeyValues | null = null;

    try {
      const totalKeyLength = activeKeyValues.keys.shape[2];
      if (totalKeyLength === undefined) {
        throw new Error("Gemma4TextAttention.forward: key states are missing a sequence axis.");
      }

      const attentionOptions =
        retainedMask === null
          ? { scale: 1.0 }
          : retainedMask === "causal"
            ? { scale: 1.0, maskMode: "causal" as const }
            : { scale: 1.0, maskArray: retainedMask };
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
      returnedKeyValues = activeKeyValues.ownsBuffers ? activeKeyValues : null;
      return {
        output: this.outputProjection.forward(mergedOutput),
        keyValues: returnedKeyValues,
      };
    } catch (error) {
      returnedKeyValues?.keys.free();
      returnedKeyValues?.values.free();
      throw error;
    } finally {
      if (retainedMask instanceof MxArray) {
        retainedMask.free();
      }
      if (activeKeyValues.ownsBuffers && returnedKeyValues === null) {
        activeKeyValues.keys.free();
        activeKeyValues.values.free();
      }
    }
  }

  private resolveActiveKeyValues(
    x: MxArray,
    batch: number,
    sequenceLength: number,
    cache: TransformerCache | undefined,
    sharedKeyValues: Gemma4SharedKeyValues | undefined,
  ): Gemma4ActiveKeyValues {
    if (sharedKeyValues !== undefined) {
      return {
        keys: sharedKeyValues.keys,
        values: sharedKeyValues.values,
        ownsBuffers: false,
      };
    }
    return {
      ...this.buildFreshKeyValues(x, batch, sequenceLength, cache),
      ownsBuffers: true,
    };
  }

  private resolveAttentionMask(
    attentionMask: AttentionMask | undefined,
    sequenceLength: number,
    keys: MxArray,
    dtype: MxArray["dtype"],
  ): AttentionMask {
    if (attentionMask === undefined) {
      return this.createMask(sequenceLength, keys, dtype);
    }
    if (attentionMask === null || attentionMask === "causal") {
      return attentionMask;
    }
    return retainArray(attentionMask);
  }

  private assertInputShape(x: MxArray): { batch: number; sequenceLength: number } {
    const [batch, sequenceLength, hiddenSize] = x.shape;
    if (batch === undefined || sequenceLength === undefined || hiddenSize === undefined) {
      throw new Error(
        `Gemma4TextAttention.forward: expected rank-3 input, got ${formatShape(x.shape)}.`,
      );
    }
    if (hiddenSize !== this.#hiddenSize) {
      throw new Error(
        `Gemma4TextAttention.forward: expected hidden size ${this.#hiddenSize}, got ${hiddenSize}.`,
      );
    }
    return { batch, sequenceLength };
  }

  private buildFreshKeyValues(
    x: MxArray,
    batch: number,
    sequenceLength: number,
    cache?: TransformerCache,
  ): Gemma4FreshKeyValues {
    using projectedKeys = this.kProjection.forward(x);
    using keyInputs = reshape(projectedKeys, [
      batch,
      sequenceLength,
      this.#numKeyValueHeads,
      this.#headDim,
    ]);
    let valueInputs: MxArray | null = null;
    try {
      if (this.vProjection === null) {
        valueInputs = retainArray(keyInputs);
      } else {
        using projectedValues = this.vProjection.forward(x);
        valueInputs = reshape(projectedValues, [
          batch,
          sequenceLength,
          this.#numKeyValueHeads,
          this.#headDim,
        ]);
      }

      using normalizedKeys = this.kNorm.forward(keyInputs);
      using normalizedValues = this.vNorm.forward(valueInputs);
      using keyHeads = transpose(normalizedKeys, [0, 2, 1, 3]);
      using valueHeads = transpose(normalizedValues, [0, 2, 1, 3]);
      using rotatedKeys = this.rope.forward(keyHeads, cache?.offset ?? 0);

      const keyValues =
        cache === undefined
          ? { keys: retainArray(rotatedKeys), values: retainArray(valueHeads) }
          : cache.updateAndFetch(this.#layerIndex, rotatedKeys, valueHeads);
      return keyValues;
    } finally {
      valueInputs?.free();
    }
  }

  private createMask(
    sequenceLength: number,
    keys: MxArray,
    dtype: MxArray["dtype"],
  ): MxArray | null {
    const totalKeyLength = keys.shape[2];
    if (totalKeyLength === undefined) {
      throw new Error("Gemma4TextAttention.forward: key states are missing a sequence axis.");
    }
    const visiblePastLength = Math.max(0, totalKeyLength - sequenceLength);
    return createCausalMask(
      sequenceLength,
      totalKeyLength,
      visiblePastLength,
      dtype,
      this.#windowSize,
    );
  }
}
