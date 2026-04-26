/**
 * Gemma 4 dense text attention.
 * @module
 */

import { formatShape, MxArray } from "@mlxts/core";
import { Linear, Module } from "@mlxts/nn";

import {
  isManagedLayerPatternBatchKVCache,
  isSingleTransformerCache,
} from "../../infrastructure/cache";
import {
  createBorrowedTransformerCacheView,
  retainTransformerCacheView,
  type TransformerCacheView,
  updateAndFetchTransformerCacheView,
} from "../../infrastructure/cache/view";
import {
  type AttentionMask,
  canOmitLeftPaddedAttentionMask,
  createCausalMask,
  createLeftPaddedAttentionMask,
} from "../../infrastructure/masks";
import { recordTransformerRuntimeCounter } from "../../infrastructure/runtime-profile";
import type { DecoderCache } from "../../types";
import { Gemma4RMSNorm } from "./norm";
import { createRoPE, type RotaryEmbedding } from "./rope";
import type { AttentionRuntimeLayout, AttentionRuntimeWeights } from "./runtime/attention";
import {
  prepareKeyValueHeads,
  prepareQueryHeadsAndRope,
  runSdpaAndOutput,
} from "./runtime/attention";
import {
  type Gemma4SharedKeyValues,
  type Gemma4TextConfig,
  gemma4UsesAlternativeAttention,
} from "./types";

type Gemma4AttentionResult = {
  output: MxArray;
  keyValues: Gemma4SharedKeyValues | null;
};

function ropeOffsetForCache(cache: DecoderCache | undefined): number | MxArray {
  if (cache === undefined) {
    return 0;
  }
  if (isManagedLayerPatternBatchKVCache(cache)) {
    return cache.offsetTensor();
  }
  if (isSingleTransformerCache(cache)) {
    return cache.offset;
  }
  throw new Error("Gemma4TextAttention.forward: unsupported batch cache implementation.");
}

/** Self-attention module used by Gemma 4 dense text decoder blocks. */
export class Gemma4TextAttention extends Module {
  qProjection: Linear;
  kProjection: Linear;
  vProjection: Linear | null;
  outputProjection: Linear;
  qNorm: Gemma4RMSNorm;
  kNorm: Gemma4RMSNorm;
  vNorm: Gemma4RMSNorm;
  rope: RotaryEmbedding;
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
    this.rope = createRoPE(
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
    cache?: DecoderCache,
    sharedKeyValues?: Gemma4SharedKeyValues,
    attentionMask?: AttentionMask,
  ): Gemma4AttentionResult {
    const { sequenceLength } = this.assertInputShape(x);
    const offset = ropeOffsetForCache(cache);
    const layout = this.runtimeLayout();
    const weights = this.runtimeWeights();

    try {
      using rotatedQueries = prepareQueryHeadsAndRope(layout, weights, x, offset);
      using activeKeyValues = this.resolveActiveKeyValues(x, cache, sharedKeyValues, offset);
      return this.runSDPAAndOutput(
        layout,
        weights,
        rotatedQueries,
        activeKeyValues,
        sharedKeyValues,
        cache,
        attentionMask,
        sequenceLength,
      );
    } finally {
      if (offset instanceof MxArray) {
        offset.free();
      }
    }
  }

  private runSDPAAndOutput(
    layout: AttentionRuntimeLayout,
    weights: AttentionRuntimeWeights,
    rotatedQueries: MxArray,
    activeKeyValues: TransformerCacheView,
    sharedKeyValues: Gemma4SharedKeyValues | undefined,
    cache: DecoderCache | undefined,
    attentionMask: AttentionMask | undefined,
    sequenceLength: number,
  ): Gemma4AttentionResult {
    const { mask: attentionMaskValue, ownedMask } = this.resolveAttentionMask(
      attentionMask,
      cache,
      sequenceLength,
      activeKeyValues.keys,
      rotatedQueries.dtype,
    );
    let returnedKeyValues: Gemma4SharedKeyValues | null = null;

    try {
      const output = runSdpaAndOutput(
        layout,
        weights,
        rotatedQueries,
        activeKeyValues.keys,
        activeKeyValues.values,
        attentionMaskValue,
      );
      returnedKeyValues =
        sharedKeyValues === undefined ? activeKeyValues.materializeOwnedPair() : null;
      return { output, keyValues: returnedKeyValues };
    } catch (error) {
      returnedKeyValues?.keys.free();
      returnedKeyValues?.values.free();
      throw error;
    } finally {
      ownedMask?.free();
    }
  }

  private runtimeLayout(): AttentionRuntimeLayout {
    return {
      numHeads: this.#numHeads,
      numKeyValueHeads: this.#numKeyValueHeads,
      headDim: this.#headDim,
    };
  }

  private runtimeWeights(): AttentionRuntimeWeights {
    return {
      qProjection: this.qProjection,
      kProjection: this.kProjection,
      vProjection: this.vProjection,
      outputProjection: this.outputProjection,
      qNorm: this.qNorm,
      kNorm: this.kNorm,
      vNorm: this.vNorm,
      rope: this.rope,
    };
  }

  private resolveActiveKeyValues(
    x: MxArray,
    cache: DecoderCache | undefined,
    sharedKeyValues: Gemma4SharedKeyValues | undefined,
    offset: number | MxArray,
  ): TransformerCacheView {
    if (sharedKeyValues !== undefined) {
      return createBorrowedTransformerCacheView(sharedKeyValues.keys, sharedKeyValues.values);
    }
    return this.buildFreshKeyValues(x, cache, offset);
  }

  private resolveAttentionMask(
    attentionMask: AttentionMask | undefined,
    cache: DecoderCache | undefined,
    sequenceLength: number,
    keys: MxArray,
    dtype: MxArray["dtype"],
  ): { mask: AttentionMask; ownedMask: MxArray | null } {
    if (attentionMask === undefined) {
      const createdMask = this.createMask(cache, sequenceLength, keys, dtype);
      if (createdMask instanceof MxArray) {
        recordTransformerRuntimeCounter("attention.mask_created");
      }
      return { mask: createdMask, ownedMask: createdMask };
    }
    if (attentionMask === null || attentionMask === "causal") {
      return { mask: attentionMask, ownedMask: null };
    }
    recordTransformerRuntimeCounter("attention.mask_borrowed");
    return { mask: attentionMask, ownedMask: null };
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
    cache: DecoderCache | undefined,
    offset: number | MxArray,
  ): TransformerCacheView {
    const layout = this.runtimeLayout();
    const weights = this.runtimeWeights();
    const keyValues = prepareKeyValueHeads(layout, weights, x, offset);

    try {
      return cache === undefined
        ? retainTransformerCacheView(keyValues.keys, keyValues.values)
        : updateAndFetchTransformerCacheView(
            cache,
            this.#layerIndex,
            keyValues.keys,
            keyValues.values,
          );
    } finally {
      keyValues.keys.free();
      keyValues.values.free();
    }
  }

  private createMask(
    cache: DecoderCache | undefined,
    sequenceLength: number,
    keys: MxArray,
    dtype: MxArray["dtype"],
  ): MxArray | null {
    const totalKeyLength = keys.shape[2];
    if (totalKeyLength === undefined) {
      throw new Error("Gemma4TextAttention.forward: key states are missing a sequence axis.");
    }
    const visiblePastLength = Math.max(0, totalKeyLength - sequenceLength);
    if (isManagedLayerPatternBatchKVCache(cache)) {
      const leftPaddingValues = cache.leftPaddingValuesForLayer(
        this.#layerIndex,
        totalKeyLength,
        sequenceLength,
      );
      if (canOmitLeftPaddedAttentionMask(sequenceLength, leftPaddingValues)) {
        return null;
      }
      using leftPadding = cache.leftPaddingTensorForLayer(
        this.#layerIndex,
        totalKeyLength,
        sequenceLength,
      );
      return createLeftPaddedAttentionMask(
        sequenceLength,
        totalKeyLength,
        visiblePastLength,
        leftPadding,
        this.#windowSize,
      );
    }
    return createCausalMask(
      sequenceLength,
      totalKeyLength,
      visiblePastLength,
      dtype,
      this.#windowSize,
    );
  }
}
