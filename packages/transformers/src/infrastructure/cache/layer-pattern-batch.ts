/**
 * Batch KV cache for models that mix full and sliding-window attention layers.
 * @module
 */

import { array, type MxArray, slice } from "@mlxts/core";

import type { TransformerBatchCache, TransformerCache } from "../../types";
import {
  extendLayerState,
  filterLayerState,
  validateBatchIndices,
  validateLayerWindows,
  validateLeftPadding,
} from "./layer-pattern-batch-state";
import {
  appendFullCacheState,
  appendSlidingCacheState,
  type CacheAppendResult,
  createEmptyLayerState,
  createTransformerCacheViewFromResult,
  disposeLayerState,
  type LayerState,
  materializeOwnedAppendResult,
  retainedLayerStateArrays,
} from "./runtime";
import { LayerPatternKVCache } from "./single";
import { INTERNAL_CACHE_VIEW, type TransformerCacheView } from "./view";

function validateUpdateBatchSize(keys: MxArray, values: MxArray, batchSize: number): void {
  const keyBatch = keys.shape[0];
  const valueBatch = values.shape[0];
  if (keyBatch !== batchSize || valueBatch !== batchSize) {
    throw new Error(
      `LayerPatternBatchKVCache.updateAndFetch: update batch size must be ${batchSize}, got ${keyBatch} and ${valueBatch}.`,
    );
  }
}

/** Batch cache that applies full or sliding retention independently per layer. */
export class LayerPatternBatchKVCache implements TransformerBatchCache {
  #layers: LayerState[];
  #layerWindowSizes: (number | undefined)[];
  #leftPadding: number[];
  #offsets: number[];
  #logicalLength = 0;

  constructor(
    layerCount: number,
    leftPadding: readonly number[],
    layerWindowSizes: readonly (number | undefined)[],
  ) {
    if (!Number.isInteger(layerCount) || layerCount <= 0) {
      throw new Error(
        `LayerPatternBatchKVCache: layerCount must be a positive integer, got ${layerCount}.`,
      );
    }
    this.#leftPadding = validateLeftPadding(leftPadding);
    this.#offsets = this.#leftPadding.map((padding) => (padding === 0 ? 0 : -padding));
    this.#layerWindowSizes = validateLayerWindows(layerCount, layerWindowSizes);
    this.#layers = Array.from({ length: layerCount }, () => createEmptyLayerState());
  }

  get layerCount(): number {
    return this.#layers.length;
  }

  get batchSize(): number {
    return this.#leftPadding.length;
  }

  get length(): number {
    return this.#logicalLength;
  }

  get leftPadding(): readonly number[] {
    return [...this.#leftPadding];
  }

  get offsets(): readonly number[] {
    return [...this.#offsets];
  }

  isEmpty(): boolean {
    return this.#layers.every((layer) => layer.length === 0);
  }

  isTrimmable(): boolean {
    return true;
  }

  advance(sequenceLength: number): void {
    if (!Number.isInteger(sequenceLength) || sequenceLength < 0) {
      throw new Error(
        `LayerPatternBatchKVCache.advance: sequenceLength must be non-negative, got ${sequenceLength}.`,
      );
    }
    this.#logicalLength += sequenceLength;
    this.#offsets = this.#offsets.map((offset) => offset + sequenceLength);
  }

  filter(batchIndices: readonly number[]): void {
    const indices = validateBatchIndices(batchIndices, this.batchSize);
    const nextLeftPadding = indices.map((index) => this.#leftPadding[index] ?? 0);
    const trimLeft = Math.min(...nextLeftPadding);
    const previousLogicalLength = this.#logicalLength;
    this.#leftPadding = nextLeftPadding.map((padding) => padding - trimLeft);
    this.#offsets = indices.map((index) => this.#offsets[index] ?? 0);
    this.#logicalLength = Math.max(0, this.#logicalLength - trimLeft);
    for (const layer of this.#layers) {
      const alreadyTrimmed = Math.max(0, previousLogicalLength - layer.length);
      const layerTrim = Math.max(0, trimLeft - alreadyTrimmed);
      filterLayerState(layer, indices, Math.min(layerTrim, layer.length));
    }
  }

  extend(other: TransformerBatchCache): void {
    if (!(other instanceof LayerPatternBatchKVCache)) {
      throw new Error(
        "LayerPatternBatchKVCache.extend: expected another layer-pattern batch cache.",
      );
    }
    if (other.layerCount !== this.layerCount) {
      throw new Error("LayerPatternBatchKVCache.extend: layer counts must match.");
    }
    if (!this.hasSameLayerWindows(other)) {
      throw new Error("LayerPatternBatchKVCache.extend: layer window sizes must match.");
    }

    const targetLogicalLength = Math.max(this.#logicalLength, other.#logicalLength);
    for (let index = 0; index < this.#layers.length; index += 1) {
      const layer = this.#layers[index];
      const otherLayer = other.#layers[index];
      if (layer === undefined || otherLayer === undefined) {
        continue;
      }
      extendLayerState(
        layer,
        otherLayer,
        this.batchSize,
        other.batchSize,
        this.retainedLengthForLogicalLength(index, targetLogicalLength),
      );
    }

    const leftAdjustment = targetLogicalLength - this.#logicalLength;
    const rightAdjustment = targetLogicalLength - other.#logicalLength;
    this.#leftPadding = [
      ...this.#leftPadding.map((padding) => padding + leftAdjustment),
      ...other.#leftPadding.map((padding) => padding + rightAdjustment),
    ];
    this.#offsets = [...this.#offsets, ...other.#offsets];
    this.#logicalLength = targetLogicalLength;
  }

  extract(batchIndex: number): TransformerCache {
    if (!Number.isInteger(batchIndex) || batchIndex < 0 || batchIndex >= this.batchSize) {
      throw new Error(
        `LayerPatternBatchKVCache.extract: batch index ${batchIndex} is out of range for batch size ${this.batchSize}.`,
      );
    }
    const cache = new LayerPatternKVCache(this.layerCount, this.#layerWindowSizes);
    for (let layerIndex = 0; layerIndex < this.#layers.length; layerIndex += 1) {
      const layer = this.#layers[layerIndex];
      if (layer === undefined || layer.keys === null || layer.values === null) {
        continue;
      }
      const padding = this.leftPaddingForLayer(batchIndex, layer.length, 0);
      const visibleLength = Math.max(0, layer.length - padding);
      if (visibleLength === 0) {
        continue;
      }
      const heads = layer.keys.shape[1] ?? 0;
      const keyWidth = layer.keys.shape[3] ?? 0;
      const valueWidth = layer.values.shape[3] ?? 0;
      using keys = slice(
        layer.keys,
        [batchIndex, 0, padding, 0],
        [batchIndex + 1, heads, layer.length, keyWidth],
      );
      using values = slice(
        layer.values,
        [batchIndex, 0, padding, 0],
        [batchIndex + 1, heads, layer.length, valueWidth],
      );
      const updated = cache.updateAndFetch(layerIndex, keys, values);
      updated.keys.free();
      updated.values.free();
    }
    cache.advance(Math.max(0, this.#offsets[batchIndex] ?? 0));
    return cache;
  }

  offsetTensor(): MxArray {
    return array([...this.#offsets], "int32");
  }

  leftPaddingTensor(): MxArray {
    return array([...this.#leftPadding], "int32");
  }

  leftPaddingTensorForLayer(
    layerIndex: number,
    activeKeyLength: number,
    queryLength: number,
  ): MxArray {
    return array(this.leftPaddingValuesForLayer(layerIndex, activeKeyLength, queryLength), "int32");
  }

  leftPaddingValuesForLayer(
    layerIndex: number,
    activeKeyLength: number,
    queryLength: number,
  ): number[] {
    this.assertLayerIndex(layerIndex);
    return this.#leftPadding.map((_, batchIndex) =>
      this.leftPaddingForLayer(batchIndex, activeKeyLength, queryLength),
    );
  }

  updateAndFetch(
    layerIndex: number,
    keys: MxArray,
    values: MxArray,
  ): { keys: MxArray; values: MxArray } {
    validateUpdateBatchSize(keys, values, this.batchSize);
    return materializeOwnedAppendResult(this.appendLayer(layerIndex, keys, values));
  }

  [INTERNAL_CACHE_VIEW](layerIndex: number, keys: MxArray, values: MxArray): TransformerCacheView {
    validateUpdateBatchSize(keys, values, this.batchSize);
    return createTransformerCacheViewFromResult(this.appendLayer(layerIndex, keys, values));
  }

  arrays(): MxArray[] {
    const arrays: MxArray[] = [];
    for (const state of this.#layers) {
      arrays.push(...retainedLayerStateArrays(state));
    }
    return arrays;
  }

  [Symbol.dispose](): void {
    for (const state of this.#layers) {
      disposeLayerState(state);
    }
  }

  private appendLayer(layerIndex: number, keys: MxArray, values: MxArray): CacheAppendResult {
    const state = this.#layers[layerIndex];
    if (state === undefined) {
      throw new Error(`LayerPatternBatchKVCache: layer ${layerIndex} is out of range.`);
    }
    const windowSize = this.#layerWindowSizes[layerIndex];
    return windowSize === undefined
      ? appendFullCacheState(state, keys, values)
      : appendSlidingCacheState(state, keys, values, windowSize);
  }

  private assertLayerIndex(layerIndex: number): void {
    if (!Number.isInteger(layerIndex) || layerIndex < 0 || layerIndex >= this.layerCount) {
      throw new Error(`LayerPatternBatchKVCache: layer ${layerIndex} is out of range.`);
    }
  }

  private hasSameLayerWindows(other: LayerPatternBatchKVCache): boolean {
    return this.#layerWindowSizes.every(
      (windowSize, index) => windowSize === other.#layerWindowSizes[index],
    );
  }

  private retainedLengthForLogicalLength(layerIndex: number, logicalLength: number): number {
    const windowSize = this.#layerWindowSizes[layerIndex];
    return windowSize === undefined ? logicalLength : Math.min(logicalLength, windowSize);
  }

  private leftPaddingForLayer(
    batchIndex: number,
    activeKeyLength: number,
    queryLength: number,
  ): number {
    const logicalTotalLength = this.#logicalLength + queryLength;
    const trimmedLeft = Math.max(0, logicalTotalLength - activeKeyLength);
    return Math.max(0, (this.#leftPadding[batchIndex] ?? 0) - trimmedLeft);
  }
}

/** Return true for the managed layer-pattern batch KV cache implementation. */
export function isManagedLayerPatternBatchKVCache(
  value: unknown,
): value is LayerPatternBatchKVCache {
  return value instanceof LayerPatternBatchKVCache;
}
