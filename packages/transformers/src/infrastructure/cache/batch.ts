/**
 * Batch-aware KV cache objects for future continuous batching engines.
 * @module
 */

import { array, type MxArray, slice } from "@mlxts/core";

import type { TransformerBatchCache, TransformerCache } from "../../types";
import {
  extendLayerState,
  filterLayerState,
  validateBatchIndices,
  validateBatchMetadata,
  validateUpdateBatchSize,
} from "./batch-state";
import { type CacheLayerKind, repeatedCacheLayerKinds } from "./layer-kind";
import {
  appendFullCacheState,
  type CacheAppendResult,
  createEmptyLayerState,
  createTransformerCacheViewFromResult,
  disposeLayerState,
  disposeLayerStateSnapshot,
  type LayerState,
  type LayerStateSnapshot,
  materializeOwnedAppendResult,
  restoreLayerStateSnapshot,
  retainedLayerStateArrays,
} from "./runtime";
import { KVCache } from "./single";
import type { FullKVBlockSnapshotSource } from "./tensor-block-snapshot";
import { INTERNAL_CACHE_VIEW, type TransformerCacheView } from "./view";

function disposeFullKVBlockSnapshotSources(
  sources: readonly (FullKVBlockSnapshotSource | null)[],
): void {
  for (const source of sources) {
    source?.[Symbol.dispose]();
  }
}

function retainFullKVBlockSnapshotSources(
  sources: readonly (FullKVBlockSnapshotSource | null)[],
): (FullKVBlockSnapshotSource | null)[] {
  const retained: (FullKVBlockSnapshotSource | null)[] = [];
  try {
    for (const source of sources) {
      retained.push(source?.retain() ?? null);
    }
    return retained;
  } catch (error) {
    disposeFullKVBlockSnapshotSources(retained);
    throw error;
  }
}

/** Full KV cache for a left-padded batch of active requests. */
export class BatchKVCache implements TransformerBatchCache {
  #layers: LayerState[];
  #leftPadding: number[];
  #offsets: number[];
  #fullKVBlockSnapshotSources: (FullKVBlockSnapshotSource | null)[];
  readonly #layerKinds: readonly CacheLayerKind[];

  constructor(layerCount: number, leftPadding: readonly number[]) {
    if (!Number.isInteger(layerCount) || layerCount <= 0) {
      throw new Error(`BatchKVCache: layerCount must be a positive integer, got ${layerCount}.`);
    }
    this.#leftPadding = validateBatchMetadata(leftPadding);
    this.#offsets = this.#leftPadding.map((padding) => (padding === 0 ? 0 : -padding));
    this.#fullKVBlockSnapshotSources = this.#leftPadding.map(() => null);
    this.#layers = Array.from({ length: layerCount }, () => createEmptyLayerState());
    this.#layerKinds = repeatedCacheLayerKinds(layerCount, "full");
  }

  get layerCount(): number {
    return this.#layers.length;
  }

  get layerKinds(): readonly CacheLayerKind[] {
    return this.#layerKinds;
  }

  get batchSize(): number {
    return this.#leftPadding.length;
  }

  get length(): number {
    return Math.max(0, ...this.#layers.map((layer) => layer.length));
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
        `BatchKVCache.advance: sequenceLength must be a non-negative integer, got ${sequenceLength}.`,
      );
    }
    this.#offsets = this.#offsets.map((offset) => offset + sequenceLength);
  }

  filter(batchIndices: readonly number[]): void {
    const indices = validateBatchIndices(batchIndices, this.batchSize);
    const nextLeftPadding = indices.map((index) => this.#leftPadding[index] ?? 0);
    const trimLeft = Math.min(...nextLeftPadding);
    const nextSources = indices.map((index) => this.#fullKVBlockSnapshotSources[index] ?? null);
    const retained = new Set(indices);
    for (let index = 0; index < this.#fullKVBlockSnapshotSources.length; index += 1) {
      if (!retained.has(index)) {
        this.#fullKVBlockSnapshotSources[index]?.[Symbol.dispose]();
      }
    }
    this.#leftPadding = nextLeftPadding.map((padding) => padding - trimLeft);
    this.#offsets = indices.map((index) => this.#offsets[index] ?? 0);
    this.#fullKVBlockSnapshotSources = nextSources;
    for (const layer of this.#layers) {
      filterLayerState(layer, indices, trimLeft);
    }
  }

  extend(other: TransformerBatchCache): void {
    if (!(other instanceof BatchKVCache)) {
      throw new Error("BatchKVCache.extend: expected another BatchKVCache.");
    }
    if (other.layerCount !== this.layerCount) {
      throw new Error("BatchKVCache.extend: layer counts must match.");
    }
    const leftAdjustments = this.#layers.map((layer, index) => {
      const otherLayer = other.#layers[index];
      if (otherLayer === undefined) {
        throw new Error("BatchKVCache.extend: missing source layer.");
      }
      return Math.max(layer.length, otherLayer.length) - layer.length;
    });
    const rightAdjustments = this.#layers.map((layer, index) => {
      const otherLayer = other.#layers[index];
      if (otherLayer === undefined) {
        throw new Error("BatchKVCache.extend: missing source layer.");
      }
      return Math.max(layer.length, otherLayer.length) - otherLayer.length;
    });

    for (let index = 0; index < this.#layers.length; index += 1) {
      const layer = this.#layers[index];
      const otherLayer = other.#layers[index];
      if (layer === undefined || otherLayer === undefined) {
        continue;
      }
      extendLayerState(layer, otherLayer, this.batchSize, other.batchSize);
    }

    const leftAdjustment = Math.max(0, ...leftAdjustments);
    const rightAdjustment = Math.max(0, ...rightAdjustments);
    this.#leftPadding = [
      ...this.#leftPadding.map((padding) => padding + leftAdjustment),
      ...other.#leftPadding.map((padding) => padding + rightAdjustment),
    ];
    this.#offsets = [...this.#offsets, ...other.#offsets];
    this.#fullKVBlockSnapshotSources = [
      ...this.#fullKVBlockSnapshotSources,
      ...retainFullKVBlockSnapshotSources(other.#fullKVBlockSnapshotSources),
    ];
  }

  restoreLayerState(layerIndex: number, snapshot: LayerStateSnapshot): void {
    const layer = this.#layers[layerIndex];
    if (layer === undefined) {
      throw new Error(`BatchKVCache.restoreLayerState: layer ${layerIndex} is out of range.`);
    }
    restoreLayerStateSnapshot(layer, snapshot);
  }

  restoreFromCache(batchIndex: number, source: TransformerCache): void {
    if (!(source instanceof KVCache)) {
      throw new Error("BatchKVCache.restoreFromCache: expected a KVCache source.");
    }
    if (this.batchSize !== 1 || batchIndex !== 0) {
      throw new Error("BatchKVCache.restoreFromCache: seeded restore requires one batch row.");
    }
    if (source.layerCount !== this.layerCount) {
      throw new Error("BatchKVCache.restoreFromCache: layer counts must match.");
    }

    for (let layerIndex = 0; layerIndex < this.#layers.length; layerIndex += 1) {
      const snapshot = source.cloneLayerState(layerIndex);
      try {
        this.restoreLayerState(layerIndex, snapshot);
      } finally {
        disposeLayerStateSnapshot(snapshot);
      }
    }
    this.#leftPadding = [0];
    this.#offsets = [source.offset];
    disposeFullKVBlockSnapshotSources(this.#fullKVBlockSnapshotSources);
    this.#fullKVBlockSnapshotSources = [source.retainFullKVBlockSnapshotSource()];
  }

  extractLayer(batchIndex: number, layerIndex: number): { keys: MxArray; values: MxArray } | null {
    if (!Number.isInteger(batchIndex) || batchIndex < 0 || batchIndex >= this.batchSize) {
      throw new Error(
        `BatchKVCache.extract: batch index ${batchIndex} is out of range for batch size ${this.batchSize}.`,
      );
    }
    const layer = this.#layers[layerIndex];
    if (layer === undefined) {
      throw new Error(`BatchKVCache.extractLayer: layer ${layerIndex} is out of range.`);
    }
    if (layer.keys === null || layer.values === null) {
      return null;
    }
    const padding = this.#leftPadding[batchIndex] ?? 0;
    if (Math.max(0, layer.length - padding) === 0) {
      return null;
    }
    const heads = layer.keys.shape[1] ?? 0;
    const keyWidth = layer.keys.shape[3] ?? 0;
    const valueWidth = layer.values.shape[3] ?? 0;
    return {
      keys: slice(
        layer.keys,
        [batchIndex, 0, padding, 0],
        [batchIndex + 1, heads, layer.length, keyWidth],
      ),
      values: slice(
        layer.values,
        [batchIndex, 0, padding, 0],
        [batchIndex + 1, heads, layer.length, valueWidth],
      ),
    };
  }

  extract(batchIndex: number): TransformerCache {
    if (!Number.isInteger(batchIndex) || batchIndex < 0 || batchIndex >= this.batchSize) {
      throw new Error(
        `BatchKVCache.extract: batch index ${batchIndex} is out of range for batch size ${this.batchSize}.`,
      );
    }
    const cache = new KVCache(this.layerCount);
    try {
      let visibleLength = 0;
      for (let layerIndex = 0; layerIndex < this.#layers.length; layerIndex += 1) {
        const pair = this.extractLayer(batchIndex, layerIndex);
        if (pair === null) {
          continue;
        }
        using keys = pair.keys;
        using values = pair.values;
        const updated = cache.updateAndFetch(layerIndex, keys, values);
        updated.keys.free();
        updated.values.free();
        visibleLength = Math.max(visibleLength, pair.keys.shape[2] ?? 0);
      }
      cache.advance(visibleLength);
      const source = this.#fullKVBlockSnapshotSources[batchIndex];
      if (source !== undefined && source !== null) {
        cache.setFullKVBlockSnapshotSource(source.retain());
      }
      return cache;
    } catch (error) {
      cache[Symbol.dispose]();
      throw error;
    }
  }

  offsetTensor(): MxArray {
    return array([...this.#offsets], "int32");
  }

  leftPaddingTensor(): MxArray {
    return array([...this.#leftPadding], "int32");
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
    disposeFullKVBlockSnapshotSources(this.#fullKVBlockSnapshotSources);
    this.#fullKVBlockSnapshotSources = [];
    for (const state of this.#layers) {
      disposeLayerState(state);
    }
  }

  private appendLayer(layerIndex: number, keys: MxArray, values: MxArray): CacheAppendResult {
    const state = this.#layers[layerIndex];
    if (state === undefined) {
      throw new Error(`BatchKVCache: layer ${layerIndex} is out of range.`);
    }
    return appendFullCacheState(state, keys, values);
  }
}

function isBatchKVCache(value: unknown): value is BatchKVCache {
  return value instanceof BatchKVCache;
}

/** Return true when a value is the managed batch KV cache implementation. */
export function isManagedBatchKVCache(value: unknown): value is BatchKVCache {
  return isBatchKVCache(value);
}
