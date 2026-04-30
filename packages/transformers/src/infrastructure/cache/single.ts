/**
 * Single-sequence KV cache objects for decoder generation.
 * @module
 */

import { type MxArray, retainArray, slice } from "@mlxts/core";

import type { TransformerCache, TransformerCacheSnapshot } from "../../types";
import {
  type CacheLayerKind,
  cacheLayerKindsFromWindowSizes,
  repeatedCacheLayerKinds,
} from "./layer-kind";
import {
  appendFullCacheState,
  appendSlidingCacheState,
  type CacheAppendResult,
  cloneCacheArray,
  cloneLayerStateSnapshot,
  createEmptyLayerState,
  createTransformerCacheViewFromResult,
  disposeLayerState,
  disposeLayerStateSnapshot,
  type LayerState,
  type LayerStateSnapshot,
  materializeOwnedAppendResult,
  retainedLayerStateArrays,
} from "./runtime";
import type { CacheFactory, SnapshotTrimPolicy } from "./snapshot";
import { createCacheSnapshot, type FullKVBlockSnapshotSource } from "./tensor-block-snapshot";
import { INTERNAL_CACHE_VIEW, type TransformerCacheView } from "./view";

function sliceSnapshotArray(value: MxArray, length: number): MxArray {
  const existingLength = value.shape[2];
  if (existingLength === undefined) {
    throw new Error("TransformerCacheSnapshot.fork: expected rank-4 cache snapshot tensors.");
  }
  if (length === existingLength) {
    return retainArray(value);
  }
  const batch = value.shape[0] ?? 0;
  const heads = value.shape[1] ?? 0;
  const width = value.shape[3] ?? 0;
  return slice(value, [0, 0, 0, 0], [batch, heads, length, width]);
}

abstract class CacheBase implements TransformerCache {
  offset = 0;
  #layers: LayerState[];
  #fullKVBlockSnapshotSource: FullKVBlockSnapshotSource | null = null;

  constructor(layerCount: number) {
    if (!Number.isInteger(layerCount) || layerCount <= 0) {
      throw new Error(`TransformerCache: layerCount must be a positive integer, got ${layerCount}`);
    }
    this.#layers = Array.from({ length: layerCount }, () => createEmptyLayerState());
  }

  get layerCount(): number {
    return this.#layers.length;
  }

  abstract get layerKinds(): readonly CacheLayerKind[];

  isEmpty(): boolean {
    return this.offset === 0;
  }

  isTrimmable(): boolean {
    return true;
  }

  snapshot(): TransformerCacheSnapshot {
    const layers: LayerStateSnapshot[] = [];
    try {
      for (const state of this.#layers) {
        layers.push(cloneLayerStateSnapshot(state));
      }
      return createCacheSnapshot({
        offset: this.offset,
        layerKinds: this.layerKinds,
        layers,
        createCache: this.snapshotCacheFactory(),
        trimPolicy: this.snapshotTrimPolicy(),
        ...(this.#fullKVBlockSnapshotSource === null
          ? {}
          : { source: this.#fullKVBlockSnapshotSource }),
      });
    } catch (error) {
      for (const layer of layers) {
        disposeLayerStateSnapshot(layer);
      }
      throw error;
    }
  }

  advance(sequenceLength: number): void {
    if (!Number.isInteger(sequenceLength) || sequenceLength < 0) {
      throw new Error(
        `TransformerCache.advance: sequenceLength must be a non-negative integer, got ${sequenceLength}`,
      );
    }
    this.offset += sequenceLength;
  }

  protected layerState(layerIndex: number): LayerState {
    const state = this.#layers[layerIndex];
    if (state === undefined) {
      throw new Error(`TransformerCache: layer ${layerIndex} is out of range`);
    }
    return state;
  }

  cloneLayerState(layerIndex: number): LayerStateSnapshot {
    return cloneLayerStateSnapshot(this.layerState(layerIndex));
  }

  /** Retain full-KV block lineage for a later prompt-boundary snapshot. */
  setFullKVBlockSnapshotSource(source: FullKVBlockSnapshotSource | null): void {
    this.#fullKVBlockSnapshotSource?.[Symbol.dispose]();
    this.#fullKVBlockSnapshotSource = source;
  }

  /** Retain full-KV block lineage for transfer into a batch cache. */
  retainFullKVBlockSnapshotSource(): FullKVBlockSnapshotSource | null {
    return this.#fullKVBlockSnapshotSource?.retain() ?? null;
  }

  protected appendLayer(layerIndex: number, keys: MxArray, values: MxArray): CacheAppendResult {
    return appendFullCacheState(this.layerState(layerIndex), keys, values);
  }

  restoreLayerSnapshot(
    layerIndex: number,
    snapshot: LayerStateSnapshot,
    length: number,
    cursor: number,
  ): void {
    this.setFullKVBlockSnapshotSource(null);
    const state = this.layerState(layerIndex);
    disposeLayerState(state);
    if (length === 0) {
      return;
    }
    if (snapshot.keys === null || snapshot.values === null) {
      throw new Error(
        "TransformerCacheSnapshot.fork: expected initialized layer snapshot tensors.",
      );
    }

    const keys = sliceSnapshotArray(snapshot.keys, length);
    const values = sliceSnapshotArray(snapshot.values, length);
    try {
      state.keys = cloneCacheArray(keys);
      state.values = cloneCacheArray(values);
      state.length = length;
      state.cursor = cursor;
    } finally {
      keys.free();
      values.free();
    }
  }

  stateArrays(): MxArray[] {
    const arrays: MxArray[] = [];
    for (const state of this.#layers) {
      arrays.push(...retainedLayerStateArrays(state));
    }
    return arrays;
  }

  arrays(): MxArray[] {
    return this.stateArrays();
  }

  updateAndFetch(
    layerIndex: number,
    keys: MxArray,
    values: MxArray,
  ): { keys: MxArray; values: MxArray } {
    return materializeOwnedAppendResult(this.appendLayer(layerIndex, keys, values));
  }

  [INTERNAL_CACHE_VIEW](layerIndex: number, keys: MxArray, values: MxArray): TransformerCacheView {
    return createTransformerCacheViewFromResult(this.appendLayer(layerIndex, keys, values));
  }

  [Symbol.dispose](): void {
    this.setFullKVBlockSnapshotSource(null);
    for (const state of this.#layers) {
      disposeLayerState(state);
    }
  }

  protected snapshotTrimPolicy(): SnapshotTrimPolicy {
    return "prefix";
  }

  protected abstract snapshotCacheFactory(): CacheFactory;
}

/** Full causal KV cache that keeps every prior token. */
export class KVCache extends CacheBase {
  readonly #layerKinds: readonly CacheLayerKind[];

  constructor(layerCount: number) {
    super(layerCount);
    this.#layerKinds = repeatedCacheLayerKinds(layerCount, "full");
  }

  get layerKinds(): readonly CacheLayerKind[] {
    return this.#layerKinds;
  }

  protected snapshotCacheFactory(): CacheFactory {
    const layerCount = this.layerCount;
    return () => new KVCache(layerCount);
  }
}

/** Sliding-window KV cache used by decoder families like Mistral. */
export class SlidingWindowKVCache extends CacheBase {
  #windowSize: number;
  readonly #layerKinds: readonly CacheLayerKind[];

  constructor(layerCount: number, windowSize: number) {
    super(layerCount);
    if (!Number.isInteger(windowSize) || windowSize <= 0) {
      throw new Error(
        `SlidingWindowKVCache: windowSize must be a positive integer, got ${windowSize}`,
      );
    }
    this.#windowSize = windowSize;
    this.#layerKinds = repeatedCacheLayerKinds(layerCount, "sliding");
  }

  get layerKinds(): readonly CacheLayerKind[] {
    return this.#layerKinds;
  }

  override isTrimmable(): boolean {
    return false;
  }

  protected override snapshotTrimPolicy(): SnapshotTrimPolicy {
    return "exact";
  }

  protected override snapshotCacheFactory(): CacheFactory {
    const layerCount = this.layerCount;
    const windowSize = this.#windowSize;
    return () => new SlidingWindowKVCache(layerCount, windowSize);
  }

  override updateAndFetch(
    layerIndex: number,
    keys: MxArray,
    values: MxArray,
  ): { keys: MxArray; values: MxArray } {
    return materializeOwnedAppendResult(
      appendSlidingCacheState(this.layerState(layerIndex), keys, values, this.#windowSize),
    );
  }

  override [INTERNAL_CACHE_VIEW](
    layerIndex: number,
    keys: MxArray,
    values: MxArray,
  ): TransformerCacheView {
    return createTransformerCacheViewFromResult(
      appendSlidingCacheState(this.layerState(layerIndex), keys, values, this.#windowSize),
    );
  }
}

/** Per-layer cache that mixes full and sliding-window retention within one model. */
export class LayerPatternKVCache extends CacheBase {
  #layerWindowSizes: (number | undefined)[];
  readonly #layerKinds: readonly CacheLayerKind[];

  constructor(layerCount: number, layerWindowSizes: readonly (number | undefined)[]) {
    super(layerCount);
    if (layerWindowSizes.length !== layerCount) {
      throw new Error(
        `LayerPatternKVCache: layerWindowSizes length ${layerWindowSizes.length} must match layerCount ${layerCount}`,
      );
    }
    for (const windowSize of layerWindowSizes) {
      if (windowSize !== undefined && (!Number.isInteger(windowSize) || windowSize <= 0)) {
        throw new Error(
          `LayerPatternKVCache: each window size must be a positive integer when present, got ${windowSize}`,
        );
      }
    }
    this.#layerWindowSizes = [...layerWindowSizes];
    this.#layerKinds = cacheLayerKindsFromWindowSizes(this.#layerWindowSizes);
  }

  get layerKinds(): readonly CacheLayerKind[] {
    return this.#layerKinds;
  }

  override isTrimmable(): boolean {
    return this.#layerWindowSizes.every((windowSize) => windowSize === undefined);
  }

  protected override snapshotTrimPolicy(): SnapshotTrimPolicy {
    return this.#layerWindowSizes.every((windowSize) => windowSize === undefined)
      ? "prefix"
      : "exact";
  }

  protected override snapshotCacheFactory(): CacheFactory {
    const layerCount = this.layerCount;
    const layerWindowSizes = [...this.#layerWindowSizes];
    return () => new LayerPatternKVCache(layerCount, layerWindowSizes);
  }

  override updateAndFetch(
    layerIndex: number,
    keys: MxArray,
    values: MxArray,
  ): { keys: MxArray; values: MxArray } {
    const state = this.layerState(layerIndex);
    const windowSize = this.#layerWindowSizes[layerIndex];
    if (windowSize !== undefined) {
      return materializeOwnedAppendResult(appendSlidingCacheState(state, keys, values, windowSize));
    }
    return materializeOwnedAppendResult(this.appendLayer(layerIndex, keys, values));
  }

  override [INTERNAL_CACHE_VIEW](
    layerIndex: number,
    keys: MxArray,
    values: MxArray,
  ): TransformerCacheView {
    const state = this.layerState(layerIndex);
    const windowSize = this.#layerWindowSizes[layerIndex];
    if (windowSize !== undefined) {
      return createTransformerCacheViewFromResult(
        appendSlidingCacheState(state, keys, values, windowSize),
      );
    }
    return createTransformerCacheViewFromResult(this.appendLayer(layerIndex, keys, values));
  }
}

/** Return retained cache-state arrays for explicit evaluation. The caller owns the views. */
export function cacheStateArrays(cache: TransformerCache): MxArray[] {
  return cache.arrays();
}
