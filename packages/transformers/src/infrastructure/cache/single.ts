/**
 * Single-sequence KV cache objects for decoder generation.
 * @module
 */

import type { MxArray } from "@mlxts/core";

import type { TransformerCache } from "../../types";
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
import { INTERNAL_CACHE_VIEW, type TransformerCacheView } from "./view";

abstract class CacheBase implements TransformerCache {
  offset = 0;
  #layers: LayerState[];

  constructor(layerCount: number) {
    if (!Number.isInteger(layerCount) || layerCount <= 0) {
      throw new Error(`TransformerCache: layerCount must be a positive integer, got ${layerCount}`);
    }
    this.#layers = Array.from({ length: layerCount }, () => createEmptyLayerState());
  }

  get layerCount(): number {
    return this.#layers.length;
  }

  isEmpty(): boolean {
    return this.offset === 0;
  }

  isTrimmable(): boolean {
    return true;
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

  protected appendLayer(layerIndex: number, keys: MxArray, values: MxArray): CacheAppendResult {
    return appendFullCacheState(this.layerState(layerIndex), keys, values);
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
    for (const state of this.#layers) {
      disposeLayerState(state);
    }
  }
}

/** Full causal KV cache that keeps every prior token. */
export class KVCache extends CacheBase {}

/** Sliding-window KV cache used by decoder families like Mistral. */
export class SlidingWindowKVCache extends CacheBase {
  #windowSize: number;

  constructor(layerCount: number, windowSize: number) {
    super(layerCount);
    if (!Number.isInteger(windowSize) || windowSize <= 0) {
      throw new Error(
        `SlidingWindowKVCache: windowSize must be a positive integer, got ${windowSize}`,
      );
    }
    this.#windowSize = windowSize;
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
