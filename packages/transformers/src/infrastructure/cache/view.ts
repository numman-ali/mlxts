import { type MxArray, retainArray } from "@mlxts/core";

import type { DecoderCache } from "../../types";

export const INTERNAL_CACHE_VIEW = Symbol("mlxts.transformers.internalCacheView");

export type OwnedKeyValuePair = {
  keys: MxArray;
  values: MxArray;
};

/**
 * Private attention-facing cache view.
 *
 * Borrowed views may alias mutable cache state and should not be kept across a
 * later mutation of the same cache layer. Call `materializeOwnedPair()` if a
 * stable owned pair is needed beyond the current forward step.
 */
export interface TransformerCacheView extends Disposable {
  readonly keys: MxArray;
  readonly values: MxArray;
  materializeOwnedPair(): OwnedKeyValuePair;
}

export type InternalTransformerCache = DecoderCache & {
  [INTERNAL_CACHE_VIEW](layerIndex: number, keys: MxArray, values: MxArray): TransformerCacheView;
};

class CacheView implements TransformerCacheView {
  #keys: MxArray | null;
  #values: MxArray | null;
  #ownsBuffers: boolean;

  constructor(keys: MxArray, values: MxArray, ownsBuffers: boolean) {
    this.#keys = keys;
    this.#values = values;
    this.#ownsBuffers = ownsBuffers;
  }

  get keys(): MxArray {
    const keys = this.#keys;
    if (keys === null) {
      throw new Error("TransformerCacheView.keys: cache view has already been released.");
    }
    return keys;
  }

  get values(): MxArray {
    const values = this.#values;
    if (values === null) {
      throw new Error("TransformerCacheView.values: cache view has already been released.");
    }
    return values;
  }

  materializeOwnedPair(): OwnedKeyValuePair {
    const keys = this.keys;
    const values = this.values;
    if (!this.#ownsBuffers) {
      return {
        keys: retainArray(keys),
        values: retainArray(values),
      };
    }
    this.#ownsBuffers = false;
    this.#keys = null;
    this.#values = null;
    return { keys, values };
  }

  [Symbol.dispose](): void {
    if (this.#ownsBuffers) {
      this.#keys?.free();
      this.#values?.free();
    }
    this.#ownsBuffers = false;
    this.#keys = null;
    this.#values = null;
  }
}

export function createOwnedTransformerCacheView(
  keys: MxArray,
  values: MxArray,
): TransformerCacheView {
  return new CacheView(keys, values, true);
}

export function createBorrowedTransformerCacheView(
  keys: MxArray,
  values: MxArray,
): TransformerCacheView {
  return new CacheView(keys, values, false);
}

export function retainTransformerCacheView(keys: MxArray, values: MxArray): TransformerCacheView {
  return createOwnedTransformerCacheView(retainArray(keys), retainArray(values));
}

function hasInternalCacheView(cache: DecoderCache): cache is InternalTransformerCache {
  return INTERNAL_CACHE_VIEW in cache;
}

export function updateAndFetchTransformerCacheView(
  cache: DecoderCache,
  layerIndex: number,
  keys: MxArray,
  values: MxArray,
): TransformerCacheView {
  if (hasInternalCacheView(cache)) {
    return cache[INTERNAL_CACHE_VIEW](layerIndex, keys, values);
  }
  const ownedPair = cache.updateAndFetch(layerIndex, keys, values);
  return createOwnedTransformerCacheView(ownedPair.keys, ownedPair.values);
}
