import type { DecoderCache, TransformerCache } from "../../types";

import { isManagedBatchKVCache } from "./batch";

export { BatchKVCache, isManagedBatchKVCache } from "./batch";
export { cacheStateArrays, KVCache, LayerPatternKVCache, SlidingWindowKVCache } from "./single";

/** Return true when a decoder cache has the single-sequence cache contract. */
export function isSingleTransformerCache(cache: DecoderCache): cache is TransformerCache {
  return "offset" in cache;
}

/** Return a single-sequence cache or throw when a model cannot consume a batch cache yet. */
export function expectSingleTransformerCache(
  cache: DecoderCache | undefined,
  context: string,
): TransformerCache | undefined {
  if (cache === undefined) {
    return undefined;
  }
  if (!isSingleTransformerCache(cache)) {
    const cacheName = isManagedBatchKVCache(cache) ? "BatchKVCache" : "batch cache";
    throw new Error(`${context}: ${cacheName} is not supported by this model yet.`);
  }
  return cache;
}
