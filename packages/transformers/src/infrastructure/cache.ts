/**
 * Explicit KV cache objects for decoder generation.
 * @module
 */

import { concatenate, type MxArray, retainArray, slice, sliceUpdate, zeros } from "@mlxts/core";

import type { TransformerCache } from "../types";

type LayerState = {
  keys: MxArray | null;
  values: MxArray | null;
  length: number;
  cursor: number;
};

const CACHE_GROWTH_STEP = 256;

function emptyLayerState(): LayerState {
  return { keys: null, values: null, length: 0, cursor: 0 };
}

function sequenceAxisLength(tensor: MxArray, context: string): number {
  const sequenceLength = tensor.shape[2];
  if (sequenceLength === undefined) {
    throw new Error(`${context}: cache tensor is missing a sequence axis.`);
  }
  return sequenceLength;
}

function roundCacheCapacity(requiredLength: number): number {
  return Math.max(
    CACHE_GROWTH_STEP,
    Math.ceil(requiredLength / CACHE_GROWTH_STEP) * CACHE_GROWTH_STEP,
  );
}

function cachePrefixView(tensor: MxArray, length: number): MxArray {
  const [batch, heads, capacity, width] = tensor.shape;
  if (batch === undefined || heads === undefined || capacity === undefined || width === undefined) {
    throw new Error("cachePrefixView: expected rank-4 cache tensor.");
  }
  if (length < 0 || length > capacity) {
    throw new Error(`cachePrefixView: length ${length} is out of range for capacity ${capacity}.`);
  }
  if (length === capacity) {
    return retainArray(tensor);
  }
  return slice(tensor, [0, 0, 0, 0], [batch, heads, length, width]);
}

function cacheTailView(tensor: MxArray, length: number): MxArray {
  const [batch, heads, capacity, width] = tensor.shape;
  if (batch === undefined || heads === undefined || capacity === undefined || width === undefined) {
    throw new Error("cacheTailView: expected rank-4 cache tensor.");
  }
  if (length < 0 || length > capacity) {
    throw new Error(`cacheTailView: length ${length} is out of range for capacity ${capacity}.`);
  }
  if (length === capacity) {
    return retainArray(tensor);
  }
  return slice(tensor, [0, 0, capacity - length, 0], [batch, heads, capacity, width]);
}

function orderedSlidingView(tensor: MxArray, length: number, cursor: number): MxArray {
  const capacity = sequenceAxisLength(tensor, "orderedSlidingView");
  if (length < capacity) {
    return cachePrefixView(tensor, length);
  }
  if (cursor === 0) {
    return retainArray(tensor);
  }

  const [batch, heads, , width] = tensor.shape;
  if (batch === undefined || heads === undefined || width === undefined) {
    throw new Error("orderedSlidingView: expected rank-4 cache tensor.");
  }
  using tail = slice(tensor, [0, 0, cursor, 0], [batch, heads, capacity, width]);
  using head = slice(tensor, [0, 0, 0, 0], [batch, heads, cursor, width]);
  return concatenate([tail, head], 2);
}

function allocateCacheBufferLike(update: MxArray, capacity: number): MxArray {
  const [batch, heads, , width] = update.shape;
  if (batch === undefined || heads === undefined || width === undefined) {
    throw new Error("allocateCacheBufferLike: expected rank-4 cache update tensor.");
  }
  return zeros([batch, heads, capacity, width], update.dtype);
}

function growCacheBuffer(
  existing: MxArray | null,
  usedLength: number,
  update: MxArray,
  capacity: number,
): MxArray {
  using base = allocateCacheBufferLike(update, capacity);
  if (existing === null || usedLength === 0) {
    return retainArray(base);
  }
  using prefix = cachePrefixView(existing, usedLength);
  return sliceUpdate(
    base,
    prefix,
    [0, 0, 0, 0],
    [prefix.shape[0] ?? 0, prefix.shape[1] ?? 0, prefix.shape[2] ?? 0, prefix.shape[3] ?? 0],
  );
}

function writeCacheRange(buffer: MxArray, update: MxArray, position: number): MxArray {
  return sliceUpdate(
    buffer,
    update,
    [0, 0, position, 0],
    [
      update.shape[0] ?? 0,
      update.shape[1] ?? 0,
      position + (update.shape[2] ?? 0),
      update.shape[3] ?? 0,
    ],
  );
}

function appendFullState(
  state: LayerState,
  keys: MxArray,
  values: MxArray,
): { keys: MxArray; values: MxArray } {
  const keyLength = sequenceAxisLength(keys, "appendFullState");
  const valueLength = sequenceAxisLength(values, "appendFullState");
  if (keyLength !== valueLength) {
    throw new Error(
      `appendFullState: key/value sequence lengths must match, got ${keyLength} and ${valueLength}.`,
    );
  }

  const requiredLength = state.length + keyLength;
  const currentCapacity =
    state.keys === null ? 0 : sequenceAxisLength(state.keys, "appendFullState");
  if (requiredLength > currentCapacity) {
    const nextCapacity = roundCacheCapacity(requiredLength);
    const nextKeys = growCacheBuffer(state.keys, state.length, keys, nextCapacity);
    const nextValues = growCacheBuffer(state.values, state.length, values, nextCapacity);
    state.keys?.free();
    state.values?.free();
    state.keys = nextKeys;
    state.values = nextValues;
  }

  const nextKeys = writeCacheRange(
    state.keys ?? growCacheBuffer(null, 0, keys, roundCacheCapacity(requiredLength)),
    keys,
    state.length,
  );
  const nextValues = writeCacheRange(
    state.values ?? growCacheBuffer(null, 0, values, roundCacheCapacity(requiredLength)),
    values,
    state.length,
  );
  if (state.keys !== nextKeys) {
    state.keys?.free();
    state.keys = nextKeys;
  }
  if (state.values !== nextValues) {
    state.values?.free();
    state.values = nextValues;
  }
  state.length = requiredLength;

  return {
    keys: cachePrefixView(state.keys, requiredLength),
    values: cachePrefixView(state.values, requiredLength),
  };
}

function appendSlidingSingleTokenAtCursor(
  state: LayerState,
  keys: MxArray,
  values: MxArray,
  windowSize: number,
): { keys: MxArray; values: MxArray } | null {
  if (
    sequenceAxisLength(keys, "appendSlidingSingleTokenAtCursor") !== 1 ||
    state.length !== windowSize ||
    state.keys === null ||
    state.values === null
  ) {
    return null;
  }

  const nextKeys = writeCacheRange(state.keys, keys, state.cursor);
  const nextValues = writeCacheRange(state.values, values, state.cursor);
  state.keys.free();
  state.values.free();
  state.keys = nextKeys;
  state.values = nextValues;
  state.cursor = (state.cursor + 1) % windowSize;

  return {
    keys: retainArray(state.keys),
    values: retainArray(state.values),
  };
}

function appendSlidingIntoExistingCapacity(
  state: LayerState,
  keys: MxArray,
  values: MxArray,
  windowSize: number,
  updateLength: number,
): { keys: MxArray; values: MxArray } | null {
  const currentCapacity =
    state.keys === null ? 0 : sequenceAxisLength(state.keys, "appendSlidingIntoExistingCapacity");
  if (
    state.length >= windowSize ||
    state.length + updateLength > windowSize ||
    currentCapacity < Math.min(windowSize, state.length + updateLength)
  ) {
    return null;
  }

  const existingKeys = state.keys;
  const existingValues = state.values;
  if (existingKeys === null || existingValues === null) {
    throw new Error("appendSlidingIntoExistingCapacity: expected allocated cache buffers.");
  }

  const nextKeys = writeCacheRange(existingKeys, keys, state.length);
  const nextValues = writeCacheRange(existingValues, values, state.length);
  existingKeys.free();
  existingValues.free();
  state.keys = nextKeys;
  state.values = nextValues;
  state.length = Math.min(windowSize, state.length + updateLength);
  state.cursor = state.length === windowSize ? 0 : state.cursor;
  const storedKeys = state.keys;
  const storedValues = state.values;

  return {
    keys: cachePrefixView(storedKeys, state.length),
    values: cachePrefixView(storedValues, state.length),
  };
}

function appendSlidingWithBufferGrowth(
  state: LayerState,
  keys: MxArray,
  values: MxArray,
  windowSize: number,
  updateLength: number,
): { keys: MxArray; values: MxArray } | null {
  if (state.length >= windowSize || state.length + updateLength > windowSize) {
    return null;
  }

  const nextCapacity = Math.min(windowSize, roundCacheCapacity(state.length + updateLength));
  const nextKeys = growCacheBuffer(state.keys, state.length, keys, nextCapacity);
  const nextValues = growCacheBuffer(state.values, state.length, values, nextCapacity);
  state.keys?.free();
  state.values?.free();
  state.keys = writeCacheRange(nextKeys, keys, state.length);
  state.values = writeCacheRange(nextValues, values, state.length);
  nextKeys.free();
  nextValues.free();
  state.length += updateLength;
  state.cursor = state.length === windowSize ? 0 : state.cursor;

  return {
    keys: cachePrefixView(state.keys, Math.min(state.length, windowSize)),
    values: cachePrefixView(state.values, Math.min(state.length, windowSize)),
  };
}

function mergeSlidingHistory(
  state: LayerState,
  keys: MxArray,
  values: MxArray,
): { keys: MxArray; values: MxArray } {
  const orderedKeysSource = state.keys;
  const orderedValuesSource = state.values;
  if (orderedKeysSource === null || orderedValuesSource === null || state.length === 0) {
    return {
      keys: retainArray(keys),
      values: retainArray(values),
    };
  }

  using orderedKeys = orderedSlidingView(orderedKeysSource, state.length, state.cursor);
  using orderedValues = orderedSlidingView(orderedValuesSource, state.length, state.cursor);
  return {
    keys: concatenate([orderedKeys, keys], 2),
    values: concatenate([orderedValues, values], 2),
  };
}

function appendSlidingState(
  state: LayerState,
  keys: MxArray,
  values: MxArray,
  windowSize: number,
): { keys: MxArray; values: MxArray } {
  const updateLength = sequenceAxisLength(keys, "appendSlidingState");
  if (updateLength !== sequenceAxisLength(values, "appendSlidingState")) {
    throw new Error("appendSlidingState: key/value sequence lengths must match.");
  }

  const singleTokenUpdate = appendSlidingSingleTokenAtCursor(state, keys, values, windowSize);
  if (singleTokenUpdate !== null) {
    return singleTokenUpdate;
  }

  const appendedInPlace = appendSlidingIntoExistingCapacity(
    state,
    keys,
    values,
    windowSize,
    updateLength,
  );
  if (appendedInPlace !== null) {
    return appendedInPlace;
  }

  const grownBuffer = appendSlidingWithBufferGrowth(state, keys, values, windowSize, updateLength);
  if (grownBuffer !== null) {
    return grownBuffer;
  }

  const mergedPair = mergeSlidingHistory(state, keys, values);
  using mergedKeys = mergedPair.keys;
  using mergedValues = mergedPair.values;
  const mergedLength = sequenceAxisLength(mergedKeys, "appendSlidingState");
  const returnedLength = Math.min(mergedLength, windowSize + updateLength - 1);
  const retainedLength = Math.min(mergedLength, windowSize);
  using returnedKeys = cacheTailView(mergedKeys, returnedLength);
  using returnedValues = cacheTailView(mergedValues, returnedLength);
  using retainedKeys = cacheTailView(mergedKeys, retainedLength);
  using retainedValues = cacheTailView(mergedValues, retainedLength);
  state.keys?.free();
  state.values?.free();
  state.keys = retainArray(retainedKeys);
  state.values = retainArray(retainedValues);
  state.length = retainedLength;
  state.cursor = 0;
  return {
    keys: retainArray(returnedKeys),
    values: retainArray(returnedValues),
  };
}

abstract class CacheBase implements TransformerCache {
  offset = 0;
  #layers: LayerState[];

  constructor(layerCount: number) {
    if (!Number.isInteger(layerCount) || layerCount <= 0) {
      throw new Error(`TransformerCache: layerCount must be a positive integer, got ${layerCount}`);
    }
    this.#layers = Array.from({ length: layerCount }, () => emptyLayerState());
  }

  get layerCount(): number {
    return this.#layers.length;
  }

  isEmpty(): boolean {
    return this.offset === 0;
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

  protected appendLayer(
    layerIndex: number,
    keys: MxArray,
    values: MxArray,
  ): { keys: MxArray; values: MxArray } {
    return appendFullState(this.layerState(layerIndex), keys, values);
  }

  stateArrays(): MxArray[] {
    const arrays: MxArray[] = [];
    for (const state of this.#layers) {
      if (state.keys !== null) {
        arrays.push(cachePrefixView(state.keys, state.length));
      }
      if (state.values !== null) {
        arrays.push(cachePrefixView(state.values, state.length));
      }
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
    return this.appendLayer(layerIndex, keys, values);
  }

  [Symbol.dispose](): void {
    for (const state of this.#layers) {
      state.keys?.free();
      state.values?.free();
      state.keys = null;
      state.values = null;
      state.length = 0;
      state.cursor = 0;
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
    return appendSlidingState(this.layerState(layerIndex), keys, values, this.#windowSize);
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
      return appendSlidingState(state, keys, values, windowSize);
    }
    return this.appendLayer(layerIndex, keys, values);
  }
}

/** Return retained cache-state arrays for explicit evaluation. The caller owns the views. */
export function cacheStateArrays(cache: TransformerCache): MxArray[] {
  return cache.arrays();
}
