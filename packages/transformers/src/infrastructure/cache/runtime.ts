/**
 * Internal KV cache state-machine helpers.
 * @module
 */

import { concatenate, type MxArray, retainArray, slice, sliceViewInPlace } from "@mlxts/core";
import { recordTransformerRuntimeCounter } from "../runtime-profile";
import {
  cacheTailView,
  growCacheBuffer,
  orderedSlidingView,
  roundCacheCapacity,
  sequenceAxisLength,
  writeCacheRangeInPlace,
} from "./ops";
import {
  createBorrowedTransformerCacheView,
  createOwnedTransformerCacheView,
  type TransformerCacheView,
} from "./view";

export type LayerState = {
  keys: MxArray | null;
  values: MxArray | null;
  visibleKeys: MxArray | null;
  visibleValues: MxArray | null;
  length: number;
  cursor: number;
};

export type CacheAppendResult = {
  keys: MxArray;
  values: MxArray;
  ownsBuffers: boolean;
};

export function createEmptyLayerState(): LayerState {
  return {
    keys: null,
    values: null,
    visibleKeys: null,
    visibleValues: null,
    length: 0,
    cursor: 0,
  };
}

function createOwnedAppendResult(keys: MxArray, values: MxArray): CacheAppendResult {
  return { keys, values, ownsBuffers: true };
}

function createBorrowedAppendResult(keys: MxArray, values: MxArray): CacheAppendResult {
  return { keys, values, ownsBuffers: false };
}

function currentBufferPair(state: LayerState, context: string): { keys: MxArray; values: MxArray } {
  const { keys, values } = state;
  if (keys === null || values === null) {
    throw new Error(`${context}: expected allocated cache buffers.`);
  }
  return { keys, values };
}

function updateVisibleStateHandles(
  state: LayerState,
  length: number,
  context: string,
): { keys: MxArray; values: MxArray } {
  const { keys, values } = currentBufferPair(state, context);
  const capacity = sequenceAxisLength(keys, context);
  if (length === capacity) {
    recordTransformerRuntimeCounter("cache.return_full_buffer", 2);
    if (state.visibleKeys !== null) {
      state.visibleKeys.free();
      state.visibleKeys = null;
    }
    if (state.visibleValues !== null) {
      state.visibleValues.free();
      state.visibleValues = null;
    }
    return { keys, values };
  }

  recordTransformerRuntimeCounter("cache.return_prefix_view", 2);
  const batch = keys.shape[0] ?? 0;
  const heads = keys.shape[1] ?? 0;
  const width = keys.shape[3] ?? 0;
  const start = [0, 0, 0, 0];
  const stop = [batch, heads, length, width];
  if (state.visibleKeys === null || state.visibleValues === null) {
    state.visibleKeys = slice(keys, start, stop);
    state.visibleValues = slice(values, start, stop);
  } else {
    sliceViewInPlace(state.visibleKeys, keys, start, stop);
    sliceViewInPlace(state.visibleValues, values, start, stop);
  }
  return { keys: state.visibleKeys, values: state.visibleValues };
}

function borrowedVisibleState(
  state: LayerState,
  length: number,
  context: string,
): CacheAppendResult {
  const visible = updateVisibleStateHandles(state, length, context);
  return createBorrowedAppendResult(visible.keys, visible.values);
}

export function materializeOwnedAppendResult(result: CacheAppendResult): {
  keys: MxArray;
  values: MxArray;
} {
  if (result.ownsBuffers) {
    return { keys: result.keys, values: result.values };
  }

  return {
    keys: retainArray(result.keys),
    values: retainArray(result.values),
  };
}

export function createTransformerCacheViewFromResult(
  result: CacheAppendResult,
): TransformerCacheView {
  return result.ownsBuffers
    ? createOwnedTransformerCacheView(result.keys, result.values)
    : createBorrowedTransformerCacheView(result.keys, result.values);
}

export function retainedLayerStateArrays(state: LayerState): MxArray[] {
  if (state.keys === null || state.values === null || state.length === 0) {
    return [];
  }

  const visible = updateVisibleStateHandles(state, state.length, "retainedLayerStateArrays");
  return [retainArray(visible.keys), retainArray(visible.values)];
}

export function disposeLayerState(state: LayerState): void {
  state.visibleKeys?.free();
  state.visibleValues?.free();
  state.keys?.free();
  state.values?.free();
  state.visibleKeys = null;
  state.visibleValues = null;
  state.keys = null;
  state.values = null;
  state.length = 0;
  state.cursor = 0;
}

export function appendFullCacheState(
  state: LayerState,
  keys: MxArray,
  values: MxArray,
): CacheAppendResult {
  const keyLength = sequenceAxisLength(keys, "appendFullCacheState");
  const valueLength = sequenceAxisLength(values, "appendFullCacheState");
  if (keyLength !== valueLength) {
    throw new Error(
      `appendFullCacheState: key/value sequence lengths must match, got ${keyLength} and ${valueLength}.`,
    );
  }

  const requiredLength = state.length + keyLength;
  recordTransformerRuntimeCounter("cache.append_full_state");
  const currentCapacity =
    state.keys === null ? 0 : sequenceAxisLength(state.keys, "appendFullCacheState");
  if (requiredLength > currentCapacity) {
    const nextCapacity = roundCacheCapacity(requiredLength);
    const nextKeys = growCacheBuffer(state.keys, state.length, keys, nextCapacity);
    const nextValues = growCacheBuffer(state.values, state.length, values, nextCapacity);
    state.keys?.free();
    state.values?.free();
    state.keys = nextKeys;
    state.values = nextValues;
  }

  if (state.keys === null) {
    state.keys = growCacheBuffer(null, 0, keys, roundCacheCapacity(requiredLength));
  }
  if (state.values === null) {
    state.values = growCacheBuffer(null, 0, values, roundCacheCapacity(requiredLength));
  }
  writeCacheRangeInPlace(state.keys, keys, state.length);
  writeCacheRangeInPlace(state.values, values, state.length);
  state.length = requiredLength;

  return borrowedVisibleState(state, requiredLength, "appendFullCacheState");
}

function appendSlidingSingleTokenAtCursor(
  state: LayerState,
  keys: MxArray,
  values: MxArray,
  windowSize: number,
): CacheAppendResult | null {
  if (
    sequenceAxisLength(keys, "appendSlidingSingleTokenAtCursor") !== 1 ||
    state.length !== windowSize ||
    state.keys === null ||
    state.values === null
  ) {
    return null;
  }

  recordTransformerRuntimeCounter("cache.sliding_single_token");
  writeCacheRangeInPlace(state.keys, keys, state.cursor);
  writeCacheRangeInPlace(state.values, values, state.cursor);
  state.cursor = (state.cursor + 1) % windowSize;
  return borrowedVisibleState(state, windowSize, "appendSlidingSingleTokenAtCursor");
}

function appendSlidingIntoExistingCapacity(
  state: LayerState,
  keys: MxArray,
  values: MxArray,
  windowSize: number,
  updateLength: number,
): CacheAppendResult | null {
  const currentCapacity =
    state.keys === null ? 0 : sequenceAxisLength(state.keys, "appendSlidingIntoExistingCapacity");
  if (
    state.length >= windowSize ||
    state.length + updateLength > windowSize ||
    currentCapacity < Math.min(windowSize, state.length + updateLength)
  ) {
    return null;
  }

  recordTransformerRuntimeCounter("cache.sliding_existing_capacity");
  const existingKeys = state.keys;
  const existingValues = state.values;
  if (existingKeys === null || existingValues === null) {
    throw new Error("appendSlidingIntoExistingCapacity: expected allocated cache buffers.");
  }

  writeCacheRangeInPlace(existingKeys, keys, state.length);
  writeCacheRangeInPlace(existingValues, values, state.length);
  state.length = Math.min(windowSize, state.length + updateLength);
  state.cursor = state.length === windowSize ? 0 : state.cursor;
  return borrowedVisibleState(state, state.length, "appendSlidingIntoExistingCapacity");
}

function appendSlidingWithBufferGrowth(
  state: LayerState,
  keys: MxArray,
  values: MxArray,
  windowSize: number,
  updateLength: number,
): CacheAppendResult | null {
  if (state.length >= windowSize || state.length + updateLength > windowSize) {
    return null;
  }

  recordTransformerRuntimeCounter("cache.sliding_growth");
  const nextCapacity = Math.min(windowSize, roundCacheCapacity(state.length + updateLength));
  const nextKeys = growCacheBuffer(state.keys, state.length, keys, nextCapacity);
  const nextValues = growCacheBuffer(state.values, state.length, values, nextCapacity);
  state.keys?.free();
  state.values?.free();
  writeCacheRangeInPlace(nextKeys, keys, state.length);
  writeCacheRangeInPlace(nextValues, values, state.length);
  state.keys = nextKeys;
  state.values = nextValues;
  recordTransformerRuntimeCounter("cache.buffer_replaced", 2);
  state.length += updateLength;
  state.cursor = state.length === windowSize ? 0 : state.cursor;
  return borrowedVisibleState(
    state,
    Math.min(state.length, windowSize),
    "appendSlidingWithBufferGrowth",
  );
}

function mergeSlidingHistory(
  state: LayerState,
  keys: MxArray,
  values: MxArray,
): { keys: MxArray; values: MxArray } {
  const orderedKeysSource = state.keys;
  const orderedValuesSource = state.values;
  if (orderedKeysSource === null || orderedValuesSource === null || state.length === 0) {
    recordTransformerRuntimeCounter("cache.return_full_buffer", 2);
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

export function appendSlidingCacheState(
  state: LayerState,
  keys: MxArray,
  values: MxArray,
  windowSize: number,
): CacheAppendResult {
  const updateLength = sequenceAxisLength(keys, "appendSlidingCacheState");
  if (updateLength !== sequenceAxisLength(values, "appendSlidingCacheState")) {
    throw new Error("appendSlidingCacheState: key/value sequence lengths must match.");
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
  recordTransformerRuntimeCounter("cache.sliding_merge");
  using mergedKeys = mergedPair.keys;
  using mergedValues = mergedPair.values;
  const mergedLength = sequenceAxisLength(mergedKeys, "appendSlidingCacheState");
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
  updateVisibleStateHandles(state, retainedLength, "appendSlidingCacheState");
  return createOwnedAppendResult(retainArray(returnedKeys), retainArray(returnedValues));
}
