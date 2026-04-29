/**
 * Internal KV cache state-machine helpers.
 * @module
 */

import { add, concatenate, type MxArray, mxEval, retainArray, slice } from "@mlxts/core";
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

const LAYER_STATE_SNAPSHOT_BRAND: unique symbol = Symbol("LayerStateSnapshot");

export type LayerState = {
  keys: MxArray | null;
  values: MxArray | null;
  length: number;
  cursor: number;
};

export type CacheAppendResult = {
  keys: MxArray;
  values: MxArray;
  ownsBuffers: boolean;
};

export type LayerStateSnapshot = {
  keys: MxArray | null;
  values: MxArray | null;
  length: number;
  cursor: number;
  readonly [LAYER_STATE_SNAPSHOT_BRAND]: true;
};

export function createEmptyLayerState(): LayerState {
  return {
    keys: null,
    values: null,
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

function visibleState(state: LayerState, length: number, context: string): CacheAppendResult {
  const { keys, values } = currentBufferPair(state, context);
  const capacity = sequenceAxisLength(keys, context);
  if (length === capacity) {
    recordTransformerRuntimeCounter("cache.return_full_buffer", 2);
    return createBorrowedAppendResult(keys, values);
  }

  recordTransformerRuntimeCounter("cache.return_prefix_view", 2);
  const batch = keys.shape[0] ?? 0;
  const heads = keys.shape[1] ?? 0;
  const width = keys.shape[3] ?? 0;
  const start = [0, 0, 0, 0];
  const stop = [batch, heads, length, width];
  return createOwnedAppendResult(slice(keys, start, stop), slice(values, start, stop));
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

  const visible = visibleState(state, state.length, "retainedLayerStateArrays");
  const owned = materializeOwnedAppendResult(visible);
  return [owned.keys, owned.values];
}

export function cloneCacheArray(value: MxArray): MxArray {
  const clone = add(value, 0);
  try {
    mxEval(clone);
    return clone;
  } catch (error) {
    clone.free();
    throw error;
  }
}

function visibleStateForSnapshot(state: LayerState): { keys: MxArray; values: MxArray } | null {
  if (state.keys === null || state.values === null || state.length === 0) {
    return null;
  }
  const visible = visibleState(state, state.length, "cloneLayerStateSnapshot");
  return materializeOwnedAppendResult(visible);
}

export function cloneLayerStateSnapshot(state: LayerState): LayerStateSnapshot {
  const visible = visibleStateForSnapshot(state);
  if (visible === null) {
    return {
      keys: null,
      values: null,
      length: 0,
      cursor: 0,
      [LAYER_STATE_SNAPSHOT_BRAND]: true,
    };
  }
  let keys: MxArray | null = null;
  let values: MxArray | null = null;
  try {
    keys = cloneCacheArray(visible.keys);
    values = cloneCacheArray(visible.values);
    return {
      keys,
      values,
      length: state.length,
      cursor: state.cursor,
      [LAYER_STATE_SNAPSHOT_BRAND]: true,
    };
  } catch (error) {
    keys?.free();
    values?.free();
    throw error;
  } finally {
    visible.keys.free();
    visible.values.free();
  }
}

export function disposeLayerStateSnapshot(snapshot: LayerStateSnapshot): void {
  snapshot.keys?.free();
  snapshot.values?.free();
  snapshot.keys = null;
  snapshot.values = null;
  snapshot.length = 0;
  snapshot.cursor = 0;
}

export function restoreLayerStateSnapshot(state: LayerState, snapshot: LayerStateSnapshot): void {
  disposeLayerState(state);
  state.keys = snapshot.keys;
  state.values = snapshot.values;
  state.length = snapshot.length;
  state.cursor = snapshot.cursor;
  snapshot.keys = null;
  snapshot.values = null;
  snapshot.length = 0;
  snapshot.cursor = 0;
}

export function disposeLayerState(state: LayerState): void {
  state.keys?.free();
  state.values?.free();
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

  return visibleState(state, requiredLength, "appendFullCacheState");
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
  return visibleState(state, windowSize, "appendSlidingSingleTokenAtCursor");
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
  return visibleState(state, state.length, "appendSlidingIntoExistingCapacity");
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
  return visibleState(state, Math.min(state.length, windowSize), "appendSlidingWithBufferGrowth");
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
  return createOwnedAppendResult(retainArray(returnedKeys), retainArray(returnedValues));
}
