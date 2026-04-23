/**
 * Batch-aware KV cache objects for future continuous batching engines.
 * @module
 */

import {
  array,
  concatenate,
  type DType,
  type MxArray,
  slice,
  sliceUpdateInPlace,
  takeAxis,
  zeros,
} from "@mlxts/core";

import type { TransformerBatchCache, TransformerCache } from "../../types";
import {
  appendFullCacheState,
  type CacheAppendResult,
  createEmptyLayerState,
  createTransformerCacheViewFromResult,
  disposeLayerState,
  type LayerState,
  materializeOwnedAppendResult,
  retainedLayerStateArrays,
} from "./runtime";
import { KVCache } from "./single";
import { INTERNAL_CACHE_VIEW, type TransformerCacheView } from "./view";

type CacheTensorSpec = {
  batchSize: number;
  heads: number;
  keyWidth: number;
  valueWidth: number;
  keyDtype: DType;
  valueDtype: DType;
};

function validateBatchMetadata(leftPadding: readonly number[]): number[] {
  if (leftPadding.length === 0) {
    throw new Error("BatchKVCache: leftPadding must contain at least one request.");
  }
  return leftPadding.map((padding, index) => {
    if (!Number.isInteger(padding) || padding < 0) {
      throw new Error(
        `BatchKVCache: leftPadding[${index}] must be a non-negative integer, got ${padding}.`,
      );
    }
    return padding;
  });
}

function validateBatchIndices(indices: readonly number[], batchSize: number): number[] {
  if (indices.length === 0) {
    throw new Error("BatchKVCache.filter: batchIndices must contain at least one index.");
  }
  const seen = new Set<number>();
  return indices.map((index) => {
    if (!Number.isInteger(index) || index < 0 || index >= batchSize) {
      throw new Error(
        `BatchKVCache.filter: batch index ${index} is out of range for batch size ${batchSize}.`,
      );
    }
    if (seen.has(index)) {
      throw new Error(`BatchKVCache.filter: duplicate batch index ${index}.`);
    }
    seen.add(index);
    return index;
  });
}

function layerTensorSpec(state: LayerState): CacheTensorSpec | null {
  if (state.keys === null || state.values === null) {
    return null;
  }
  const [batchSize, heads, , keyWidth] = state.keys.shape;
  const valueWidth = state.values.shape[3];
  if (
    batchSize === undefined ||
    heads === undefined ||
    keyWidth === undefined ||
    valueWidth === undefined
  ) {
    throw new Error("BatchKVCache: expected rank-4 cache tensors.");
  }
  return {
    batchSize,
    heads,
    keyWidth,
    valueWidth,
    keyDtype: state.keys.dtype,
    valueDtype: state.values.dtype,
  };
}

function compatibleSpec(
  target: CacheTensorSpec | null,
  source: CacheTensorSpec | null,
): CacheTensorSpec | null {
  const spec = target ?? source;
  if (spec === null) {
    return null;
  }
  if (target === null || source === null) {
    return spec;
  }
  if (
    target.heads !== source.heads ||
    target.keyWidth !== source.keyWidth ||
    target.valueWidth !== source.valueWidth ||
    target.keyDtype !== source.keyDtype ||
    target.valueDtype !== source.valueDtype
  ) {
    throw new Error("BatchKVCache.extend: cache tensor shapes and dtypes must match.");
  }
  return spec;
}

function emptyBatchTensor(
  batchSize: number,
  heads: number,
  length: number,
  width: number,
  dtype: DType,
): MxArray {
  return zeros([batchSize, heads, length, width], dtype);
}

function sliceBatchAxis(tensor: MxArray, indices: readonly number[]): MxArray {
  using indexTensor = array([...indices], "int32");
  return takeAxis(tensor, indexTensor, 0);
}

function sliceSequenceAxis(tensor: MxArray, start: number, stop: number): MxArray {
  const [batchSize, heads, , width] = tensor.shape;
  if (batchSize === undefined || heads === undefined || width === undefined) {
    throw new Error("BatchKVCache: expected rank-4 cache tensor.");
  }
  return slice(tensor, [0, 0, start, 0], [batchSize, heads, stop, width]);
}

function padStateTensor(
  tensor: MxArray | null,
  batchSize: number,
  heads: number,
  sourceLength: number,
  targetLength: number,
  width: number,
  dtype: DType,
): MxArray {
  if (targetLength < sourceLength) {
    throw new Error("BatchKVCache.extend: target length cannot be shorter than source length.");
  }
  if (tensor !== null && sourceLength === targetLength) {
    return sliceSequenceAxis(tensor, 0, targetLength);
  }
  const padded = emptyBatchTensor(batchSize, heads, targetLength, width, dtype);
  if (tensor === null || sourceLength === 0) {
    return padded;
  }
  const leftPadding = targetLength - sourceLength;
  using visible = sliceSequenceAxis(tensor, 0, sourceLength);
  sliceUpdateInPlace(
    padded,
    visible,
    [0, 0, leftPadding, 0],
    [batchSize, heads, targetLength, width],
  );
  return padded;
}

function retainedStatePair(state: LayerState): { keys: MxArray; values: MxArray } | null {
  if (state.keys === null || state.values === null || state.length === 0) {
    return null;
  }
  return {
    keys: sliceSequenceAxis(state.keys, 0, state.length),
    values: sliceSequenceAxis(state.values, 0, state.length),
  };
}

function filterLayerState(state: LayerState, indices: readonly number[], trimLeft: number): void {
  const retained = retainedStatePair(state);
  if (retained === null) {
    return;
  }

  using filteredKeys = sliceBatchAxis(retained.keys, indices);
  using filteredValues = sliceBatchAxis(retained.values, indices);
  const nextLength = state.length - trimLeft;
  const nextKeys =
    trimLeft === 0
      ? sliceSequenceAxis(filteredKeys, 0, state.length)
      : sliceSequenceAxis(filteredKeys, trimLeft, state.length);
  const nextValues =
    trimLeft === 0
      ? sliceSequenceAxis(filteredValues, 0, state.length)
      : sliceSequenceAxis(filteredValues, trimLeft, state.length);

  disposeLayerState(state);
  state.keys = nextKeys;
  state.values = nextValues;
  state.length = nextLength;
  state.cursor = 0;
  retained.keys.free();
  retained.values.free();
}

function extendLayerState(
  target: LayerState,
  source: LayerState,
  targetBatchSize: number,
  sourceBatchSize: number,
): void {
  const spec = compatibleSpec(layerTensorSpec(target), layerTensorSpec(source));
  if (spec === null) {
    return;
  }
  const targetLength = Math.max(target.length, source.length);
  using leftKeys = padStateTensor(
    target.keys,
    targetBatchSize,
    spec.heads,
    target.length,
    targetLength,
    spec.keyWidth,
    spec.keyDtype,
  );
  using leftValues = padStateTensor(
    target.values,
    targetBatchSize,
    spec.heads,
    target.length,
    targetLength,
    spec.valueWidth,
    spec.valueDtype,
  );
  using rightKeys = padStateTensor(
    source.keys,
    sourceBatchSize,
    spec.heads,
    source.length,
    targetLength,
    spec.keyWidth,
    spec.keyDtype,
  );
  using rightValues = padStateTensor(
    source.values,
    sourceBatchSize,
    spec.heads,
    source.length,
    targetLength,
    spec.valueWidth,
    spec.valueDtype,
  );
  const nextKeys = concatenate([leftKeys, rightKeys], 0);
  const nextValues = concatenate([leftValues, rightValues], 0);
  disposeLayerState(target);
  target.keys = nextKeys;
  target.values = nextValues;
  target.length = targetLength;
  target.cursor = 0;
}

function validateUpdateBatchSize(keys: MxArray, values: MxArray, batchSize: number): void {
  const keyBatch = keys.shape[0];
  const valueBatch = values.shape[0];
  if (keyBatch !== batchSize || valueBatch !== batchSize) {
    throw new Error(
      `BatchKVCache.updateAndFetch: update batch size must be ${batchSize}, got ${keyBatch} and ${valueBatch}.`,
    );
  }
}

/** Full KV cache for a left-padded batch of active requests. */
export class BatchKVCache implements TransformerBatchCache {
  #layers: LayerState[];
  #leftPadding: number[];
  #offsets: number[];

  constructor(layerCount: number, leftPadding: readonly number[]) {
    if (!Number.isInteger(layerCount) || layerCount <= 0) {
      throw new Error(`BatchKVCache: layerCount must be a positive integer, got ${layerCount}.`);
    }
    this.#leftPadding = validateBatchMetadata(leftPadding);
    this.#offsets = this.#leftPadding.map((padding) => (padding === 0 ? 0 : -padding));
    this.#layers = Array.from({ length: layerCount }, () => createEmptyLayerState());
  }

  get layerCount(): number {
    return this.#layers.length;
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
    this.#leftPadding = nextLeftPadding.map((padding) => padding - trimLeft);
    this.#offsets = indices.map((index) => this.#offsets[index] ?? 0);
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
  }

  extract(batchIndex: number): TransformerCache {
    if (!Number.isInteger(batchIndex) || batchIndex < 0 || batchIndex >= this.batchSize) {
      throw new Error(
        `BatchKVCache.extract: batch index ${batchIndex} is out of range for batch size ${this.batchSize}.`,
      );
    }
    const cache = new KVCache(this.layerCount);
    let visibleLength = 0;
    for (let layerIndex = 0; layerIndex < this.#layers.length; layerIndex += 1) {
      const layer = this.#layers[layerIndex];
      if (layer === undefined || layer.keys === null || layer.values === null) {
        continue;
      }
      const padding = this.#leftPadding[batchIndex] ?? 0;
      const layerVisibleLength = Math.max(0, layer.length - padding);
      if (layerVisibleLength === 0) {
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
      visibleLength = Math.max(visibleLength, layerVisibleLength);
    }
    cache.advance(visibleLength);
    return cache;
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
