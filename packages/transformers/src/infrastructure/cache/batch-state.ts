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

import { disposeLayerState, type LayerState } from "./runtime";

type CacheTensorSpec = {
  batchSize: number;
  heads: number;
  keyWidth: number;
  valueWidth: number;
  keyDtype: DType;
  valueDtype: DType;
};

export function validateBatchMetadata(leftPadding: readonly number[]): number[] {
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

export function validateBatchIndices(indices: readonly number[], batchSize: number): number[] {
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

export function filterLayerState(
  state: LayerState,
  indices: readonly number[],
  trimLeft: number,
): void {
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

export function extendLayerState(
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

export function validateUpdateBatchSize(keys: MxArray, values: MxArray, batchSize: number): void {
  const keyBatch = keys.shape[0];
  const valueBatch = values.shape[0];
  if (keyBatch !== batchSize || valueBatch !== batchSize) {
    throw new Error(
      `BatchKVCache.updateAndFetch: update batch size must be ${batchSize}, got ${keyBatch} and ${valueBatch}.`,
    );
  }
}
