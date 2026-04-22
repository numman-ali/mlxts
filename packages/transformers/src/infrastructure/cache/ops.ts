import {
  concatenate,
  type MxArray,
  retainArray,
  slice,
  sliceUpdateInPlace,
  zeros,
} from "@mlxts/core";

import { recordTransformerRuntimeCounter } from "../runtime-profile";

export function sequenceAxisLength(tensor: MxArray, context: string): number {
  const sequenceLength = tensor.shape[2];
  if (sequenceLength === undefined) {
    throw new Error(`${context}: cache tensor is missing a sequence axis.`);
  }
  return sequenceLength;
}

export function roundCacheCapacity(requiredLength: number): number {
  return Math.max(256, Math.ceil(requiredLength / 256) * 256);
}

export function cachePrefixView(tensor: MxArray, length: number): MxArray {
  const [batch, heads, capacity, width] = tensor.shape;
  if (batch === undefined || heads === undefined || capacity === undefined || width === undefined) {
    throw new Error("cachePrefixView: expected rank-4 cache tensor.");
  }
  if (length < 0 || length > capacity) {
    throw new Error(`cachePrefixView: length ${length} is out of range for capacity ${capacity}.`);
  }
  if (length === capacity) {
    recordTransformerRuntimeCounter("cache.return_full_buffer");
    return retainArray(tensor);
  }
  recordTransformerRuntimeCounter("cache.return_prefix_view");
  return slice(tensor, [0, 0, 0, 0], [batch, heads, length, width]);
}

export function cacheTailView(tensor: MxArray, length: number): MxArray {
  const [batch, heads, capacity, width] = tensor.shape;
  if (batch === undefined || heads === undefined || capacity === undefined || width === undefined) {
    throw new Error("cacheTailView: expected rank-4 cache tensor.");
  }
  if (length < 0 || length > capacity) {
    throw new Error(`cacheTailView: length ${length} is out of range for capacity ${capacity}.`);
  }
  if (length === capacity) {
    recordTransformerRuntimeCounter("cache.return_full_buffer");
    return retainArray(tensor);
  }
  recordTransformerRuntimeCounter("cache.return_tail_view");
  return slice(tensor, [0, 0, capacity - length, 0], [batch, heads, capacity, width]);
}

export function orderedSlidingView(tensor: MxArray, length: number, cursor: number): MxArray {
  const capacity = sequenceAxisLength(tensor, "orderedSlidingView");
  if (length < capacity) {
    return cachePrefixView(tensor, length);
  }
  if (cursor === 0) {
    recordTransformerRuntimeCounter("cache.return_full_buffer");
    return retainArray(tensor);
  }

  const [batch, heads, , width] = tensor.shape;
  if (batch === undefined || heads === undefined || width === undefined) {
    throw new Error("orderedSlidingView: expected rank-4 cache tensor.");
  }
  using tail = slice(tensor, [0, 0, cursor, 0], [batch, heads, capacity, width]);
  using head = slice(tensor, [0, 0, 0, 0], [batch, heads, cursor, width]);
  recordTransformerRuntimeCounter("cache.return_ordered_concat_view");
  return concatenate([tail, head], 2);
}

function allocateCacheBufferLike(update: MxArray, capacity: number): MxArray {
  const [batch, heads, , width] = update.shape;
  if (batch === undefined || heads === undefined || width === undefined) {
    throw new Error("allocateCacheBufferLike: expected rank-4 cache update tensor.");
  }
  return zeros([batch, heads, capacity, width], update.dtype);
}

export function growCacheBuffer(
  existing: MxArray | null,
  usedLength: number,
  update: MxArray,
  capacity: number,
): MxArray {
  if (existing === null || usedLength === 0) {
    return allocateCacheBufferLike(update, capacity);
  }

  const existingCapacity = sequenceAxisLength(existing, "growCacheBuffer");
  if (usedLength === existingCapacity && capacity > existingCapacity) {
    using extra = allocateCacheBufferLike(update, capacity - existingCapacity);
    return concatenate([existing, extra], 2);
  }

  const base = allocateCacheBufferLike(update, capacity);
  using prefix = cachePrefixView(existing, usedLength);
  sliceUpdateInPlace(
    base,
    prefix,
    [0, 0, 0, 0],
    [prefix.shape[0] ?? 0, prefix.shape[1] ?? 0, prefix.shape[2] ?? 0, prefix.shape[3] ?? 0],
  );
  return base;
}

export function writeCacheRangeInPlace(buffer: MxArray, update: MxArray, position: number): void {
  recordTransformerRuntimeCounter("cache.write_range");
  sliceUpdateInPlace(
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
