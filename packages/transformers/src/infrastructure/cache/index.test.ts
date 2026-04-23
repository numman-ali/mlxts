import { describe, expect, test } from "bun:test";

import { array, mxEval } from "@mlxts/core";
import {
  BatchKVCache,
  cacheStateArrays,
  KVCache,
  LayerPatternKVCache,
  SlidingWindowKVCache,
} from "./index";
import { updateAndFetchTransformerCacheView } from "./view";

function runFullAppendScenario(cache: KVCache): void {
  using firstKeys = array([[[[1, 2]]]], "float32");
  using firstValues = array([[[[3, 4]]]], "float32");
  using secondKeys = array([[[[5, 6]]]], "float32");
  using secondValues = array([[[[7, 8]]]], "float32");
  using firstView = cache.updateAndFetch(0, firstKeys, firstValues).keys;
  using secondView = cache.updateAndFetch(0, secondKeys, secondValues).keys;

  cache.advance(1);
  cache.advance(1);
  mxEval(firstView, secondView);

  expect(cache.isEmpty()).toBe(false);
  expect(cache.offset).toBe(2);
  expect(firstView.toList()).toEqual([[[[1, 2]]]]);
  expect(secondView.toList()).toEqual([
    [
      [
        [1, 2],
        [5, 6],
      ],
    ],
  ]);
}

function runSlidingSingleTokenScenario(cache: SlidingWindowKVCache): void {
  using firstKeys = array([[[[1], [2]]]], "float32");
  using firstValues = array([[[[10], [20]]]], "float32");
  using secondKeys = array([[[[3]]]], "float32");
  using secondValues = array([[[[30]]]], "float32");
  using firstView = cache.updateAndFetch(0, firstKeys, firstValues).keys;
  using secondView = cache.updateAndFetch(0, secondKeys, secondValues).keys;

  cache.advance(2);
  cache.advance(1);
  mxEval(firstView, secondView);

  expect(cache.offset).toBe(3);
  expect(firstView.toList()).toEqual([[[[1], [2]]]]);
  expect(secondView.toList()).toEqual([[[[3], [2]]]]);
}

function runLayerPatternScenario(cache: LayerPatternKVCache): void {
  using firstKeys = array([[[[1], [2]]]], "float32");
  using firstValues = array([[[[10], [20]]]], "float32");
  using secondKeys = array([[[[3]]]], "float32");
  using secondValues = array([[[[30]]]], "float32");

  using fullLayerView = cache.updateAndFetch(0, firstKeys, firstValues).keys;
  using slidingLayerFirstView = cache.updateAndFetch(1, firstKeys, firstValues).keys;
  using slidingLayerSecondView = cache.updateAndFetch(1, secondKeys, secondValues).keys;

  mxEval(fullLayerView, slidingLayerFirstView, slidingLayerSecondView);

  expect(fullLayerView.toList()).toEqual([[[[1], [2]]]]);
  expect(slidingLayerFirstView.toList()).toEqual([[[[1], [2]]]]);
  expect(slidingLayerSecondView.toList()).toEqual([[[[3], [2]]]]);
}

function runSlidingPrefillScenario(cache: SlidingWindowKVCache): void {
  using firstKeys = array([[[[1], [2], [3], [4]]]], "float32");
  using firstValues = array([[[[10], [20], [30], [40]]]], "float32");
  using secondKeys = array([[[[5], [6], [7]]]], "float32");
  using secondValues = array([[[[50], [60], [70]]]], "float32");

  using firstView = cache.updateAndFetch(0, firstKeys, firstValues).keys;
  using secondView = cache.updateAndFetch(0, secondKeys, secondValues).keys;
  const retainedState = cache.arrays()[0];
  if (retainedState === undefined) {
    throw new Error("expected retained sliding cache state");
  }
  using ownedRetainedState = retainedState;

  mxEval(firstView, secondView, ownedRetainedState);

  expect(firstView.toList()).toEqual([[[[1], [2], [3], [4]]]]);
  expect(secondView.toList()).toEqual([[[[2], [3], [4], [5], [6], [7]]]]);
  expect(ownedRetainedState.toList()).toEqual([[[[4], [5], [6], [7]]]]);
}

function runChunkGrowthScenario(cache: LayerPatternKVCache): void {
  const firstSequence = Array.from({ length: 200 }, (_, index) => [index + 1]);
  const secondSequence = Array.from({ length: 120 }, (_, index) => [index + 201]);
  using firstKeys = array([[[...firstSequence]]], "float32");
  using firstValues = array([[[...firstSequence]]], "float32");
  using secondKeys = array([[[...secondSequence]]], "float32");
  using secondValues = array([[[...secondSequence]]], "float32");

  using firstView = cache.updateAndFetch(0, firstKeys, firstValues).keys;
  using secondView = cache.updateAndFetch(0, secondKeys, secondValues).keys;

  mxEval(firstView, secondView);

  expect(firstView.shape).toEqual([1, 1, 200, 1]);
  expect(secondView.shape).toEqual([1, 1, 320, 1]);
  expect(secondView.toList()).toEqual([[[...firstSequence, ...secondSequence]]]);
}

describe("Transformer caches", () => {
  test("managed caches expose counts, offset helpers, and constructor validation", () => {
    using cache = new KVCache(2);

    expect(cache.layerCount).toBe(2);
    expect(cache.isEmpty()).toBe(true);
    expect(cache.isTrimmable()).toBe(true);
    expect(cache.arrays()).toEqual([]);
    expect(cacheStateArrays(cache)).toEqual([]);

    cache.advance(0);
    cache.advance(2);

    expect(cache.offset).toBe(2);
    expect(cache.isEmpty()).toBe(false);
    expect(() => cache.advance(-1)).toThrow("sequenceLength must be a non-negative integer");
    expect(() => new KVCache(0)).toThrow("layerCount must be a positive integer");
    expect(() => new SlidingWindowKVCache(1, 0)).toThrow("windowSize must be a positive integer");
    expect(() => new LayerPatternKVCache(1, [])).toThrow("must match layerCount");
    expect(() => new LayerPatternKVCache(1, [0])).toThrow(
      "each window size must be a positive integer",
    );
  });

  test("managed KVCache appends keys and values across updates", () => {
    using cache = new KVCache(1);
    runFullAppendScenario(cache);
  });

  test("managed SlidingWindowKVCache reuses ring-buffer storage for single-token updates once full", () => {
    using cache = new SlidingWindowKVCache(1, 2);
    runSlidingSingleTokenScenario(cache);
  });

  test("managed LayerPatternKVCache reuses ring-buffer storage only for configured sliding layers", () => {
    using cache = new LayerPatternKVCache(2, [undefined, 2]);
    runLayerPatternScenario(cache);
  });

  test("managed SlidingWindowKVCache keeps extra within-chunk context during multi-token prefill", () => {
    using cache = new SlidingWindowKVCache(1, 4);
    runSlidingPrefillScenario(cache);
  });

  test("managed LayerPatternKVCache full layers grow in chunks without losing earlier tokens", () => {
    using cache = new LayerPatternKVCache(1, [undefined]);
    runChunkGrowthScenario(cache);
  });

  test("managed LayerPatternKVCache full layers support both owned fetches and borrowed cache views", () => {
    using cache = new LayerPatternKVCache(1, [undefined]);
    using firstKeys = array([[[[1], [2]]]], "float32");
    using firstValues = array([[[[10], [20]]]], "float32");
    using secondKeys = array([[[[3]]]], "float32");
    using secondValues = array([[[[30]]]], "float32");
    using firstOwnedKeys = cache.updateAndFetch(0, firstKeys, firstValues).keys;
    using secondView = updateAndFetchTransformerCacheView(cache, 0, secondKeys, secondValues);

    mxEval(firstOwnedKeys, secondView.keys);

    expect(firstOwnedKeys.toList()).toEqual([[[[1], [2]]]]);
    expect(secondView.keys.toList()).toEqual([[[[1], [2], [3]]]]);
  });

  test("managed BatchKVCache tracks per-request offsets and appends batched updates", () => {
    using cache = new BatchKVCache(1, [1, 0]);
    using firstKeys = array([[[[0], [1]]], [[[2], [3]]]], "float32");
    using firstValues = array([[[[10], [11]]], [[[12], [13]]]], "float32");
    using secondKeys = array([[[[4]]], [[[5]]]], "float32");
    using secondValues = array([[[[14]]], [[[15]]]], "float32");

    using firstView = cache.updateAndFetch(0, firstKeys, firstValues).keys;
    cache.advance(2);
    using secondView = cache.updateAndFetch(0, secondKeys, secondValues).keys;
    cache.advance(1);
    mxEval(firstView, secondView);

    expect(cache.batchSize).toBe(2);
    expect(cache.length).toBe(3);
    expect(cache.leftPadding).toEqual([1, 0]);
    expect(cache.offsets).toEqual([2, 3]);
    expect(firstView.toList()).toEqual([[[[0], [1]]], [[[2], [3]]]]);
    expect(secondView.toList()).toEqual([[[[0], [1], [4]]], [[[2], [3], [5]]]]);
  });

  test("managed BatchKVCache filters active requests and removes shared left padding", () => {
    using cache = new BatchKVCache(1, [2, 1, 0]);
    using keys = array([[[[0], [0], [1]]], [[[0], [2], [3]]], [[[4], [5], [6]]]], "float32");
    using values = array(
      [[[[10], [10], [11]]], [[[10], [12], [13]]], [[[14], [15], [16]]]],
      "float32",
    );
    using view = cache.updateAndFetch(0, keys, values).keys;
    cache.advance(3);
    mxEval(view);

    cache.filter([0, 1]);
    const stateArrays = cache.arrays();
    const retainedKeys = stateArrays[0];
    if (retainedKeys === undefined) {
      throw new Error("expected retained batch cache keys");
    }
    using ownedRetainedKeys = retainedKeys;
    mxEval(ownedRetainedKeys);

    expect(cache.batchSize).toBe(2);
    expect(cache.length).toBe(2);
    expect(cache.leftPadding).toEqual([1, 0]);
    expect(cache.offsets).toEqual([1, 2]);
    expect(ownedRetainedKeys.toList()).toEqual([[[[0], [1]]], [[[2], [3]]]]);
  });

  test("managed BatchKVCache extracts single-request caches", () => {
    using cache = new BatchKVCache(1, [1, 0]);
    using keys = array([[[[0], [1], [2]]], [[[3], [4], [5]]]], "float32");
    using values = array([[[[10], [11], [12]]], [[[13], [14], [15]]]], "float32");
    using view = cache.updateAndFetch(0, keys, values).keys;
    cache.advance(3);
    mxEval(view);

    using extracted = cache.extract(0);
    const stateArrays = extracted.arrays();
    const extractedKeys = stateArrays[0];
    if (extractedKeys === undefined) {
      throw new Error("expected extracted cache keys");
    }
    using ownedExtractedKeys = extractedKeys;
    mxEval(ownedExtractedKeys);

    expect(extracted.offset).toBe(2);
    expect(ownedExtractedKeys.toList()).toEqual([[[[1], [2]]]]);
  });

  test("managed BatchKVCache extends batches by left-aligning shorter histories", () => {
    using first = new BatchKVCache(1, [0]);
    using firstKeys = array([[[[1], [2], [3]]]], "float32");
    using firstValues = array([[[[11], [12], [13]]]], "float32");
    using firstView = first.updateAndFetch(0, firstKeys, firstValues).keys;
    first.advance(3);

    using second = new BatchKVCache(1, [0]);
    using secondKeys = array([[[[4]]]], "float32");
    using secondValues = array([[[[14]]]], "float32");
    using secondView = second.updateAndFetch(0, secondKeys, secondValues).keys;
    second.advance(1);
    mxEval(firstView, secondView);

    first.extend(second);
    const stateArrays = first.arrays();
    const extendedKeys = stateArrays[0];
    if (extendedKeys === undefined) {
      throw new Error("expected extended cache keys");
    }
    using ownedExtendedKeys = extendedKeys;
    mxEval(ownedExtendedKeys);

    expect(first.batchSize).toBe(2);
    expect(first.length).toBe(3);
    expect(first.leftPadding).toEqual([0, 2]);
    expect(first.offsets).toEqual([3, 1]);
    expect(ownedExtendedKeys.toList()).toEqual([[[[1], [2], [3]]], [[[0], [0], [4]]]]);
  });

  test("managed BatchKVCache exposes disposable metadata tensors for batched model calls", () => {
    using cache = new BatchKVCache(1, [2, 0]);
    cache.advance(3);

    using offsets = cache.offsetTensor();
    using leftPadding = cache.leftPaddingTensor();

    expect(offsets.shape).toEqual([2]);
    expect(offsets.toList()).toEqual([1, 3]);
    expect(leftPadding.shape).toEqual([2]);
    expect(leftPadding.toList()).toEqual([2, 0]);
  });

  test("managed BatchKVCache extends empty and populated cache states", () => {
    using empty = new BatchKVCache(1, [0]);
    using populated = new BatchKVCache(1, [0]);
    using keys = array([[[[7], [8]]]], "float32");
    using values = array([[[[17], [18]]]], "float32");
    using view = populated.updateAndFetch(0, keys, values).keys;
    populated.advance(2);
    mxEval(view);

    empty.extend(populated);
    const stateArrays = empty.arrays();
    const extendedKeys = stateArrays[0];
    if (extendedKeys === undefined) {
      throw new Error("expected extended cache keys");
    }
    using ownedExtendedKeys = extendedKeys;
    mxEval(ownedExtendedKeys);

    expect(empty.batchSize).toBe(2);
    expect(empty.length).toBe(2);
    expect(empty.leftPadding).toEqual([2, 0]);
    expect(empty.offsets).toEqual([0, 2]);
    expect(ownedExtendedKeys.toList()).toEqual([[[[0], [0]]], [[[7], [8]]]]);
  });
});
