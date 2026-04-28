import { describe, expect, test } from "bun:test";

import { array, type MxArray, mxEval } from "@mlxts/core";
import {
  BatchKVCache,
  cacheLayerKindFromAttentionType,
  cacheStateArrays,
  KVCache,
  LayerPatternBatchKVCache,
  LayerPatternKVCache,
  SlidingWindowKVCache,
} from "./index";
import { updateAndFetchTransformerCacheView } from "./view";

function cacheArrayLists(cache: { arrays(): MxArray[] }): unknown[] {
  const arrays = cache.arrays();
  try {
    mxEval(...arrays);
    return arrays.map((value) => value.toList());
  } finally {
    for (const value of arrays) {
      value.free();
    }
  }
}

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

  test("managed caches expose shared per-layer cache taxonomy", () => {
    using fullCache = new KVCache(2);
    using slidingCache = new SlidingWindowKVCache(2, 4);
    using layerPatternCache = new LayerPatternKVCache(3, [undefined, 2, undefined]);
    using layerPatternSnapshot = layerPatternCache.snapshot();
    using batchCache = new BatchKVCache(2, [0]);
    using layerPatternBatchCache = new LayerPatternBatchKVCache(3, [0], [undefined, 2, undefined]);

    expect(cacheLayerKindFromAttentionType("full_attention")).toBe("full");
    expect(cacheLayerKindFromAttentionType("sliding_attention")).toBe("sliding");
    expect(cacheLayerKindFromAttentionType("linear_attention")).toBe("linear-recurrent");
    expect(fullCache.layerKinds).toEqual(["full", "full"]);
    expect(slidingCache.layerKinds).toEqual(["sliding", "sliding"]);
    expect(layerPatternCache.layerKinds).toEqual(["full", "sliding", "full"]);
    expect(layerPatternSnapshot.layerKinds).toEqual(["full", "sliding", "full"]);
    expect(batchCache.layerKinds).toEqual(["full", "full"]);
    expect(layerPatternBatchCache.layerKinds).toEqual(["full", "sliding", "full"]);
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

  test("managed KVCache snapshots fork exact and prefix-trimmed cache state", () => {
    using cache = new KVCache(1);
    using keys = array([[[[1], [2], [3]]]], "float32");
    using values = array([[[[10], [20], [30]]]], "float32");
    using view = cache.updateAndFetch(0, keys, values).keys;
    cache.advance(3);
    mxEval(view);

    using snapshot = cache.snapshot();
    expect(snapshot.offset).toBe(3);
    expect(snapshot.trimmable).toBe(true);
    expect(snapshot.canFork({ offset: 2 })).toBe(true);

    using exact = snapshot.fork();
    expect(exact.offset).toBe(3);
    expect(cacheArrayLists(exact)).toEqual([[[[[1], [2], [3]]]], [[[[10], [20], [30]]]]]);

    using prefix = snapshot.fork({ offset: 2 });
    expect(prefix.offset).toBe(2);
    expect(cacheArrayLists(prefix)).toEqual([[[[[1], [2]]]], [[[[10], [20]]]]]);
  });

  test("managed SlidingWindowKVCache snapshots preserve exact ring-buffer continuation", () => {
    using cache = new SlidingWindowKVCache(1, 2);
    using firstKeys = array([[[[1], [2]]]], "float32");
    using firstValues = array([[[[10], [20]]]], "float32");
    using secondKeys = array([[[[3]]]], "float32");
    using secondValues = array([[[[30]]]], "float32");
    using firstView = cache.updateAndFetch(0, firstKeys, firstValues).keys;
    cache.advance(2);
    using secondView = cache.updateAndFetch(0, secondKeys, secondValues).keys;
    cache.advance(1);
    mxEval(firstView, secondView);

    using snapshot = cache.snapshot();
    expect(snapshot.offset).toBe(3);
    expect(snapshot.trimmable).toBe(false);
    expect(snapshot.canFork()).toBe(true);
    expect(snapshot.canFork({ offset: 2 })).toBe(false);

    using fork = snapshot.fork();
    expect(fork.offset).toBe(3);
    expect(cacheArrayLists(fork)).toEqual([[[[[3], [2]]]], [[[[30], [20]]]]]);

    using nextKeys = array([[[[4]]]], "float32");
    using nextValues = array([[[[40]]]], "float32");
    using originalNextView = cache.updateAndFetch(0, nextKeys, nextValues).keys;
    using forkNextView = fork.updateAndFetch(0, nextKeys, nextValues).keys;
    mxEval(originalNextView, forkNextView);
    expect(originalNextView.toList()).toEqual([[[[3], [4]]]]);
    expect(forkNextView.toList()).toEqual([[[[3], [4]]]]);
  });

  test("managed LayerPatternKVCache snapshots preserve mixed full and sliding layers exactly", () => {
    using cache = new LayerPatternKVCache(2, [undefined, 2]);
    using fullKeys = array([[[[1], [2], [3]]]], "float32");
    using fullValues = array([[[[10], [20], [30]]]], "float32");
    using slidingFirstKeys = array([[[[4], [5]]]], "float32");
    using slidingFirstValues = array([[[[40], [50]]]], "float32");
    using slidingSecondKeys = array([[[[6]]]], "float32");
    using slidingSecondValues = array([[[[60]]]], "float32");
    using fullView = cache.updateAndFetch(0, fullKeys, fullValues).keys;
    using slidingFirstView = cache.updateAndFetch(1, slidingFirstKeys, slidingFirstValues).keys;
    cache.advance(2);
    using slidingSecondView = cache.updateAndFetch(1, slidingSecondKeys, slidingSecondValues).keys;
    cache.advance(1);
    mxEval(fullView, slidingFirstView, slidingSecondView);

    expect(cache.isTrimmable()).toBe(false);
    using snapshot = cache.snapshot();
    expect(snapshot.trimmable).toBe(false);
    expect(snapshot.canFork()).toBe(true);
    expect(snapshot.canFork({ offset: 2 })).toBe(false);

    using fork = snapshot.fork();
    expect(fork.offset).toBe(3);
    expect(cacheArrayLists(fork)).toEqual([
      [[[[1], [2], [3]]]],
      [[[[10], [20], [30]]]],
      [[[[6], [5]]]],
      [[[[60], [50]]]],
    ]);

    using nextKeys = array([[[[7]]]], "float32");
    using nextValues = array([[[[70]]]], "float32");
    using forkNextView = fork.updateAndFetch(1, nextKeys, nextValues).keys;
    mxEval(forkNextView);
    expect(forkNextView.toList()).toEqual([[[[6], [7]]]]);
  });

  test("managed KVCache snapshots cannot fork after disposal", () => {
    using cache = new KVCache(1);
    using keys = array([[[[1]]]], "float32");
    using values = array([[[[10]]]], "float32");
    using view = cache.updateAndFetch(0, keys, values).keys;
    cache.advance(1);
    mxEval(view);

    const snapshot = cache.snapshot();
    snapshot[Symbol.dispose]();
    expect(snapshot.canFork()).toBe(false);
    expect(() => snapshot.fork()).toThrow("cannot fork offset 1");
    snapshot[Symbol.dispose]();
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

  test("managed LayerPatternBatchKVCache retains full and sliding layers independently", () => {
    using cache = new LayerPatternBatchKVCache(2, [2, 0], [undefined, 2]);
    using keys = array([[[[0], [0], [1], [2]]], [[[3], [4], [5], [6]]]], "float32");
    using values = array([[[[10], [10], [11], [12]]], [[[13], [14], [15], [16]]]], "float32");
    using fullView = cache.updateAndFetch(0, keys, values).keys;
    using slidingView = cache.updateAndFetch(1, keys, values).keys;
    cache.advance(4);
    const stateArrays = cache.arrays();
    const fullKeys = stateArrays[0];
    const fullValues = stateArrays[1];
    const slidingKeys = stateArrays[2];
    const slidingValues = stateArrays[3];
    if (
      fullKeys === undefined ||
      fullValues === undefined ||
      slidingKeys === undefined ||
      slidingValues === undefined
    ) {
      throw new Error("expected retained layer-pattern batch cache keys");
    }
    using ownedFullKeys = fullKeys;
    using _ownedFullValues = fullValues;
    using ownedSlidingKeys = slidingKeys;
    using _ownedSlidingValues = slidingValues;
    using fullPadding = cache.leftPaddingTensorForLayer(0, 4, 0);
    using slidingPaddingAfterDecode = cache.leftPaddingTensorForLayer(1, 2, 1);
    mxEval(fullView, slidingView, ownedFullKeys, ownedSlidingKeys);

    expect(fullView.shape).toEqual([2, 1, 4, 1]);
    expect(slidingView.shape).toEqual([2, 1, 4, 1]);
    expect(ownedFullKeys.shape).toEqual([2, 1, 4, 1]);
    expect(ownedSlidingKeys.shape).toEqual([2, 1, 2, 1]);
    expect(cache.leftPaddingValuesForLayer(1, 2, 1)).toEqual([0, 0]);
    expect(fullPadding.toList()).toEqual([2, 0]);
    expect(slidingPaddingAfterDecode.toList()).toEqual([0, 0]);
  });

  test("managed LayerPatternBatchKVCache filters active requests with retained sliding state", () => {
    using cache = new LayerPatternBatchKVCache(2, [2, 1, 0], [undefined, 2]);
    using keys = array([[[[0], [0], [1]]], [[[0], [2], [3]]], [[[4], [5], [6]]]], "float32");
    using values = array(
      [[[[10], [10], [11]]], [[[10], [12], [13]]], [[[14], [15], [16]]]],
      "float32",
    );
    using fullView = cache.updateAndFetch(0, keys, values).keys;
    using slidingView = cache.updateAndFetch(1, keys, values).keys;
    cache.advance(3);
    mxEval(fullView, slidingView);

    cache.filter([0, 1]);
    const stateArrays = cache.arrays();
    const fullKeys = stateArrays[0];
    const fullValues = stateArrays[1];
    const slidingKeys = stateArrays[2];
    const slidingValues = stateArrays[3];
    if (
      fullKeys === undefined ||
      fullValues === undefined ||
      slidingKeys === undefined ||
      slidingValues === undefined
    ) {
      throw new Error("expected retained layer-pattern batch cache keys");
    }
    using ownedFullKeys = fullKeys;
    using _ownedFullValues = fullValues;
    using ownedSlidingKeys = slidingKeys;
    using _ownedSlidingValues = slidingValues;
    mxEval(ownedFullKeys, ownedSlidingKeys);

    expect(cache.batchSize).toBe(2);
    expect(cache.length).toBe(2);
    expect(cache.leftPadding).toEqual([1, 0]);
    expect(cache.offsets).toEqual([1, 2]);
    expect(ownedFullKeys.toList()).toEqual([[[[0], [1]]], [[[2], [3]]]]);
    expect(ownedSlidingKeys.toList()).toEqual([[[[0], [1]]], [[[2], [3]]]]);
  });

  test("managed LayerPatternBatchKVCache extends batches and extracts single caches", () => {
    using first = new LayerPatternBatchKVCache(2, [0], [undefined, 2]);
    using firstKeys = array([[[[1], [2], [3]]]], "float32");
    using firstValues = array([[[[11], [12], [13]]]], "float32");
    using firstFullView = first.updateAndFetch(0, firstKeys, firstValues).keys;
    using firstSlidingView = first.updateAndFetch(1, firstKeys, firstValues).keys;
    first.advance(3);

    using second = new LayerPatternBatchKVCache(2, [0], [undefined, 2]);
    using secondKeys = array([[[[4]]]], "float32");
    using secondValues = array([[[[14]]]], "float32");
    using secondFullView = second.updateAndFetch(0, secondKeys, secondValues).keys;
    using secondSlidingView = second.updateAndFetch(1, secondKeys, secondValues).keys;
    second.advance(1);
    mxEval(firstFullView, firstSlidingView, secondFullView, secondSlidingView);

    first.extend(second);
    const stateArrays = first.arrays();
    const fullKeys = stateArrays[0];
    const fullValues = stateArrays[1];
    const slidingKeys = stateArrays[2];
    const slidingValues = stateArrays[3];
    if (
      fullKeys === undefined ||
      fullValues === undefined ||
      slidingKeys === undefined ||
      slidingValues === undefined
    ) {
      throw new Error("expected extended layer-pattern batch cache keys");
    }
    using ownedFullKeys = fullKeys;
    using _ownedFullValues = fullValues;
    using ownedSlidingKeys = slidingKeys;
    using _ownedSlidingValues = slidingValues;
    mxEval(ownedFullKeys, ownedSlidingKeys);

    expect(first.batchSize).toBe(2);
    expect(first.length).toBe(3);
    expect(first.leftPadding).toEqual([0, 2]);
    expect(first.offsets).toEqual([3, 1]);
    expect(ownedFullKeys.toList()).toEqual([[[[1], [2], [3]]], [[[0], [0], [4]]]]);
    expect(ownedSlidingKeys.toList()).toEqual([[[[2], [3]]], [[[0], [4]]]]);

    using extracted = first.extract(1);
    const extractedArrays = extracted.arrays();
    const extractedFullKeys = extractedArrays[0];
    const extractedFullValues = extractedArrays[1];
    const extractedSlidingKeys = extractedArrays[2];
    const extractedSlidingValues = extractedArrays[3];
    if (
      extractedFullKeys === undefined ||
      extractedFullValues === undefined ||
      extractedSlidingKeys === undefined ||
      extractedSlidingValues === undefined
    ) {
      throw new Error("expected extracted layer-pattern cache keys");
    }
    using ownedExtractedFullKeys = extractedFullKeys;
    using _ownedExtractedFullValues = extractedFullValues;
    using ownedExtractedSlidingKeys = extractedSlidingKeys;
    using _ownedExtractedSlidingValues = extractedSlidingValues;
    mxEval(ownedExtractedFullKeys, ownedExtractedSlidingKeys);

    expect(extracted.offset).toBe(1);
    expect(ownedExtractedFullKeys.toList()).toEqual([[[[4]]]]);
    expect(ownedExtractedSlidingKeys.toList()).toEqual([[[[4]]]]);
  });

  test("managed LayerPatternBatchKVCache validates metadata and incompatible operations", () => {
    expect(() => new LayerPatternBatchKVCache(0, [0], [undefined])).toThrow(
      "layerCount must be a positive integer",
    );
    expect(() => new LayerPatternBatchKVCache(1, [], [undefined])).toThrow(
      "leftPadding must contain",
    );
    expect(() => new LayerPatternBatchKVCache(1, [-1], [undefined])).toThrow("leftPadding[0]");
    expect(() => new LayerPatternBatchKVCache(2, [0], [undefined])).toThrow(
      "must match layerCount",
    );
    expect(() => new LayerPatternBatchKVCache(1, [0], [0])).toThrow(
      "each window size must be positive",
    );

    using cache = new LayerPatternBatchKVCache(1, [0, 0], [undefined]);
    expect(() => cache.advance(-1)).toThrow("sequenceLength must be non-negative");
    expect(() => cache.filter([])).toThrow("batchIndices must contain");
    expect(() => cache.filter([2])).toThrow("out of range");
    expect(() => cache.filter([0, 0])).toThrow("duplicate");
    expect(() => cache.extract(2)).toThrow("out of range");
    expect(() => cache.extend(new BatchKVCache(1, [0]))).toThrow(
      "expected another layer-pattern batch cache",
    );

    using incompatible = new LayerPatternBatchKVCache(1, [0], [2]);
    expect(() => cache.extend(incompatible)).toThrow("layer window sizes must match");

    using wrongBatchKeys = array([[[[1]]]], "float32");
    using wrongBatchValues = array([[[[1]]]], "float32");
    expect(() => cache.updateAndFetch(0, wrongBatchKeys, wrongBatchValues)).toThrow(
      "update batch size must be 2",
    );
  });
});
