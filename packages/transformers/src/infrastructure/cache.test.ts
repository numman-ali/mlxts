import { describe, expect, test } from "bun:test";

import { array, mxEval } from "@mlxts/core";

import { KVCache, LayerPatternKVCache, SlidingWindowKVCache } from "./cache";

describe("Transformer caches", () => {
  test("KVCache appends keys and values across updates", () => {
    using cache = new KVCache(1);
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
  });

  test("SlidingWindowKVCache reuses ring-buffer storage for single-token updates once full", () => {
    using cache = new SlidingWindowKVCache(1, 2);
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
  });

  test("LayerPatternKVCache reuses ring-buffer storage only for configured sliding layers", () => {
    using cache = new LayerPatternKVCache(2, [undefined, 2]);
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
  });

  test("SlidingWindowKVCache keeps extra within-chunk context during multi-token prefill", () => {
    using cache = new SlidingWindowKVCache(1, 4);
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
  });

  test("LayerPatternKVCache full layers grow in chunks without losing earlier tokens", () => {
    using cache = new LayerPatternKVCache(1, [undefined]);
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
  });
});
