import { describe, expect, test } from "bun:test";
import { array, mxEval } from "@mlxts/core";

import { Qwen3_5TextBatchCache } from "./batch-cache";
import { Qwen3_5TextCache } from "./cache";

describe("Qwen3_5TextCache", () => {
  test("tracks mixed full-attention and linear-attention state", () => {
    using cache = new Qwen3_5TextCache(["linear_attention", "full_attention", "linear_attention"]);

    expect(cache.layerCount).toBe(3);
    expect(cache.isEmpty()).toBe(true);
    expect(cache.isTrimmable()).toBe(false);

    using fullKeys = array([[[[1], [2]]]], "float32");
    using fullValues = array([[[[10], [20]]]], "float32");
    using fullView = cache.updateAndFetch(1, fullKeys, fullValues).keys;
    mxEval(fullView);
    expect(fullView.toList()).toEqual([[[[1], [2]]]]);

    using convState = array(
      [
        [
          [1, 2],
          [3, 4],
        ],
      ],
      "float32",
    );
    using recurrentState = array([[[[5, 6]]]], "float32");
    cache.updateLinearState(0, convState, recurrentState);
    expect(cache.linearState(0).convState?.toList()).toEqual([
      [
        [1, 2],
        [3, 4],
      ],
    ]);
    expect(cache.linearState(0).recurrentState?.toList()).toEqual([[[[5, 6]]]]);

    const stateArrays = cache.arrays();
    try {
      expect(stateArrays).toHaveLength(4);
      mxEval(...stateArrays);
      expect(stateArrays.map((value) => value.toList())).toEqual([
        [
          [
            [1, 2],
            [3, 4],
          ],
        ],
        [[[[5, 6]]]],
        [[[[1], [2]]]],
        [[[[10], [20]]]],
      ]);
    } finally {
      for (const stateArray of stateArrays) {
        stateArray.free();
      }
    }

    cache.advance(2);
    expect(cache.offset).toBe(2);
    expect(cache.isEmpty()).toBe(false);
  });

  test("rejects incompatible cache access patterns", () => {
    using cache = new Qwen3_5TextCache(["linear_attention", "full_attention"]);
    using keys = array([[[[1]]]], "float32");
    using values = array([[[[2]]]], "float32");

    expect(() => cache.updateAndFetch(0, keys, values)).toThrow("KV updates only apply");
    expect(() => cache.linearState(1)).toThrow("linear state only exists");
    expect(() => cache.advance(-1)).toThrow("sequenceLength must be a non-negative integer");
    expect(() => new Qwen3_5TextCache([])).toThrow("must contain at least one layer");
  });
});

describe("Qwen3_5TextBatchCache", () => {
  test("tracks hybrid batched state, masks left padding, and extracts single caches", () => {
    using cache = new Qwen3_5TextBatchCache(["linear_attention", "full_attention"], [2, 0]);

    expect(cache.batchSize).toBe(2);
    expect(cache.layerCount).toBe(2);
    expect(cache.leftPadding).toEqual([2, 0]);
    expect(cache.offsets).toEqual([-2, 0]);

    using fullKeys = array([[[[0], [0], [7]]], [[[1], [2], [3]]]], "float32");
    using fullValues = array([[[[0], [0], [70]]], [[[10], [20], [30]]]], "float32");
    using activeKeys = cache.updateAndFetch(1, fullKeys, fullValues).keys;
    mxEval(activeKeys);
    expect(activeKeys.shape).toEqual([2, 1, 3, 1]);

    using convState = array([[[7, 8]], [[1, 2]]], "float32");
    using recurrentState = array([[[[9]]], [[[3]]]], "float32");
    cache.updateLinearState(0, convState, recurrentState);

    using initialMask = cache.linearAttentionMask(3);
    expect(initialMask?.toList()).toEqual([
      [0, 0, 1],
      [1, 1, 1],
    ]);

    cache.advance(3);
    expect(cache.offsets).toEqual([1, 3]);
    using decodeMask = cache.linearAttentionMask(1);
    expect(decodeMask).toBeNull();

    const extracted = cache.extract(0);
    try {
      expect(extracted).toBeInstanceOf(Qwen3_5TextCache);
      expect(extracted.offset).toBe(1);
      const arrays = extracted.arrays();
      try {
        mxEval(...arrays);
        expect(arrays.map((value) => value.toList())).toEqual([
          [[[7, 8]]],
          [[[[9]]]],
          [[[[7]]]],
          [[[[70]]]],
        ]);
      } finally {
        for (const value of arrays) {
          value.free();
        }
      }
    } finally {
      extracted[Symbol.dispose]();
    }
  });

  test("filters full and linear state by active rows", () => {
    using cache = new Qwen3_5TextBatchCache(["linear_attention", "full_attention"], [1, 0]);
    using convState = array([[[4, 5]], [[6, 7]]], "float32");
    using recurrentState = array([[[[8]]], [[[9]]]], "float32");
    cache.updateLinearState(0, convState, recurrentState);

    cache.filter([1]);

    expect(cache.batchSize).toBe(1);
    expect(cache.leftPadding).toEqual([0]);
    expect(cache.offsets).toEqual([0]);
    expect(cache.linearState(0).convState?.toList()).toEqual([[[6, 7]]]);
    expect(cache.linearState(0).recurrentState?.toList()).toEqual([[[[9]]]]);
  });

  test("extends full-attention and linear-attention batch state", () => {
    using left = new Qwen3_5TextBatchCache(["linear_attention", "full_attention"], [0]);
    using leftKeys = array([[[[1], [2]]]], "float32");
    using leftValues = array([[[[10], [20]]]], "float32");
    using leftConvState = array([[[3, 4]]], "float32");
    using leftRecurrentState = array([[[[5]]]], "float32");
    using leftView = left.updateAndFetch(1, leftKeys, leftValues).keys;
    left.updateLinearState(0, leftConvState, leftRecurrentState);
    mxEval(leftView);
    left.advance(2);

    using right = new Qwen3_5TextBatchCache(["linear_attention", "full_attention"], [0]);
    using rightKeys = array([[[[7]]]], "float32");
    using rightValues = array([[[[70]]]], "float32");
    using rightConvState = array([[[8, 9]]], "float32");
    using rightRecurrentState = array([[[[11]]]], "float32");
    using rightView = right.updateAndFetch(1, rightKeys, rightValues).keys;
    right.updateLinearState(0, rightConvState, rightRecurrentState);
    mxEval(rightView);
    right.advance(1);

    left.extend(right);

    expect(left.batchSize).toBe(2);
    expect(left.leftPadding).toEqual([0, 1]);
    expect(left.offsets).toEqual([2, 1]);
    expect(left.linearState(0).convState?.toList()).toEqual([[[3, 4]], [[8, 9]]]);
    expect(left.linearState(0).recurrentState?.toList()).toEqual([[[[5]]], [[[11]]]]);

    const extracted = left.extract(1);
    try {
      const arrays = extracted.arrays();
      try {
        mxEval(...arrays);
        expect(arrays.map((value) => value.toList())).toEqual([
          [[[8, 9]]],
          [[[[11]]]],
          [[[[7]]]],
          [[[[70]]]],
        ]);
      } finally {
        for (const value of arrays) {
          value.free();
        }
      }
    } finally {
      extracted[Symbol.dispose]();
    }
  });
});
