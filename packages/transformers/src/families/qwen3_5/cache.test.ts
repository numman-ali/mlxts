import { describe, expect, test } from "bun:test";
import { array, mxEval } from "@mlxts/core";

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
