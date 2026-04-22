import { describe, expect, test } from "bun:test";

import { array, mxEval, retainArray } from "@mlxts/core";

import {
  cachePrefixView,
  cacheTailView,
  growCacheBuffer,
  orderedSlidingView,
  roundCacheCapacity,
  sequenceAxisLength,
  writeCacheRangeInPlace,
} from "./ops";

describe("cache ops", () => {
  test("roundCacheCapacity rounds up in 256-token chunks", () => {
    expect(roundCacheCapacity(1)).toBe(256);
    expect(roundCacheCapacity(256)).toBe(256);
    expect(roundCacheCapacity(257)).toBe(512);
  });

  test("sequenceAxisLength reads the cache sequence axis and rejects malformed tensors", () => {
    using cacheUpdate = array([[[[1], [2], [3]]]], "float32");
    using malformed = array([1, 2, 3], "float32");

    expect(sequenceAxisLength(cacheUpdate, "cache-ops.test")).toBe(3);
    expect(() => sequenceAxisLength(malformed, "cache-ops.test")).toThrow(
      "cache-ops.test: cache tensor is missing a sequence axis.",
    );
  });

  test("cachePrefixView and cacheTailView return the requested windows", () => {
    using tensor = array([[[[1], [2], [3], [4]]]], "float32");
    using prefix = cachePrefixView(tensor, 2);
    using fullPrefix = cachePrefixView(tensor, 4);
    using tail = cacheTailView(tensor, 2);
    using fullTail = cacheTailView(tensor, 4);

    mxEval(prefix, fullPrefix, tail, fullTail);

    expect(prefix.toList()).toEqual([[[[1], [2]]]]);
    expect(fullPrefix.toList()).toEqual([[[[1], [2], [3], [4]]]]);
    expect(tail.toList()).toEqual([[[[3], [4]]]]);
    expect(fullTail.toList()).toEqual([[[[1], [2], [3], [4]]]]);
  });

  test("orderedSlidingView rotates a saturated ring buffer into logical order", () => {
    using tensor = array([[[[1], [2], [3], [4]]]], "float32");
    using rotated = orderedSlidingView(tensor, 4, 2);
    using prefix = orderedSlidingView(tensor, 2, 0);

    mxEval(rotated, prefix);

    expect(rotated.toList()).toEqual([[[[3], [4], [1], [2]]]]);
    expect(prefix.toList()).toEqual([[[[1], [2]]]]);
  });

  test("growCacheBuffer copies the retained prefix and writeCacheRange inserts updates", () => {
    using existing = array([[[[1], [2], [3], [4]]]], "float32");
    using update = array([[[[9], [10]]]], "float32");
    using grown = growCacheBuffer(existing, 2, update, 6);
    using prefixCopied = retainArray(grown);
    writeCacheRangeInPlace(grown, update, 2);

    mxEval(prefixCopied, grown);

    expect(prefixCopied.toList()).toEqual([[[[1], [2], [0], [0], [0], [0]]]]);
    expect(grown.toList()).toEqual([[[[1], [2], [9], [10], [0], [0]]]]);
  });

  test("growCacheBuffer extends a fully used buffer without rewriting the existing prefix", () => {
    using existing = array([[[[1], [2], [3], [4]]]], "float32");
    using grown = growCacheBuffer(existing, 4, existing, 6);

    mxEval(grown);

    expect(grown.toList()).toEqual([[[[1], [2], [3], [4], [0], [0]]]]);
  });
});
