import { describe, expect, test } from "bun:test";

import { array, mxEval } from "@mlxts/core";
import type { TransformerCache } from "../../types";
import { KVCache, SlidingWindowKVCache } from "./index";
import { createBorrowedTransformerCacheView, updateAndFetchTransformerCacheView } from "./view";

function runStableMutationScenario(cache: TransformerCache): void {
  using firstKeys = array([[[[1], [2]]]], "float32");
  using firstValues = array([[[[10], [20]]]], "float32");
  using secondKeys = array([[[[3]]]], "float32");
  using secondValues = array([[[[30]]]], "float32");
  const retainedFirstPair = (() => {
    using firstView = updateAndFetchTransformerCacheView(cache, 0, firstKeys, firstValues);
    return firstView.materializeOwnedPair();
  })();
  using ownedFirstKeys = retainedFirstPair.keys;
  using secondView = updateAndFetchTransformerCacheView(cache, 0, secondKeys, secondValues);

  mxEval(ownedFirstKeys, secondView.keys);

  expect(ownedFirstKeys.toList()).toEqual([[[[1], [2]]]]);
  expect(secondView.keys.toList()).toEqual([[[[3], [2]]]]);
}

function runGrowthStabilityScenario(cache: TransformerCache): void {
  const firstSequence = Array.from({ length: 256 }, (_, index) => [index + 1]);
  const secondSequence = [[257]];
  using firstKeys = array([[[...firstSequence]]], "float32");
  using firstValues = array([[[...firstSequence]]], "float32");
  using secondKeys = array([[[...secondSequence]]], "float32");
  using secondValues = array([[[...secondSequence]]], "float32");
  const retainedFirstPair = (() => {
    using firstView = updateAndFetchTransformerCacheView(cache, 0, firstKeys, firstValues);
    return firstView.materializeOwnedPair();
  })();
  using ownedFirstKeys = retainedFirstPair.keys;
  using secondView = updateAndFetchTransformerCacheView(cache, 0, secondKeys, secondValues);

  mxEval(ownedFirstKeys, secondView.keys);

  expect(ownedFirstKeys.shape).toEqual([1, 1, 256, 1]);
  expect(ownedFirstKeys.toList()).toEqual([[[...firstSequence]]]);
  expect(secondView.keys.shape).toEqual([1, 1, 257, 1]);
  expect(secondView.keys.toList()).toEqual([[[...firstSequence, ...secondSequence]]]);
}

describe("TransformerCacheView", () => {
  test("borrowed prefix views stay stable across later cache mutations", () => {
    using cache = new KVCache(1);
    using firstKeys = array([[[[1], [2]]]], "float32");
    using firstValues = array([[[[10], [20]]]], "float32");
    using secondKeys = array([[[[3]]]], "float32");
    using secondValues = array([[[[30]]]], "float32");

    using firstView = updateAndFetchTransformerCacheView(cache, 0, firstKeys, firstValues);
    using secondView = updateAndFetchTransformerCacheView(cache, 0, secondKeys, secondValues);

    mxEval(firstView.keys, secondView.keys);

    expect(firstView.keys.toList()).toEqual([[[[1], [2]]]]);
    expect(secondView.keys.toList()).toEqual([[[[1], [2], [3]]]]);
  });

  test("managed cache views stay stable across later cache mutations", () => {
    using cache = new SlidingWindowKVCache(1, 2);
    runStableMutationScenario(cache);
  });

  test("managed full-cache views stay stable after later chunk growth", () => {
    using cache = new KVCache(1);
    runGrowthStabilityScenario(cache);
  });

  test("materializes owned arrays from a borrowed cache view without taking source ownership", () => {
    using keys = array([[[[1], [2]]]], "float32");
    using values = array([[[[10], [20]]]], "float32");
    const retainedPair = (() => {
      using view = createBorrowedTransformerCacheView(keys, values);
      return view.materializeOwnedPair();
    })();
    using ownedKeys = retainedPair.keys;
    using ownedValues = retainedPair.values;

    mxEval(keys, values, ownedKeys, ownedValues);

    expect(keys.toList()).toEqual([[[[1], [2]]]]);
    expect(values.toList()).toEqual([[[[10], [20]]]]);
    expect(ownedKeys.toList()).toEqual([[[[1], [2]]]]);
    expect(ownedValues.toList()).toEqual([[[[10], [20]]]]);
  });
});
