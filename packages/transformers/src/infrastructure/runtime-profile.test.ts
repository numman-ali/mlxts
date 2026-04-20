import { afterEach, describe, expect, test } from "bun:test";

import { array, mxEval } from "@mlxts/core";

import { SlidingWindowKVCache } from "./cache";
import {
  resetTransformerRuntimeProfile,
  snapshotTransformerRuntimeProfile,
} from "./runtime-profile";

const ORIGINAL_PROFILE_FLAG = process.env.MLXTS_RUNTIME_PROFILE;

afterEach(() => {
  if (ORIGINAL_PROFILE_FLAG === undefined) {
    delete process.env.MLXTS_RUNTIME_PROFILE;
  } else {
    process.env.MLXTS_RUNTIME_PROFILE = ORIGINAL_PROFILE_FLAG;
  }
  resetTransformerRuntimeProfile();
});

function runProfileScenario(cache: SlidingWindowKVCache): void {
  process.env.MLXTS_RUNTIME_PROFILE = "1";
  resetTransformerRuntimeProfile();

  using firstKeys = array([[[[1], [2]]]], "float32");
  using firstValues = array([[[[10], [20]]]], "float32");
  using secondKeys = array([[[[3]]]], "float32");
  using secondValues = array([[[[30]]]], "float32");
  using firstView = cache.updateAndFetch(0, firstKeys, firstValues).keys;
  using secondView = cache.updateAndFetch(0, secondKeys, secondValues).keys;

  mxEval(firstView, secondView);

  const snapshot = snapshotTransformerRuntimeProfile();
  expect(snapshot.enabled).toBe(true);
  expect(snapshot.counters["cache.sliding_growth"]).toBe(1);
  expect(snapshot.counters["cache.sliding_single_token"]).toBe(1);
  expect(snapshot.counters["cache.write_range"]).toBe(4);
  expect(snapshot.counters["cache.buffer_replaced"]).toBe(2);
  expect(snapshot.counters["cache.return_prefix_view"]).toBe(0);
  expect(snapshot.counters["cache.return_full_buffer"]).toBe(4);
}

describe("transformer runtime profile", () => {
  test("captures the saturated sliding cache path counters for the managed cache", () => {
    using cache = new SlidingWindowKVCache(1, 2);
    runProfileScenario(cache);
  });
});
