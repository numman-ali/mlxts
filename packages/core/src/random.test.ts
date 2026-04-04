import { describe, expect, test } from "bun:test";

import { array } from "./array";
import { bernoulli, categorical, key, normal, seed, split, uniform } from "./random";

describe("random", () => {
  test("key creates a deterministic key", () => {
    const k = key(42);
    expect(k.ndim).toBeGreaterThanOrEqual(0);
    expect(k.size).toBeGreaterThan(0);
    k.free();
  });

  test("split produces two different keys", () => {
    const k = key(42);
    const [k1, k2] = split(k);
    expect(k1._ctx).not.toBe(k2._ctx);
    k.free();
    k1.free();
    k2.free();
  });

  test("normal produces correct shape", () => {
    const k = key(42);
    const a = normal([3, 4], "float32", 0, 1, k);
    expect(a.shape).toEqual([3, 4]);
    expect(a.dtype).toBe("float32");
    expect(a.size).toBe(12);
    k.free();
    a.free();
  });

  test("normal without explicit key uses default", () => {
    seed(123);
    const a = normal([5]);
    expect(a.shape).toEqual([5]);
    a.free();
  });

  test("re-seeding resets the default generator deterministically", () => {
    seed(777);
    const first = normal([8], "float32");
    first.eval();
    const firstList = first.toList();
    first.free();

    seed(777);
    const second = normal([8], "float32");
    second.eval();
    expect(second.toList()).toEqual(firstList);
    second.free();
  });

  test("uniform values are in range", () => {
    const k = key(42);
    const a = uniform(0, 1, [100], "float32", k);
    a.eval();
    const data = a.toList() as number[];
    for (const v of data) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
    k.free();
    a.free();
  });

  test("normal has roughly zero mean and unit variance over many samples", () => {
    seed(1234);
    const samples = normal([4096], "float32");
    samples.eval();
    const data = samples.toList() as number[];
    const meanValue = data.reduce((total, value) => total + value, 0) / data.length;
    const variance =
      data.reduce((total, value) => total + (value - meanValue) ** 2, 0) / data.length;

    expect(Math.abs(meanValue)).toBeLessThan(0.1);
    expect(variance).toBeGreaterThan(0.8);
    expect(variance).toBeLessThan(1.2);
    samples.free();
  });

  test("bernoulli produces boolean-like values", () => {
    const k = key(42);
    const a = bernoulli(0.5, [100], k);
    a.eval();
    const data = a.toList() as number[];
    // All values should be 0 or 1
    for (const v of data) {
      expect(v === 0 || v === 1).toBe(true);
    }
    k.free();
    a.free();
  });

  test("same key produces same results", () => {
    const k1 = key(42);
    const k2 = key(42);
    const a = normal([5], "float32", 0, 1, k1);
    const b = normal([5], "float32", 0, 1, k2);
    a.eval();
    b.eval();
    const aList = a.toList() as number[];
    const bList = b.toList() as number[];
    for (let i = 0; i < 5; i++) {
      const bValue = bList[i];
      if (bValue === undefined) {
        throw new Error(`Missing random value at index ${i}`);
      }
      expect(aList[i]).toBeCloseTo(bValue);
    }
    k1.free();
    k2.free();
    a.free();
    b.free();
  });

  test("categorical returns int32 indices", () => {
    seed(42);
    // Logits strongly favor index 2
    const logits = array([[-100, -100, 10]]);
    const result = categorical(logits, -1);
    result.eval();
    expect(result.dtype).toBe("uint32");
    expect(result.item()).toBe(2);
    logits.free();
    result.free();
  });

  test("categorical produces valid indices within range", () => {
    seed(42);
    const logits = array([[1, 2, 3, 4, 5]]);
    const result = categorical(logits, -1);
    result.eval();
    const idx = result.item();
    expect(idx).toBeGreaterThanOrEqual(0);
    expect(idx).toBeLessThan(5);
    logits.free();
    result.free();
  });

  test("categorical output shape matches input without sampled axis", () => {
    seed(42);
    // [2, 5] logits → sample along axis=-1 → [2] output
    const logits = array([
      [1, 2, 3, 4, 5],
      [5, 4, 3, 2, 1],
    ]);
    const result = categorical(logits, -1);
    result.eval();
    expect(result.shape).toEqual([2]);
    logits.free();
    result.free();
  });

  test("categorical follows the stronger logit more often across many rows", () => {
    seed(314);
    const logits = array(
      Array.from({ length: 512 }, () => [0, 3]),
      "float32",
    );
    const result = categorical(logits, -1);
    result.eval();
    const indices = result.toList() as number[];
    const countOfOne = indices.filter((value) => value === 1).length;

    expect(countOfOne).toBeGreaterThan(400);
    logits.free();
    result.free();
  });
});
