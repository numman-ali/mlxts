import { describe, expect, test } from "bun:test";

import { bernoulli, key, normal, seed, split, uniform } from "./random";

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
});
