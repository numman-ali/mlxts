import { describe, expect, test } from "bun:test";

import {
  assertElementCount,
  createDataBuffer,
  expectPresent,
  getDirectStorageDtype,
  normalizeInput,
  reshapeFlat,
} from "./array-data";

describe("array-data", () => {
  test("normalizeInput infers shape and dtype from typed arrays", () => {
    const normalized = normalizeInput(new Int16Array([1, 2, 3]));
    expect(normalized.inferredShape).toEqual([3]);
    expect(normalized.inferredDtype).toBe("int16");
    expect(normalized.values).toEqual([1, 2, 3]);
  });

  test("normalizeInput handles nested arrays and explicit overrides", () => {
    const nested = normalizeInput(
      [
        [1, 2],
        [3, 4],
      ],
      [4],
      "float64",
    );

    expect(nested.inferredShape).toEqual([4]);
    expect(nested.inferredDtype).toBe("float64");
    expect(nested.values).toEqual([1, 2, 3, 4]);
  });

  test("normalizeInput rejects ragged nested arrays", () => {
    expect(() => normalizeInput([[1], [2, 3]])).toThrow("uniform shape");
  });

  test("getDirectStorageDtype maps indirect dtypes to a supported storage dtype", () => {
    expect(getDirectStorageDtype("uint64")).toBe("float64");
    expect(getDirectStorageDtype("float16")).toBe("float32");
    expect(getDirectStorageDtype("bfloat16")).toBe("float32");
    expect(getDirectStorageDtype("complex64")).toBe("float32");
  });

  test("assertElementCount rejects mismatched shapes", () => {
    expect(() => assertElementCount(3, [2, 2])).toThrow("Data length 3");
  });

  test("createDataBuffer supports bool and floating-point storage", () => {
    expect(Array.from(createDataBuffer([0, 1, -1], "bool"))).toEqual([0, 1, 1]);
    expect(Array.from(createDataBuffer([1.5, 2.5], "float64"))).toEqual([1.5, 2.5]);
  });

  test("reshapeFlat rebuilds scalar and nested output shapes", () => {
    expect(reshapeFlat([7], [])).toBe(7);
    expect(reshapeFlat([1, 2, 3, 4], [2, 2])).toEqual([
      [1, 2],
      [3, 4],
    ]);
  });

  test("expectPresent throws on undefined values", () => {
    expect(() => expectPresent(undefined, "demo")).toThrow("demo was unexpectedly undefined");
  });
});
