import { describe, expect, test } from "bun:test";

import { array } from "./array";
import { ffi } from "./ffi/lib";
import { normalizeArgnums, sameIndices, withArrayVector } from "./transforms-base";

describe("transforms-base", () => {
  test("normalizeArgnums supports default, single, and multi-arg configurations", () => {
    expect(normalizeArgnums(undefined, 2)).toEqual([0]);
    expect(normalizeArgnums(1, 2)).toEqual([1]);
    expect(normalizeArgnums([0, 2], 3)).toEqual([0, 2]);
  });

  test("normalizeArgnums rejects invalid indices", () => {
    expect(() => normalizeArgnums(-1, 2)).toThrow("non-negative integer");
    expect(() => normalizeArgnums([0, 0], 2)).toThrow("Duplicate argnums");
    expect(() => normalizeArgnums(2, 2)).toThrow("out of range");
  });

  test("sameIndices compares arrays positionally", () => {
    expect(sameIndices([0, 1], [0, 1])).toBe(true);
    expect(sameIndices([0, 1], [1, 0])).toBe(false);
    expect(sameIndices([0, 1], [0])).toBe(false);
  });

  test("withArrayVector builds a temporary native vector and frees it afterward", () => {
    using left = array([1, 2, 3], "float32");
    using right = array([4, 5, 6], "float32");

    const size = withArrayVector([left, right], "test", (vec) =>
      Number(ffi.mlx_vector_array_size(vec)),
    );

    expect(size).toBe(2);
  });
});
