import { describe, expect, test } from "bun:test";

import { array } from "./array";
import { layerNorm } from "./fast";

describe("fast.layerNorm", () => {
  test("returns the same shape without affine parameters", () => {
    using input = array(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      "float32",
    );
    using output = layerNorm(input);

    expect(output.shape).toEqual([2, 3]);
  });

  test("supports affine weight, bias, and explicit epsilon", () => {
    using input = array(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      "float32",
    );
    using weight = array([1, 1, 1], "float32");
    using bias = array([0.5, 0.5, 0.5], "float32");
    using output = layerNorm(input, weight, bias, { eps: 1e-4 });

    expect(output.shape).toEqual([2, 3]);
    expect(output.toTypedArray().every((value) => Number.isFinite(value))).toBe(true);
  });
});
