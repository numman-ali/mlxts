import { describe, expect, test } from "bun:test";

import { array, mxEval, ones } from "@mlxts/core";
import { GroupNorm } from "./group-norm";

describe("GroupNorm", () => {
  test("matches manual channel-last group normalization", () => {
    using norm = new GroupNorm(2, 4, 1e-5);
    using input = array([
      [
        [
          [1, 2, 3, 4],
          [5, 6, 7, 8],
        ],
      ],
    ]);
    using output = norm.forward(input);
    mxEval(output);

    const values = output.toList() as number[][][][];
    const row = values[0]?.[0];
    expect(row?.[0]?.[0]).toBeCloseTo(-1.2127, 4);
    expect(row?.[0]?.[1]).toBeCloseTo(-0.7276, 4);
    expect(row?.[0]?.[2]).toBeCloseTo(-1.2127, 4);
    expect(row?.[0]?.[3]).toBeCloseTo(-0.7276, 4);
    expect(row?.[1]?.[0]).toBeCloseTo(0.7276, 4);
    expect(row?.[1]?.[1]).toBeCloseTo(1.2127, 4);
    expect(row?.[1]?.[2]).toBeCloseTo(0.7276, 4);
    expect(row?.[1]?.[3]).toBeCloseTo(1.2127, 4);
  });

  test("applies affine parameters and exposes them", () => {
    using norm = new GroupNorm(1, 2);
    const initialWeight = norm.weight;
    norm.weight = array([2, 3]);
    initialWeight.free();
    const initialBias = norm.bias;
    norm.bias = array([10, 20]);
    initialBias.free();

    const params = norm.parameters();
    expect(params.weight).toBe(norm.weight);
    expect(params.bias).toBe(norm.bias);
    expect(norm.groups).toBe(1);
    expect(norm.channels).toBe(2);

    using input = array([[[[1, 3]]]]);
    using output = norm.forward(input);
    mxEval(output);
    const values = output.toList() as number[][][][];
    expect(values[0]?.[0]?.[0]?.[0]).toBeCloseTo(8.00001, 4);
    expect(values[0]?.[0]?.[0]?.[1]).toBeCloseTo(22.99999, 4);
  });

  test("rejects invalid constructor, input, and parameter shapes", () => {
    expect(() => new GroupNorm(0, 4)).toThrow("groups");
    expect(() => new GroupNorm(2, 0)).toThrow("channels");
    expect(() => new GroupNorm(3, 4)).toThrow("divide");
    expect(() => new GroupNorm(2, 4, 0)).toThrow("eps");

    using norm = new GroupNorm(2, 4);
    using wrongChannels = ones([1, 2, 2, 3]);
    expect(() => norm.forward(wrongChannels)).toThrow("last dimension");

    const initialWeight = norm.weight;
    norm.weight = ones([3]);
    initialWeight.free();
    using input = ones([1, 2, 2, 4]);
    expect(() => norm.forward(input)).toThrow("weight shape");
  });
});
