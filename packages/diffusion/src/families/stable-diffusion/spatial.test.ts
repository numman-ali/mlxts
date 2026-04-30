import { describe, expect, test } from "bun:test";

import { MxArray, mxEval } from "@mlxts/core";
import { padBottomRight2d, upsampleNearest2d } from "./spatial";

describe("Stable Diffusion spatial helpers", () => {
  test("upsampleNearest2d repeats NHWC feature maps over height and width", () => {
    using input = MxArray.fromData([1, 2, 3, 4], [1, 2, 2, 1]);
    using output = upsampleNearest2d(input);
    mxEval(output);

    expect(output.shape).toEqual([1, 4, 4, 1]);
    expect(output.toList()).toEqual([
      [
        [[1], [1], [2], [2]],
        [[1], [1], [2], [2]],
        [[3], [3], [4], [4]],
        [[3], [3], [4], [4]],
      ],
    ]);
  });

  test("padBottomRight2d matches the Stable Diffusion downsample pre-padding shape", () => {
    using input = MxArray.fromData([1, 2, 3, 4], [1, 2, 2, 1]);
    using output = padBottomRight2d(input);
    mxEval(output);

    expect(output.shape).toEqual([1, 3, 3, 1]);
    expect(output.toList()).toEqual([
      [
        [[1], [2], [0]],
        [[3], [4], [0]],
        [[0], [0], [0]],
      ],
    ]);
  });

  test("rejects malformed image inputs and scale values", () => {
    using input = MxArray.fromData([1, 2], [1, 2]);
    expect(() => upsampleNearest2d(input)).toThrow("rank-4");
    expect(() => padBottomRight2d(input)).toThrow("rank-4");

    using image = MxArray.fromData([1], [1, 1, 1, 1]);
    expect(() => upsampleNearest2d(image, 0)).toThrow("scale");
    expect(() => padBottomRight2d(image, -1)).toThrow("non-negative");
  });
});
