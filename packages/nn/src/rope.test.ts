import { describe, expect, test } from "bun:test";
import { array, mxEval } from "@mlxts/core";
import { RoPE } from "./rope";

describe("RoPE", () => {
  test("preserves shape and applies a positional rotation", () => {
    using rope = new RoPE(4, true, 10000);
    using input = array(
      [
        [
          [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
          ],
        ],
      ],
      "float32",
    );
    using output = rope.forward(input, 2);

    mxEval(output);
    expect(output.shape).toEqual([1, 1, 2, 4]);
    expect(output.toList()).not.toEqual(input.toList());
  });

  test("supports per-example offset arrays", () => {
    using rope = new RoPE(4, false, 10000);
    using input = array(
      [
        [
          [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
          ],
        ],
        [
          [
            [2, 1, 4, 3],
            [6, 5, 8, 7],
          ],
        ],
      ],
      "float32",
    );
    using offsets = array([0, 3], "int32");
    using output = rope.forward(input, offsets);

    mxEval(output);
    expect(output.shape).toEqual([2, 1, 2, 4]);
  });
});
