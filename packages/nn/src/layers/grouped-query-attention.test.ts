import { describe, expect, test } from "bun:test";
import { array, mxEval } from "@mlxts/core";
import { GroupedQueryAttention } from "./grouped-query-attention";
import { RoPE } from "./rope";

describe("GroupedQueryAttention", () => {
  test("forward works when numKeyValueHeads is smaller than numHeads", () => {
    using attention = new GroupedQueryAttention(8, 4, 2, {
      rope: new RoPE(2, true),
    });
    using input = array(
      [
        [
          [1, 2, 3, 4, 5, 6, 7, 8],
          [2, 3, 4, 5, 6, 7, 8, 9],
          [3, 4, 5, 6, 7, 8, 9, 10],
        ],
      ],
      "float32",
    );
    using mask = array(
      [
        [
          [
            [0, -1e9, -1e9],
            [0, 0, -1e9],
            [0, 0, 0],
          ],
          [
            [0, -1e9, -1e9],
            [0, 0, -1e9],
            [0, 0, 0],
          ],
          [
            [0, -1e9, -1e9],
            [0, 0, -1e9],
            [0, 0, 0],
          ],
          [
            [0, -1e9, -1e9],
            [0, 0, -1e9],
            [0, 0, 0],
          ],
        ],
      ],
      "float32",
    );

    using output = attention.forward(input, { attentionMask: mask, offset: 1 });
    mxEval(output);

    expect(output.shape).toEqual([1, 3, 8]);
    const values = (output.toList() as number[][][]).flat(2);
    expect(values.every((value) => Number.isFinite(value))).toBe(true);
  });

  test("rejects head configurations that do not divide cleanly", () => {
    expect(() => new GroupedQueryAttention(8, 4, 3)).toThrow("must be divisible");
  });
});
