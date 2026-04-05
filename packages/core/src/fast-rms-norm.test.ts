import { describe, expect, test } from "bun:test";

import { array } from "./array";
import { rmsNorm } from "./fast";
import { mxEval } from "./transforms";

function manualRmsNorm(values: number[], weight: number[], eps: number): number[] {
  const meanSquare = values.reduce((sum, value) => sum + value * value, 0) / values.length;
  const scale = 1 / Math.sqrt(meanSquare + eps);
  return values.map((value, index) => value * scale * (weight[index] ?? 1));
}

describe("fast.rmsNorm", () => {
  test("matches a manual RMSNorm computation", () => {
    const eps = 1e-5;
    using input = array(
      [
        [1, 2, 3, 4],
        [2, 4, 6, 8],
      ],
      "float32",
    );
    using weight = array([1, 0.5, 2, 1.5], "float32");
    using output = rmsNorm(input, weight, { eps });

    mxEval(output);
    const actual = output.toList() as number[][];
    const expected = [
      manualRmsNorm([1, 2, 3, 4], [1, 0.5, 2, 1.5], eps),
      manualRmsNorm([2, 4, 6, 8], [1, 0.5, 2, 1.5], eps),
    ];

    for (let row = 0; row < expected.length; row++) {
      const expectedRow = expected[row] ?? [];
      const actualRow = actual[row] ?? [];
      for (let column = 0; column < expectedRow.length; column++) {
        expect(actualRow[column]).toBeCloseTo(expectedRow[column] ?? 0, 5);
      }
    }
  });
});
