import { describe, expect, test } from "bun:test";

import { array } from "./array";
import { rope } from "./fast";
import { mxEval } from "./transforms";

function invFrequency(pairIndex: number, dims: number, base: number): number {
  return base ** (-(2 * pairIndex) / dims);
}

function rotateTraditional(
  values: number[],
  position: number,
  dims: number,
  base: number,
): number[] {
  const result = [...values];
  for (let pairIndex = 0; pairIndex < dims / 2; pairIndex++) {
    const angle = position * invFrequency(pairIndex, dims, base);
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const leftIndex = pairIndex * 2;
    const rightIndex = leftIndex + 1;
    const left = values[leftIndex] ?? 0;
    const right = values[rightIndex] ?? 0;
    result[leftIndex] = left * cos - right * sin;
    result[rightIndex] = left * sin + right * cos;
  }
  return result;
}

function rotateSplitHalf(values: number[], position: number, dims: number, base: number): number[] {
  const result = [...values];
  const half = dims / 2;
  for (let pairIndex = 0; pairIndex < half; pairIndex++) {
    const angle = position * invFrequency(pairIndex, dims, base);
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const left = values[pairIndex] ?? 0;
    const right = values[pairIndex + half] ?? 0;
    result[pairIndex] = left * cos - right * sin;
    result[pairIndex + half] = right * cos + left * sin;
  }
  return result;
}

function manualRoPE(
  input: number[][][][],
  dims: number,
  base: number,
  traditional: boolean,
  offsets: number[],
): number[][][][] {
  return input.map((batchRows, batchIndex) =>
    batchRows.map((headRows) =>
      headRows.map((values, tokenIndex) => {
        const offset = offsets[batchIndex] ?? 0;
        const position = offset + tokenIndex;
        return traditional
          ? rotateTraditional(values, position, dims, base)
          : rotateSplitHalf(values, position, dims, base);
      }),
    ),
  );
}

function flatten4d(values: number[][][][]): number[] {
  return values.flatMap((batches) => batches.flatMap((heads) => heads.flat()));
}

function expectNestedClose(actual: number[][][][], expected: number[][][][]): void {
  const actualFlat = flatten4d(actual);
  const expectedFlat = flatten4d(expected);
  expect(actualFlat).toHaveLength(expectedFlat.length);

  for (let index = 0; index < expectedFlat.length; index++) {
    expect(actualFlat[index]).toBeCloseTo(expectedFlat[index] ?? 0, 5);
  }
}

describe("fast.rope", () => {
  const input = [
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
  ] satisfies number[][][][];

  test("matches the traditional RoPE formula with a numeric offset", () => {
    const dims = 4;
    const base = 10000;
    using tensor = array(input, "float32");
    using output = rope(tensor, dims, {
      traditional: true,
      base,
      offset: 3,
    });

    mxEval(output);
    expectNestedClose(
      output.toList() as number[][][][],
      manualRoPE(input, dims, base, true, [3, 3]),
    );
  });

  test("matches the split-half RoPE formula with batch-specific offsets", () => {
    const dims = 4;
    const base = 10000;
    using tensor = array(input, "float32");
    using offsets = array([0, 2], "int32");
    using output = rope(tensor, dims, {
      traditional: false,
      base,
      offset: offsets,
    });

    mxEval(output);
    expectNestedClose(
      output.toList() as number[][][][],
      manualRoPE(input, dims, base, false, [0, 2]),
    );
  });
});
