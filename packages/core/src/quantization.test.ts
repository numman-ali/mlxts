import { describe, expect, test } from "bun:test";

import { array } from "./array";
import { matmul, transpose } from "./ops";
import { dequantize, quantize, quantizedMatmul } from "./quantization";
import { mxEval } from "./transforms";

function meanAbsoluteError(actual: number[][], expected: number[][]): number {
  let sum = 0;
  let count = 0;

  for (let row = 0; row < expected.length; row++) {
    const expectedRow = expected[row] ?? [];
    const actualRow = actual[row] ?? [];
    for (let column = 0; column < expectedRow.length; column++) {
      sum += Math.abs((actualRow[column] ?? 0) - (expectedRow[column] ?? 0));
      count += 1;
    }
  }

  return sum / count;
}

describe("quantization", () => {
  test("affine quantize/dequantize round-trips approximately", () => {
    const rows = Array.from({ length: 2 }, (_, row) =>
      Array.from({ length: 64 }, (_, column) => (row * 64 + column - 32) / 16),
    );

    using weight = array(rows, "float32");
    const result = quantize(weight, {
      groupSize: 32,
      bits: 4,
      mode: "affine",
    });
    using quantized = result.weight;
    using quantizedScales = result.scales;
    using quantizedBiases = result.biases ?? array([0], "float32");
    using dequantized =
      result.biases === undefined
        ? dequantize(quantized, quantizedScales, {
            groupSize: 32,
            bits: 4,
            mode: "affine",
            dtype: "float32",
          })
        : dequantize(quantized, quantizedScales, {
            biases: quantizedBiases,
            groupSize: 32,
            bits: 4,
            mode: "affine",
            dtype: "float32",
          });

    mxEval(dequantized);
    expect(quantized.dtype).toBe("uint32");
    expect(quantizedScales.shape.length).toBeGreaterThan(0);
    expect(meanAbsoluteError(dequantized.toList() as number[][], rows)).toBeLessThan(0.15);
  });

  test("mxfp4 quantize returns weights and scales without biases", () => {
    using weight = array(
      Array.from({ length: 2 }, (_, row) =>
        Array.from({ length: 32 }, (_, column) => (row * 32 + column) / 8),
      ),
      "float32",
    );

    const result = quantize(weight, { mode: "mxfp4" });
    using quantized = result.weight;
    using scales = result.scales;

    expect(result.biases).toBeUndefined();
    expect(quantized.shape[0]).toBe(2);
    expect(scales.shape.length).toBeGreaterThan(0);
  });

  test("quantizedMatmul matches dense matmul with transpose", () => {
    using input = array([Array.from({ length: 32 }, (_, index) => (index + 1) / 8)], "float32");
    using weight = array(
      [
        Array.from({ length: 32 }, (_, index) => (index - 8) / 16),
        Array.from({ length: 32 }, (_, index) => (16 - index) / 20),
      ],
      "float32",
    );
    const result = quantize(weight, {
      groupSize: 32,
      bits: 4,
      mode: "affine",
    });
    using quantized = result.weight;
    using scales = result.scales;
    using biases = result.biases ?? array([[0]], "float32");
    const dequantizeOptions =
      result.biases === undefined
        ? {
            groupSize: 32,
            bits: 4,
            mode: "affine" as const,
            dtype: "float32" as const,
          }
        : {
            biases,
            groupSize: 32,
            bits: 4,
            mode: "affine" as const,
            dtype: "float32" as const,
          };
    using dense = dequantize(quantized, scales, {
      ...dequantizeOptions,
    });
    using denseWeight = transpose(dense);
    using expected = matmul(input, denseWeight);
    const matmulOptions =
      result.biases === undefined
        ? {
            groupSize: 32,
            bits: 4,
            mode: "affine" as const,
            transpose: true,
          }
        : {
            biases,
            groupSize: 32,
            bits: 4,
            mode: "affine" as const,
            transpose: true,
          };
    using actual = quantizedMatmul(input, quantized, scales, matmulOptions);

    mxEval(expected, actual);

    expectCloseLists(actual.toList(), expected.toList());
  });
});

function flattenNumbers(value: unknown): number[] {
  if (typeof value === "number") {
    return [value];
  }
  if (!Array.isArray(value)) {
    throw new Error("Expected nested numeric arrays");
  }
  return value.flatMap((entry) => flattenNumbers(entry));
}

function expectCloseLists(actual: unknown, expected: unknown): void {
  const actualValues = flattenNumbers(actual);
  const expectedValues = flattenNumbers(expected);
  expect(actualValues.length).toBe(expectedValues.length);
  for (let index = 0; index < actualValues.length; index += 1) {
    expect(actualValues[index]).toBeCloseTo(expectedValues[index] ?? 0, 1);
  }
}
