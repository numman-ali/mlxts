import { describe, expect, test } from "bun:test";

import { array } from "./array";
import { expandDims, gatherMm, matmul, transpose } from "./ops";
import { dequantize, gatherQmm, quantize, quantizedMatmul } from "./quantization";
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
  test("affine quantize preserves the source floating dtype for scales and biases", () => {
    for (const dtype of ["float16", "bfloat16", "float32"] as const) {
      using weight = array(
        Array.from({ length: 2 }, (_, row) =>
          Array.from({ length: 64 }, (_, column) => (row * 64 + column - 32) / 16),
        ),
        dtype,
      );

      const result = quantize(weight, {
        groupSize: 32,
        bits: 4,
        mode: "affine",
      });
      using quantized = result.weight;
      using quantizedScales = result.scales;
      using quantizedBiases = result.biases ?? array([0], dtype);

      expect(quantized.dtype).toBe("uint32");
      expect(quantizedScales.dtype).toBe(dtype);
      expect(quantizedBiases.dtype).toBe(dtype);
    }
  });

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
    expect(scales.dtype).toBe("uint8");
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

  test("gatherQmm matches dequantized gatherMm with repeated expert indices", () => {
    using inputRows = array(
      [
        Array.from({ length: 32 }, (_, index) => (index + 1) / 16),
        Array.from({ length: 32 }, (_, index) => (32 - index) / 20),
      ],
      "float32",
    );
    using inputWithTopK = expandDims(inputRows, 1);
    using input = expandDims(inputWithTopK, 2);
    using weight = array(
      Array.from({ length: 2 }, (_, expert) =>
        Array.from({ length: 3 }, (_, row) =>
          Array.from({ length: 32 }, (_, column) => (expert * 96 + row * 32 + column - 40) / 32),
        ),
      ),
      "float32",
    );
    const result = quantize(weight, {
      groupSize: 32,
      bits: 4,
      mode: "affine",
    });
    using quantized = result.weight;
    using scales = result.scales;
    using indices = array(
      [
        [1, 0],
        [0, 1],
      ],
      "int32",
    );
    using dense =
      result.biases === undefined
        ? dequantize(quantized, scales, {
            groupSize: 32,
            bits: 4,
            mode: "affine",
            dtype: "float32",
          })
        : dequantize(quantized, scales, {
            biases: result.biases,
            groupSize: 32,
            bits: 4,
            mode: "affine",
            dtype: "float32",
          });
    using denseTransposed = transpose(dense, [0, 2, 1]);
    using expected = gatherMm(input, denseTransposed, { rhsIndices: indices });
    using actual = gatherQmm(input, quantized, scales, {
      ...(result.biases === undefined ? {} : { biases: result.biases }),
      rhsIndices: indices,
      transpose: true,
      groupSize: 32,
      bits: 4,
      mode: "affine",
    });

    mxEval(expected, actual);

    expectCloseLists(actual.toList(), expected.toList());
    result.biases?.free();
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
