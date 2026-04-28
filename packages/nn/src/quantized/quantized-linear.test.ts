import { describe, expect, test } from "bun:test";
import { array, concatenate, mxEval } from "@mlxts/core";
import { Linear } from "../layers/linear";
import { fuseQuantizedLinears, QuantizedLinear } from "./quantized-linear";

describe("QuantizedLinear", () => {
  test("forward approximates dense linear output", () => {
    const linear = new Linear(64, 4, true);
    linear.weight.free();
    linear.weight = array(
      Array.from({ length: 4 }, (_, row) =>
        Array.from({ length: 64 }, (_, column) => (row * 64 + column - 32) / 32),
      ),
      "float32",
    );
    if (linear.bias !== null) {
      linear.bias.free();
      linear.bias = array([0.25, -0.5, 0.75, 1], "float32");
    }

    using input = array([Array.from({ length: 64 }, (_, index) => index / 16)], "float32");
    using dense = linear.forward(input);
    const quantized = QuantizedLinear.fromLinear(linear, {
      bits: 4,
      groupSize: 32,
      mode: "affine",
    });
    using actual = quantized.forward(input);

    mxEval(dense, actual);

    const denseValues = dense.toList() as number[][];
    const actualValues = actual.toList() as number[][];
    expect(actualValues[0]?.length).toBe(denseValues[0]?.length);
    let absoluteError = 0;
    for (let index = 0; index < (denseValues[0]?.length ?? 0); index += 1) {
      absoluteError += Math.abs((actualValues[0]?.[index] ?? 0) - (denseValues[0]?.[index] ?? 0));
    }
    expect(absoluteError / (denseValues[0]?.length ?? 1)).toBeLessThan(0.3);

    quantized[Symbol.dispose]();
    linear[Symbol.dispose]();
  });

  test("toLinear preserves output bias", () => {
    const linear = new Linear(64, 2, true);
    const quantized = QuantizedLinear.fromLinear(linear, {
      bits: 4,
      groupSize: 32,
      mode: "affine",
    });
    const dense = quantized.toLinear();

    expect(dense.bias?.shape).toEqual(linear.bias?.shape);

    dense[Symbol.dispose]();
    quantized[Symbol.dispose]();
    linear[Symbol.dispose]();
  });

  test("fuseQuantizedLinears concatenates compatible projection outputs", () => {
    const left = new Linear(32, 3, false);
    left.weight.free();
    left.weight = array(
      Array.from({ length: 3 }, (_, row) =>
        Array.from({ length: 32 }, (_, column) => (row * 32 + column + 1) / 256),
      ),
    );
    const right = new Linear(32, 2, false);
    right.weight.free();
    right.weight = array(
      Array.from({ length: 2 }, (_, row) =>
        Array.from({ length: 32 }, (_, column) => (row * 32 + column + 9) / 192),
      ),
    );

    const quantizedLeft = QuantizedLinear.fromLinear(left, { groupSize: 32, bits: 4 });
    const quantizedRight = QuantizedLinear.fromLinear(right, { groupSize: 32, bits: 4 });
    const fused = fuseQuantizedLinears([quantizedLeft, quantizedRight]);
    if (fused === null) {
      throw new Error("Expected compatible quantized projections to fuse.");
    }

    using input = array([
      Array.from({ length: 32 }, (_, index) => (index + 1) / 32),
      Array.from({ length: 32 }, (_, index) => (32 - index) / 32),
    ]);
    using leftOut = quantizedLeft.forward(input);
    using rightOut = quantizedRight.forward(input);
    using expected = concatenate([leftOut, rightOut], 1);
    using actual = fused.forward(input);
    mxEval(expected, actual);

    expect(actual.shape).toEqual([2, 5]);
    const expectedValues = expected.toList() as number[][];
    const actualValues = actual.toList() as number[][];
    for (let row = 0; row < expectedValues.length; row += 1) {
      const expectedRow = expectedValues[row] ?? [];
      const actualRow = actualValues[row] ?? [];
      for (let column = 0; column < expectedRow.length; column += 1) {
        expect(actualRow[column]).toBeCloseTo(expectedRow[column] ?? 0, 4);
      }
    }

    fused[Symbol.dispose]();
    quantizedLeft[Symbol.dispose]();
    quantizedRight[Symbol.dispose]();
    left[Symbol.dispose]();
    right[Symbol.dispose]();
  });
});
