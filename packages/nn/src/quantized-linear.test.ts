import { describe, expect, test } from "bun:test";
import { array, mxEval } from "@mlxts/core";
import { Linear } from "./linear";
import { QuantizedLinear } from "./quantized-linear";

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
});
