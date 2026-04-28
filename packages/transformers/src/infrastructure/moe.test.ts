import { describe, expect, test } from "bun:test";

import { array, dequantize, type MxArray, multiply, mxEval, quantize } from "@mlxts/core";
import {
  PackedSwitchGLUExperts,
  SwitchGLUExperts,
  type SwitchLinear,
  topKFromRouterProbabilities,
} from "./moe";

function replaceWeight(projection: SwitchLinear, weight: MxArray): void {
  projection.weight.free();
  projection.weight = weight;
}

function replaceQuantizedWeight(projection: SwitchLinear, source: MxArray): MxArray {
  const result = quantize(source, {
    groupSize: 32,
    bits: 4,
    mode: "affine",
  });
  const dense =
    result.biases === undefined
      ? dequantize(result.weight, result.scales, {
          groupSize: 32,
          bits: 4,
          mode: "affine",
          dtype: "float32",
        })
      : dequantize(result.weight, result.scales, {
          biases: result.biases,
          groupSize: 32,
          bits: 4,
          mode: "affine",
          dtype: "float32",
        });

  projection.prepareQuantized({ groupSize: 32, bits: 4, mode: "affine" });
  projection.weight.free();
  projection.scales?.free();
  projection.biases?.free();
  projection.weight = result.weight;
  projection.scales = result.scales;
  projection.biases = result.biases ?? null;
  return dense;
}

function flattenNumbers(value: unknown): number[] {
  if (typeof value === "number") {
    return [value];
  }
  if (!Array.isArray(value)) {
    throw new Error("Expected nested numeric arrays.");
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

describe("MoE infrastructure", () => {
  test("selects and normalizes top-k router probabilities", () => {
    using probabilities = array(
      [
        [0.1, 0.7, 0.2],
        [0.6, 0.3, 0.1],
      ],
      "float32",
    );
    const routing = topKFromRouterProbabilities(probabilities, 2, 3);
    try {
      expect(routing.indices.shape).toEqual([2, 2]);
      expect(routing.weights.shape).toEqual([2, 2]);
      const weights = routing.weights.toList() as number[][];
      const first = weights[0];
      const second = weights[1];
      if (first === undefined || second === undefined) {
        throw new Error("Expected two routed rows.");
      }
      expect((first[0] ?? 0) + (first[1] ?? 0)).toBeCloseTo(1);
      expect((second[0] ?? 0) + (second[1] ?? 0)).toBeCloseTo(1);
    } finally {
      routing.indices.free();
      routing.weights.free();
    }
  });

  test("runs packed SwitchGLU experts with repeated expert assignments", () => {
    using experts = new PackedSwitchGLUExperts(2, 1, 1, multiply);
    experts.gateUpProjection.free();
    experts.downProjection.free();
    experts.gateUpProjection = array(
      [
        [[1], [1]],
        [[2], [1]],
      ],
      "float32",
    );
    experts.downProjection = array([[[1]], [[1]]], "float32");

    using hidden = array([[2], [3]], "float32");
    using indices = array(
      [
        [0, 1],
        [1, 0],
      ],
      "int32",
    );
    using weights = array(
      [
        [0.25, 0.75],
        [1, 0],
      ],
      "float32",
    );
    using output = experts.forward(hidden, indices, weights);

    expect(output.toList()).toEqual([[7], [18]]);
  });

  test("split SwitchGLU experts match packed gate-up expert math", () => {
    using packed = new PackedSwitchGLUExperts(2, 1, 1, multiply);
    using split = new SwitchGLUExperts(2, 1, 1, multiply);
    packed.gateUpProjection.free();
    packed.downProjection.free();
    replaceWeight(split.gateProjection, array([[[1]], [[2]]], "float32"));
    replaceWeight(split.upProjection, array([[[1]], [[1]]], "float32"));
    replaceWeight(split.downProjection, array([[[1]], [[1]]], "float32"));
    packed.gateUpProjection = array(
      [
        [[1], [1]],
        [[2], [1]],
      ],
      "float32",
    );
    packed.downProjection = array([[[1]], [[1]]], "float32");

    using hidden = array([[2], [3]], "float32");
    using indices = array(
      [
        [0, 1],
        [1, 0],
      ],
      "int32",
    );
    using weights = array(
      [
        [0.25, 0.75],
        [1, 0],
      ],
      "float32",
    );
    using packedOutput = packed.forward(hidden, indices, weights);
    using splitOutput = split.forward(hidden, indices, weights);

    expect(splitOutput.toList()).toEqual(packedOutput.toList());
  });

  test("quantized split SwitchGLU experts match dequantized switch weights", () => {
    using quantized = new SwitchGLUExperts(2, 32, 32, multiply);
    using dense = new SwitchGLUExperts(2, 32, 32, multiply);
    using gateSource = array(
      Array.from({ length: 2 }, (_, expert) =>
        Array.from({ length: 32 }, (_, row) =>
          Array.from(
            { length: 32 },
            (_, column) => (expert * 1024 + row * 32 + column - 400) / 2048,
          ),
        ),
      ),
      "float32",
    );
    using upSource = array(
      Array.from({ length: 2 }, (_, expert) =>
        Array.from({ length: 32 }, (_, row) =>
          Array.from(
            { length: 32 },
            (_, column) => (300 - expert * 1024 - row * 32 - column) / 2048,
          ),
        ),
      ),
      "float32",
    );
    using downSource = array(
      Array.from({ length: 2 }, (_, expert) =>
        Array.from({ length: 32 }, (_, row) =>
          Array.from({ length: 32 }, (_, column) => (expert * 512 + row * 16 - column) / 2048),
        ),
      ),
      "float32",
    );
    replaceWeight(
      dense.gateProjection,
      replaceQuantizedWeight(quantized.gateProjection, gateSource),
    );
    replaceWeight(dense.upProjection, replaceQuantizedWeight(quantized.upProjection, upSource));
    replaceWeight(
      dense.downProjection,
      replaceQuantizedWeight(quantized.downProjection, downSource),
    );

    using hidden = array(
      [
        Array.from({ length: 32 }, (_, index) => (index + 1) / 32),
        Array.from({ length: 32 }, (_, index) => (32 - index) / 40),
      ],
      "float32",
    );
    using indices = array(
      [
        [1, 0],
        [0, 1],
      ],
      "int32",
    );
    using weights = array(
      [
        [0.6, 0.4],
        [0.25, 0.75],
      ],
      "float32",
    );
    using expected = dense.forward(hidden, indices, weights);
    using actual = quantized.forward(hidden, indices, weights);

    mxEval(expected, actual);

    expectCloseLists(actual.toList(), expected.toList());
  });
});
