import { describe, expect, test } from "bun:test";

import { array, geluApprox, multiply, mxEval } from "@mlxts/core";

import { gegluApprox } from "./index";

describe("transformer gated activations", () => {
  test("GEGLU gating matches the eager helper output", () => {
    using gate = array([[[1, -1, 0.5]]], "float32");
    using value = array([[[2, 3, 4]]], "float32");
    using output = gegluApprox(gate, value);
    using eagerGelu = geluApprox(gate);
    using eagerOutput = multiply(eagerGelu, value);

    mxEval(output, eagerOutput);

    expect(output.toList()).toEqual(eagerOutput.toList());
  });

  test("GEGLU gating can be reused across shapes", () => {
    using firstGate = array([[[1, 2]]], "float32");
    using firstValue = array([[[3, 4]]], "float32");
    using secondGate = array([[[[1, 2, 3]]]], "float32");
    using secondValue = array([[[[4, 5, 6]]]], "float32");
    using firstOutput = gegluApprox(firstGate, firstValue);
    using secondOutput = gegluApprox(secondGate, secondValue);

    mxEval(firstOutput, secondOutput);

    expect(firstOutput.shape).toEqual([1, 1, 2]);
    expect(secondOutput.shape).toEqual([1, 1, 1, 3]);
  });
});
