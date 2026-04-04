import { describe, expect, test } from "bun:test";
import { array, mxEval } from "@mlxts/core";
import { gelu, relu, silu } from "./activations";

describe("relu", () => {
  test("positive values pass through", () => {
    const x = array([1, 2, 3]);
    const result = relu(x);
    mxEval(result);
    expect(result.toList()).toEqual([1, 2, 3]);
    x.free();
    result.free();
  });

  test("negative values become zero", () => {
    const x = array([-1, -2, -3]);
    const result = relu(x);
    mxEval(result);
    expect(result.toList()).toEqual([0, 0, 0]);
    x.free();
    result.free();
  });

  test("mixed values", () => {
    const x = array([-2, -1, 0, 1, 2]);
    const result = relu(x);
    mxEval(result);
    expect(result.toList()).toEqual([0, 0, 0, 1, 2]);
    x.free();
    result.free();
  });
});

describe("gelu", () => {
  test("gelu(0) = 0", () => {
    const x = array([0.0]);
    const result = gelu(x);
    mxEval(result);
    expect(result.item()).toBeCloseTo(0, 5);
    x.free();
    result.free();
  });

  test("gelu is approximately relu for large positive", () => {
    const x = array([5.0]);
    const result = gelu(x);
    mxEval(result);
    // For large positive x, gelu(x) ≈ x
    expect(result.item()).toBeCloseTo(5.0, 2);
    x.free();
    result.free();
  });

  test("gelu of negative is small but not zero", () => {
    const x = array([-1.0]);
    const result = gelu(x);
    mxEval(result);
    // gelu(-1) ≈ -0.1587
    expect(result.item()).toBeCloseTo(-0.1587, 2);
    x.free();
    result.free();
  });
});

describe("silu", () => {
  test("silu(0) = 0", () => {
    const x = array([0.0]);
    const result = silu(x);
    mxEval(result);
    expect(result.item()).toBeCloseTo(0, 5);
    x.free();
    result.free();
  });

  test("silu is approximately identity for large positive", () => {
    const x = array([10.0]);
    const result = silu(x);
    mxEval(result);
    // For large positive x, sigmoid(x) ≈ 1, so silu(x) ≈ x
    expect(result.item()).toBeCloseTo(10.0, 2);
    x.free();
    result.free();
  });

  test("silu of negative is small negative", () => {
    const x = array([-1.0]);
    const result = silu(x);
    mxEval(result);
    // silu(-1) = -1 * sigmoid(-1) ≈ -0.2689
    expect(result.item()).toBeCloseTo(-0.2689, 2);
    x.free();
    result.free();
  });
});
