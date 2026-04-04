import { describe, expect, test } from "bun:test";
import { array, mxEval, zeros } from "@mlxts/core";
import { crossEntropy, mse } from "./losses";

describe("crossEntropy", () => {
  test("basic correctness — single sample", () => {
    const logits = array([[2.0, 1.0, 0.1]]);
    const targets = array([0], "int32");
    const loss = crossEntropy(logits, targets);
    mxEval(loss);

    // Manual: logsumexp([2.0, 1.0, 0.1]) = log(e^2 + e^1 + e^0.1) ≈ 2.417
    // logprob[0] = 2.0 - 2.417 ≈ -0.417
    // loss = -(-0.417) ≈ 0.417
    expect(loss.item()).toBeCloseTo(0.417, 2);

    logits.free();
    targets.free();
    loss.free();
  });

  test("batch — two samples", () => {
    const logits = array([
      [2.0, 1.0, 0.1],
      [0.1, 1.0, 2.0],
    ]);
    const targets = array([0, 2], "int32");
    const loss = crossEntropy(logits, targets);
    mxEval(loss);

    // Both samples pick the highest-logit class, so loss should be
    // relatively small (symmetric setup, both ≈ 0.417).
    // Mean of two identical losses ≈ 0.417
    expect(loss.item()).toBeCloseTo(0.417, 2);

    logits.free();
    targets.free();
    loss.free();
  });

  test("numerical stability — large logits", () => {
    const logits = array([[1000.0, 1001.0, 999.0]]);
    const targets = array([1], "int32");
    const loss = crossEntropy(logits, targets);
    mxEval(loss);

    const value = loss.item();
    expect(Number.isFinite(value)).toBe(true);
    // logits[1]=1001 is the largest, so loss for target=1 should be small
    expect(value).toBeGreaterThanOrEqual(0);
    expect(value).toBeLessThan(5);

    logits.free();
    targets.free();
    loss.free();
  });

  test("float targets throw with descriptive message", () => {
    const logits = array([[1.0, 2.0, 3.0]]);
    const targets = array([0.0, 1.0, 2.0]);

    expect(() => crossEntropy(logits, targets)).toThrow("integer dtype");

    logits.free();
    targets.free();
  });

  test("shape mismatch throws before broadcasting can hide the bug", () => {
    const logits = array([
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
    ]);
    const targets = array([0], "int32");

    expect(() => crossEntropy(logits, targets)).toThrow("targets shape must match");

    logits.free();
    targets.free();
  });

  test("rank mismatch throws a clear error", () => {
    const logits = array([[1.0, 2.0, 3.0]]);
    const targets = array([[0]], "int32");

    expect(() => crossEntropy(logits, targets)).toThrow("targets rank must be logits rank - 1");

    logits.free();
    targets.free();
  });

  test("empty class axis throws a clear error", () => {
    const logits = zeros([2, 0]);
    const targets = array([0, 0], "int32");

    expect(() => crossEntropy(logits, targets)).toThrow("class axis must have size > 0");

    logits.free();
    targets.free();
  });
});

describe("mse", () => {
  test("zero loss when predictions match targets", () => {
    const predictions = array([1.0, 2.0, 3.0]);
    const targets = array([1.0, 2.0, 3.0]);
    const loss = mse(predictions, targets);
    mxEval(loss);

    expect(loss.item()).toBeCloseTo(0.0, 5);

    predictions.free();
    targets.free();
    loss.free();
  });

  test("non-zero loss — constant offset of 1", () => {
    const predictions = array([1.0, 2.0, 3.0]);
    const targets = array([2.0, 3.0, 4.0]);
    const loss = mse(predictions, targets);
    mxEval(loss);

    // Each diff = -1, squared = 1, mean = 1.0
    expect(loss.item()).toBeCloseTo(1.0, 5);

    predictions.free();
    targets.free();
    loss.free();
  });

  test("result is scalar", () => {
    const predictions = array([1.0, 2.0, 3.0]);
    const targets = array([4.0, 5.0, 6.0]);
    const loss = mse(predictions, targets);
    mxEval(loss);

    expect(loss.ndim).toBe(0);
    expect(loss.shape).toEqual([]);

    predictions.free();
    targets.free();
    loss.free();
  });
});
