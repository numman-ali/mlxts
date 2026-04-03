import { describe, expect, test } from "bun:test";

import { array, MxArray } from "../core/array";
import { multiply, square, subtract } from "../core/ops/arithmetic";
import { mean, sum } from "../core/ops/reduction";
import { grad, mxEval } from "../core/transforms";
import { LayerNorm } from "./layer-norm";

describe("LayerNorm", () => {
  test("output has same shape as input", () => {
    using ln = new LayerNorm(4);
    const x = MxArray.fromData([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
    const out = ln.forward(x);
    mxEval(out);

    expect(out.shape).toEqual([2, 4]);

    x.free();
    out.free();
  });

  test("normalized output has mean near 0 and variance near 1 per row", () => {
    using ln = new LayerNorm(4);
    const x = MxArray.fromData([10, 20, 30, 40, 1, 2, 3, 4], [2, 4]);
    const out = ln.forward(x);

    // Check mean along last axis is close to 0
    const rowMeans = mean(out, -1);
    mxEval(rowMeans);
    const meanValues = rowMeans.toList() as number[];
    for (const m of meanValues) {
      expect(m).toBeCloseTo(0, 4);
    }

    // Check variance along last axis is close to 1
    // var = mean((x - mean(x))^2)
    using rowMeansKept = mean(out, -1, true);
    using centered = subtract(out, rowMeansKept);
    using centeredSq = square(centered);
    const rowVars = mean(centeredSq, -1);
    mxEval(rowVars);
    const varValues = rowVars.toList() as number[];
    for (const v of varValues) {
      expect(v).toBeCloseTo(1, 1);
    }

    x.free();
    out.free();
    rowMeans.free();
    rowVars.free();
  });

  test("weight and bias are in parameters", () => {
    using ln = new LayerNorm(4);
    const params = ln.parameters();

    expect(Object.keys(params)).toContain("weight");
    expect(Object.keys(params)).toContain("bias");
    expect(params.weight).toBe(ln.weight);
    expect(params.bias).toBe(ln.bias);
  });

  test("invalid dims throws", () => {
    expect(() => new LayerNorm(0)).toThrow();
  });

  test("forward validates the last dimension", () => {
    using ln = new LayerNorm(2);
    const x = MxArray.fromData([1, 2, 3], [3]);

    expect(() => ln.forward(x)).toThrow("expected last dimension 2");

    x.free();
  });

  test("gradient flows through layer norm", () => {
    using ln = new LayerNorm(4);
    const x = MxArray.fromData([1, 10, 100, 1000], [1, 4]);
    // Use a weighted sum so the gradient is not trivially zero.
    // (Plain sum of a normalized vector is ~0, so its gradient is ~0 too.)
    const weights = array([1, 0, 0, 0]);

    const lossFn = (input: MxArray) => {
      const out = ln.forward(input);
      using weighted = multiply(out, weights);
      return sum(weighted);
    };

    const gradFn = grad(lossFn);
    const g = gradFn(x);
    mxEval(g);

    expect(g.shape).toEqual([1, 4]);

    // Verify at least one gradient element is non-zero
    const values = (g.toList() as number[][]).flat();
    const hasNonZero = values.some((v) => Math.abs(v) > 1e-10);
    expect(hasNonZero).toBe(true);

    x.free();
    g.free();
    weights.free();
  });
});
