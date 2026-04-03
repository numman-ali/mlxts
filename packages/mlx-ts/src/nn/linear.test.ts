import { describe, expect, test } from "bun:test";

import { MxArray } from "../core/array";
import { sum } from "../core/ops/reduction";
import { grad, mxEval } from "../core/transforms";
import { Linear } from "./linear";

describe("Linear", () => {
  test("forward output has correct shape", () => {
    using linear = new Linear(4, 8);
    const x = MxArray.fromData([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
    const out = linear.forward(x);
    mxEval(out);

    expect(out.shape).toEqual([2, 8]);

    x.free();
    out.free();
  });

  test("forward without bias", () => {
    using linear = new Linear(3, 5, false);
    const x = MxArray.fromData([1, 2, 3, 4, 5, 6], [2, 3]);
    const out = linear.forward(x);
    mxEval(out);

    expect(out.shape).toEqual([2, 5]);
    expect(linear.bias).toBeNull();

    x.free();
    out.free();
  });

  test("parameters include weight and bias", () => {
    using linear = new Linear(3, 5);
    const params = linear.parameters();

    expect(Object.keys(params)).toContain("weight");
    expect(Object.keys(params)).toContain("bias");
    expect(params.weight).toBe(linear.weight);
  });

  test("parameters exclude bias when hasBias=false", () => {
    using linear = new Linear(3, 5, false);
    const params = linear.parameters();

    expect(Object.keys(params)).toContain("weight");
    expect(Object.keys(params)).not.toContain("bias");
  });

  test("invalid dims throw", () => {
    expect(() => new Linear(0, 5)).toThrow();
    expect(() => new Linear(5, -1)).toThrow();
  });

  test("forward validates the input feature dimension", () => {
    using linear = new Linear(2, 3);
    const x = MxArray.fromData([1, 2, 3], [3]);

    expect(() => linear.forward(x)).toThrow("expected input last dimension 2");

    x.free();
  });

  test("gradient flows through linear", () => {
    using linear = new Linear(4, 3);
    const x = MxArray.fromData([1, 2, 3, 4], [1, 4]);

    const lossFn = (input: MxArray) => sum(linear.forward(input));
    const gradFn = grad(lossFn);
    const g = gradFn(x);
    mxEval(g);

    expect(g.shape).toEqual([1, 4]);

    x.free();
    g.free();
  });
});
