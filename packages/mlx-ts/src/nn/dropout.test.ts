import { describe, expect, test } from "bun:test";

import { array, ones } from "../core/array";
import * as random from "../core/random";
import { mxEval } from "../core/transforms";
import { Dropout } from "./dropout";

describe("Dropout", () => {
  test("eval mode returns same values (identity)", () => {
    using dropout = new Dropout(0.5);
    dropout.eval();

    const x = array([1, 2, 3, 4, 5]);
    const out = dropout.forward(x);
    mxEval(out);

    expect(out.toList()).toEqual([1, 2, 3, 4, 5]);
    expect(out).not.toBe(x);

    out.free();
    expect(x.isDisposed).toBe(false);

    x.free();
  });

  test("training mode with p=0 returns same values", () => {
    using dropout = new Dropout(0.0);
    dropout.train();

    const x = array([1, 2, 3, 4, 5]);
    const out = dropout.forward(x);
    mxEval(out);

    expect(out.toList()).toEqual([1, 2, 3, 4, 5]);
    expect(out).not.toBe(x);

    out.free();
    expect(x.isDisposed).toBe(false);

    x.free();
  });

  test("training mode with p=0.5 zeros some elements", () => {
    random.seed(42);
    using dropout = new Dropout(0.5);
    dropout.train();

    // Large array of ones to make the statistical test reliable
    const x = ones([100]);
    const out = dropout.forward(x);
    mxEval(out);

    const values = out.toList() as number[];
    const zeroCount = values.filter((v) => v === 0).length;
    const nonZeroCount = values.filter((v) => v !== 0).length;

    // With p=0.5, roughly half should be zero and half non-zero.
    // Use a wide margin for stochastic test stability.
    expect(zeroCount).toBeGreaterThan(10);
    expect(nonZeroCount).toBeGreaterThan(10);

    x.free();
    out.free();
  });

  test("invalid p throws", () => {
    expect(() => new Dropout(-0.1)).toThrow();
    expect(() => new Dropout(1.0)).toThrow();
  });

  test("no parameters", () => {
    using dropout = new Dropout(0.5);
    const params = dropout.parameters();

    expect(Object.keys(params)).toEqual([]);
  });
});
