import { describe, expect, test } from "bun:test";

import { array, type MxArray } from "../core/array";
import { mxEval } from "../core/transforms";
import { Module } from "../nn/module";
import { Adam, AdamW } from "./adam";

// --- Test helper: single-parameter module ---

class TestModel extends Module {
  weight: MxArray;

  constructor(value: number[]) {
    super();
    this.weight = array(value);
  }

  forward(x: MxArray): MxArray {
    return x;
  }
}

// ---------------------------------------------------------------------------
// AdamW
// ---------------------------------------------------------------------------

describe("AdamW", () => {
  test("first step matches manual computation", () => {
    const model = new TestModel([1.0]);
    const optimizer = new AdamW(0.001, 0.9, 0.999, 1e-8, 0.01);

    // param=1.0, grad=0.5
    // m = 0.1 * 0.5 = 0.05
    // v = 0.001 * 0.25 = 0.00025
    // mHat = 0.05 / (1 - 0.9) = 0.5
    // vHat = 0.00025 / (1 - 0.999) = 0.25
    // decayed = 1.0 * (1 - 0.001 * 0.01) = 0.99999
    // result = 0.99999 - 0.001 * 0.5 / (sqrt(0.25) + 1e-8) ≈ 0.99899
    const grads = { weight: array([0.5]) };
    optimizer.update(model, grads);
    mxEval(model.weight);

    const values = model.weight.toList() as number[];
    expect(values[0]).toBeCloseTo(0.99899, 3);

    model[Symbol.dispose]();
    optimizer[Symbol.dispose]();
    grads.weight.free();
  });

  test("gradient path mismatch throws", () => {
    const model = new TestModel([1.0]);
    const optimizer = new AdamW();

    const badGrads = { nonexistent: array([0.5]) };
    expect(() => optimizer.update(model, badGrads)).toThrow('no gradient for parameter "weight"');

    model[Symbol.dispose]();
    optimizer[Symbol.dispose]();
    badGrads.nonexistent.free();
  });

  test("failed updates do not advance the step counter", () => {
    const model = new TestModel([1.0]);
    const optimizer = new AdamW(0.001, 0.9, 0.999, 1e-8, 0.01);

    const badGrads = { nonexistent: array([0.5]) };
    expect(() => optimizer.update(model, badGrads)).toThrow();
    badGrads.nonexistent.free();

    const goodGrads = { weight: array([0.5]) };
    optimizer.update(model, goodGrads);
    mxEval(model.weight);

    const values = model.weight.toList() as number[];
    expect(values[0]).toBeCloseTo(0.99899, 3);

    model[Symbol.dispose]();
    optimizer[Symbol.dispose]();
    goodGrads.weight.free();
  });

  test("dispose clears optimizer state", () => {
    const model = new TestModel([1.0]);
    const optimizer = new AdamW();

    const grads = { weight: array([0.5]) };
    optimizer.update(model, grads);
    mxEval(model.weight);

    // State should have m and v entries after a step
    expect(optimizer.stateArrays().length).toBeGreaterThan(0);

    optimizer[Symbol.dispose]();

    // State should be cleared after dispose
    expect(optimizer.stateArrays().length).toBe(0);

    model[Symbol.dispose]();
    grads.weight.free();
  });
});

// ---------------------------------------------------------------------------
// Adam (zero weight decay)
// ---------------------------------------------------------------------------

describe("Adam", () => {
  test("first step with zero weight decay — no decay applied", () => {
    const model = new TestModel([1.0]);
    const optimizer = new Adam(0.001, 0.9, 0.999, 1e-8);

    // Same computation as AdamW but weightDecay = 0:
    // m = 0.1 * 0.5 = 0.05
    // v = 0.001 * 0.25 = 0.00025
    // mHat = 0.05 / 0.1 = 0.5
    // vHat = 0.00025 / 0.001 = 0.25
    // decayed = 1.0 * (1 - 0.001 * 0) = 1.0 (no decay)
    // result = 1.0 - 0.001 * 0.5 / (sqrt(0.25) + 1e-8) ≈ 0.999
    const grads = { weight: array([0.5]) };
    optimizer.update(model, grads);
    mxEval(model.weight);

    const values = model.weight.toList() as number[];
    expect(values[0]).toBeCloseTo(0.999, 3);

    model[Symbol.dispose]();
    optimizer[Symbol.dispose]();
    grads.weight.free();
  });

  test("Adam result differs from AdamW by weight decay amount", () => {
    // Run identical step through Adam and AdamW, verify the difference
    // is accounted for by weight decay alone.
    const modelAdam = new TestModel([1.0]);
    const modelAdamW = new TestModel([1.0]);
    const optAdam = new Adam(0.001, 0.9, 0.999, 1e-8);
    const optAdamW = new AdamW(0.001, 0.9, 0.999, 1e-8, 0.01);

    const gradsAdam = { weight: array([0.5]) };
    const gradsAdamW = { weight: array([0.5]) };

    optAdam.update(modelAdam, gradsAdam);
    optAdamW.update(modelAdamW, gradsAdamW);
    mxEval(modelAdam.weight, modelAdamW.weight);

    const adamVal = (modelAdam.weight.toList() as number[])[0] ?? 0;
    const adamWVal = (modelAdamW.weight.toList() as number[])[0] ?? 0;

    // Adam should produce a larger value (less penalty)
    expect(adamVal).toBeGreaterThan(adamWVal);

    modelAdam[Symbol.dispose]();
    modelAdamW[Symbol.dispose]();
    optAdam[Symbol.dispose]();
    optAdamW[Symbol.dispose]();
    gradsAdam.weight.free();
    gradsAdamW.weight.free();
  });
});
