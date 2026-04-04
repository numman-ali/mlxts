import { describe, expect, test } from "bun:test";

import { array, type MxArray } from "../core/array";
import { mxEval } from "../core/transforms";
import { Module } from "../nn/module";
import { SGD } from "./sgd";

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
// Basic SGD step
// ---------------------------------------------------------------------------

describe("SGD", () => {
  test("basic step: param -= lr * grad", () => {
    const model = new TestModel([10]);
    const optimizer = new SGD({ learningRate: 0.1 });

    const grads = { weight: array([2]) };
    optimizer.update(model, grads);
    mxEval(model.weight);

    // 10 - 0.1 * 2 = 9.8
    const values = model.weight.toList() as number[];
    expect(values[0]).toBeCloseTo(9.8, 5);

    model[Symbol.dispose]();
    optimizer[Symbol.dispose]();
    grads.weight.free();
  });

  // ---------------------------------------------------------------------------
  // Momentum accumulation
  // ---------------------------------------------------------------------------

  test("momentum accumulates velocity across steps", () => {
    const model = new TestModel([10]);
    const optimizer = new SGD({ learningRate: 0.1, momentum: 0.9 });

    // Step 1: v = 0.9*0 + 2 = 2; param = 10 - 0.1*2 = 9.8
    const grads1 = { weight: array([2]) };
    optimizer.update(model, grads1);
    mxEval(model.weight);

    const afterStep1 = model.weight.toList() as number[];
    expect(afterStep1[0]).toBeCloseTo(9.8, 5);

    // Step 2: v = 0.9*2 + 2 = 3.8; param = 9.8 - 0.1*3.8 = 9.42
    const grads2 = { weight: array([2]) };
    optimizer.update(model, grads2);
    mxEval(model.weight);

    const afterStep2 = model.weight.toList() as number[];
    expect(afterStep2[0]).toBeCloseTo(9.42, 5);

    model[Symbol.dispose]();
    optimizer[Symbol.dispose]();
    grads1.weight.free();
    grads2.weight.free();
  });

  // ---------------------------------------------------------------------------
  // Weight decay
  // ---------------------------------------------------------------------------

  test("weight decay adds L2 penalty to gradient", () => {
    const model = new TestModel([10]);
    const optimizer = new SGD({ learningRate: 0.1, weightDecay: 0.1 });

    // effectiveGrad = 1 + 0.1 * 10 = 2
    // param = 10 - 0.1 * 2 = 9.8
    const grads = { weight: array([1]) };
    optimizer.update(model, grads);
    mxEval(model.weight);

    const values = model.weight.toList() as number[];
    expect(values[0]).toBeCloseTo(9.8, 5);

    model[Symbol.dispose]();
    optimizer[Symbol.dispose]();
    grads.weight.free();
  });

  // ---------------------------------------------------------------------------
  // Gradient path mismatch
  // ---------------------------------------------------------------------------

  test("throws on gradient path mismatch", () => {
    const model = new TestModel([10]);
    const optimizer = new SGD({ learningRate: 0.1 });

    const badGrads = { bogus: array([1]) };
    expect(() => optimizer.update(model, badGrads)).toThrow('no gradient for parameter "weight"');

    model[Symbol.dispose]();
    optimizer[Symbol.dispose]();
    badGrads.bogus.free();
  });

  // ---------------------------------------------------------------------------
  // Dispose clears state
  // ---------------------------------------------------------------------------

  test("dispose clears optimizer state", () => {
    const model = new TestModel([10]);
    const optimizer = new SGD({ learningRate: 0.1, momentum: 0.9 });

    const grads = { weight: array([2]) };
    optimizer.update(model, grads);
    mxEval(model.weight);

    // State should have entries after a momentum step
    expect(optimizer.stateArrays().length).toBeGreaterThan(0);

    optimizer[Symbol.dispose]();

    // State should be cleared
    expect(optimizer.stateArrays().length).toBe(0);

    model[Symbol.dispose]();
    grads.weight.free();
  });
});
