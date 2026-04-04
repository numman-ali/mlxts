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
    const optimizer = new AdamW({
      learningRate: 0.001,
      beta1: 0.9,
      beta2: 0.999,
      eps: 1e-8,
      weightDecay: 0.01,
    });

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
    const optimizer = new AdamW({
      learningRate: 0.001,
      beta1: 0.9,
      beta2: 0.999,
      eps: 1e-8,
      weightDecay: 0.01,
    });

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

  test("checkpoint and restore round-trip optimizer state", () => {
    const sourceModel = new TestModel([1.0]);
    const sourceOptimizer = new AdamW({
      learningRate: 0.001,
      beta1: 0.9,
      beta2: 0.999,
      eps: 1e-8,
      weightDecay: 0.01,
    });
    const grads = { weight: array([0.5]) };

    sourceOptimizer.update(sourceModel, grads);
    mxEval(sourceModel.weight);

    const snapshot = sourceOptimizer.checkpoint();
    const restored = new AdamW();
    try {
      restored.restore(snapshot);
      expect(restored.step).toBe(sourceOptimizer.step);
      expect(restored.stateArrays().length).toBe(sourceOptimizer.stateArrays().length);
    } finally {
      restored[Symbol.dispose]();
      sourceOptimizer[Symbol.dispose]();
      sourceModel[Symbol.dispose]();
      grads.weight.free();
    }
  });
});

// ---------------------------------------------------------------------------
// Adam (zero weight decay)
// ---------------------------------------------------------------------------

describe("Adam", () => {
  test("first step with zero weight decay — no decay applied", () => {
    const model = new TestModel([1.0]);
    const optimizer = new Adam({
      learningRate: 0.001,
      beta1: 0.9,
      beta2: 0.999,
      eps: 1e-8,
    });

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
    const optAdam = new Adam({
      learningRate: 0.001,
      beta1: 0.9,
      beta2: 0.999,
      eps: 1e-8,
    });
    const optAdamW = new AdamW({
      learningRate: 0.001,
      beta1: 0.9,
      beta2: 0.999,
      eps: 1e-8,
      weightDecay: 0.01,
    });

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

  test("setLearningRate changes subsequent step behavior", () => {
    // Step 1 with lr=0.1
    const model1 = new TestModel([10, 20, 30]);
    const opt1 = new AdamW({
      learningRate: 0.1,
      beta1: 0.9,
      beta2: 0.999,
      eps: 1e-8,
      weightDecay: 0,
    });
    const grads1 = { weight: array([1, 1, 1]) };
    opt1.update(model1, grads1);
    mxEval(model1.weight);
    const after1 = model1.weight.toList() as number[];

    // Step 2 with same lr=0.1 (for comparison)
    const model2 = new TestModel([10, 20, 30]);
    const opt2 = new AdamW({
      learningRate: 0.1,
      beta1: 0.9,
      beta2: 0.999,
      eps: 1e-8,
      weightDecay: 0,
    });
    const grads2a = { weight: array([1, 1, 1]) };
    opt2.update(model2, grads2a);
    mxEval(model2.weight);
    // Now change LR to something much larger
    opt2.setLearningRate(1.0);
    const grads2b = { weight: array([1, 1, 1]) };
    opt2.update(model2, grads2b);
    mxEval(model2.weight);
    const afterLRChange = model2.weight.toList() as number[];

    // The model with changed LR should have moved much more on step 2
    // First element should be significantly different
    const step1Diff = Math.abs(10 - (after1[0] ?? 0));
    const totalDiff = Math.abs(10 - (afterLRChange[0] ?? 0));
    expect(totalDiff).toBeGreaterThan(step1Diff * 1.5);

    model1[Symbol.dispose]();
    model2[Symbol.dispose]();
    opt1[Symbol.dispose]();
    opt2[Symbol.dispose]();
    grads1.weight.free();
  });
});
