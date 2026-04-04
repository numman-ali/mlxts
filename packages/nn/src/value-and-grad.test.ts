import { describe, expect, test } from "bun:test";
import type { ParameterTree } from "@mlxts/core";
import {
  add,
  array,
  clearMemoryCache,
  getActiveMemoryBytes,
  MxArray,
  multiply,
  mxEval,
  sum,
  synchronize,
  treeFlatten,
} from "@mlxts/core";
import { Module } from "./module";
import { valueAndGrad } from "./value-and-grad";

// --- Test helpers ---

class SimpleModel extends Module {
  weight: MxArray;

  constructor() {
    super();
    this.weight = array([2.0, 3.0]);
  }

  forward(x: MxArray): MxArray {
    return multiply(this.weight, x);
  }
}

class TwoParamModel extends Module {
  weight: MxArray;
  bias: MxArray;

  constructor() {
    super();
    this.weight = array([2.0]);
    this.bias = array([1.0]);
  }

  forward(x: MxArray): MxArray {
    return add(multiply(this.weight, x), this.bias);
  }
}

class NestedModel extends Module {
  inner: SimpleModel;

  constructor() {
    super();
    this.inner = new SimpleModel();
  }

  forward(x: MxArray): MxArray {
    return this.inner.forward(x);
  }
}

class ModuleArrayModel extends Module {
  layers: SimpleModel[];

  constructor() {
    super();
    this.layers = [new SimpleModel(), new SimpleModel()];
  }

  forward(x: MxArray): MxArray {
    const firstLayer = this.layers[0];
    const secondLayer = this.layers[1];
    if (firstLayer === undefined || secondLayer === undefined) {
      throw new Error("ModuleArrayModel: expected two layers");
    }
    using first = firstLayer.forward(x);
    return secondLayer.forward(first);
  }
}

// ---------------------------------------------------------------------------
// Gradient tree structure
// ---------------------------------------------------------------------------

describe("nn.valueAndGrad — structure", () => {
  test("gradient tree matches trainableParameters structure", () => {
    const model = new TwoParamModel();
    const lossFn = (x: MxArray) => sum(model.forward(x));
    const vgFn = valueAndGrad(model, lossFn);

    const x = array([3.0]);
    const [loss, grads] = vgFn(x);

    expect(Object.keys(grads)).toEqual(["weight", "bias"]);
    expect(grads.weight).toBeInstanceOf(MxArray);
    expect(grads.bias).toBeInstanceOf(MxArray);

    mxEval(loss);
    loss.free();
    for (const [, g] of treeFlatten(grads)) g.free();
    x.free();
    model[Symbol.dispose]();
  });

  test("nested model produces nested gradient tree", () => {
    const model = new NestedModel();
    const lossFn = (x: MxArray) => sum(model.forward(x));
    const vgFn = valueAndGrad(model, lossFn);

    const x = array([1.0, 1.0]);
    const [loss, grads] = vgFn(x);

    // Gradient tree should have { inner: { weight: ... } }
    expect(Object.keys(grads)).toEqual(["inner"]);
    const innerGrads = grads.inner as ParameterTree;
    expect(Object.keys(innerGrads)).toEqual(["weight"]);

    mxEval(loss);
    loss.free();
    for (const [, g] of treeFlatten(grads)) g.free();
    x.free();
    model[Symbol.dispose]();
  });

  test("Module[] models produce numeric-keyed gradient subtrees", () => {
    const model = new ModuleArrayModel();
    const lossFn = (x: MxArray) => sum(model.forward(x));
    const vgFn = valueAndGrad(model, lossFn);

    const x = array([1.0, 1.0]);
    const [loss, grads] = vgFn(x);

    expect(Object.keys(grads)).toEqual(["layers"]);
    const layerGrads = grads.layers as ParameterTree;
    expect(Object.keys(layerGrads)).toEqual(["0", "1"]);
    expect(Object.keys(layerGrads["0"] as ParameterTree)).toEqual(["weight"]);
    expect(Object.keys(layerGrads["1"] as ParameterTree)).toEqual(["weight"]);

    mxEval(loss);
    loss.free();
    for (const [, g] of treeFlatten(grads)) g.free();
    x.free();
    model[Symbol.dispose]();
  });
});

// ---------------------------------------------------------------------------
// Gradient correctness
// ---------------------------------------------------------------------------

describe("nn.valueAndGrad — correctness", () => {
  test("gradient of sum(w * x) w.r.t. w is x", () => {
    const model = new SimpleModel();
    const lossFn = (x: MxArray) => sum(model.forward(x));
    const vgFn = valueAndGrad(model, lossFn);

    const x = array([4.0, 5.0]);
    const [loss, grads] = vgFn(x);

    mxEval(loss);
    expect(loss.item()).toBeCloseTo(2 * 4 + 3 * 5, 2); // 8 + 15 = 23

    const wGrad = grads.weight;
    expect(wGrad).toBeInstanceOf(MxArray);
    mxEval(wGrad as MxArray);
    // d/dw sum(w * x) = x
    expect((wGrad as MxArray).toList()).toEqual([4, 5]);

    loss.free();
    for (const [, g] of treeFlatten(grads)) g.free();
    x.free();
    model[Symbol.dispose]();
  });

  test("gradient of (w*x + b) w.r.t. w and b", () => {
    const model = new TwoParamModel();
    // loss = sum(w*x + b) = w*x + b (for scalar x)
    const lossFn = (x: MxArray) => sum(model.forward(x));
    const vgFn = valueAndGrad(model, lossFn);

    const x = array([3.0]);
    const [loss, grads] = vgFn(x);

    mxEval(loss);
    expect(loss.item()).toBeCloseTo(2 * 3 + 1, 2); // 7

    const wGrad = grads.weight as MxArray;
    const bGrad = grads.bias as MxArray;
    mxEval(wGrad, bGrad);
    // d/dw sum(w*x+b) = x = 3
    expect(wGrad.toList()).toEqual([3]);
    // d/db sum(w*x+b) = 1
    expect(bGrad.toList()).toEqual([1]);

    loss.free();
    wGrad.free();
    bGrad.free();
    x.free();
    model[Symbol.dispose]();
  });

  test("the same transformed function can be called repeatedly", () => {
    const model = new TwoParamModel();
    const lossFn = (x: MxArray) => sum(model.forward(x));
    const vgFn = valueAndGrad(model, lossFn);

    const firstInput = array([3.0]);
    const secondInput = array([4.0]);
    const [firstLoss, firstGrads] = vgFn(firstInput);
    const [secondLoss, secondGrads] = vgFn(secondInput);

    mxEval(firstLoss, secondLoss, secondGrads.weight as MxArray, secondGrads.bias as MxArray);
    expect(firstLoss.item()).toBeCloseTo(7, 2);
    expect(secondLoss.item()).toBeCloseTo(9, 2);
    expect((secondGrads.weight as MxArray).toList()).toEqual([4]);
    expect((secondGrads.bias as MxArray).toList()).toEqual([1]);

    firstLoss.free();
    secondLoss.free();
    for (const [, g] of treeFlatten(firstGrads)) g.free();
    for (const [, g] of treeFlatten(secondGrads)) g.free();
    firstInput.free();
    secondInput.free();
    model[Symbol.dispose]();
  });
});

// ---------------------------------------------------------------------------
// Model restore
// ---------------------------------------------------------------------------

describe("nn.valueAndGrad — restore", () => {
  test("model params are restored after call", () => {
    const model = new SimpleModel();
    const originalWeight = model.weight;

    const lossFn = (x: MxArray) => sum(model.forward(x));
    const vgFn = valueAndGrad(model, lossFn);

    const x = array([1.0, 1.0]);
    const [loss, grads] = vgFn(x);

    // Same MxArray reference — model was restored
    expect(model.weight).toBe(originalWeight);

    mxEval(loss);
    loss.free();
    for (const [, g] of treeFlatten(grads)) g.free();
    x.free();
    model[Symbol.dispose]();
  });

  test("model params are restored even if lossFn throws", () => {
    const model = new SimpleModel();
    const originalWeight = model.weight;

    const lossFn = (): MxArray => {
      throw new Error("intentional test error");
    };
    const vgFn = valueAndGrad(model, lossFn);

    const x = array([1.0]);
    expect(() => vgFn(x)).toThrow("intentional test error");

    // Model must be restored even after throw
    expect(model.weight).toBe(originalWeight);

    x.free();
    model[Symbol.dispose]();
  });
});

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

describe("nn.valueAndGrad — edge cases", () => {
  test("zero trainable params returns loss and empty grad tree", () => {
    const model = new SimpleModel();
    model.freeze(); // Freeze all params

    const lossFn = (x: MxArray) => sum(x);
    const vgFn = valueAndGrad(model, lossFn);

    const x = array([1.0, 2.0, 3.0]);
    const [loss, grads] = vgFn(x);

    mxEval(loss);
    expect(loss.item()).toBeCloseTo(6.0, 2);
    expect(Object.keys(grads)).toEqual([]);

    loss.free();
    x.free();
    model[Symbol.dispose]();
  });

  test("frozen params excluded from gradients", () => {
    const model = new TwoParamModel();
    model.freeze(["bias"]);

    const lossFn = (x: MxArray) => sum(model.forward(x));
    const vgFn = valueAndGrad(model, lossFn);

    const x = array([3.0]);
    const [loss, grads] = vgFn(x);

    // Only weight gradient, not bias
    expect(Object.keys(grads)).toEqual(["weight"]);

    mxEval(loss);
    loss.free();
    for (const [, g] of treeFlatten(grads)) g.free();
    x.free();
    model[Symbol.dispose]();
  });

  test("changing the trainable structure after creating the transform throws clearly", () => {
    const model = new TwoParamModel();
    const lossFn = (x: MxArray) => sum(model.forward(x));
    const vgFn = valueAndGrad(model, lossFn);

    const firstInput = array([3.0]);
    const [firstLoss, firstGrads] = vgFn(firstInput);
    model.freeze(["bias"]);

    const secondInput = array([3.0]);
    expect(() => vgFn(secondInput)).toThrow("trainable parameter structure changed");

    firstLoss.free();
    for (const [, g] of treeFlatten(firstGrads)) {
      g.free();
    }
    firstInput.free();
    secondInput.free();
    model[Symbol.dispose]();
  });

  test("repeated calls do not grow active memory for a tiny model", () => {
    synchronize();
    clearMemoryCache();

    const model = new SimpleModel();
    const lossFn = (x: MxArray) => sum(model.forward(x));
    const vgFn = valueAndGrad(model, lossFn);
    const x = array([1.0, 2.0]);

    mxEval(...treeFlatten(model.parameters()).map(([, value]) => value));
    synchronize();
    const baseline = getActiveMemoryBytes();

    for (let index = 0; index < 5; index++) {
      const [loss, grads] = vgFn(x);
      mxEval(loss, ...treeFlatten(grads).map(([, value]) => value));
      loss.free();
      for (const [, g] of treeFlatten(grads)) {
        g.free();
      }
    }

    synchronize();
    clearMemoryCache();
    expect(getActiveMemoryBytes()).toBeLessThanOrEqual(baseline + 64);

    x.free();
    model[Symbol.dispose]();
  });
});
