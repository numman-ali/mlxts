/**
 * End-to-end integration test: MLP trains to convergence on XOR.
 *
 * This exercises the exact training shape Phase 4 will use:
 * integer targets, crossEntropy loss, multi-class logits, nn.valueAndGrad,
 * and optimizer parameter updates.
 */
import { describe, expect, test } from "bun:test";

import { array, type MxArray } from "../core/array";
import { argmax } from "../core/ops/reduction";
import * as random from "../core/random";
import { mxEval } from "../core/transforms";
import { SGD } from "../optimizers/sgd";
import { treeFlatten, treeLeaves } from "../utils/tree";
import { gelu } from "./activations";
import { Linear } from "./linear";
import { crossEntropy } from "./losses";
import { Module } from "./module";
import { valueAndGrad } from "./value-and-grad";

class MLP extends Module {
  layer1: Linear;
  layer2: Linear;

  constructor() {
    super();
    this.layer1 = new Linear(2, 16);
    this.layer2 = new Linear(16, 2); // 2 output logits for 2 classes
  }

  forward(x: MxArray): MxArray {
    using h = this.layer1.forward(x);
    using activated = gelu(h);
    return this.layer2.forward(activated);
  }
}

describe("MLP convergence on XOR", () => {
  test("2-class classification with crossEntropy converges", () => {
    // Deterministic
    random.seed(42);

    // XOR as 2-class classification
    const X = array([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ]);
    const Y = array([0, 1, 1, 0], "int32"); // Integer class labels

    const model = new MLP();
    using optimizer = new SGD(0.1);

    const lossFn = (x: MxArray, y: MxArray) => crossEntropy(model.forward(x), y);
    const trainStep = valueAndGrad(model, lossFn);

    let lastLoss = Number.POSITIVE_INFINITY;

    for (let step = 0; step < 500; step++) {
      const [loss, grads] = trainStep(X, Y);
      optimizer.update(model, grads);

      // Eval loss + model params + optimizer state together
      mxEval(loss, ...treeLeaves(model.parameters()), ...optimizer.stateArrays());

      lastLoss = loss.item();
      loss.free();
      for (const [, g] of treeFlatten(grads)) g.free();
    }

    // Exit criterion: loss < 0.05
    expect(lastLoss).toBeLessThan(0.05);

    // Verify predictions match XOR truth table
    const logits = model.forward(X);
    const preds = argmax(logits, -1);
    mxEval(preds);
    const predList = preds.toList() as number[];

    // XOR: [0,0]→0, [0,1]→1, [1,0]→1, [1,1]→0
    expect(predList[0]).toBe(0);
    expect(predList[1]).toBe(1);
    expect(predList[2]).toBe(1);
    expect(predList[3]).toBe(0);

    // Cleanup
    X.free();
    Y.free();
    logits.free();
    preds.free();
    model[Symbol.dispose]();
  });
});
