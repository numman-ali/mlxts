/**
 * End-to-end integration test: MLP trains to convergence on XOR.
 *
 * This exercises the exact training shape the repo now uses:
 * integer targets, crossEntropy loss, multi-class logits, nn.valueAndGrad,
 * and optimizer parameter updates.
 */
import { describe, expect, test } from "bun:test";
import {
  add,
  argmax,
  array,
  type MxArray,
  mxEval,
  random,
  treeFlatten,
  treeLeaves,
} from "@mlxts/core";
import { SGD } from "@mlxts/optimizers";
import { gelu, swiglu } from "./activations";
import { GroupedQueryAttention } from "./grouped-query-attention";
import { Linear } from "./linear";
import { crossEntropy } from "./losses";
import { Module } from "./module";
import { RMSNorm } from "./rms-norm";
import { RoPE } from "./rope";
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

class TinyLlamaBlock extends Module {
  inputNorm: RMSNorm;
  attention: GroupedQueryAttention;
  postAttentionNorm: RMSNorm;
  gateProjection: Linear;
  upProjection: Linear;
  downProjection: Linear;

  constructor() {
    super();
    this.inputNorm = new RMSNorm(8, 1e-6);
    this.attention = new GroupedQueryAttention(8, 4, 2, {
      rope: new RoPE(2, true),
    });
    this.postAttentionNorm = new RMSNorm(8, 1e-6);
    this.gateProjection = new Linear(8, 16, false);
    this.upProjection = new Linear(8, 16, false);
    this.downProjection = new Linear(16, 8, false);
  }

  forward(x: MxArray): MxArray {
    using attentionInput = this.inputNorm.forward(x);
    using attentionOutput = this.attention.forward(attentionInput);
    using afterAttention = add(x, attentionOutput);
    using mlpInput = this.postAttentionNorm.forward(afterAttention);
    using gate = this.gateProjection.forward(mlpInput);
    using up = this.upProjection.forward(mlpInput);
    using activated = swiglu(gate, up);
    using down = this.downProjection.forward(activated);
    return add(afterAttention, down);
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
    using optimizer = new SGD({ learningRate: 0.1 });

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

describe("LLaMA-style block composition", () => {
  test("RMSNorm, GQA, RoPE, and SwiGLU compose into a decoder block", () => {
    using block = new TinyLlamaBlock();
    using input = array(
      [
        [
          [1, 2, 3, 4, 5, 6, 7, 8],
          [2, 3, 4, 5, 6, 7, 8, 9],
          [3, 4, 5, 6, 7, 8, 9, 10],
        ],
      ],
      "float32",
    );

    using output = block.forward(input);
    mxEval(output);

    expect(output.shape).toEqual([1, 3, 8]);
    const values = (output.toList() as number[][][]).flat(2);
    expect(values.every((value) => Number.isFinite(value))).toBe(true);
  });
});
