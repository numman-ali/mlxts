import { describe, expect, test } from "bun:test";
import { array } from "../core/array";
import { add, multiply } from "../core/ops/arithmetic";
import { sum } from "../core/ops/reduction";
import { mxEval } from "../core/transforms";
import { treeFlatten } from "../utils/tree";
import { checkpoint } from "./checkpoint";
import { Module } from "./module";
import { valueAndGrad } from "./value-and-grad";

class AffineModule extends Module {
  weight = array([2], "float32");
  bias = array([1], "float32");

  forward(x: ReturnType<typeof array>): ReturnType<typeof array> {
    using scaled = multiply(x, this.weight);
    return add(scaled, this.bias);
  }
}

describe("nn.checkpoint", () => {
  test("returns the same value as the underlying module function", () => {
    const module = new AffineModule();
    const checkpointed = checkpoint(module);
    using input = array([3], "float32");
    using output = checkpointed(input);

    mxEval(output);
    expect(output.toList()).toEqual([7]);

    module[Symbol.dispose]();
  });

  test("restores the original module parameter handles after each call", () => {
    const module = new AffineModule();
    const checkpointed = checkpoint(module);
    const before = treeFlatten(module.parameters());
    using input = array([4], "float32");
    using output = checkpointed(input);

    mxEval(output);
    const after = treeFlatten(module.parameters());

    expect(after).toHaveLength(before.length);
    for (let index = 0; index < before.length; index++) {
      expect(after[index]?.[0]).toEqual(before[index]?.[0]);
      expect(after[index]?.[1]).toBe(before[index]?.[1]);
    }

    module[Symbol.dispose]();
  });

  test("restores the original module parameter handles when the wrapped function throws", () => {
    const module = new AffineModule();
    const originalWeight = module.weight;
    const originalBias = module.bias;
    const checkpointed = checkpoint(module, () => {
      throw new Error("checkpoint boom");
    });

    using input = array([4], "float32");
    expect(() => checkpointed(input)).toThrow("checkpoint boom");
    expect(module.weight).toBe(originalWeight);
    expect(module.bias).toBe(originalBias);

    module[Symbol.dispose]();
  });

  test("works with nn.valueAndGrad over the module parameter tree", () => {
    const module = new AffineModule();
    const checkpointed = checkpoint(module, (x) => sum(module.forward(x)));
    const valueAndGradFn = valueAndGrad(module, (x) => checkpointed(x));

    using input = array([3], "float32");
    const [loss, grads] = valueAndGradFn(input);

    try {
      const entries = treeFlatten(grads);
      const weightGrad = entries.find(([path]) => path.join(".") === "weight")?.[1];
      const biasGrad = entries.find(([path]) => path.join(".") === "bias")?.[1];
      if (weightGrad === undefined || biasGrad === undefined) {
        throw new Error("expected gradients for weight and bias");
      }

      mxEval(loss, weightGrad, biasGrad);
      expect(loss.item()).toBeCloseTo(7, 5);
      expect(weightGrad.toList()).toEqual([3]);
      expect(biasGrad.toList()).toEqual([1]);
    } finally {
      loss.free();
      for (const [, grad] of treeFlatten(grads)) {
        grad.free();
      }
      module[Symbol.dispose]();
    }
  });

  test("throws clearly if the trainable parameter structure changes after creation", () => {
    const module = new AffineModule();
    const checkpointed = checkpoint(module, (x) => sum(module.forward(x)));

    using input = array([3], "float32");
    using output = checkpointed(input);
    mxEval(output);

    module.freeze(["bias"]);

    using secondInput = array([3], "float32");
    expect(() => checkpointed(secondInput)).toThrow("trainable parameter structure changed");

    module[Symbol.dispose]();
  });

  test("checkpointed module functions can be explicitly disposed", () => {
    const module = new AffineModule();
    using checkpointed = checkpoint(module);
    using input = array([2], "float32");
    using output = checkpointed(input);
    mxEval(output);
    expect(output.toList()).toEqual([5]);

    checkpointed[Symbol.dispose]();
    expect(() => checkpointed(input)).toThrow("disposed");

    module[Symbol.dispose]();
  });
});
