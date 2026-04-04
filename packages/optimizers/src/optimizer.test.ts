import { describe, expect, test } from "bun:test";
import { add, array, type MxArray } from "@mlxts/core";
import { Module } from "@mlxts/nn";
import { Optimizer } from "./optimizer";

class SingleParamModel extends Module {
  weight: MxArray;

  constructor() {
    super();
    this.weight = array([1]);
  }

  forward(x: MxArray): MxArray {
    return x;
  }
}

class TwoParamModel extends Module {
  weight: MxArray;
  bias: MxArray;

  constructor() {
    super();
    this.weight = array([1]);
    this.bias = array([2]);
  }

  forward(x: MxArray): MxArray {
    return x;
  }
}

class SimpleOptimizer extends Optimizer {
  protected applySingle(
    _key: string,
    param: MxArray,
    grad: MxArray,
  ): { parameter: MxArray; state?: Record<string, MxArray> } {
    return { parameter: add(param, grad) };
  }
}

class FailingOptimizer extends Optimizer {
  primeState(): void {
    this.state.set("weight", { cache: array([10]) });
    this.state.set("bias", { cache: array([20]) });
  }

  stateRef(key: string): MxArray | undefined {
    return this.state.get(key)?.cache;
  }

  protected applySingle(
    key: string,
    param: MxArray,
    grad: MxArray,
    previousState?: Readonly<Record<string, MxArray>>,
  ): { parameter: MxArray; state?: Record<string, MxArray> } {
    if (key === "bias") {
      throw new Error("synthetic failure");
    }

    const baseState = previousState?.cache ?? grad;
    return {
      parameter: add(param, grad),
      state: { cache: add(baseState, 1) },
    };
  }
}

describe("Optimizer base behavior", () => {
  test("extra gradient paths throw", () => {
    const model = new SingleParamModel();
    const optimizer = new SimpleOptimizer();

    const grads = { weight: array([1]), extra: array([2]) };
    expect(() => optimizer.update(model, grads)).toThrow(
      'unexpected gradient for parameter "extra"',
    );

    model[Symbol.dispose]();
    optimizer[Symbol.dispose]();
    grads.weight.free();
    grads.extra.free();
  });

  test("failed updates leave model parameters and state unchanged", () => {
    const model = new TwoParamModel();
    const optimizer = new FailingOptimizer();
    optimizer.primeState();

    const originalWeight = model.weight;
    const originalBias = model.bias;
    const originalWeightState = optimizer.stateRef("weight");
    const originalBiasState = optimizer.stateRef("bias");
    const grads = { weight: array([1]), bias: array([1]) };

    expect(() => optimizer.update(model, grads)).toThrow("synthetic failure");

    expect(model.weight).toBe(originalWeight);
    expect(model.bias).toBe(originalBias);
    expect(optimizer.stateRef("weight")).toBe(originalWeightState);
    expect(optimizer.stateRef("bias")).toBe(originalBiasState);

    model[Symbol.dispose]();
    optimizer[Symbol.dispose]();
    grads.weight.free();
    grads.bias.free();
  });
});
