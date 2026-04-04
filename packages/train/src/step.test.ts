import { describe, expect, test } from "bun:test";
import { full, treeFlatten } from "@mlxts/core";
import { Module } from "@mlxts/nn";

import { applyGradientStep, materializeTrainingState } from "./step";

class TinyModel extends Module {
  weight = full([2], 1);

  override forward() {
    return this.weight;
  }
}

describe("step helpers", () => {
  test("applyGradientStep averages and applies accumulated gradients", () => {
    const seen: number[] = [];
    const model = new TinyModel();

    try {
      const result = applyGradientStep({
        gradAccumSteps: 2,
        maxGradNorm: null,
        takeMicroStep() {
          return {
            lossValue: 2,
            gradients: {
              weight: full([2], 1),
            },
          };
        },
        applyGradients(gradients) {
          const weight = treeFlatten(gradients)[0]?.[1];
          if (weight === undefined) {
            throw new Error("expected weight gradient");
          }
          seen.push(...Array.from(weight.toTypedArray(), Number));
        },
      });

      expect(result.averageLoss).toBe(2);
      expect(seen).toEqual([1, 1]);
    } finally {
      model[Symbol.dispose]();
    }
  });

  test("materializeTrainingState evaluates model and optimizer arrays", () => {
    const model = new TinyModel();
    const optimizerState = full([2], 3);

    try {
      materializeTrainingState(model, {
        stateArrays() {
          return [optimizerState];
        },
      });

      const parameter = treeFlatten(model.parameters())[0]?.[1];
      expect(parameter?.toList()).toEqual([1, 1]);
      expect(optimizerState.toList()).toEqual([3, 3]);
    } finally {
      optimizerState.free();
      model[Symbol.dispose]();
    }
  });
});
