/**
 * Stochastic gradient descent optimizer.
 *
 * Supports optional momentum and weight decay.
 *
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { add, multiply, stopGradient, subtract, zeros } from "@mlxts/core";
import { Optimizer } from "./optimizer";

/** SGD constructor options. */
export interface SGDOptions {
  learningRate: number;
  momentum?: number;
  weightDecay?: number;
}

/** SGD optimizer with optional momentum and weight decay. */
export class SGD extends Optimizer {
  #lr: number;
  #momentum: number;
  #weightDecay: number;

  /**
   * @param learningRate - Learning rate.
   * @param momentum - Momentum factor. Defaults to 0 (no momentum).
   * @param weightDecay - Weight decay (L2 penalty). Defaults to 0.
   */
  constructor({ learningRate, momentum = 0, weightDecay = 0 }: SGDOptions) {
    super();
    this.#lr = learningRate;
    this.#momentum = momentum;
    this.#weightDecay = weightDecay;
  }

  protected applySingle(
    _key: string,
    param: MxArray,
    grad: MxArray,
    previousState?: Readonly<Record<string, MxArray>>,
  ): { parameter: MxArray; state?: Record<string, MxArray> } {
    // Apply weight decay: effectiveGrad = grad + weightDecay * param
    let effectiveGrad = grad;
    let ownedGrad = false;
    if (this.#weightDecay !== 0) {
      using decayTerm = multiply(param, this.#weightDecay);
      effectiveGrad = add(grad, decayTerm);
      ownedGrad = true;
    }

    try {
      if (this.#momentum === 0) {
        // Simple SGD: param = param - lr * grad
        using step = multiply(effectiveGrad, this.#lr);
        using rawParameter = subtract(param, step);
        return { parameter: stopGradient(rawParameter) };
      }

      // Momentum: v = momentum * v + grad; param = param - lr * v
      const savedVelocity = previousState?.velocity;
      const velocity = savedVelocity ?? zeros([...param.shape]);

      try {
        // v = momentum * v + effectiveGrad
        using scaledV = multiply(velocity, this.#momentum);
        using rawVelocity = add(scaledV, effectiveGrad);

        // param = param - lr * newVelocity
        using step = multiply(rawVelocity, this.#lr);
        using rawParameter = subtract(param, step);
        const parameter = stopGradient(rawParameter);
        try {
          const velocityState = stopGradient(rawVelocity);
          return {
            parameter,
            state: { velocity: velocityState },
          };
        } catch (error) {
          parameter.free();
          throw error;
        }
      } finally {
        if (savedVelocity === undefined) {
          velocity.free();
        }
      }
    } finally {
      if (ownedGrad) effectiveGrad.free();
    }
  }
}
