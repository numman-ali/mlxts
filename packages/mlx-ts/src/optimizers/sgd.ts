/**
 * Stochastic gradient descent optimizer.
 *
 * Supports optional momentum and weight decay.
 *
 * @module
 */

import type { MxArray } from "../core/array";
import { zeros } from "../core/array";
import { add, multiply, subtract } from "../core/ops/arithmetic";
import { Optimizer } from "./optimizer";

/** SGD optimizer with optional momentum and weight decay. */
export class SGD extends Optimizer {
  #lr: number;
  #momentum: number;
  #weightDecay: number;

  /**
   * @param lr - Learning rate.
   * @param momentum - Momentum factor. Defaults to 0 (no momentum).
   * @param weightDecay - Weight decay (L2 penalty). Defaults to 0.
   */
  constructor(lr: number, momentum = 0, weightDecay = 0) {
    super();
    this.#lr = lr;
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
        return { parameter: subtract(param, step) };
      }

      // Momentum: v = momentum * v + grad; param = param - lr * v
      const savedVelocity = previousState?.velocity;
      const velocity = savedVelocity ?? zeros([...param.shape]);

      try {
        // v = momentum * v + effectiveGrad
        using scaledV = multiply(velocity, this.#momentum);
        const newVelocity = add(scaledV, effectiveGrad);

        // param = param - lr * newVelocity
        using step = multiply(newVelocity, this.#lr);
        return {
          parameter: subtract(param, step),
          state: { velocity: newVelocity },
        };
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
