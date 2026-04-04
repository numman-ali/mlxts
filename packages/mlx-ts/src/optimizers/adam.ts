/**
 * Adam and AdamW optimizers.
 *
 * AdamW implements Adam with decoupled weight decay.
 * Adam is a zero-weight-decay wrapper around AdamW.
 *
 * @module
 */

import type { MxArray } from "../core/array";
import { zeros } from "../core/array";
import { add, divide, multiply, sqrt, square, subtract } from "../core/ops/arithmetic";
import { stopGradient } from "../core/ops/shape";
import type { Module } from "../nn/module";
import type { ParameterTree } from "../utils/tree";
import { Optimizer } from "./optimizer";

/** Serializable AdamW optimizer state snapshot. */
export interface AdamWCheckpoint {
  kind: "adamw";
  step: number;
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
  state: Record<string, Record<string, MxArray>>;
}

/** AdamW constructor options. */
export interface AdamWOptions {
  learningRate?: number;
  beta1?: number;
  beta2?: number;
  eps?: number;
  weightDecay?: number;
}

/** Adam constructor options. */
export interface AdamOptions {
  learningRate?: number;
  beta1?: number;
  beta2?: number;
  eps?: number;
}

/** AdamW optimizer — Adam with decoupled weight decay. */
export class AdamW extends Optimizer {
  #lr: number;
  #beta1: number;
  #beta2: number;
  #eps: number;
  #weightDecay: number;
  #step = 0;

  /**
   * @param learningRate - Learning rate. Defaults to 0.001.
   * @param beta1 - First moment decay rate. Defaults to 0.9.
   * @param beta2 - Second moment decay rate. Defaults to 0.999.
   * @param eps - Small constant for numerical stability. Defaults to 1e-8.
   * @param weightDecay - Decoupled weight decay. Defaults to 0.01.
   */
  constructor({
    learningRate = 0.001,
    beta1 = 0.9,
    beta2 = 0.999,
    eps = 1e-8,
    weightDecay = 0.01,
  }: AdamWOptions = {}) {
    super();
    this.#lr = learningRate;
    this.#beta1 = beta1;
    this.#beta2 = beta2;
    this.#eps = eps;
    this.#weightDecay = weightDecay;
  }

  /** Update the learning rate for subsequent steps. */
  setLearningRate(lr: number): void {
    this.#lr = lr;
  }

  /** Current optimization step count (used for bias correction). */
  get step(): number {
    return this.#step;
  }

  /** Export the optimizer state for checkpointing. */
  checkpoint(): AdamWCheckpoint {
    return {
      kind: "adamw",
      step: this.#step,
      lr: this.#lr,
      beta1: this.#beta1,
      beta2: this.#beta2,
      eps: this.#eps,
      weightDecay: this.#weightDecay,
      state: this.exportStateSnapshot(),
    };
  }

  /** Restore the optimizer state from a checkpoint. */
  restore(checkpoint: AdamWCheckpoint): void {
    this.#step = checkpoint.step;
    this.#lr = checkpoint.lr;
    this.#beta1 = checkpoint.beta1;
    this.#beta2 = checkpoint.beta2;
    this.#eps = checkpoint.eps;
    this.#weightDecay = checkpoint.weightDecay;
    this.replaceStateSnapshot(checkpoint.state);
  }

  override update(model: Module, gradients: ParameterTree): void {
    this.#step += 1;
    try {
      super.update(model, gradients);
    } catch (error) {
      this.#step -= 1;
      throw error;
    }
  }

  protected applySingle(
    key: string,
    param: MxArray,
    grad: MxArray,
    previousState?: Readonly<Record<string, MxArray>>,
  ): { parameter: MxArray; state: Record<string, MxArray> } {
    // Initialize state on first call
    const oldM = previousState?.m ?? zeros([...param.shape]);
    const oldV = previousState?.v ?? zeros([...param.shape]);
    if (oldM === undefined || oldV === undefined) {
      throw new Error(`AdamW: missing state for "${key}"`);
    }
    const ownsOldM = previousState?.m === undefined;
    const ownsOldV = previousState?.v === undefined;

    try {
      // m = beta1 * m + (1 - beta1) * grad
      using scaledM = multiply(oldM, this.#beta1);
      using scaledGrad = multiply(grad, 1 - this.#beta1);
      using rawM = add(scaledM, scaledGrad);

      // v = beta2 * v + (1 - beta2) * grad^2
      using scaledV = multiply(oldV, this.#beta2);
      using gradSq = square(grad);
      using scaledGradSq = multiply(gradSq, 1 - this.#beta2);
      using rawV = add(scaledV, scaledGradSq);

      // Bias correction
      using mHat = divide(rawM, 1 - this.#beta1 ** this.#step);
      using vHat = divide(rawV, 1 - this.#beta2 ** this.#step);

      // Decoupled weight decay: decay param before Adam update
      using decayed = multiply(param, 1 - this.#lr * this.#weightDecay);

      // param = decayed - lr * mHat / (sqrt(vHat) + eps)
      using sqrtVHat = sqrt(vHat);
      using denom = add(sqrtVHat, this.#eps);
      using ratio = divide(mHat, denom);
      using step = multiply(ratio, this.#lr);
      using rawParameter = subtract(decayed, step);
      const parameter = stopGradient(rawParameter);
      try {
        const nextM = stopGradient(rawM);
        try {
          const nextV = stopGradient(rawV);
          return {
            parameter,
            state: { m: nextM, v: nextV },
          };
        } catch (error) {
          nextM.free();
          throw error;
        }
      } catch (error) {
        parameter.free();
        throw error;
      }
    } finally {
      if (ownsOldM) {
        oldM.free();
      }
      if (ownsOldV) {
        oldV.free();
      }
    }
  }
}

/** Adam optimizer — AdamW with zero weight decay. */
export class Adam extends AdamW {
  /**
   * @param learningRate - Learning rate. Defaults to 0.001.
   * @param beta1 - First moment decay rate. Defaults to 0.9.
   * @param beta2 - Second moment decay rate. Defaults to 0.999.
   * @param eps - Small constant for numerical stability. Defaults to 1e-8.
   */
  constructor(options: AdamOptions = {}) {
    super({ ...options, weightDecay: 0 });
  }
}
