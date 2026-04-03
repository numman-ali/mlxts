/**
 * Layer normalization.
 *
 * Normalizes inputs over the last axis using mean and variance,
 * then applies a learnable affine transform (weight and bias).
 * Composed from existing core ops (no fused kernel).
 *
 * @module
 */

import type { MxArray } from "../core/array";
import { ones, zeros } from "../core/array";
import { add, divide, multiply, sqrt, square, subtract } from "../core/ops/arithmetic";
import { mean } from "../core/ops/reduction";
import { Module } from "./module";

function formatShape(shape: readonly number[]): string {
  return shape.length === 0 ? "[]" : `[${shape.join(", ")}]`;
}

/** Layer normalization over the last axis. */
export class LayerNorm extends Module {
  weight: MxArray;
  bias: MxArray;
  #dims: number;
  #eps: number;

  /**
   * @param dims - Size of the last axis to normalize over. Must be > 0.
   * @param eps - Small constant for numerical stability. Defaults to 1e-5.
   */
  constructor(dims: number, eps = 1e-5) {
    super();
    if (dims <= 0) {
      throw new Error(`LayerNorm: dims must be > 0, got ${dims}`);
    }
    this.#dims = dims;
    this.#eps = eps;
    this.weight = ones([dims]);
    this.bias = zeros([dims]);
  }

  forward(x: MxArray): MxArray {
    const lastDimension = x.shape[x.shape.length - 1];
    if (lastDimension !== this.#dims) {
      throw new Error(
        `LayerNorm.forward: expected last dimension ${this.#dims}, got ${lastDimension ?? "undefined"} ` +
          `for shape ${formatShape(x.shape)}.`,
      );
    }

    // mu = mean(x, axis=-1, keepdims=true)
    using mu = mean(x, -1, true);
    // centered = x - mu
    using centered = subtract(x, mu);
    // variance = mean(centered^2, axis=-1, keepdims=true)
    using centeredSq = square(centered);
    using variance = mean(centeredSq, -1, true);
    // stdInv = 1 / sqrt(variance + eps)
    using varPlusEps = add(variance, this.#eps);
    using std = sqrt(varPlusEps);
    using stdInv = divide(1.0, std);
    // normalized = centered * stdInv
    using normalized = multiply(centered, stdInv);
    // output = weight * normalized + bias
    using scaled = multiply(this.weight, normalized);
    return add(scaled, this.bias);
  }
}
