/**
 * Root mean square normalization.
 *
 * Normalizes inputs over the last axis using the root mean square and then
 * applies a learnable weight. There is no additive bias term.
 *
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { fastRmsNorm, formatShape, ones } from "@mlxts/core";
import { Module } from "./module";

/** Root mean square normalization over the last axis. */
export class RMSNorm extends Module {
  weight: MxArray;
  #dims: number;
  #eps: number;

  /**
   * @param dims - Size of the last axis to normalize over. Must be > 0.
   * @param eps - Small constant for numerical stability. Defaults to 1e-5.
   */
  constructor(dims: number, eps = 1e-5) {
    super();
    if (dims <= 0) {
      throw new Error(`RMSNorm: dims must be > 0, got ${dims}`);
    }
    this.#dims = dims;
    this.#eps = eps;
    this.weight = ones([dims]);
  }

  forward(x: MxArray): MxArray {
    const lastDimension = x.shape[x.shape.length - 1];
    if (lastDimension !== this.#dims) {
      throw new Error(
        `RMSNorm.forward: expected last dimension ${this.#dims}, got ${lastDimension ?? "undefined"} ` +
          `for shape ${formatShape(x.shape)}.`,
      );
    }

    return fastRmsNorm(x, this.weight, { eps: this.#eps });
  }
}
