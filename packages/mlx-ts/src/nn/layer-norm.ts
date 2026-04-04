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
import { layerNorm as fastLayerNorm } from "../core/fast";
import { formatShape } from "../utils/format-shape";
import { Module } from "./module";

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

    return fastLayerNorm(x, this.weight, this.bias, { eps: this.#eps });
  }
}
