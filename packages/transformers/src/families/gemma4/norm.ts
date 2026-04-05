/**
 * Gemma 4 RMSNorm variants.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { fastRmsNorm, formatShape, ones } from "@mlxts/core";
import { Module } from "@mlxts/nn";

/** RMSNorm used by Gemma 4 hidden states, q/k heads, and value heads. */
export class Gemma4RMSNorm extends Module {
  weight: MxArray | null;
  #dims: number;
  #eps: number;

  constructor(dims: number, eps: number, withScale = true) {
    super();
    this.#dims = dims;
    this.#eps = eps;
    this.weight = withScale ? ones([dims]) : null;
  }

  forward(x: MxArray): MxArray {
    const lastDimension = x.shape[x.shape.length - 1];
    if (lastDimension !== this.#dims) {
      throw new Error(
        `Gemma4RMSNorm.forward: expected last dimension ${this.#dims}, got ${lastDimension ?? "undefined"} for shape ${formatShape(x.shape)}.`,
      );
    }

    return fastRmsNorm(x, this.weight ?? undefined, { eps: this.#eps });
  }
}
