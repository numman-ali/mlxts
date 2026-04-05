/**
 * Gemma 3 RMSNorm modules with explicit `1 + weight` semantics.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { add, fastRmsNorm, formatShape, zeros } from "@mlxts/core";
import { Module } from "@mlxts/nn";

/** RMSNorm variant used by Gemma 3 for hidden states and q/k normalization. */
export class Gemma3RMSNorm extends Module {
  weight: MxArray;
  #dims: number;
  #eps: number;
  #effectiveWeight: MxArray | null = null;
  #effectiveWeightSource: MxArray | null = null;

  constructor(dims: number, eps: number) {
    super();
    this.#dims = dims;
    this.#eps = eps;
    this.weight = zeros([dims]);
  }

  forward(x: MxArray): MxArray {
    const lastDimension = x.shape[x.shape.length - 1];
    if (lastDimension !== this.#dims) {
      throw new Error(
        `Gemma3RMSNorm.forward: expected last dimension ${this.#dims}, got ${lastDimension ?? "undefined"} for shape ${formatShape(x.shape)}.`,
      );
    }

    const effectiveWeight = this.effectiveWeight();
    return fastRmsNorm(x, effectiveWeight, { eps: this.#eps });
  }

  private effectiveWeight(): MxArray {
    if (this.#effectiveWeight === null || this.#effectiveWeightSource !== this.weight) {
      this.#effectiveWeight?.free();
      this.#effectiveWeight = add(this.weight, 1.0);
      this.#effectiveWeightSource = this.weight;
    }
    return this.#effectiveWeight;
  }

  override [Symbol.dispose](): void {
    this.#effectiveWeight?.free();
    this.#effectiveWeight = null;
    this.#effectiveWeightSource = null;
    super[Symbol.dispose]();
  }
}
