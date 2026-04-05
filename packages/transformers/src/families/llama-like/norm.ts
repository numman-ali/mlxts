/**
 * LLaMA-style RMSNorm with optional Gemma weight offset semantics.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { add, fastRmsNorm, formatShape, ones, zeros } from "@mlxts/core";
import { Module } from "@mlxts/nn";

import type { LlamaLikeConfig } from "./types";

/** RMSNorm wrapper that keeps Gemma's `1 + weight` offset visible. */
export class LlamaLikeNorm extends Module {
  weight: MxArray;
  #dims: number;
  #eps: number;
  #weightOffset: boolean;
  #effectiveWeight: MxArray | null = null;
  #effectiveWeightSource: MxArray | null = null;

  constructor(config: Pick<LlamaLikeConfig, "hiddenSize" | "rmsNormEps" | "normWeightOffset">) {
    super();
    this.#dims = config.hiddenSize;
    this.#eps = config.rmsNormEps;
    this.#weightOffset = config.normWeightOffset === true;
    this.weight = this.#weightOffset ? zeros([config.hiddenSize]) : ones([config.hiddenSize]);
  }

  forward(x: MxArray): MxArray {
    const lastDimension = x.shape[x.shape.length - 1];
    if (lastDimension !== this.#dims) {
      throw new Error(
        `LlamaLikeNorm.forward: expected last dimension ${this.#dims}, got ${lastDimension ?? "undefined"} for shape ${formatShape(x.shape)}.`,
      );
    }

    if (!this.#weightOffset) {
      return fastRmsNorm(x, this.weight, { eps: this.#eps });
    }

    return fastRmsNorm(x, this.effectiveWeight(), { eps: this.#eps });
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
