/**
 * Dropout regularization layer.
 *
 * During training, randomly zeros elements with probability `p` and
 * scales remaining elements by `1/(1-p)` to preserve expectation.
 * During evaluation, acts as identity (no-op).
 *
 * @module
 */

import type { MxArray } from "../core/array";
import { retainArray } from "../core/array";
import { multiply } from "../core/ops/arithmetic";
import * as random from "../core/random";
import { Module } from "./module";

/** Dropout layer — no learnable parameters. */
export class Dropout extends Module {
  #p: number;

  /**
   * @param p - Probability of zeroing each element. Must satisfy 0 <= p < 1.
   */
  constructor(p = 0.0) {
    super();
    if (p < 0 || p >= 1) {
      throw new Error(`Dropout: p must satisfy 0 <= p < 1, got ${p}`);
    }
    this.#p = p;
  }

  /** Configured dropout probability. */
  get probability(): number {
    return this.#p;
  }

  /**
   * Apply dropout.
   *
   * In training mode: elements are zeroed with probability p, and
   * remaining elements are scaled by 1/(1-p).
   * In eval mode: returns input unchanged.
   */
  forward(x: MxArray): MxArray {
    if (!this.isTraining || this.#p === 0) {
      return retainArray(x);
    }

    // Bernoulli mask: 1 with probability (1-p), 0 with probability p
    using mask = random.bernoulli(1 - this.#p, [...x.shape]);
    // Scale by 1/(1-p) to preserve expectation
    const scale = 1 / (1 - this.#p);
    using masked = multiply(x, mask);
    return multiply(masked, scale);
  }
}
