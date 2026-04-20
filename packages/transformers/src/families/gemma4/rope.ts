/**
 * Gemma 4 rotary embeddings.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { array, fastRoPE } from "@mlxts/core";
import { Module, RoPE } from "@mlxts/nn";

export interface RotaryEmbedding extends Module {
  forward(x: MxArray, offset: number | MxArray): MxArray;
}

/** Proportional RoPE used by Gemma 4 full-attention layers. */
export class ProportionalRoPE extends Module implements RotaryEmbedding {
  #dims: number;
  #freqs: MxArray;

  constructor(dims: number, rotatedDims: number, base: number, factor = 1.0) {
    super();
    if (!Number.isInteger(dims) || dims <= 0 || dims % 2 !== 0) {
      throw new Error(`ProportionalRoPE: dims must be a positive even integer, got ${dims}.`);
    }
    if (!Number.isInteger(rotatedDims) || rotatedDims <= 0 || rotatedDims % 2 !== 0) {
      throw new Error(
        `ProportionalRoPE: rotatedDims must be a positive even integer, got ${rotatedDims}.`,
      );
    }
    if (rotatedDims > dims) {
      throw new Error(`ProportionalRoPE: rotatedDims ${rotatedDims} must not exceed dims ${dims}.`);
    }

    this.#dims = dims;

    const halfDimensions = dims / 2;
    const rotatedPairs = rotatedDims / 2;
    const freqs = Array.from({ length: halfDimensions }, (_, pairIndex) => {
      if (pairIndex >= rotatedPairs) {
        return Number.POSITIVE_INFINITY;
      }
      return factor * base ** ((pairIndex * 2) / dims);
    });
    this.#freqs = array(freqs, "float32");
  }

  /** Pre-computed per-dimension frequency tensor (borrowed — caller must not free). */
  get freqs(): MxArray {
    return this.#freqs;
  }

  forward(x: MxArray, offset: number | MxArray): MxArray {
    return fastRoPE(x, this.#dims, {
      traditional: false,
      freqs: this.#freqs,
      offset,
    });
  }

  override [Symbol.dispose](): void {
    this.#freqs.free();
  }
}

export function createRoPE(dims: number, ropeTheta: number, rotatedDims?: number): RotaryEmbedding {
  if (rotatedDims === undefined || rotatedDims === dims) {
    return new RoPE(dims, false, ropeTheta);
  }
  return new ProportionalRoPE(dims, rotatedDims, ropeTheta);
}
