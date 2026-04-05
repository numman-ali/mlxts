/**
 * Rotary positional embeddings.
 *
 * Thin module wrapper over the fused MLX RoPE kernel.
 *
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { fastRoPE } from "@mlxts/core";
import { Module } from "./module";

/** Rotary positional embeddings applied over the last axis. */
export class RoPE extends Module {
  #dims: number;
  #traditional: boolean;
  #base: number;
  #scale: number;

  /**
   * @param dims - Number of feature dimensions to rotate. Must be a positive even integer.
   * @param traditional - Whether to use the traditional consecutive-dimension rotation.
   * @param base - RoPE base frequency. Defaults to 10000.
   * @param scale - Position scaling factor. Defaults to 1.0.
   */
  constructor(dims: number, traditional = false, base = 10000, scale = 1.0) {
    super();
    if (!Number.isInteger(dims) || dims <= 0 || dims % 2 !== 0) {
      throw new Error(`RoPE: dims must be a positive even integer, got ${dims}`);
    }
    this.#dims = dims;
    this.#traditional = traditional;
    this.#base = base;
    this.#scale = scale;
  }

  forward(x: MxArray, offset: number | MxArray = 0): MxArray {
    return fastRoPE(x, this.#dims, {
      traditional: this.#traditional,
      base: this.#base,
      scale: this.#scale,
      offset,
    });
  }
}
