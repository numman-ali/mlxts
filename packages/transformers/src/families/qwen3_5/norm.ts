/**
 * Qwen 3.5 normalization layers.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { asType, fastRmsNorm, formatShape, ones } from "@mlxts/core";
import { Module, swiglu } from "@mlxts/nn";

/** Direct RMSNorm used by Qwen 3.5 text layers and q/k norms. */
export class Qwen3_5RMSNorm extends Module {
  weight: MxArray;
  #dims: number;
  #eps: number;

  constructor(dims: number, eps = 1e-6) {
    super();
    if (!Number.isInteger(dims) || dims <= 0) {
      throw new Error(`Qwen3_5RMSNorm: dims must be a positive integer, got ${dims}.`);
    }
    this.weight = ones([dims]);
    this.#dims = dims;
    this.#eps = eps;
  }

  forward(x: MxArray): MxArray {
    const lastDimension = x.shape[x.shape.length - 1];
    if (lastDimension !== this.#dims) {
      throw new Error(
        `Qwen3_5RMSNorm.forward: expected last dimension ${this.#dims}, got ${lastDimension ?? "undefined"} for shape ${formatShape(x.shape)}.`,
      );
    }

    return fastRmsNorm(x, this.weight, { eps: this.#eps });
  }
}

/** Gated RMSNorm used by Qwen 3.5 linear-attention outputs. */
export class Qwen3_5RMSNormGated extends Module {
  weight: MxArray;
  #dims: number;
  #eps: number;

  constructor(dims: number, eps = 1e-6) {
    super();
    if (!Number.isInteger(dims) || dims <= 0) {
      throw new Error(`Qwen3_5RMSNormGated: dims must be a positive integer, got ${dims}.`);
    }
    this.weight = ones([dims]);
    this.#dims = dims;
    this.#eps = eps;
  }

  forward(x: MxArray, gate: MxArray): MxArray {
    const lastDimension = x.shape[x.shape.length - 1];
    const gateLastDimension = gate.shape[gate.shape.length - 1];
    if (lastDimension !== this.#dims) {
      throw new Error(
        `Qwen3_5RMSNormGated.forward: expected hidden last dimension ${this.#dims}, got ${lastDimension ?? "undefined"} for shape ${formatShape(x.shape)}.`,
      );
    }
    if (gateLastDimension !== this.#dims) {
      throw new Error(
        `Qwen3_5RMSNormGated.forward: expected gate last dimension ${this.#dims}, got ${gateLastDimension ?? "undefined"} for shape ${formatShape(gate.shape)}.`,
      );
    }

    using normalized = fastRmsNorm(x, this.weight, { eps: this.#eps });
    using floatGate = asType(gate, "float32");
    using floatNormalized = asType(normalized, "float32");
    using precise = swiglu(floatGate, floatNormalized);
    return asType(precise, x.dtype);
  }
}
