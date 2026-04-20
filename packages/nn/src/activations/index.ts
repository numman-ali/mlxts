/**
 * Activation functions.
 *
 * Free functions that compose from existing core ops.
 * These are not Module subclasses — they have no learnable parameters.
 *
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { maximum } from "@mlxts/core";
import { runGelu, runSilu, runSwiglu } from "./runtime";

/** Exact GELU activation. Prefer `geluApprox` from `@mlxts/core` on inference hot paths. */
export function gelu(x: MxArray): MxArray {
  return runGelu(x);
}

/** ReLU activation: max(x, 0). */
export function relu(x: MxArray): MxArray {
  return maximum(x, 0);
}

/** SiLU (Swish) activation: x * sigmoid(x). */
export function silu(x: MxArray): MxArray {
  return runSilu(x);
}

/** SwiGLU activation: silu(gate) * value. */
export function swiglu(gate: MxArray, value: MxArray): MxArray {
  return runSwiglu(gate, value);
}
