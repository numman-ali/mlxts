/**
 * Activation functions.
 *
 * Free functions that compose from existing core ops.
 * These are not Module subclasses — they have no learnable parameters.
 *
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { add, divide, erf, maximum, multiply, sigmoid } from "@mlxts/core";

/** GELU activation: x * 0.5 * (1 + erf(x / sqrt(2))). */
export function gelu(x: MxArray): MxArray {
  using xScaled = divide(x, Math.sqrt(2));
  using erfResult = erf(xScaled);
  using inner = add(erfResult, 1.0);
  using scaled = multiply(x, 0.5);
  return multiply(scaled, inner);
}

/** ReLU activation: max(x, 0). */
export function relu(x: MxArray): MxArray {
  return maximum(x, 0);
}

/** SiLU (Swish) activation: x * sigmoid(x). */
export function silu(x: MxArray): MxArray {
  using sig = sigmoid(x);
  return multiply(x, sig);
}
