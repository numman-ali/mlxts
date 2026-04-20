/**
 * Shared gated activation helpers for decoder families.
 * @module
 */

import type { MxArray } from "@mlxts/core";
import { runGegluApprox } from "./runtime";

/** GELU-approx gated activation used by Gemma and GELU-gated LLaMA-like MLPs. */
export function gegluApprox(gate: MxArray, value: MxArray): MxArray {
  return runGegluApprox(gate, value);
}
