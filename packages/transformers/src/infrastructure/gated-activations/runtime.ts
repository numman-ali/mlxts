import { compile, geluApprox, type MxArray, multiply } from "@mlxts/core";

const gegluApproxTransform = compile(
  (gate: MxArray, value: MxArray) => {
    using geluResult = geluApprox(gate);
    return multiply(geluResult, value);
  },
  { shapeless: true },
);

export function runGegluApprox(gate: MxArray, value: MxArray): MxArray {
  return gegluApproxTransform(gate, value);
}
