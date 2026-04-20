import type { MxArray } from "@mlxts/core";
import { add, compile, divide, erf, multiply, sigmoid } from "@mlxts/core";

const geluTransform = compile(
  (x: MxArray) => {
    using xScaled = divide(x, Math.sqrt(2));
    using erfResult = erf(xScaled);
    using inner = add(erfResult, 1.0);
    using scaled = multiply(x, 0.5);
    return multiply(scaled, inner);
  },
  { shapeless: true },
);

const siluTransform = compile(
  (x: MxArray) => {
    using sig = sigmoid(x);
    return multiply(x, sig);
  },
  { shapeless: true },
);

const swigluTransform = compile(
  (gate: MxArray, value: MxArray) => {
    using sig = sigmoid(gate);
    using activatedGate = multiply(gate, sig);
    return multiply(activatedGate, value);
  },
  { shapeless: true },
);

export function runGelu(x: MxArray): MxArray {
  return geluTransform(x);
}

export function runSilu(x: MxArray): MxArray {
  return siluTransform(x);
}

export function runSwiglu(gate: MxArray, value: MxArray): MxArray {
  return swigluTransform(gate, value);
}
