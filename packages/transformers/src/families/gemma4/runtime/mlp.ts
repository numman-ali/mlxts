/**
 * Gemma 4 MLP runtime helpers.
 * @module
 */

import { compile, type DisposableTransform, type MxArray, matmul, transpose } from "@mlxts/core";
import { gegluApprox } from "../../../infrastructure/gated-activations";

function shapeKey(shape: readonly number[]): string {
  return shape.join("x");
}

type MlpTransform = DisposableTransform<
  (x: MxArray, gateWeight: MxArray, upWeight: MxArray, downWeight: MxArray) => MxArray
>;

function createMlpVariant(): MlpTransform {
  return compile((x: MxArray, gateWeight: MxArray, upWeight: MxArray, downWeight: MxArray) => {
    using gateWeightTranspose = transpose(gateWeight);
    using upWeightTranspose = transpose(upWeight);
    using downWeightTranspose = transpose(downWeight);
    using gate = matmul(x, gateWeightTranspose);
    using value = matmul(x, upWeightTranspose);
    using activated = gegluApprox(gate, value);
    return matmul(activated, downWeightTranspose);
  });
}

const mlpVariants = new Map<string, MlpTransform>();

function mlpVariantFor(
  xShape: readonly number[],
  gateWeightShape: readonly number[],
  upWeightShape: readonly number[],
  downWeightShape: readonly number[],
): MlpTransform {
  const key =
    `${shapeKey(xShape)}|` +
    `${shapeKey(gateWeightShape)}|` +
    `${shapeKey(upWeightShape)}|` +
    `${shapeKey(downWeightShape)}`;
  let variant = mlpVariants.get(key);
  if (variant !== undefined) {
    return variant;
  }
  variant = createMlpVariant();
  mlpVariants.set(key, variant);
  return variant;
}

export function runMlp(
  x: MxArray,
  gateWeight: MxArray,
  upWeight: MxArray,
  downWeight: MxArray,
): MxArray {
  return mlpVariantFor(x.shape, gateWeight.shape, upWeight.shape, downWeight.shape)(
    x,
    gateWeight,
    upWeight,
    downWeight,
  );
}
