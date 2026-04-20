import type { MxArray } from "@mlxts/core";
import {
  compile,
  type DisposableTransform,
  expandDims,
  logsumexp,
  mean,
  multiply,
  squeeze,
  subtract,
  takeAlongAxis,
} from "@mlxts/core";

function shapeKey(shape: readonly number[]): string {
  return shape.join("x");
}

function createCrossEntropyVariant(): DisposableTransform<
  (logits: MxArray, targets: MxArray) => MxArray
> {
  return compile((logits: MxArray, targets: MxArray) => {
    using lse = logsumexp(logits, -1, true);
    using logProbs = subtract(logits, lse);
    using targetIndices = expandDims(targets, -1);
    using gathered = takeAlongAxis(logProbs, targetIndices, -1);
    using squeezed = squeeze(gathered, -1);
    using meanLogProb = mean(squeezed);
    return multiply(meanLogProb, -1.0);
  });
}

const crossEntropyVariants = new Map<
  string,
  DisposableTransform<(logits: MxArray, targets: MxArray) => MxArray>
>();

function crossEntropyVariantForShapes(
  logitsShape: readonly number[],
  targetShape: readonly number[],
): DisposableTransform<(logits: MxArray, targets: MxArray) => MxArray> {
  const key = `${shapeKey(logitsShape)}|${shapeKey(targetShape)}`;
  let variant = crossEntropyVariants.get(key);
  if (variant !== undefined) {
    return variant;
  }
  variant = createCrossEntropyVariant();
  crossEntropyVariants.set(key, variant);
  return variant;
}

export function runCrossEntropy(logits: MxArray, targets: MxArray): MxArray {
  return crossEntropyVariantForShapes(logits.shape, targets.shape)(logits, targets);
}
