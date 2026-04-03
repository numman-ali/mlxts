/**
 * Loss functions for neural network training.
 *
 * All losses return scalar MxArray values suitable for autograd.
 *
 * @module
 */

import type { MxArray } from "../core/array";
import { multiply, square, subtract } from "../core/ops/arithmetic";
import { logsumexp, mean } from "../core/ops/reduction";
import { expandDims, squeeze, takeAlongAxis } from "../core/ops/shape";

const INTEGER_DTYPES: ReadonlySet<string> = new Set([
  "int8",
  "int16",
  "int32",
  "int64",
  "uint8",
  "uint16",
  "uint32",
  "uint64",
]);

function assertIntegerDtype(arr: MxArray, name: string): void {
  if (!INTEGER_DTYPES.has(arr.dtype)) {
    throw new Error(
      `${name}: targets must be integer dtype (int32, uint32, etc.), got ${arr.dtype}.\n` +
        `  Hint: use array([1, 2, 3], "int32") to create integer indices.`,
    );
  }
}

function formatShape(shape: readonly number[]): string {
  return shape.length === 0 ? "[]" : `[${shape.join(", ")}]`;
}

function shapesEqual(left: readonly number[], right: readonly number[]): boolean {
  if (left.length !== right.length) {
    return false;
  }

  for (let index = 0; index < left.length; index++) {
    if (left[index] !== right[index]) {
      return false;
    }
  }

  return true;
}

function assertCrossEntropyShapes(logits: MxArray, targets: MxArray): void {
  if (logits.ndim < 1) {
    throw new Error(
      `crossEntropy: logits must have at least 1 dimension for the class axis, got shape ${formatShape(logits.shape)}`,
    );
  }

  const classCount = logits.shape[logits.shape.length - 1];
  if (classCount === undefined || classCount <= 0) {
    throw new Error(
      `crossEntropy: logits class axis must have size > 0, got shape ${formatShape(logits.shape)}`,
    );
  }

  const expectedTargetShape = logits.shape.slice(0, -1);
  if (targets.ndim !== expectedTargetShape.length) {
    throw new Error(
      `crossEntropy: targets rank must be logits rank - 1. ` +
        `Got logits shape ${formatShape(logits.shape)} and targets shape ${formatShape(targets.shape)}.`,
    );
  }

  if (!shapesEqual(targets.shape, expectedTargetShape)) {
    throw new Error(
      `crossEntropy: targets shape must match logits shape without the class axis. ` +
        `Expected ${formatShape(expectedTargetShape)}, got ${formatShape(targets.shape)}.`,
    );
  }
}

/**
 * Cross-entropy loss for classification.
 *
 * Computes the mean negative log-probability of the correct class.
 * Uses logsumexp for numerical stability.
 *
 * @param logits - Unnormalized predictions, shape [..., numClasses].
 * @param targets - Integer class indices, shape [...]. Must be integer dtype.
 * @returns Scalar loss (mean cross-entropy over all elements).
 */
export function crossEntropy(logits: MxArray, targets: MxArray): MxArray {
  assertIntegerDtype(targets, "crossEntropy");
  assertCrossEntropyShapes(logits, targets);

  // Log-softmax: logits - logsumexp(logits, axis=-1, keepdims=true)
  using lse = logsumexp(logits, -1, true);
  using logProbs = subtract(logits, lse);

  // Gather log-prob of the correct class
  using targetIndices = expandDims(targets, -1); // [...] → [..., 1]
  using gathered = takeAlongAxis(logProbs, targetIndices, -1); // [..., 1]
  using squeezed = squeeze(gathered, -1); // [..., 1] → [...]

  // Mean negative log-probability
  using meanLogProb = mean(squeezed);
  return multiply(meanLogProb, -1.0);
}

/**
 * Mean squared error loss.
 *
 * @param predictions - Predicted values.
 * @param targets - Target values (same shape as predictions).
 * @returns Scalar MSE loss.
 */
export function mse(predictions: MxArray, targets: MxArray): MxArray {
  using diff = subtract(predictions, targets);
  using sq = square(diff);
  return mean(sq);
}
