import {
  divide,
  expandDims,
  log,
  logsumexp,
  type MxArray,
  mean,
  multiply,
  sigmoid,
  squeeze,
  stopGradient,
  subtract,
  sum,
  takeAlongAxis,
} from "@mlxts/core";

import type { PreferenceBatch, TokenBatch } from "@mlxts/data";
import type { CausalLM } from "@mlxts/transformers";

function assertMatchingBatchShapes(logits: MxArray, batch: TokenBatch, context: string): void {
  const expectedShape = logits.shape.slice(0, -1);
  const shapes = [batch.inputIds.shape, batch.targetIds.shape, batch.lossMask.shape];
  for (const shape of shapes) {
    if (shape.length !== expectedShape.length) {
      throw new Error(`${context}: batch tensors must match logits rank without the class axis.`);
    }
    for (let index = 0; index < expectedShape.length; index += 1) {
      if (shape[index] !== expectedShape[index]) {
        throw new Error(
          `${context}: batch tensors must match logits shape without the class axis.`,
        );
      }
    }
  }
}

function maskedTokenNegativeLogProbs(logits: MxArray, batch: TokenBatch, context: string): MxArray {
  assertMatchingBatchShapes(logits, batch, context);
  using lse = logsumexp(logits, -1, true);
  using logProbs = subtract(logits, lse);
  using targetIndices = expandDims(batch.targetIds, -1);
  using gathered = takeAlongAxis(logProbs, targetIndices, -1);
  using squeezed = squeeze(gathered, -1);
  using negated = multiply(squeezed, -1);
  return multiply(negated, batch.lossMask);
}

/** Masked mean next-token loss for SFT-style batches. */
export function maskedTokenLoss(logits: MxArray, batch: TokenBatch): MxArray {
  using masked = maskedTokenNegativeLogProbs(logits, batch, "align.maskedTokenLoss");
  using total = sum(masked);
  using count = sum(batch.lossMask);
  return divide(total, count);
}

function maskedSequenceLogProbs(logits: MxArray, batch: TokenBatch, context: string): MxArray {
  using masked = maskedTokenNegativeLogProbs(logits, batch, context);
  using signed = multiply(masked, -1);
  return sum(signed, 1);
}

/** Sequence-level chosen and rejected log-probs for DPO-style batches. */
export function preferenceLogProbSums(
  model: CausalLM,
  batch: PreferenceBatch,
): {
  chosen: MxArray;
  rejected: MxArray;
} {
  using chosenLogits = model.forward(batch.chosen.inputIds);
  using rejectedLogits = model.forward(batch.rejected.inputIds);
  return {
    chosen: maskedSequenceLogProbs(chosenLogits, batch.chosen, "align.preferenceLogProbSums"),
    rejected: maskedSequenceLogProbs(rejectedLogits, batch.rejected, "align.preferenceLogProbSums"),
  };
}

/** DPO loss over a preference batch. */
export function directPreferenceLoss(
  policyModel: CausalLM,
  referenceModel: CausalLM,
  batch: PreferenceBatch,
  beta: number,
): MxArray {
  const policy = preferenceLogProbSums(policyModel, batch);
  const reference = preferenceLogProbSums(referenceModel, batch);
  try {
    using policyMargin = subtract(policy.chosen, policy.rejected);
    using referenceMargin = subtract(
      stopGradient(reference.chosen),
      stopGradient(reference.rejected),
    );
    using margin = subtract(policyMargin, referenceMargin);
    using scaledMargin = multiply(margin, beta);
    using probabilities = sigmoid(scaledMargin);
    using logProbabilities = log(probabilities);
    using negative = multiply(logProbabilities, -1);
    return mean(negative);
  } finally {
    policy.chosen.free();
    policy.rejected.free();
    reference.chosen.free();
    reference.rejected.free();
  }
}
