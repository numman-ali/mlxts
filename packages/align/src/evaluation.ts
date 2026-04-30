import { type MxArray, mxEval } from "@mlxts/core";
import { collatePreferenceBatch, collateTokenSupervisionBatch } from "@mlxts/data";
import type { CausalLM } from "@mlxts/transformers";

import { dpoLoss } from "./dpo";
import { preferenceRewardSums } from "./loss-utils";
import type { PreferenceDatasetOptions, SupervisionDatasetOptions } from "./recipes";
import { sftLoss } from "./sft";

export type PreferenceEvalMetrics = {
  rewardAccuracy: number;
  rewardMargin: number;
  chosenReward: number;
  rejectedReward: number;
  chosenLogProb: number;
  rejectedLogProb: number;
  rawPreferenceAccuracy: number;
};

type PreferenceMetricArrays = {
  chosenLogProbs: number[];
  rejectedLogProbs: number[];
  chosenRewards: number[];
  rejectedRewards: number[];
  rewardMargins: number[];
};

type PreferenceMetricTotals = {
  rewardWins: number;
  rawWins: number;
  total: number;
  rewardMarginTotal: number;
  chosenRewardTotal: number;
  rejectedRewardTotal: number;
  chosenLogProbTotal: number;
  rejectedLogProbTotal: number;
};

function expectExamples<T>(examples: readonly T[], context: string): void {
  if (examples.length === 0) {
    throw new Error(`${context}: expected at least one example.`);
  }
}

function expectPositiveInteger(value: number, name: string, context: string): void {
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`${context}: ${name} must be a positive integer.`);
  }
}

function chunkExamples<T>(examples: readonly T[], batchSize: number): T[][] {
  const chunks: T[][] = [];
  for (let index = 0; index < examples.length; index += batchSize) {
    chunks.push(examples.slice(index, index + batchSize));
  }
  return chunks;
}

function evaluateLoss(loss: ReturnType<typeof sftLoss> | ReturnType<typeof dpoLoss>): number {
  mxEval(loss);
  const value = loss.item();
  loss.free();
  return value;
}

function freeTokenBatch(batch: ReturnType<typeof collateTokenSupervisionBatch>): void {
  batch.inputIds.free();
  batch.targetIds.free();
  batch.lossMask.free();
}

function freePreferenceBatch(batch: ReturnType<typeof collatePreferenceBatch>): void {
  freeTokenBatch(batch.chosen);
  freeTokenBatch(batch.rejected);
}

function evaluateSftBatchLoss(
  model: CausalLM,
  batch: ReturnType<typeof collateTokenSupervisionBatch>,
): number {
  try {
    return evaluateLoss(sftLoss(model, batch));
  } finally {
    freeTokenBatch(batch);
  }
}

function evaluateDpoBatchLoss(
  policyModel: CausalLM,
  referenceModel: CausalLM,
  batch: ReturnType<typeof collatePreferenceBatch>,
  beta: number | undefined,
): number {
  try {
    return evaluateLoss(dpoLoss(policyModel, referenceModel, batch, beta));
  } finally {
    freePreferenceBatch(batch);
  }
}

function readNumberList(array: MxArray, context: string): number[] {
  const values = array.toList();
  if (!Array.isArray(values)) {
    throw new Error(`${context}: expected a flat numeric array.`);
  }
  const numbers: number[] = [];
  for (const value of values) {
    if (typeof value !== "number") {
      throw new Error(`${context}: expected numeric values.`);
    }
    numbers.push(value);
  }
  return numbers;
}

function createPreferenceMetricTotals(): PreferenceMetricTotals {
  return {
    rewardWins: 0,
    rawWins: 0,
    total: 0,
    rewardMarginTotal: 0,
    chosenRewardTotal: 0,
    rejectedRewardTotal: 0,
    chosenLogProbTotal: 0,
    rejectedLogProbTotal: 0,
  };
}

function emptyPreferenceEvalMetrics(): PreferenceEvalMetrics {
  return {
    rewardAccuracy: 0,
    rewardMargin: 0,
    chosenReward: 0,
    rejectedReward: 0,
    chosenLogProb: 0,
    rejectedLogProb: 0,
    rawPreferenceAccuracy: 0,
  };
}

function metricValue(values: readonly number[], index: number, context: string): number {
  const value = values[index];
  if (value === undefined) {
    throw new Error(`${context}: missing metric value at index ${index}.`);
  }
  return value;
}

function readPreferenceMetricArrays(
  metrics: ReturnType<typeof preferenceRewardSums>,
  context: string,
): PreferenceMetricArrays {
  mxEval(
    metrics.chosenLogProbs,
    metrics.rejectedLogProbs,
    metrics.chosenRewards,
    metrics.rejectedRewards,
    metrics.rewardMargins,
  );

  return {
    chosenLogProbs: readNumberList(metrics.chosenLogProbs, context),
    rejectedLogProbs: readNumberList(metrics.rejectedLogProbs, context),
    chosenRewards: readNumberList(metrics.chosenRewards, context),
    rejectedRewards: readNumberList(metrics.rejectedRewards, context),
    rewardMargins: readNumberList(metrics.rewardMargins, context),
  };
}

function freePreferenceRewardMetrics(metrics: ReturnType<typeof preferenceRewardSums>): void {
  metrics.chosenLogProbs.free();
  metrics.rejectedLogProbs.free();
  metrics.chosenRewards.free();
  metrics.rejectedRewards.free();
  metrics.rewardMargins.free();
}

function addPreferenceMetricTotals(
  totals: PreferenceMetricTotals,
  arrays: PreferenceMetricArrays,
  context: string,
): void {
  for (let index = 0; index < arrays.chosenLogProbs.length; index += 1) {
    const chosenLogProb = metricValue(arrays.chosenLogProbs, index, context);
    const rejectedLogProb = metricValue(arrays.rejectedLogProbs, index, context);
    const chosenReward = metricValue(arrays.chosenRewards, index, context);
    const rejectedReward = metricValue(arrays.rejectedRewards, index, context);
    const rewardMargin = metricValue(arrays.rewardMargins, index, context);

    if (rewardMargin > 0) {
      totals.rewardWins += 1;
    }
    if (chosenLogProb > rejectedLogProb) {
      totals.rawWins += 1;
    }
    totals.rewardMarginTotal += rewardMargin;
    totals.chosenRewardTotal += chosenReward;
    totals.rejectedRewardTotal += rejectedReward;
    totals.chosenLogProbTotal += chosenLogProb;
    totals.rejectedLogProbTotal += rejectedLogProb;
    totals.total += 1;
  }
}

/** Evaluate mean SFT loss over a supervision-example dataset. */
export function evaluateSupervisionDatasetLoss(
  model: CausalLM,
  options: SupervisionDatasetOptions,
): number {
  expectExamples(options.examples, "align.evaluateSupervisionDatasetLoss");
  expectPositiveInteger(options.batchSize, "batchSize", "align.evaluateSupervisionDatasetLoss");

  let totalLoss = 0;
  let batches = 0;
  for (const chunk of chunkExamples(options.examples, options.batchSize)) {
    totalLoss += evaluateSftBatchLoss(
      model,
      collateTokenSupervisionBatch(chunk, options.padTokenId),
    );
    batches += 1;
  }
  return totalLoss / batches;
}

/** Evaluate mean DPO loss over a preference-example dataset. */
export function evaluatePreferenceDatasetLoss(
  policyModel: CausalLM,
  options: PreferenceDatasetOptions,
): number {
  expectExamples(options.examples, "align.evaluatePreferenceDatasetLoss");
  expectPositiveInteger(options.batchSize, "batchSize", "align.evaluatePreferenceDatasetLoss");

  let totalLoss = 0;
  let batches = 0;
  for (const chunk of chunkExamples(options.examples, options.batchSize)) {
    totalLoss += evaluateDpoBatchLoss(
      policyModel,
      options.referenceModel,
      collatePreferenceBatch(chunk, options.padTokenId),
      options.beta,
    );
    batches += 1;
  }
  return totalLoss / batches;
}

/** Evaluate reward-aware DPO metrics over a preference-example dataset. */
export function evaluatePreferenceMetrics(
  policyModel: CausalLM,
  options: PreferenceDatasetOptions,
): PreferenceEvalMetrics {
  const context = "align.evaluatePreferenceMetrics";
  expectExamples(options.examples, context);
  expectPositiveInteger(options.batchSize, "batchSize", context);
  const totals = createPreferenceMetricTotals();

  for (const chunk of chunkExamples(options.examples, options.batchSize)) {
    const batch = collatePreferenceBatch(chunk, options.padTokenId);
    try {
      const metrics = preferenceRewardSums(
        policyModel,
        options.referenceModel,
        batch,
        options.beta ?? 0.1,
      );
      try {
        const arrays = readPreferenceMetricArrays(metrics, context);
        addPreferenceMetricTotals(totals, arrays, context);
      } finally {
        freePreferenceRewardMetrics(metrics);
      }
    } finally {
      freePreferenceBatch(batch);
    }
  }

  if (totals.total === 0) {
    return emptyPreferenceEvalMetrics();
  }

  return {
    rewardAccuracy: totals.rewardWins / totals.total,
    rewardMargin: totals.rewardMarginTotal / totals.total,
    chosenReward: totals.chosenRewardTotal / totals.total,
    rejectedReward: totals.rejectedRewardTotal / totals.total,
    chosenLogProb: totals.chosenLogProbTotal / totals.total,
    rejectedLogProb: totals.rejectedLogProbTotal / totals.total,
    rawPreferenceAccuracy: totals.rawWins / totals.total,
  };
}
