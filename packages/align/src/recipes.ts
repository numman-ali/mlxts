import { type MxArray, mxEval } from "@mlxts/core";
import {
  collatePreferenceBatch,
  collateTokenSupervisionBatch,
  createRandomSource,
  type PreferenceExample,
  type TokenSupervisionExample,
} from "@mlxts/data";
import type { CausalLM } from "@mlxts/transformers";

import { dpoLoss, dpoTrain } from "./dpo";
import { preferenceRewardSums } from "./loss-utils";
import { sftLoss, sftTrain } from "./sft";
import type { OptimizerLike } from "./sft-types";

export type SupervisionDatasetOptions = {
  examples: readonly TokenSupervisionExample[];
  padTokenId: number;
  batchSize: number;
};

export type PreferenceDatasetOptions = {
  examples: readonly PreferenceExample[];
  referenceModel: CausalLM;
  padTokenId: number;
  batchSize: number;
};

export type SupervisionTrainingStepsOptions = SupervisionDatasetOptions & {
  optimizer: OptimizerLike;
  steps: number;
  seed: number;
  learningRate?: number;
  maxGradNorm?: number | null;
};

export type PreferenceTrainingStepsOptions = PreferenceDatasetOptions & {
  optimizer: OptimizerLike;
  steps: number;
  seed: number;
  beta?: number;
  learningRate?: number;
  maxGradNorm?: number | null;
};

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
): number {
  try {
    return evaluateLoss(dpoLoss(policyModel, referenceModel, batch));
  } finally {
    freePreferenceBatch(batch);
  }
}

function createShuffledOrder(length: number, seed: number): number[] {
  const nextRandom = createRandomSource(seed);
  const order = Array.from({ length }, (_, index) => index);
  for (let index = order.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(nextRandom() * (index + 1));
    const current = order[index];
    const swap = order[swapIndex];
    if (current === undefined || swap === undefined) {
      throw new Error("align.createShuffledOrder: selected an undefined shuffle index.");
    }
    order[index] = swap;
    order[swapIndex] = current;
  }
  return order;
}

function createBatchPicker<T>(
  examples: readonly T[],
  batchSize: number,
  seed: number,
  context: string,
): () => readonly T[] {
  const order = createShuffledOrder(examples.length, seed);
  let cursor = 0;
  return () => {
    const batch: T[] = [];
    for (let index = 0; index < batchSize; index += 1) {
      const exampleIndex = order[cursor % order.length];
      if (exampleIndex === undefined) {
        throw new Error(`${context}: selected an undefined batch index.`);
      }
      const example = examples[exampleIndex];
      if (example === undefined) {
        throw new Error(`${context}: selected an undefined batch example.`);
      }
      batch.push(example);
      cursor += 1;
    }
    return batch;
  };
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
    );
    batches += 1;
  }
  return totalLoss / batches;
}

/** Evaluate reward-aware DPO metrics over a preference-example dataset. */
export function evaluatePreferenceMetrics(
  policyModel: CausalLM,
  options: PreferenceDatasetOptions & { beta?: number },
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

/** Run several SFT steps over supervision examples with deterministic batch picking. */
export function runSupervisionTrainingSteps(
  model: CausalLM,
  options: SupervisionTrainingStepsOptions,
): { averageLoss: number } {
  expectExamples(options.examples, "align.runSupervisionTrainingSteps");
  expectPositiveInteger(options.batchSize, "batchSize", "align.runSupervisionTrainingSteps");
  expectPositiveInteger(options.steps, "steps", "align.runSupervisionTrainingSteps");

  const nextBatch = createBatchPicker(
    options.examples,
    options.batchSize,
    options.seed,
    "align.runSupervisionTrainingSteps",
  );
  let totalTrainingLoss = 0;
  for (let step = 0; step < options.steps; step += 1) {
    const trainOptions: {
      optimizer: OptimizerLike;
      batches: ReturnType<typeof collateTokenSupervisionBatch>[];
      learningRate?: number;
      maxGradNorm?: number | null;
    } = {
      optimizer: options.optimizer,
      batches: [collateTokenSupervisionBatch(nextBatch(), options.padTokenId)],
    };
    if (options.learningRate !== undefined) {
      trainOptions.learningRate = options.learningRate;
    }
    if (options.maxGradNorm !== undefined) {
      trainOptions.maxGradNorm = options.maxGradNorm;
    }
    const result = sftTrain(model, trainOptions);
    totalTrainingLoss += result.averageLoss;
  }

  return {
    averageLoss: totalTrainingLoss / options.steps,
  };
}

/** Run several DPO steps over preference examples with deterministic batch picking. */
export function runPreferenceTrainingSteps(
  policyModel: CausalLM,
  options: PreferenceTrainingStepsOptions,
): { averageLoss: number } {
  expectExamples(options.examples, "align.runPreferenceTrainingSteps");
  expectPositiveInteger(options.batchSize, "batchSize", "align.runPreferenceTrainingSteps");
  expectPositiveInteger(options.steps, "steps", "align.runPreferenceTrainingSteps");

  const nextBatch = createBatchPicker(
    options.examples,
    options.batchSize,
    options.seed,
    "align.runPreferenceTrainingSteps",
  );
  let totalTrainingLoss = 0;
  for (let step = 0; step < options.steps; step += 1) {
    const trainOptions: {
      referenceModel: CausalLM;
      optimizer: OptimizerLike;
      batches: ReturnType<typeof collatePreferenceBatch>[];
      beta?: number;
      learningRate?: number;
      maxGradNorm?: number | null;
    } = {
      referenceModel: options.referenceModel,
      optimizer: options.optimizer,
      batches: [collatePreferenceBatch(nextBatch(), options.padTokenId)],
    };
    if (options.beta !== undefined) {
      trainOptions.beta = options.beta;
    }
    if (options.learningRate !== undefined) {
      trainOptions.learningRate = options.learningRate;
    }
    if (options.maxGradNorm !== undefined) {
      trainOptions.maxGradNorm = options.maxGradNorm;
    }
    const result = dpoTrain(policyModel, trainOptions);
    totalTrainingLoss += result.averageLoss;
  }

  return {
    averageLoss: totalTrainingLoss / options.steps,
  };
}
