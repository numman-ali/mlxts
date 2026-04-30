import {
  collatePreferenceBatch,
  collateTokenSupervisionBatch,
  createRandomSource,
  type PreferenceExample,
  type TokenSupervisionExample,
} from "@mlxts/data";
import type { CausalLM } from "@mlxts/transformers";

import { dpoTrain } from "./dpo";
import { sftTrain } from "./sft";
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
  beta?: number;
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
  learningRate?: number;
  maxGradNorm?: number | null;
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
