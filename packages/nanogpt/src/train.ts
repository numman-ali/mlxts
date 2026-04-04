/**
 * GPT training loop with gradient accumulation and cosine LR schedule.
 *
 * Designed as a structured API: the train() function accepts an event
 * callback, returns a summary, and leaves rendering concerns to the CLI.
 *
 * @module
 */

import {
  add,
  type MxArray,
  multiply,
  mxEval,
  type ParameterTree,
  random,
  reshape,
  square,
  sum,
  synchronize,
  treeFlatten,
  treeLeaves,
  treeUnflatten,
} from "@mlxts/core";
import { crossEntropy, valueAndGrad as moduleValueAndGrad } from "@mlxts/nn";
import type { AdamW } from "@mlxts/optimizers";

import type { GPTConfig } from "./config";
import { createRandomSource, getBatch } from "./data";
import type { GPT } from "./model/gpt";
import { createDefaultAdamW } from "./optimizer-defaults";

/** Training hyperparameters. */
export interface TrainConfig {
  startStep?: number;
  maxSteps: number;
  batchSize: number;
  learningRate: number;
  weightDecay: number;
  warmupSteps: number;
  minLearningRate: number;
  gradAccumSteps: number;
  evalInterval: number;
  evalSteps: number;
  logInterval: number;
  maxGradNorm: number | null;
  seed: number;
}

/** Events emitted during training for structured rendering. */
export type TrainEvent =
  | { type: "step"; step: number; loss: number; learningRate: number; tokensPerSec: number }
  | {
      type: "progress";
      phase: "eval";
      split: "train" | "val";
      step: number;
      completed: number;
      total: number;
    }
  | { type: "eval"; step: number; trainLoss: number; valLoss: number }
  | { type: "done"; totalSteps: number };

/** All inputs needed for a training run. */
export interface TrainOptions {
  model: GPT;
  optimizer?: AdamW;
  config: GPTConfig;
  trainConfig: TrainConfig;
  trainTokens: Int32Array;
  valTokens: Int32Array;
  onEvent?: (event: TrainEvent) => void;
  shouldStop?: () => boolean;
}

/** Structured result of a completed training run. */
export interface TrainSummary {
  totalSteps: number;
  lastStepLoss: number | null;
  lastTrainLoss: number | null;
  lastValLoss: number | null;
}

/** Cosine decay with linear warmup. */
export function getLearningRate(step: number, config: TrainConfig): number {
  if (step < config.warmupSteps) {
    return config.learningRate * (step / config.warmupSteps);
  }
  if (step >= config.maxSteps) {
    return config.minLearningRate;
  }

  const decayRatio = (step - config.warmupSteps) / (config.maxSteps - config.warmupSteps);
  const coeff = 0.5 * (1 + Math.cos(Math.PI * decayRatio));
  return config.minLearningRate + coeff * (config.learningRate - config.minLearningRate);
}

function restoreTrainingMode(model: GPT, wasTraining: boolean): void {
  if (wasTraining) {
    model.train();
    return;
  }
  model.eval();
}

function readIntegerAtLeast(
  value: number | undefined,
  minimum: number,
  name: string,
  requirement: string,
): void {
  if (value === undefined) {
    return;
  }
  if (!Number.isInteger(value) || value < minimum) {
    throw new Error(`TrainConfig: ${name} must be ${requirement}`);
  }
}

function readFiniteAtLeast(
  value: number,
  minimum: number,
  name: string,
  requirement: string,
): void {
  if (!Number.isFinite(value) || value < minimum) {
    throw new Error(`TrainConfig: ${name} must be ${requirement}`);
  }
}

function readPositiveFinite(value: number, name: string): void {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`TrainConfig: ${name} must be > 0`);
  }
}

function validateTrainConfig(config: TrainConfig): void {
  readIntegerAtLeast(config.startStep, 0, "startStep", "a non-negative integer");
  readIntegerAtLeast(config.maxSteps, 1, "maxSteps", "> 0");
  readIntegerAtLeast(config.batchSize, 1, "batchSize", "> 0");
  readPositiveFinite(config.learningRate, "learningRate");
  readFiniteAtLeast(config.weightDecay, 0, "weightDecay", ">= 0");
  readIntegerAtLeast(config.warmupSteps, 0, "warmupSteps", ">= 0");
  if (config.warmupSteps >= config.maxSteps) {
    throw new Error("TrainConfig: warmupSteps must be < maxSteps");
  }
  if ((config.startStep ?? 0) >= config.maxSteps) {
    throw new Error("TrainConfig: startStep must be < maxSteps");
  }
  readPositiveFinite(config.minLearningRate, "minLearningRate");
  readIntegerAtLeast(config.gradAccumSteps, 1, "gradAccumSteps", ">= 1");
  readIntegerAtLeast(config.evalInterval, 1, "evalInterval", "> 0");
  readIntegerAtLeast(config.evalSteps, 1, "evalSteps", "> 0");
  readIntegerAtLeast(config.logInterval, 1, "logInterval", "> 0");
  if (config.maxGradNorm !== null) {
    readPositiveFinite(config.maxGradNorm, "maxGradNorm");
  }
  if (!Number.isFinite(config.seed)) throw new Error("TrainConfig: seed must be a finite number");
}

function validateTokenSplitLength(
  label: "train" | "validation",
  tokens: Int32Array,
  blockSize: number,
): void {
  const minimumLength = blockSize + 1;
  if (tokens.length < minimumLength) {
    throw new Error(
      `train: ${label} token split length ${tokens.length} is too short for blockSize ${blockSize}; need at least ${minimumLength} tokens`,
    );
  }
}

/** Free all MxArray leaves in a gradient tree. */
function freeGradTree(tree: ParameterTree): void {
  for (const [, value] of treeFlatten(tree)) {
    value.free();
  }
}

function evalGradTree(tree: ParameterTree): void {
  const leaves = treeLeaves(tree);
  if (leaves.length > 0) {
    mxEval(...leaves);
  }
}

type LossAndGradFn = (input: MxArray, target: MxArray) => [MxArray, ParameterTree];
type FlatGradientEntry = [path: string[], value: MxArray];
type StepResult = {
  loss: number;
  tokensPerSec: number;
};

function formatGradientPath(path: readonly string[]): string {
  return path.length === 0 ? "<root>" : path.join(".");
}

function pathKey(path: readonly string[]): string {
  return path.join(".");
}

function assertMatchingGradientEntries(
  left: readonly FlatGradientEntry[],
  right: readonly FlatGradientEntry[],
  context: string,
): void {
  if (left.length !== right.length) {
    throw new Error(`${context}: gradient tree leaf counts do not match`);
  }

  for (let index = 0; index < left.length; index++) {
    const leftEntry = left[index];
    const rightEntry = right[index];
    if (leftEntry === undefined || rightEntry === undefined) {
      throw new Error(`${context}: missing gradient leaf at index ${index}`);
    }
    if (pathKey(leftEntry[0]) !== pathKey(rightEntry[0])) {
      throw new Error(
        `${context}: gradient path mismatch at index ${index} (${formatGradientPath(leftEntry[0])} vs ${formatGradientPath(rightEntry[0])})`,
      );
    }
  }
}

function mapGradientEntries(
  entries: readonly FlatGradientEntry[],
  mapper: (value: MxArray, path: readonly string[]) => MxArray,
  context: string,
): FlatGradientEntry[] {
  const mapped: FlatGradientEntry[] = [];
  try {
    for (const [path, value] of entries) {
      mapped.push([[...path], mapper(value, path)]);
    }
    return mapped;
  } catch (error) {
    for (const [, value] of mapped) {
      value.free();
    }
    throw new Error(
      error instanceof Error ? `${context}: ${error.message}` : `${context}: ${String(error)}`,
    );
  }
}

function gradientEntriesToTree(entries: FlatGradientEntry[], context: string): ParameterTree {
  try {
    return treeUnflatten(entries);
  } catch (error) {
    for (const [, value] of entries) {
      value.free();
    }
    throw new Error(
      error instanceof Error ? `${context}: ${error.message}` : `${context}: ${String(error)}`,
    );
  }
}

function accumulateGradients(accumulated: ParameterTree, next: ParameterTree): ParameterTree {
  const accumulatedEntries = treeFlatten(accumulated);
  const nextEntries = treeFlatten(next);
  assertMatchingGradientEntries(accumulatedEntries, nextEntries, "train.accumulateGradients");

  const summedEntries: FlatGradientEntry[] = [];
  try {
    for (let index = 0; index < accumulatedEntries.length; index++) {
      const accumulatedEntry = accumulatedEntries[index];
      const nextEntry = nextEntries[index];
      if (accumulatedEntry === undefined || nextEntry === undefined) {
        throw new Error(`missing gradient leaf at index ${index}`);
      }
      summedEntries.push([[...accumulatedEntry[0]], add(accumulatedEntry[1], nextEntry[1])]);
    }
  } catch (error) {
    for (const [, value] of summedEntries) {
      value.free();
    }
    throw new Error(
      error instanceof Error
        ? `train.accumulateGradients: ${error.message}`
        : `train.accumulateGradients: ${String(error)}`,
    );
  }

  return gradientEntriesToTree(summedEntries, "train.accumulateGradients");
}

function scaleGradientTree(tree: ParameterTree, factor: number): ParameterTree {
  if (factor === 1) {
    return tree;
  }

  const entries = treeFlatten(tree);
  const scaledEntries = mapGradientEntries(
    entries,
    (grad) => multiply(grad, factor),
    "train.scaleGradientTree",
  );
  return gradientEntriesToTree(scaledEntries, "train.scaleGradientTree");
}

function gradientNorm(tree: ParameterTree): number {
  let totalSquared = 0;
  for (const grad of treeLeaves(tree)) {
    using squared = square(grad);
    using squaredSum = sum(squared);
    mxEval(squaredSum);
    totalSquared += squaredSum.item();
  }
  return Math.sqrt(totalSquared);
}

function assertFiniteValue(value: number, context: string): void {
  if (!Number.isFinite(value)) {
    throw new Error(`${context}: encountered non-finite value (${String(value)})`);
  }
}

function materializeStepState(model: GPT, optimizer: AdamW): void {
  const arrays = [...treeLeaves(model.parameters()), ...optimizer.stateArrays()];
  if (arrays.length > 0) {
    mxEval(...arrays);
  }
  synchronize();
}

function takeMicroStep(
  lossAndGrad: LossAndGradFn,
  tokens: Int32Array,
  batchSize: number,
  blockSize: number,
  nextRandom: () => number,
): { lossValue: number; gradients: ParameterTree } {
  const { input, target } = getBatch(tokens, batchSize, blockSize, nextRandom);
  let loss: MxArray | null = null;
  let gradients: ParameterTree | null = null;

  try {
    [loss, gradients] = lossAndGrad(input, target);
    mxEval(loss);
    if (gradients === null) {
      throw new Error("train: lossAndGrad produced no gradients");
    }
    evalGradTree(gradients);
    const lossValue = loss.item();
    assertFiniteValue(lossValue, "train");
    const result = { lossValue, gradients };
    gradients = null;
    return result;
  } finally {
    loss?.free();
    if (gradients !== null) {
      freeGradTree(gradients);
    }
    input.free();
    target.free();
  }
}

function accumulateMicroSteps(
  lossAndGrad: LossAndGradFn,
  trainTokens: Int32Array,
  config: GPTConfig,
  trainConfig: TrainConfig,
  nextRandom: () => number,
): { averageLoss: number; gradients: ParameterTree } {
  let accumulatedGrads: ParameterTree | null = null;
  let preparedGradients: ParameterTree | null = null;
  let totalLoss = 0;

  try {
    for (let microStep = 0; microStep < trainConfig.gradAccumSteps; microStep++) {
      const { lossValue, gradients } = takeMicroStep(
        lossAndGrad,
        trainTokens,
        trainConfig.batchSize,
        config.blockSize,
        nextRandom,
      );
      totalLoss += lossValue;
      if (accumulatedGrads === null) {
        accumulatedGrads = gradients;
        continue;
      }

      try {
        const combined = accumulateGradients(accumulatedGrads, gradients);
        freeGradTree(accumulatedGrads);
        freeGradTree(gradients);
        accumulatedGrads = combined;
      } catch (error) {
        freeGradTree(accumulatedGrads);
        freeGradTree(gradients);
        accumulatedGrads = null;
        throw error;
      }
    }

    if (accumulatedGrads === null) {
      throw new Error("train: gradient accumulation produced no gradients");
    }

    preparedGradients = accumulatedGrads;
    accumulatedGrads = null;
    if (trainConfig.gradAccumSteps > 1) {
      const scaledGradients = scaleGradientTree(preparedGradients, 1 / trainConfig.gradAccumSteps);
      freeGradTree(preparedGradients);
      preparedGradients = scaledGradients;
    }

    const norm = gradientNorm(preparedGradients);
    assertFiniteValue(norm, "train: gradient norm");
    if (trainConfig.maxGradNorm !== null && norm > 0 && norm > trainConfig.maxGradNorm) {
      const clippedGradients = scaleGradientTree(preparedGradients, trainConfig.maxGradNorm / norm);
      freeGradTree(preparedGradients);
      preparedGradients = clippedGradients;
    }

    evalGradTree(preparedGradients);

    return {
      averageLoss: totalLoss / trainConfig.gradAccumSteps,
      gradients: preparedGradients,
    };
  } catch (error) {
    if (accumulatedGrads !== null) {
      freeGradTree(accumulatedGrads);
    }
    if (preparedGradients !== null) {
      freeGradTree(preparedGradients);
    }
    throw error;
  }
}

function runTrainingStep(
  model: GPT,
  optimizer: AdamW,
  config: GPTConfig,
  trainConfig: TrainConfig,
  trainTokens: Int32Array,
  nextRandom: () => number,
  lossAndGrad: LossAndGradFn,
): StepResult {
  const stepStart = performance.now();
  const { averageLoss, gradients } = accumulateMicroSteps(
    lossAndGrad,
    trainTokens,
    config,
    trainConfig,
    nextRandom,
  );

  try {
    optimizer.update(model, gradients);
  } finally {
    freeGradTree(gradients);
  }

  materializeStepState(model, optimizer);
  const stepEnd = performance.now();
  return {
    loss: averageLoss,
    tokensPerSec:
      (trainConfig.batchSize * config.blockSize * trainConfig.gradAccumSteps) /
      ((stepEnd - stepStart) / 1000),
  };
}

function maybeReportStep(
  step: number,
  trainConfig: TrainConfig,
  learningRate: number,
  stepResult: StepResult,
  onEvent?: (event: TrainEvent) => void,
): void {
  if (step % trainConfig.logInterval !== 0) {
    return;
  }

  onEvent?.({
    type: "step",
    step,
    loss: stepResult.loss,
    learningRate,
    tokensPerSec: stepResult.tokensPerSec,
  });
}

function evaluateIfDue(
  step: number,
  model: GPT,
  config: GPTConfig,
  trainConfig: TrainConfig,
  trainTokens: Int32Array,
  valTokens: Int32Array,
  nextRandom: () => number,
  onEvent?: (event: TrainEvent) => void,
): { trainLoss: number; valLoss: number } | null {
  if (step % trainConfig.evalInterval !== 0) {
    return null;
  }

  const trainLoss = estimateLoss(
    model,
    trainTokens,
    config,
    trainConfig.batchSize,
    trainConfig.evalSteps,
    nextRandom,
    (completed, total) => {
      onEvent?.({ type: "progress", phase: "eval", split: "train", step, completed, total });
    },
  );
  const valLoss = estimateLoss(
    model,
    valTokens,
    config,
    trainConfig.batchSize,
    trainConfig.evalSteps,
    nextRandom,
    (completed, total) => {
      onEvent?.({ type: "progress", phase: "eval", split: "val", step, completed, total });
    },
  );
  onEvent?.({ type: "eval", step, trainLoss, valLoss });
  return { trainLoss, valLoss };
}

function estimateLoss(
  model: GPT,
  tokens: Int32Array,
  config: GPTConfig,
  batchSize: number,
  evalSteps: number,
  nextRandom: () => number,
  reportProgress?: (completed: number, total: number) => void,
): number {
  const wasTraining = model.isTraining;
  model.eval();

  try {
    let totalLoss = 0;
    for (let index = 0; index < evalSteps; index++) {
      const { input, target } = getBatch(tokens, batchSize, config.blockSize, nextRandom);
      try {
        using logits = model.forward(input);
        const [batch, time, vocab] = logits.shape;
        if (batch === undefined || time === undefined || vocab === undefined) {
          throw new Error("train.estimateLoss: unexpected logits shape");
        }

        using flatLogits = reshape(logits, [batch * time, vocab]);
        using flatTargets = reshape(target, [batch * time]);
        using loss = crossEntropy(flatLogits, flatTargets);
        mxEval(loss);
        totalLoss += loss.item();
      } finally {
        input.free();
        target.free();
      }

      const completed = index + 1;
      if (reportProgress !== undefined && (completed % 5 === 0 || completed === evalSteps)) {
        reportProgress(completed, evalSteps);
      }
    }

    return totalLoss / evalSteps;
  } finally {
    restoreTrainingMode(model, wasTraining);
  }
}

/** Run the training loop. */
export function train(options: TrainOptions): TrainSummary {
  const { model, config, trainConfig, trainTokens, valTokens, onEvent, shouldStop } = options;
  validateTrainConfig(trainConfig);
  validateTokenSplitLength("train", trainTokens, config.blockSize);
  validateTokenSplitLength("validation", valTokens, config.blockSize);

  random.seed(trainConfig.seed);
  const trainRandom = createRandomSource(trainConfig.seed);
  const evalRandom = createRandomSource(trainConfig.seed ^ 0x9e3779b9);
  const optimizer =
    options.optimizer ?? createDefaultAdamW(trainConfig.learningRate, trainConfig.weightDecay);
  const ownsOptimizer = options.optimizer === undefined;
  const startStep = trainConfig.startStep ?? 0;
  const wasTraining = model.isTraining;
  model.train();

  const lossFn = (input: MxArray, target: MxArray): MxArray => {
    using logits = model.forward(input);
    const [batch, time, vocab] = logits.shape;
    if (batch === undefined || time === undefined || vocab === undefined) {
      throw new Error("train: unexpected logits shape");
    }
    using flatLogits = reshape(logits, [batch * time, vocab]);
    using flatTargets = reshape(target, [batch * time]);
    return crossEntropy(flatLogits, flatTargets);
  };

  const lossAndGrad = moduleValueAndGrad(model, lossFn);
  let lastStepLoss: number | null = null;
  let lastTrainLoss: number | null = null;
  let lastValLoss: number | null = null;
  let completedSteps = startStep;

  try {
    for (let step = startStep + 1; step <= trainConfig.maxSteps; step++) {
      const learningRate = getLearningRate(step, trainConfig);
      optimizer.setLearningRate(learningRate);

      const stepResult = runTrainingStep(
        model,
        optimizer,
        config,
        trainConfig,
        trainTokens,
        trainRandom,
        lossAndGrad,
      );
      lastStepLoss = stepResult.loss;
      completedSteps = step;

      maybeReportStep(step, trainConfig, learningRate, stepResult, onEvent);

      const evaluation = evaluateIfDue(
        step,
        model,
        config,
        trainConfig,
        trainTokens,
        valTokens,
        evalRandom,
        onEvent,
      );
      if (evaluation !== null) {
        lastTrainLoss = evaluation.trainLoss;
        lastValLoss = evaluation.valLoss;
      }

      if (shouldStop?.() === true) {
        break;
      }
    }

    onEvent?.({ type: "done", totalSteps: completedSteps });
    return {
      totalSteps: completedSteps,
      lastStepLoss,
      lastTrainLoss,
      lastValLoss,
    };
  } finally {
    if (ownsOptimizer) {
      optimizer[Symbol.dispose]();
    }
    restoreTrainingMode(model, wasTraining);
  }
}
