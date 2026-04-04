/**
 * GPT-specific training wrapper over @mlxts/train.
 *
 * nanogpt keeps the GPT loss, eval policy, and event surface local while
 * delegating gradient preparation and loop orchestration to the canonical
 * training package.
 *
 * @module
 */

import { type MxArray, mxEval, type ParameterTree, random, reshape } from "@mlxts/core";
import { createRandomSource, getBatch } from "@mlxts/data";
import { crossEntropy, valueAndGrad as moduleValueAndGrad } from "@mlxts/nn";
import type { AdamW } from "@mlxts/optimizers";
import {
  applyGradientStep,
  freeGradientTree,
  getLearningRate as getSharedLearningRate,
  materializeTrainingState,
  type TrainLoopOptions,
  trainLoop,
  validateTrainLoopConfig,
} from "@mlxts/train";

import type { GPTConfig } from "./config";
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

type LossAndGradFn = (input: MxArray, target: MxArray) => [MxArray, ParameterTree];
type StepResult = {
  loss: number;
  tokensPerSec: number;
};

type EvalResult = {
  trainLoss: number;
  valLoss: number;
};

/** Cosine decay with linear warmup. */
export function getLearningRate(step: number, config: TrainConfig): number {
  return getSharedLearningRate(step, config);
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
  validateTrainLoopConfig(config);
  readIntegerAtLeast(config.batchSize, 1, "batchSize", "> 0");
  readFiniteAtLeast(config.weightDecay, 0, "weightDecay", ">= 0");
  readIntegerAtLeast(config.gradAccumSteps, 1, "gradAccumSteps", ">= 1");
  readIntegerAtLeast(config.evalSteps, 1, "evalSteps", "> 0");
  if (config.maxGradNorm !== null) {
    readPositiveFinite(config.maxGradNorm, "maxGradNorm");
  }
  if (!Number.isFinite(config.seed)) {
    throw new Error("TrainConfig: seed must be a finite number");
  }
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

function assertFiniteValue(value: number, context: string): void {
  if (!Number.isFinite(value)) {
    throw new Error(`${context}: encountered non-finite value (${String(value)})`);
  }
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
    const lossValue = loss.item();
    assertFiniteValue(lossValue, "train");
    const stepResult = { lossValue, gradients };
    gradients = null;
    return stepResult;
  } finally {
    loss?.free();
    if (gradients !== null) {
      freeGradientTree(gradients);
    }
    input.free();
    target.free();
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
  learningRate: number,
): StepResult {
  const stepStart = performance.now();
  optimizer.setLearningRate(learningRate);

  const { averageLoss } = applyGradientStep({
    gradAccumSteps: trainConfig.gradAccumSteps,
    maxGradNorm: trainConfig.maxGradNorm,
    takeMicroStep() {
      return takeMicroStep(
        lossAndGrad,
        trainTokens,
        trainConfig.batchSize,
        config.blockSize,
        nextRandom,
      );
    },
    applyGradients(gradients: ParameterTree) {
      optimizer.update(model, gradients);
    },
    materialize() {
      materializeTrainingState(model, optimizer);
    },
  });

  const stepEnd = performance.now();
  return {
    loss: averageLoss,
    tokensPerSec:
      (trainConfig.batchSize * config.blockSize * trainConfig.gradAccumSteps) /
      ((stepEnd - stepStart) / 1000),
  };
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

  try {
    const loopOptions: TrainLoopOptions<StepResult, EvalResult> = {
      config: trainConfig,
      schedule(step: number) {
        return getSharedLearningRate(step, trainConfig);
      },
      runStep(_step: number, learningRate: number) {
        const result = runTrainingStep(
          model,
          optimizer,
          config,
          trainConfig,
          trainTokens,
          trainRandom,
          lossAndGrad,
          learningRate,
        );
        lastStepLoss = result.loss;
        return result;
      },
      evaluate(step: number): EvalResult {
        const result = {
          trainLoss: estimateLoss(
            model,
            trainTokens,
            config,
            trainConfig.batchSize,
            trainConfig.evalSteps,
            evalRandom,
            (completed, total) => {
              onEvent?.({
                type: "progress",
                phase: "eval",
                split: "train",
                step,
                completed,
                total,
              });
            },
          ),
          valLoss: estimateLoss(
            model,
            valTokens,
            config,
            trainConfig.batchSize,
            trainConfig.evalSteps,
            evalRandom,
            (completed, total) => {
              onEvent?.({ type: "progress", phase: "eval", split: "val", step, completed, total });
            },
          ),
        };
        lastTrainLoss = result.trainLoss;
        lastValLoss = result.valLoss;
        return result;
      },
      onStep(step: number, learningRate: number, result: StepResult) {
        onEvent?.({
          type: "step",
          step,
          loss: result.loss,
          learningRate,
          tokensPerSec: result.tokensPerSec,
        });
      },
      onEval(step: number, result: EvalResult) {
        onEvent?.({
          type: "eval",
          step,
          trainLoss: result.trainLoss,
          valLoss: result.valLoss,
        });
      },
      onDone(totalSteps: number) {
        onEvent?.({ type: "done", totalSteps });
      },
    };
    if (shouldStop !== undefined) {
      loopOptions.shouldStop = shouldStop;
    }

    const totalSteps = trainLoop(loopOptions);

    return {
      totalSteps,
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
