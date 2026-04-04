/**
 * Generic training-loop orchestration.
 *
 * @module
 */

import {
  type LearningRateSchedule,
  validateLearningRateConfig,
  warmupCosineSchedule,
} from "./schedule";

/** Shared loop-level configuration independent of model architecture. */
export interface TrainLoopConfig {
  startStep?: number;
  maxSteps: number;
  learningRate: number;
  warmupSteps: number;
  minLearningRate: number;
  evalInterval: number;
  logInterval: number;
}

/** Generic loop options with host-defined step and eval behavior. */
export interface TrainLoopOptions<TStepResult, TEvalResult> {
  config: TrainLoopConfig;
  runStep: (step: number, learningRate: number) => TStepResult;
  evaluate?: (step: number) => TEvalResult;
  onStep?: (step: number, learningRate: number, result: TStepResult) => void;
  onEval?: (step: number, result: TEvalResult) => void;
  onDone?: (totalSteps: number) => void;
  shouldStop?: () => boolean;
  schedule?: LearningRateSchedule;
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
    throw new Error(`TrainLoopConfig: ${name} must be ${requirement}`);
  }
}

/** Validate loop-level configuration before training starts. */
export function validateTrainLoopConfig(config: TrainLoopConfig): void {
  validateLearningRateConfig(config);
  readIntegerAtLeast(config.startStep, 0, "startStep", "a non-negative integer");
  if ((config.startStep ?? 0) >= config.maxSteps) {
    throw new Error("TrainLoopConfig: startStep must be < maxSteps");
  }
  readIntegerAtLeast(config.evalInterval, 1, "evalInterval", "> 0");
  readIntegerAtLeast(config.logInterval, 1, "logInterval", "> 0");
}

/** Run a generic step/eval loop and return the final completed step. */
export function trainLoop<TStepResult, TEvalResult>(
  options: TrainLoopOptions<TStepResult, TEvalResult>,
): number {
  const { config, runStep, evaluate, onStep, onEval, onDone, shouldStop } = options;
  validateTrainLoopConfig(config);

  const schedule = options.schedule ?? warmupCosineSchedule(config);
  const startStep = config.startStep ?? 0;
  let completedSteps = startStep;

  for (let step = startStep + 1; step <= config.maxSteps; step++) {
    const learningRate = schedule(step);
    const stepResult = runStep(step, learningRate);
    completedSteps = step;

    if (step % config.logInterval === 0) {
      onStep?.(step, learningRate, stepResult);
    }

    if (evaluate !== undefined && step % config.evalInterval === 0) {
      const evaluation = evaluate(step);
      onEval?.(step, evaluation);
    }

    if (shouldStop?.() === true) {
      break;
    }
  }

  onDone?.(completedSteps);
  return completedSteps;
}
