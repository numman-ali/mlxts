/**
 * Learning-rate schedules for training loops.
 *
 * @module
 */

/** Shared inputs for learning-rate schedules. */
export interface LearningRateConfig {
  learningRate: number;
  warmupSteps: number;
  minLearningRate: number;
  maxSteps: number;
}

/** Step-indexed learning-rate callback. */
export type LearningRateSchedule = (step: number) => number;

function readIntegerAtLeast(
  value: number,
  minimum: number,
  name: string,
  requirement: string,
): void {
  if (!Number.isInteger(value) || value < minimum) {
    throw new Error(`LearningRateConfig: ${name} must be ${requirement}`);
  }
}

function readPositiveFinite(value: number, name: string): void {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`LearningRateConfig: ${name} must be > 0`);
  }
}

/** Validate schedule inputs before the loop starts. */
export function validateLearningRateConfig(config: LearningRateConfig): void {
  readPositiveFinite(config.learningRate, "learningRate");
  readIntegerAtLeast(config.warmupSteps, 0, "warmupSteps", ">= 0");
  readPositiveFinite(config.minLearningRate, "minLearningRate");
  readIntegerAtLeast(config.maxSteps, 1, "maxSteps", "> 0");
  if (config.warmupSteps >= config.maxSteps) {
    throw new Error("LearningRateConfig: warmupSteps must be < maxSteps");
  }
}

/** Cosine decay with linear warmup. */
export function getLearningRate(step: number, config: LearningRateConfig): number {
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

/** Create a reusable warmup + cosine schedule callback. */
export function warmupCosineSchedule(config: LearningRateConfig): LearningRateSchedule {
  validateLearningRateConfig(config);
  return (step) => getLearningRate(step, config);
}
