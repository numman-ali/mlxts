export type BetaSchedule = "linear" | "scaled_linear";

export type TimestepSpacing = "leading" | "linspace" | "trailing";

export type DiffusionScheduleConfig = {
  numTrainTimesteps?: number;
  betaStart?: number;
  betaEnd?: number;
  betaSchedule?: BetaSchedule;
};

const DEFAULT_NUM_TRAIN_TIMESTEPS = 1000;
const DEFAULT_BETA_START = 0.00085;
const DEFAULT_BETA_END = 0.012;
const DEFAULT_BETA_SCHEDULE: BetaSchedule = "scaled_linear";

function assertFinitePositive(name: string, value: number): void {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${name} must be a finite positive number.`);
  }
}

function assertPositiveInteger(name: string, value: number): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
}

export function resolveDiffusionScheduleConfig(
  config: DiffusionScheduleConfig = {},
): Required<DiffusionScheduleConfig> {
  const numTrainTimesteps = config.numTrainTimesteps ?? DEFAULT_NUM_TRAIN_TIMESTEPS;
  const betaStart = config.betaStart ?? DEFAULT_BETA_START;
  const betaEnd = config.betaEnd ?? DEFAULT_BETA_END;
  const betaSchedule = config.betaSchedule ?? DEFAULT_BETA_SCHEDULE;

  assertPositiveInteger("numTrainTimesteps", numTrainTimesteps);
  assertFinitePositive("betaStart", betaStart);
  assertFinitePositive("betaEnd", betaEnd);
  if (betaEnd < betaStart) {
    throw new Error("betaEnd must be greater than or equal to betaStart.");
  }

  return { numTrainTimesteps, betaStart, betaEnd, betaSchedule };
}

export function linspace(start: number, end: number, count: number): Float64Array {
  assertPositiveInteger("count", count);
  if (count === 1) {
    return Float64Array.of(start);
  }

  const values = new Float64Array(count);
  const denominator = count - 1;
  for (let index = 0; index < count; index += 1) {
    const fraction = index / denominator;
    values[index] = start + (end - start) * fraction;
  }
  return values;
}

export function makeBetaSchedule(config: DiffusionScheduleConfig = {}): Float64Array {
  const resolved = resolveDiffusionScheduleConfig(config);
  if (resolved.betaSchedule === "linear") {
    return linspace(resolved.betaStart, resolved.betaEnd, resolved.numTrainTimesteps);
  }

  const scaled = linspace(
    Math.sqrt(resolved.betaStart),
    Math.sqrt(resolved.betaEnd),
    resolved.numTrainTimesteps,
  );
  for (let index = 0; index < scaled.length; index += 1) {
    const value = scaled[index];
    if (value === undefined) {
      throw new Error("makeBetaSchedule: missing scaled beta value.");
    }
    scaled[index] = value * value;
  }
  return scaled;
}

export function makeAlphaCumprodSchedule(config: DiffusionScheduleConfig = {}): Float64Array {
  const betas = makeBetaSchedule(config);
  const alphas = new Float64Array(betas.length);
  let product = 1;
  for (let index = 0; index < betas.length; index += 1) {
    const beta = betas[index];
    if (beta === undefined) {
      throw new Error("makeAlphaCumprodSchedule: missing beta value.");
    }
    product *= 1 - beta;
    alphas[index] = product;
  }
  return alphas;
}

export function makeSigmaSchedule(config: DiffusionScheduleConfig = {}): Float64Array {
  const alphaCumprod = makeAlphaCumprodSchedule(config);
  const sigmas = new Float64Array(alphaCumprod.length + 1);
  sigmas[0] = 0;
  for (let index = 0; index < alphaCumprod.length; index += 1) {
    const alpha = alphaCumprod[index];
    if (alpha === undefined) {
      throw new Error("makeSigmaSchedule: missing alpha value.");
    }
    sigmas[index + 1] = Math.sqrt((1 - alpha) / alpha);
  }
  return sigmas;
}

export function interpolateSchedule(schedule: Float64Array, position: number): number {
  if (!Number.isFinite(position)) {
    throw new Error("position must be finite.");
  }
  if (position < 0 || position > schedule.length - 1) {
    throw new Error(
      `position ${position} is outside the schedule range 0..${schedule.length - 1}.`,
    );
  }

  const lowIndex = Math.floor(position);
  const highIndex = Math.min(lowIndex + 1, schedule.length - 1);
  const low = schedule[lowIndex];
  const high = schedule[highIndex];
  if (low === undefined || high === undefined) {
    throw new Error("interpolateSchedule: missing schedule value.");
  }
  const fraction = position - lowIndex;
  return low * (1 - fraction) + high * fraction;
}

export function makeDiscreteTimesteps(
  numInferenceSteps: number,
  numTrainTimesteps: number,
  spacing: TimestepSpacing,
  stepsOffset = 0,
): readonly number[] {
  assertPositiveInteger("numInferenceSteps", numInferenceSteps);
  assertPositiveInteger("numTrainTimesteps", numTrainTimesteps);
  if (numInferenceSteps > numTrainTimesteps) {
    throw new Error("numInferenceSteps cannot exceed numTrainTimesteps.");
  }

  if (spacing === "linspace") {
    return Array.from(linspace(0, numTrainTimesteps - 1, numInferenceSteps), (value) =>
      Math.round(value),
    ).reverse();
  }

  if (spacing === "leading") {
    const stepRatio = Math.floor(numTrainTimesteps / numInferenceSteps);
    return Array.from({ length: numInferenceSteps }, (_, index) =>
      Math.round(index * stepRatio + stepsOffset),
    ).reverse();
  }

  const stepRatio = numTrainTimesteps / numInferenceSteps;
  return Array.from(
    { length: numInferenceSteps },
    (_, index) => Math.round(numTrainTimesteps - index * stepRatio) - 1,
  );
}
