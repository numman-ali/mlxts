import {
  add,
  type DType,
  type MxArray,
  multiply,
  random,
  retainArray,
  subtract,
} from "@mlxts/core";

import { linspace } from "./schedule";

export type FlowMatchEulerTimeShiftType = "exponential" | "linear";

/** Config for Flux-style FlowMatch Euler denoising. */
export type FlowMatchEulerSchedulerConfig = {
  numTrainTimesteps?: number;
  shift?: number;
  shiftTerminal?: number;
  useDynamicShifting?: boolean;
  baseShift?: number;
  maxShift?: number;
  baseImageSeqLen?: number;
  maxImageSeqLen?: number;
  timeShiftType?: FlowMatchEulerTimeShiftType;
};

/** Options for constructing explicit FlowMatch Euler denoising steps. */
export type FlowMatchEulerTimestepsOptions = {
  sigmas?: readonly number[];
  mu?: number;
  imageSequenceLength?: number;
};

export type FlowMatchEulerStep = {
  timestep: number;
  previousTimestep: number;
  sigma: number;
  nextSigma: number;
};

type ResolvedFlowMatchEulerConfig = Required<
  Omit<FlowMatchEulerSchedulerConfig, "shiftTerminal">
> & {
  shiftTerminal?: number;
};

const DEFAULT_FLOW_MATCH_EULER_CONFIG: ResolvedFlowMatchEulerConfig = {
  numTrainTimesteps: 1000,
  shift: 1,
  useDynamicShifting: false,
  baseShift: 0.5,
  maxShift: 1.15,
  baseImageSeqLen: 256,
  maxImageSeqLen: 4096,
  timeShiftType: "exponential",
};

function expectPositiveInteger(value: number, name: string): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
}

function expectPositiveFinite(value: number, name: string): void {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${name} must be a positive finite number.`);
  }
}

function expectNonNegativeFinite(value: number, name: string): void {
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`${name} must be a finite non-negative number.`);
  }
}

function resolveConfig(config: FlowMatchEulerSchedulerConfig): ResolvedFlowMatchEulerConfig {
  const resolved: ResolvedFlowMatchEulerConfig = {
    ...DEFAULT_FLOW_MATCH_EULER_CONFIG,
    ...config,
  };

  expectPositiveInteger(resolved.numTrainTimesteps, "numTrainTimesteps");
  expectPositiveFinite(resolved.shift, "shift");
  expectPositiveFinite(resolved.baseShift, "baseShift");
  expectPositiveFinite(resolved.maxShift, "maxShift");
  expectPositiveInteger(resolved.baseImageSeqLen, "baseImageSeqLen");
  expectPositiveInteger(resolved.maxImageSeqLen, "maxImageSeqLen");
  if (resolved.maxImageSeqLen <= resolved.baseImageSeqLen) {
    throw new Error("maxImageSeqLen must be greater than baseImageSeqLen.");
  }
  if (resolved.timeShiftType !== "exponential" && resolved.timeShiftType !== "linear") {
    throw new Error("timeShiftType must be exponential or linear.");
  }
  if (resolved.shiftTerminal !== undefined) {
    if (
      !Number.isFinite(resolved.shiftTerminal) ||
      resolved.shiftTerminal < 0 ||
      resolved.shiftTerminal >= 1
    ) {
      throw new Error("shiftTerminal must be within [0, 1).");
    }
  }
  return resolved;
}

function expectSigma(value: number, name: string): void {
  if (!Number.isFinite(value) || value <= 0 || value > 1) {
    throw new Error(`${name} must be within (0, 1].`);
  }
}

function shiftedSigma(sigma: number, shift: number): number {
  return (shift * sigma) / (1 + (shift - 1) * sigma);
}

function exponentialTimeShift(mu: number, sigma: number): number {
  const expMu = Math.exp(mu);
  return expMu / (expMu + (1 / sigma - 1));
}

function linearTimeShift(mu: number, sigma: number): number {
  return mu / (mu + (1 / sigma - 1));
}

function sigmaFromStep(stepOrSigma: FlowMatchEulerStep | number): number {
  if (typeof stepOrSigma === "number") {
    expectSigma(stepOrSigma, "sigma");
    return stepOrSigma;
  }
  return stepOrSigma.sigma;
}

function resolveMu(
  config: ResolvedFlowMatchEulerConfig,
  options: FlowMatchEulerTimestepsOptions,
): number {
  if (options.mu !== undefined) {
    expectNonNegativeFinite(options.mu, "mu");
    return options.mu;
  }
  if (options.imageSequenceLength !== undefined) {
    return calculateFlowMatchShift(options.imageSequenceLength, {
      baseImageSeqLen: config.baseImageSeqLen,
      maxImageSeqLen: config.maxImageSeqLen,
      baseShift: config.baseShift,
      maxShift: config.maxShift,
    });
  }
  throw new Error("mu or imageSequenceLength is required when useDynamicShifting is true.");
}

function resolveBaseSigmas(
  numInferenceSteps: number,
  options: FlowMatchEulerTimestepsOptions,
): number[] {
  if (options.sigmas !== undefined) {
    if (options.sigmas.length !== numInferenceSteps) {
      throw new Error("sigmas must have numInferenceSteps entries.");
    }
    return options.sigmas.map((sigma, index) => {
      expectSigma(sigma, `sigmas[${index}]`);
      return sigma;
    });
  }

  return Array.from(linspace(1, 1 / numInferenceSteps, numInferenceSteps));
}

function shiftSigmas(
  sigmas: readonly number[],
  config: ResolvedFlowMatchEulerConfig,
  options: FlowMatchEulerTimestepsOptions,
): number[] {
  if (!config.useDynamicShifting) {
    return sigmas.map((sigma) => shiftedSigma(sigma, config.shift));
  }

  const mu = resolveMu(config, options);
  if (config.timeShiftType === "linear") {
    return sigmas.map((sigma) => linearTimeShift(mu, sigma));
  }
  return sigmas.map((sigma) => exponentialTimeShift(mu, sigma));
}

function stretchShiftToTerminal(sigmas: readonly number[], shiftTerminal: number): number[] {
  const finalSigma = sigmas.at(-1);
  if (finalSigma === undefined) {
    throw new Error("stretchShiftToTerminal: missing terminal sigma.");
  }
  const scaleFactor = (1 - finalSigma) / (1 - shiftTerminal);
  return sigmas.map((sigma) => 1 - (1 - sigma) / scaleFactor);
}

/** Calculate the Flux resolution-dependent time-shift value. */
export function calculateFlowMatchShift(
  imageSequenceLength: number,
  config: Pick<
    FlowMatchEulerSchedulerConfig,
    "baseImageSeqLen" | "maxImageSeqLen" | "baseShift" | "maxShift"
  > = {},
): number {
  const resolved = resolveConfig(config);
  expectPositiveInteger(imageSequenceLength, "imageSequenceLength");
  const slope =
    (resolved.maxShift - resolved.baseShift) / (resolved.maxImageSeqLen - resolved.baseImageSeqLen);
  return imageSequenceLength * slope + resolved.baseShift - slope * resolved.baseImageSeqLen;
}

/** FlowMatch Euler scheduler for Flux-style rectified-flow denoising. */
export class FlowMatchEulerScheduler {
  readonly #config: ResolvedFlowMatchEulerConfig;

  constructor(config: FlowMatchEulerSchedulerConfig = {}) {
    this.#config = resolveConfig(config);
  }

  /** Number of training timesteps represented by this scheduler. */
  get maxTimestep(): number {
    return this.#config.numTrainTimesteps;
  }

  /** Initial noise scale for flow-matching denoising loops. */
  get initNoiseSigma(): number {
    return 1;
  }

  /** Create explicit denoising steps from high noise to terminal zero sigma. */
  timesteps(
    numInferenceSteps: number,
    options: FlowMatchEulerTimestepsOptions = {},
  ): readonly FlowMatchEulerStep[] {
    expectPositiveInteger(numInferenceSteps, "numInferenceSteps");

    const baseSigmas = resolveBaseSigmas(numInferenceSteps, options);
    const shiftedSigmas = shiftSigmas(baseSigmas, this.#config, options);
    const denoisingSigmas =
      this.#config.shiftTerminal === undefined
        ? shiftedSigmas
        : stretchShiftToTerminal(shiftedSigmas, this.#config.shiftTerminal);
    const sigmas = [...denoisingSigmas, 0];
    return Array.from({ length: numInferenceSteps }, (_, index) => {
      const sigma = sigmas[index];
      const nextSigma = sigmas[index + 1];
      if (sigma === undefined || nextSigma === undefined) {
        throw new Error("FlowMatchEulerScheduler.timesteps: missing sigma.");
      }
      return {
        timestep: sigma * this.#config.numTrainTimesteps,
        previousTimestep: nextSigma * this.#config.numTrainTimesteps,
        sigma,
        nextSigma,
      };
    });
  }

  /** Create an initial latent sample from unscaled normal noise. */
  samplePrior(shape: readonly number[], dtype: DType = "float32", key?: MxArray): MxArray {
    return random.normal([...shape], dtype, 0, 1, key);
  }

  /** Retain caller-provided normal noise as the scheduler's initial latent space. */
  scaleInitialNoise(noise: MxArray): MxArray {
    return retainArray(noise);
  }

  /** Retain a latent before passing it to a flow-matching denoiser. */
  scaleModelInput(sample: MxArray): MxArray {
    return retainArray(sample);
  }

  /** Add forward-process flow noise at a sigma or step. */
  scaleNoise(sample: MxArray, noise: MxArray, stepOrSigma: FlowMatchEulerStep | number): MxArray {
    const sigma = sigmaFromStep(stepOrSigma);
    using scaledSample = multiply(sample, 1 - sigma);
    using scaledNoise = multiply(noise, sigma);
    return add(scaledSample, scaledNoise);
  }

  /** Add forward-process flow noise at a sigma or step. */
  addNoise(sample: MxArray, noise: MxArray, stepOrSigma: FlowMatchEulerStep | number): MxArray {
    return this.scaleNoise(sample, noise, stepOrSigma);
  }

  /** Move one deterministic Euler step from `sigma` to `nextSigma`. */
  step(modelOutput: MxArray, sample: MxArray, step: FlowMatchEulerStep): MxArray {
    using delta = multiply(modelOutput, step.nextSigma - step.sigma);
    return add(sample, delta);
  }

  /** Recover denoised sample prediction from flow velocity and current sample. */
  predictDenoised(
    modelOutput: MxArray,
    sample: MxArray,
    stepOrSigma: FlowMatchEulerStep | number,
  ): MxArray {
    const sigma = sigmaFromStep(stepOrSigma);
    using velocity = multiply(modelOutput, sigma);
    return subtract(sample, velocity);
  }
}
