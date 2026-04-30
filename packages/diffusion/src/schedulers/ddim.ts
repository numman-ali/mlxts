import { add, divide, type MxArray, maximum, minimum, multiply, subtract } from "@mlxts/core";

import {
  type DiffusionScheduleConfig,
  makeAlphaCumprodSchedule,
  makeDiscreteTimesteps,
  type TimestepSpacing,
} from "./schedule";

export type DDIMSchedulerConfig = DiffusionScheduleConfig & {
  setAlphaToOne?: boolean;
  clipSample?: boolean;
  clipSampleRange?: number;
  timestepSpacing?: TimestepSpacing;
  stepsOffset?: number;
};

export type DDIMSchedulerStep = {
  timestep: number;
  previousTimestep: number;
};

export type DDIMStepOutput = {
  prevSample: MxArray;
  predOriginalSample: MxArray;
};

/** DDIM scheduler for deterministic latent denoising. */
export class DDIMScheduler {
  readonly #alphaCumprod: Float64Array;
  readonly #finalAlphaCumprod: number;
  readonly #numTrainTimesteps: number;
  readonly #clipSample: boolean;
  readonly #clipSampleRange: number;
  readonly #timestepSpacing: TimestepSpacing;
  readonly #stepsOffset: number;

  constructor(config: DDIMSchedulerConfig = {}) {
    this.#alphaCumprod = makeAlphaCumprodSchedule(config);
    this.#numTrainTimesteps = this.#alphaCumprod.length;
    this.#finalAlphaCumprod = config.setAlphaToOne === false ? this.requireAlphaCumprod(0) : 1;
    this.#clipSample = config.clipSample ?? true;
    this.#clipSampleRange = config.clipSampleRange ?? 1;
    this.#timestepSpacing = config.timestepSpacing ?? "leading";
    this.#stepsOffset = config.stepsOffset ?? 0;
    if (!Number.isFinite(this.#clipSampleRange) || this.#clipSampleRange <= 0) {
      throw new Error("clipSampleRange must be a finite positive number.");
    }
  }

  /** Training timestep count represented by the alpha schedule. */
  get numTrainTimesteps(): number {
    return this.#numTrainTimesteps;
  }

  /** Discrete timesteps used for a denoising run. */
  timesteps(numInferenceSteps: number): readonly number[] {
    return makeDiscreteTimesteps(
      numInferenceSteps,
      this.#numTrainTimesteps,
      this.#timestepSpacing,
      this.#stepsOffset,
    );
  }

  /** Paired DDIM steps for a denoising run. */
  steps(numInferenceSteps: number): readonly DDIMSchedulerStep[] {
    const timesteps = this.timesteps(numInferenceSteps);
    const stepRatio = Math.floor(this.#numTrainTimesteps / numInferenceSteps);
    return timesteps.map((timestep) => ({
      timestep,
      previousTimestep: timestep - stepRatio,
    }));
  }

  /** Variance term for a current/previous timestep pair. */
  variance(timestep: number, previousTimestep: number): number {
    const alphaProdT = this.requireAlphaCumprod(timestep);
    const alphaProdPrev =
      previousTimestep >= 0 ? this.requireAlphaCumprod(previousTimestep) : this.#finalAlphaCumprod;
    const betaProdT = 1 - alphaProdT;
    const betaProdPrev = 1 - alphaProdPrev;
    return (betaProdPrev / betaProdT) * (1 - alphaProdT / alphaProdPrev);
  }

  /** DDIM does not scale model input. */
  scaleModelInput(sample: MxArray): MxArray {
    return sample;
  }

  /** Add forward-process noise at a discrete timestep. */
  addNoise(originalSample: MxArray, noise: MxArray, timestep: number): MxArray {
    const alpha = this.requireAlphaCumprod(timestep);
    using scaledOriginal = multiply(originalSample, Math.sqrt(alpha));
    using scaledNoise = multiply(noise, Math.sqrt(1 - alpha));
    return add(scaledOriginal, scaledNoise);
  }

  /** Move one deterministic DDIM step using epsilon prediction. */
  step(modelOutput: MxArray, sample: MxArray, step: DDIMSchedulerStep): DDIMStepOutput {
    const alphaProdT = this.requireAlphaCumprod(step.timestep);
    const alphaProdPrev =
      step.previousTimestep >= 0
        ? this.requireAlphaCumprod(step.previousTimestep)
        : this.#finalAlphaCumprod;
    const betaProdT = 1 - alphaProdT;

    using scaledNoise = multiply(modelOutput, Math.sqrt(betaProdT));
    using denoisedNumerator = subtract(sample, scaledNoise);
    const rawPredOriginalSample = divide(denoisedNumerator, Math.sqrt(alphaProdT));
    const predOriginalSample = this.#clipSample
      ? clipSample(rawPredOriginalSample, this.#clipSampleRange)
      : rawPredOriginalSample;
    if (predOriginalSample !== rawPredOriginalSample) {
      rawPredOriginalSample.free();
    }

    try {
      using direction = multiply(modelOutput, Math.sqrt(1 - alphaProdPrev));
      using previousOriginal = multiply(predOriginalSample, Math.sqrt(alphaProdPrev));
      const prevSample = add(previousOriginal, direction);
      return { prevSample, predOriginalSample };
    } catch (error) {
      predOriginalSample.free();
      throw error;
    }
  }

  requireAlphaCumprod(timestep: number): number {
    if (!Number.isInteger(timestep) || timestep < 0 || timestep >= this.#alphaCumprod.length) {
      throw new Error(`timestep ${timestep} is outside the alpha schedule.`);
    }
    const alpha = this.#alphaCumprod[timestep];
    if (alpha === undefined) {
      throw new Error("DDIMScheduler: missing alpha schedule value.");
    }
    return alpha;
  }
}

function clipSample(sample: MxArray, range: number): MxArray {
  using clippedLow = maximum(sample, -range);
  return minimum(clippedLow, range);
}
