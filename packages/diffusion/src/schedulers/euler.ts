import { add, type DType, divide, type MxArray, multiply, random } from "@mlxts/core";

import {
  type DiffusionScheduleConfig,
  interpolateSchedule,
  linspace,
  makeSigmaSchedule,
  resolveDiffusionScheduleConfig,
} from "./schedule";

export type EulerSchedulerConfig = DiffusionScheduleConfig;

export type EulerTimestepPair = {
  timestep: number;
  previousTimestep: number;
};

/** Euler scheduler for Stable Diffusion-style epsilon prediction. */
export class EulerScheduler {
  readonly #sigmas: Float64Array;

  constructor(config: EulerSchedulerConfig = {}) {
    resolveDiffusionScheduleConfig(config);
    this.#sigmas = makeSigmaSchedule(config);
  }

  /** Largest training-time position represented by this scheduler. */
  get maxTimestep(): number {
    return this.#sigmas.length - 1;
  }

  /** Initial noise scale for Euler denoising loops. */
  get initNoiseSigma(): number {
    const sigma = this.#sigmas[this.#sigmas.length - 1];
    if (sigma === undefined) {
      throw new Error("EulerScheduler: missing terminal sigma.");
    }
    return sigma / Math.sqrt(sigma * sigma + 1);
  }

  /** Interpolated sigma value for a training timestep. */
  sigmaAt(timestep: number): number {
    return interpolateSchedule(this.#sigmas, timestep);
  }

  /** Create denoising timestep pairs from high noise to zero noise. */
  timesteps(
    numInferenceSteps: number,
    startTimestep = this.maxTimestep,
  ): readonly EulerTimestepPair[] {
    if (!Number.isInteger(numInferenceSteps) || numInferenceSteps <= 0) {
      throw new Error("numInferenceSteps must be a positive integer.");
    }
    if (!Number.isFinite(startTimestep) || startTimestep <= 0 || startTimestep > this.maxTimestep) {
      throw new Error(`startTimestep must be within 1..${this.maxTimestep}.`);
    }

    const steps = linspace(startTimestep, 0, numInferenceSteps + 1);
    return Array.from({ length: numInferenceSteps }, (_, index) => {
      const timestep = steps[index];
      const previousTimestep = steps[index + 1];
      if (timestep === undefined || previousTimestep === undefined) {
        throw new Error("EulerScheduler.timesteps: missing timestep.");
      }
      return { timestep, previousTimestep };
    });
  }

  /** Create an initial latent sample from normal noise. */
  samplePrior(shape: readonly number[], dtype: DType = "float32", key?: MxArray): MxArray {
    const noise = random.normal([...shape], dtype, 0, 1, key);
    try {
      return this.scaleInitialNoise(noise);
    } finally {
      noise.free();
    }
  }

  /** Scale caller-provided normal noise into the scheduler's initial latent space. */
  scaleInitialNoise(noise: MxArray): MxArray {
    return multiply(noise, this.initNoiseSigma);
  }

  /** Scale a latent before passing it to an epsilon-prediction denoiser. */
  scaleModelInput(sample: MxArray, timestep: number): MxArray {
    const sigma = this.sigmaAt(timestep);
    return divide(sample, Math.sqrt(sigma * sigma + 1));
  }

  /** Add forward-process noise at a timestep. */
  addNoise(sample: MxArray, noise: MxArray, timestep: number): MxArray {
    const sigma = this.sigmaAt(timestep);
    using scaledNoise = multiply(noise, sigma);
    using noisy = add(sample, scaledNoise);
    return divide(noisy, Math.sqrt(sigma * sigma + 1));
  }

  /** Move one Euler step from `timestep` to `previousTimestep`. */
  step(modelOutput: MxArray, sample: MxArray, step: EulerTimestepPair): MxArray {
    const sigma = this.sigmaAt(step.timestep);
    const previousSigma = this.sigmaAt(step.previousTimestep);

    using scaledSample = multiply(sample, Math.sqrt(sigma * sigma + 1));
    using delta = multiply(modelOutput, previousSigma - sigma);
    using updated = add(scaledSample, delta);
    return divide(updated, Math.sqrt(previousSigma * previousSigma + 1));
  }
}
