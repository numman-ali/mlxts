import { add, type DType, divide, type MxArray, multiply, random } from "@mlxts/core";

import {
  type DiffusionScheduleConfig,
  interpolateSchedule,
  linspace,
  makeSigmaSchedule,
  resolveDiffusionScheduleConfig,
  type TimestepSpacing,
} from "./schedule";

export type EulerFinalSigmasType = "zero" | "sigma_min";

export type EulerSchedulerConfig = DiffusionScheduleConfig & {
  timestepSpacing?: TimestepSpacing;
  stepsOffset?: number;
  finalSigmasType?: EulerFinalSigmasType;
};

export type EulerTimestepPair = {
  timestep: number;
  previousTimestep: number;
  sigma: number;
  previousSigma: number;
};

/** Euler scheduler for Stable Diffusion-style epsilon prediction. */
export class EulerScheduler {
  readonly #sigmas: Float64Array;
  readonly #numTrainTimesteps: number;
  readonly #timestepSpacing: TimestepSpacing;
  readonly #stepsOffset: number;
  readonly #finalSigmasType: EulerFinalSigmasType;

  constructor(config: EulerSchedulerConfig = {}) {
    const resolved = resolveDiffusionScheduleConfig(config);
    this.#sigmas = makeSigmaSchedule(config).slice(1);
    this.#numTrainTimesteps = resolved.numTrainTimesteps;
    this.#timestepSpacing = config.timestepSpacing ?? "linspace";
    this.#stepsOffset = config.stepsOffset ?? 0;
    this.#finalSigmasType = config.finalSigmasType ?? "zero";
    if (!Number.isInteger(this.#stepsOffset)) {
      throw new Error("stepsOffset must be an integer.");
    }
  }

  /** Largest training-time position represented by this scheduler. */
  get maxTimestep(): number {
    return this.#numTrainTimesteps - 1;
  }

  /** Initial noise scale before inference timesteps are selected. */
  get initNoiseSigma(): number {
    return this.initialNoiseSigma();
  }

  /** Initial noise scale for a denoising run. */
  initialNoiseSigma(numInferenceSteps?: number): number {
    const maxSigma =
      numInferenceSteps === undefined
        ? this.requireSigma(this.maxTimestep)
        : this.maxInferenceSigma(numInferenceSteps);
    if (this.#timestepSpacing === "leading") {
      return Math.sqrt(maxSigma * maxSigma + 1);
    }
    return maxSigma;
  }

  /** Interpolated sigma value for a training timestep. */
  sigmaAt(timestep: number): number {
    return interpolateSchedule(this.#sigmas, timestep);
  }

  /** Create denoising timestep pairs from high noise to zero noise. */
  timesteps(numInferenceSteps: number): readonly EulerTimestepPair[] {
    if (!Number.isInteger(numInferenceSteps) || numInferenceSteps <= 0) {
      throw new Error("numInferenceSteps must be a positive integer.");
    }
    if (numInferenceSteps > this.#numTrainTimesteps) {
      throw new Error("numInferenceSteps cannot exceed numTrainTimesteps.");
    }

    const timesteps = this.inferenceTimesteps(numInferenceSteps);
    const sigmas = Array.from(timesteps, (timestep) => this.sigmaAt(timestep));
    sigmas.push(this.finalSigma());
    return Array.from({ length: numInferenceSteps }, (_, index) => {
      const timestep = timesteps[index];
      const previousTimestep = timesteps[index + 1] ?? 0;
      const sigma = sigmas[index];
      const previousSigma = sigmas[index + 1];
      if (timestep === undefined || sigma === undefined || previousSigma === undefined) {
        throw new Error("EulerScheduler.timesteps: missing timestep.");
      }
      return { timestep, previousTimestep, sigma, previousSigma };
    });
  }

  /** Create an initial latent sample from normal noise. */
  samplePrior(
    shape: readonly number[],
    dtype: DType = "float32",
    key?: MxArray,
    numInferenceSteps?: number,
  ): MxArray {
    const noise = random.normal([...shape], dtype, 0, 1, key);
    try {
      return this.scaleInitialNoise(noise, numInferenceSteps);
    } finally {
      noise.free();
    }
  }

  /** Scale caller-provided normal noise into the scheduler's initial latent space. */
  scaleInitialNoise(noise: MxArray, numInferenceSteps?: number): MxArray {
    return multiply(noise, this.initialNoiseSigma(numInferenceSteps));
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
    return add(sample, scaledNoise);
  }

  /** Move one Euler step from `timestep` to `previousTimestep`. */
  step(modelOutput: MxArray, sample: MxArray, step: EulerTimestepPair): MxArray {
    using delta = multiply(modelOutput, step.previousSigma - step.sigma);
    return add(sample, delta);
  }

  private inferenceTimesteps(numInferenceSteps: number): Float64Array {
    if (this.#timestepSpacing === "linspace") {
      return linspace(0, this.maxTimestep, numInferenceSteps).reverse();
    }

    const timesteps = new Float64Array(numInferenceSteps);
    if (this.#timestepSpacing === "leading") {
      const stepRatio = Math.floor(this.#numTrainTimesteps / numInferenceSteps);
      for (let index = 0; index < numInferenceSteps; index += 1) {
        timesteps[index] =
          Math.round((numInferenceSteps - 1 - index) * stepRatio) + this.#stepsOffset;
      }
    } else {
      const stepRatio = this.#numTrainTimesteps / numInferenceSteps;
      for (let index = 0; index < numInferenceSteps; index += 1) {
        timesteps[index] = Math.round(this.#numTrainTimesteps - index * stepRatio) - 1;
      }
    }

    for (const timestep of timesteps) {
      if (timestep < 0 || timestep > this.maxTimestep) {
        throw new Error(`timestep ${timestep} is outside the sigma schedule.`);
      }
    }
    return timesteps;
  }

  private maxInferenceSigma(numInferenceSteps: number): number {
    let maxSigma = 0;
    for (const timestep of this.inferenceTimesteps(numInferenceSteps)) {
      maxSigma = Math.max(maxSigma, this.sigmaAt(timestep));
    }
    return maxSigma;
  }

  private finalSigma(): number {
    if (this.#finalSigmasType === "sigma_min") {
      return this.requireSigma(0);
    }
    return 0;
  }

  private requireSigma(timestep: number): number {
    const sigma = this.#sigmas[timestep];
    if (sigma === undefined) {
      throw new Error("EulerScheduler: missing sigma schedule value.");
    }
    return sigma;
  }
}
