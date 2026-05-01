import {
  add,
  concatenate,
  type DType,
  divide,
  formatShape,
  full,
  type MxArray,
  maximum,
  minimum,
  multiply,
  mxEval,
  retainArray,
  split,
  subtract,
} from "@mlxts/core";

import type {
  FlowMatchEulerScheduler,
  FlowMatchEulerStep,
} from "../../schedulers/flow-match-euler";
import {
  createStableDiffusion3InitialLatents,
  type StableDiffusion3InitialLatentOptions,
  stableDiffusion3LatentShape,
} from "./latents";
import type { StableDiffusion3DenoiserInput } from "./transformer";

/** Scheduler implementation supported by the Stable Diffusion 3 sampling loop. */
export type StableDiffusion3Scheduler = FlowMatchEulerScheduler;

/** Prepared text conditioning tensors consumed by the SD3 transformer. */
export type StableDiffusion3Conditioning = {
  encoderHiddenStates: MxArray;
  pooledProjections: MxArray;
};

/** Conditional denoiser shape required by Stable Diffusion 3 sampling. */
export type StableDiffusion3Denoiser = {
  forward(input: StableDiffusion3DenoiserInput): MxArray;
};

/** VAE decoder surface required by Stable Diffusion 3 image decoding. */
export type StableDiffusion3LatentDecoder = {
  readonly scalingFactor: number;
  readonly shiftFactor: number;
  readonly latentChannels: number;
  readonly vaeScaleFactor?: number;
  decode(latents: MxArray): MxArray;
};

/** Options for denoising existing SD3 latents with prepared conditioning tensors. */
export type StableDiffusion3DenoiseOptions = {
  denoiser: StableDiffusion3Denoiser;
  scheduler: StableDiffusion3Scheduler;
  initialLatents: MxArray;
  conditioning: StableDiffusion3Conditioning;
  negativeConditioning?: StableDiffusion3Conditioning;
  guidanceScale?: number;
  numInferenceSteps: number;
  evaluateEachStep?: boolean;
  onStep?: (event: StableDiffusion3DenoisingStepEvent) => void;
};

/** Step event emitted after a denoising update has produced the next latent. */
export type StableDiffusion3DenoisingStepEvent = {
  stepIndex: number;
  timestep: number;
  previousTimestep: number;
  sigma: number;
  nextSigma: number;
  latents: MxArray;
};

/** Options for complete Stable Diffusion 3 image generation from prepared conditioning. */
export type StableDiffusion3ImageGenerationOptions = Omit<
  StableDiffusion3DenoiseOptions,
  "initialLatents"
> & {
  vae: StableDiffusion3LatentDecoder;
  batchSize: number;
  height: number;
  width: number;
  dtype?: DType;
  rngKey?: MxArray;
  vaeScaleFactor?: number;
};

type OwnedStableDiffusion3Conditioning = {
  encoderHiddenStates: MxArray;
  pooledProjections: MxArray;
};

function assertPositiveInteger(name: string, value: number): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
}

function resolveGuidanceScale(value: number | undefined): number {
  const scale = value ?? 1;
  if (!Number.isFinite(scale) || scale < 0) {
    throw new Error("guidanceScale must be a finite non-negative number.");
  }
  return scale;
}

function disposeOwnedConditioning(conditioning: OwnedStableDiffusion3Conditioning): void {
  conditioning.encoderHiddenStates.free();
  conditioning.pooledProjections.free();
}

function assertConditioningBatch(
  conditioning: StableDiffusion3Conditioning,
  batchSize: number,
  owner: string,
): void {
  if (
    conditioning.encoderHiddenStates.shape.length !== 3 ||
    conditioning.encoderHiddenStates.shape[0] !== batchSize
  ) {
    throw new Error(
      `${owner}: expected encoderHiddenStates batch ${batchSize}, got ${formatShape(
        conditioning.encoderHiddenStates.shape,
      )}.`,
    );
  }
  if (
    conditioning.pooledProjections.shape.length !== 2 ||
    conditioning.pooledProjections.shape[0] !== batchSize
  ) {
    throw new Error(
      `${owner}: expected pooledProjections batch ${batchSize}, got ${formatShape(
        conditioning.pooledProjections.shape,
      )}.`,
    );
  }
}

function assertMatchingConditioning(
  positive: StableDiffusion3Conditioning,
  negative: StableDiffusion3Conditioning,
): void {
  assertMatchingTensorShapeExceptBatch(
    positive.encoderHiddenStates,
    negative.encoderHiddenStates,
    "negativeConditioning encoderHiddenStates",
  );
  assertMatchingTensorShapeExceptBatch(
    positive.pooledProjections,
    negative.pooledProjections,
    "negativeConditioning pooledProjections",
  );
}

function assertMatchingTensorShapeExceptBatch(
  positive: MxArray,
  negative: MxArray,
  owner: string,
): void {
  if (positive.shape.length !== negative.shape.length) {
    throw new Error(`${owner} rank must match conditioning.`);
  }
  for (let index = 1; index < positive.shape.length; index += 1) {
    if (positive.shape[index] !== negative.shape[index]) {
      throw new Error(`${owner} shape must match conditioning.`);
    }
  }
}

function makeDenoiserInput(
  hiddenStates: MxArray,
  conditioning: StableDiffusion3Conditioning,
  timestep: MxArray,
): StableDiffusion3DenoiserInput {
  return {
    hiddenStates,
    encoderHiddenStates: conditioning.encoderHiddenStates,
    pooledProjections: conditioning.pooledProjections,
    timestep,
  };
}

function makeGuidedConditioning(
  positive: StableDiffusion3Conditioning,
  negative: StableDiffusion3Conditioning,
): OwnedStableDiffusion3Conditioning {
  const encoderHiddenStates = concatenate(
    [negative.encoderHiddenStates, positive.encoderHiddenStates],
    0,
  );
  try {
    return {
      encoderHiddenStates,
      pooledProjections: concatenate([negative.pooledProjections, positive.pooledProjections], 0),
    };
  } catch (error) {
    encoderHiddenStates.free();
    throw error;
  }
}

function predictStableDiffusion3Velocity(
  options: StableDiffusion3DenoiseOptions,
  latents: MxArray,
  step: FlowMatchEulerStep,
  batchSize: number,
): MxArray {
  using scaledLatents = options.scheduler.scaleModelInput(latents);
  const guidanceScale = resolveGuidanceScale(options.guidanceScale);
  if (guidanceScale <= 1) {
    using timestep = full([batchSize], step.timestep, scaledLatents.dtype);
    return options.denoiser.forward(
      makeDenoiserInput(scaledLatents, options.conditioning, timestep),
    );
  }
  if (options.negativeConditioning === undefined) {
    throw new Error("negativeConditioning is required when guidanceScale is greater than 1.");
  }
  assertConditioningBatch(options.negativeConditioning, batchSize, "negativeConditioning");
  assertMatchingConditioning(options.conditioning, options.negativeConditioning);
  using guidedLatents = concatenate([scaledLatents, scaledLatents], 0);
  const guidedConditioning = makeGuidedConditioning(
    options.conditioning,
    options.negativeConditioning,
  );
  try {
    using timestep = full([batchSize * 2], step.timestep, scaledLatents.dtype);
    using prediction = options.denoiser.forward(
      makeDenoiserInput(guidedLatents, guidedConditioning, timestep),
    );
    return applyStableDiffusion3ClassifierFreeGuidance(prediction, guidanceScale);
  } finally {
    disposeOwnedConditioning(guidedConditioning);
  }
}

function denoiseStep(
  options: StableDiffusion3DenoiseOptions,
  latents: MxArray,
  step: FlowMatchEulerStep,
  batchSize: number,
): MxArray {
  using prediction = predictStableDiffusion3Velocity(options, latents, step, batchSize);
  return options.scheduler.step(prediction, latents, step);
}

function latentSequenceLength(latents: MxArray): number {
  const [, height, width] = latents.shape;
  if (latents.shape.length !== 4 || height === undefined || width === undefined) {
    throw new Error(`initialLatents must be NHWC latents, got ${formatShape(latents.shape)}.`);
  }
  return height * width;
}

/** Apply SD3 classifier-free guidance to a paired prediction batch. */
export function applyStableDiffusion3ClassifierFreeGuidance(
  prediction: MxArray,
  guidanceScale: number,
): MxArray {
  const parts = split(prediction, 2, 0);
  try {
    const unconditional = parts[0];
    const conditional = parts[1];
    if (parts.length !== 2 || unconditional === undefined || conditional === undefined) {
      throw new Error("classifier-free guidance prediction must split into two batches.");
    }
    using delta = subtract(conditional, unconditional);
    using scaledDelta = multiply(delta, guidanceScale);
    return add(unconditional, scaledDelta);
  } finally {
    for (const part of parts) {
      part.free();
    }
  }
}

/** Denoise SD3 NHWC latents with prepared prompt embedding tensors. */
export function denoiseStableDiffusion3Latents(options: StableDiffusion3DenoiseOptions): MxArray {
  assertPositiveInteger("numInferenceSteps", options.numInferenceSteps);
  const [batchSize] = options.initialLatents.shape;
  if (options.initialLatents.shape.length !== 4 || batchSize === undefined) {
    throw new Error(
      `initialLatents must be NHWC latents, got ${formatShape(options.initialLatents.shape)}.`,
    );
  }
  assertConditioningBatch(options.conditioning, batchSize, "conditioning");
  const imageSequenceLength = latentSequenceLength(options.initialLatents);

  let current = retainArray(options.initialLatents);
  try {
    const steps = options.scheduler.timesteps(options.numInferenceSteps, {
      imageSequenceLength,
    });
    for (let index = 0; index < steps.length; index += 1) {
      const step = steps[index];
      if (step === undefined) {
        throw new Error("denoiseStableDiffusion3Latents: missing scheduler step.");
      }
      const next = denoiseStep(options, current, step, batchSize);
      current.free();
      current = next;
      if (options.evaluateEachStep ?? true) {
        mxEval(current);
      }
      options.onStep?.({
        stepIndex: index,
        timestep: step.timestep,
        previousTimestep: step.previousTimestep,
        sigma: step.sigma,
        nextSigma: step.nextSigma,
        latents: current,
      });
    }
    return current;
  } catch (error) {
    current.free();
    throw error;
  }
}

/** Decode Stable Diffusion 3 latents into an NHWC image tensor in the `0..1` range. */
export function decodeStableDiffusion3Latents(
  vae: StableDiffusion3LatentDecoder,
  latents: MxArray,
): MxArray {
  using scaled = divide(latents, vae.scalingFactor);
  using shifted = add(scaled, vae.shiftFactor);
  using decoded = vae.decode(shifted);
  using positive = add(decoded, 1);
  using normalized = divide(positive, 2);
  using clippedLow = maximum(normalized, 0);
  return minimum(clippedLow, 1);
}

/** Generate an image from prepared Stable Diffusion 3 conditioning tensors. */
export function generateStableDiffusion3Image(
  options: StableDiffusion3ImageGenerationOptions,
): MxArray {
  const initialLatentOptions: StableDiffusion3InitialLatentOptions = {
    scheduler: options.scheduler,
    batchSize: options.batchSize,
    height: options.height,
    width: options.width,
    latentChannels: options.vae.latentChannels,
  };
  const vaeScaleFactor = options.vaeScaleFactor ?? options.vae.vaeScaleFactor;
  if (vaeScaleFactor !== undefined) {
    initialLatentOptions.vaeScaleFactor = vaeScaleFactor;
  }
  if (options.dtype !== undefined) {
    initialLatentOptions.dtype = options.dtype;
  }
  if (options.rngKey !== undefined) {
    initialLatentOptions.rngKey = options.rngKey;
  }
  using initialLatents = createStableDiffusion3InitialLatents(initialLatentOptions);
  const denoiseOptions: StableDiffusion3DenoiseOptions = {
    denoiser: options.denoiser,
    scheduler: options.scheduler,
    initialLatents,
    conditioning: options.conditioning,
    numInferenceSteps: options.numInferenceSteps,
  };
  if (options.negativeConditioning !== undefined) {
    denoiseOptions.negativeConditioning = options.negativeConditioning;
  }
  if (options.guidanceScale !== undefined) {
    denoiseOptions.guidanceScale = options.guidanceScale;
  }
  if (options.evaluateEachStep !== undefined) {
    denoiseOptions.evaluateEachStep = options.evaluateEachStep;
  }
  if (options.onStep !== undefined) {
    denoiseOptions.onStep = options.onStep;
  }
  using denoised = denoiseStableDiffusion3Latents(denoiseOptions);
  return decodeStableDiffusion3Latents(options.vae, denoised);
}

export { createStableDiffusion3InitialLatents, stableDiffusion3LatentShape };
