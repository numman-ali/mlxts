/**
 * Stable Diffusion pipeline assembly over package-owned diffusion components.
 * @module
 */

import {
  add,
  concatenate,
  type DType,
  divide,
  formatShape,
  type MxArray,
  maximum,
  minimum,
  multiply,
  mxEval,
  random,
  retainArray,
  split,
  subtract,
} from "@mlxts/core";

import type { DDIMScheduler } from "../../schedulers/ddim";
import { EulerScheduler } from "../../schedulers/euler";
import type {
  StableDiffusionUNetForwardOptions,
  StableDiffusionUNetTextTimeConditioning,
} from "./unet";

/** Scheduler implementations supported by the Stable Diffusion sampling loop. */
export type StableDiffusionScheduler = DDIMScheduler | EulerScheduler;

/** Conditional denoiser shape required by Stable Diffusion sampling. */
export type StableDiffusionDenoiser = {
  forward(
    x: MxArray,
    timestep: number | MxArray,
    encoderHiddenStates: MxArray,
    options?: StableDiffusionUNetForwardOptions,
  ): MxArray;
};

/** VAE decoder shape required by Stable Diffusion image decoding. */
export type StableDiffusionLatentDecoder = {
  readonly scalingFactor: number;
  readonly latentChannels: number;
  readonly vaeScaleFactor?: number;
  decode(latents: MxArray): MxArray;
};

/** Text conditioning tensors consumed by the UNet cross-attention path. */
export type StableDiffusionConditioning = {
  encoderHiddenStates: MxArray;
  textTime?: StableDiffusionUNetTextTimeConditioning;
};

/** Shape options for initial Stable Diffusion latent sampling. */
export type StableDiffusionInitialLatentOptions = {
  scheduler: StableDiffusionScheduler;
  batchSize: number;
  height: number;
  width: number;
  latentChannels: number;
  vaeScaleFactor?: number;
  dtype?: DType;
  rngKey?: MxArray;
};

/** Options for denoising an existing latent tensor. */
export type StableDiffusionDenoiseOptions = {
  unet: StableDiffusionDenoiser;
  scheduler: StableDiffusionScheduler;
  initialLatents: MxArray;
  conditioning: StableDiffusionConditioning;
  negativeConditioning?: StableDiffusionConditioning;
  guidanceScale?: number;
  numInferenceSteps: number;
  evaluateEachStep?: boolean;
  onStep?: (event: StableDiffusionDenoisingStepEvent) => void;
};

/** Step event emitted after a denoising update has produced the next latent. */
export type StableDiffusionDenoisingStepEvent = {
  stepIndex: number;
  timestep: number;
  previousTimestep: number;
  latents: MxArray;
};

/** Options for complete Stable Diffusion image generation from supplied conditioning. */
export type StableDiffusionImageGenerationOptions = Omit<
  StableDiffusionDenoiseOptions,
  "initialLatents"
> & {
  vae: StableDiffusionLatentDecoder;
  batchSize: number;
  height: number;
  width: number;
  dtype?: DType;
  rngKey?: MxArray;
  vaeScaleFactor?: number;
};

type OwnedStableDiffusionConditioning = {
  encoderHiddenStates: MxArray;
  textTime?: StableDiffusionUNetTextTimeConditioning;
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

function usesClassifierFreeGuidance(options: StableDiffusionDenoiseOptions): boolean {
  return resolveGuidanceScale(options.guidanceScale) > 1;
}

function assertConditioningBatch(
  conditioning: StableDiffusionConditioning,
  batchSize: number,
  owner: string,
): void {
  const [batch] = conditioning.encoderHiddenStates.shape;
  if (conditioning.encoderHiddenStates.shape.length !== 3 || batch !== batchSize) {
    throw new Error(
      `${owner}: expected encoderHiddenStates batch ${batchSize}, got ${formatShape(
        conditioning.encoderHiddenStates.shape,
      )}.`,
    );
  }
  if (
    conditioning.textTime !== undefined &&
    (conditioning.textTime.textEmbeds.shape[0] !== batchSize ||
      conditioning.textTime.timeIds.shape[0] !== batchSize)
  ) {
    throw new Error(`${owner}: textTime batch must match encoderHiddenStates batch.`);
  }
}

function assertMatchingConditioning(
  positive: StableDiffusionConditioning,
  negative: StableDiffusionConditioning,
): void {
  if (positive.encoderHiddenStates.shape.length !== negative.encoderHiddenStates.shape.length) {
    throw new Error("negativeConditioning encoderHiddenStates rank must match conditioning.");
  }
  for (let index = 1; index < positive.encoderHiddenStates.shape.length; index += 1) {
    if (positive.encoderHiddenStates.shape[index] !== negative.encoderHiddenStates.shape[index]) {
      throw new Error("negativeConditioning encoderHiddenStates shape must match conditioning.");
    }
  }
  if ((positive.textTime === undefined) !== (negative.textTime === undefined)) {
    throw new Error("negativeConditioning textTime must match conditioning textTime.");
  }
  if (positive.textTime !== undefined && negative.textTime !== undefined) {
    assertMatchingTensorShapeExceptBatch(
      positive.textTime.textEmbeds,
      negative.textTime.textEmbeds,
      "negativeConditioning textEmbeds",
    );
    assertMatchingTensorShapeExceptBatch(
      positive.textTime.timeIds,
      negative.textTime.timeIds,
      "negativeConditioning timeIds",
    );
  }
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

function disposeOwnedConditioning(conditioning: OwnedStableDiffusionConditioning): void {
  conditioning.encoderHiddenStates.free();
  conditioning.textTime?.textEmbeds.free();
  conditioning.textTime?.timeIds.free();
}

function makeUNetForwardOptions(
  textTime: StableDiffusionUNetTextTimeConditioning | undefined,
): StableDiffusionUNetForwardOptions | undefined {
  return textTime === undefined ? undefined : { textTime };
}

function concatenateTextTime(
  negative: StableDiffusionUNetTextTimeConditioning | undefined,
  positive: StableDiffusionUNetTextTimeConditioning | undefined,
): StableDiffusionUNetTextTimeConditioning | undefined {
  if (negative === undefined || positive === undefined) {
    return undefined;
  }
  const textEmbeds = concatenate([negative.textEmbeds, positive.textEmbeds], 0);
  try {
    return {
      textEmbeds,
      timeIds: concatenate([negative.timeIds, positive.timeIds], 0),
    };
  } catch (error) {
    textEmbeds.free();
    throw error;
  }
}

function makeGuidedConditioning(
  positive: StableDiffusionConditioning,
  negative: StableDiffusionConditioning,
): OwnedStableDiffusionConditioning {
  const encoderHiddenStates = concatenate(
    [negative.encoderHiddenStates, positive.encoderHiddenStates],
    0,
  );
  try {
    const textTime = concatenateTextTime(negative.textTime, positive.textTime);
    if (textTime === undefined) {
      return { encoderHiddenStates };
    }
    return {
      encoderHiddenStates,
      textTime,
    };
  } catch (error) {
    encoderHiddenStates.free();
    throw error;
  }
}

function scaleSchedulerInput(
  scheduler: StableDiffusionScheduler,
  latents: MxArray,
  timestep: number,
): MxArray {
  if (scheduler instanceof EulerScheduler) {
    return scheduler.scaleModelInput(latents, timestep);
  }
  return retainArray(latents);
}

function stepScheduler(
  scheduler: StableDiffusionScheduler,
  modelOutput: MxArray,
  latents: MxArray,
  timestep: number,
  previousTimestep: number,
): MxArray {
  if (scheduler instanceof EulerScheduler) {
    return scheduler.step(modelOutput, latents, { timestep, previousTimestep });
  }
  const output = scheduler.step(modelOutput, latents, { timestep, previousTimestep });
  output.predOriginalSample.free();
  return output.prevSample;
}

function schedulerSteps(
  scheduler: StableDiffusionScheduler,
  numInferenceSteps: number,
): readonly { timestep: number; previousTimestep: number }[] {
  return scheduler instanceof EulerScheduler
    ? scheduler.timesteps(numInferenceSteps)
    : scheduler.steps(numInferenceSteps);
}

function predictNoise(
  options: StableDiffusionDenoiseOptions,
  latents: MxArray,
  timestep: number,
  batchSize: number,
): MxArray {
  using scaledLatents = scaleSchedulerInput(options.scheduler, latents, timestep);
  if (!usesClassifierFreeGuidance(options)) {
    return options.unet.forward(
      scaledLatents,
      timestep,
      options.conditioning.encoderHiddenStates,
      makeUNetForwardOptions(options.conditioning.textTime),
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
    using prediction = options.unet.forward(
      guidedLatents,
      timestep,
      guidedConditioning.encoderHiddenStates,
      makeUNetForwardOptions(guidedConditioning.textTime),
    );
    return applyStableDiffusionClassifierFreeGuidance(
      prediction,
      resolveGuidanceScale(options.guidanceScale),
    );
  } finally {
    disposeOwnedConditioning(guidedConditioning);
  }
}

function denoiseStep(
  options: StableDiffusionDenoiseOptions,
  latents: MxArray,
  step: { timestep: number; previousTimestep: number },
  batchSize: number,
): MxArray {
  using prediction = predictNoise(options, latents, step.timestep, batchSize);
  return stepScheduler(
    options.scheduler,
    prediction,
    latents,
    step.timestep,
    step.previousTimestep,
  );
}

function resolveVaeScaleFactor(options: StableDiffusionImageGenerationOptions): number | undefined {
  return options.vaeScaleFactor ?? options.vae.vaeScaleFactor;
}

/** Return the NHWC latent shape for an image size and VAE scale factor. */
export function stableDiffusionLatentShape(
  options: Omit<StableDiffusionInitialLatentOptions, "scheduler" | "dtype" | "rngKey">,
): readonly [number, number, number, number] {
  assertPositiveInteger("batchSize", options.batchSize);
  assertPositiveInteger("height", options.height);
  assertPositiveInteger("width", options.width);
  assertPositiveInteger("latentChannels", options.latentChannels);
  const vaeScaleFactor = options.vaeScaleFactor ?? 8;
  assertPositiveInteger("vaeScaleFactor", vaeScaleFactor);
  if (options.height % vaeScaleFactor !== 0 || options.width % vaeScaleFactor !== 0) {
    throw new Error("height and width must be divisible by vaeScaleFactor.");
  }
  return [
    options.batchSize,
    options.height / vaeScaleFactor,
    options.width / vaeScaleFactor,
    options.latentChannels,
  ];
}

/** Create scheduler-scaled initial noise latents for text-to-image sampling. */
export function createStableDiffusionInitialLatents(
  options: StableDiffusionInitialLatentOptions,
): MxArray {
  const shape = stableDiffusionLatentShape(options);
  const dtype = options.dtype ?? "float32";
  if (options.scheduler instanceof EulerScheduler) {
    return options.scheduler.samplePrior([...shape], dtype, options.rngKey);
  }
  return random.normal([...shape], dtype, 0, 1, options.rngKey);
}

/** Apply Stable Diffusion classifier-free guidance to a paired prediction batch. */
export function applyStableDiffusionClassifierFreeGuidance(
  prediction: MxArray,
  guidanceScale: number,
): MxArray {
  const parts = split(prediction, 2, 0);
  try {
    if (parts.length !== 2) {
      throw new Error("classifier-free guidance prediction must split into two batches.");
    }
    const unconditional = parts[0];
    const conditional = parts[1];
    if (unconditional === undefined || conditional === undefined) {
      throw new Error("classifier-free guidance prediction split returned empty batches.");
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

/** Denoise an initial latent tensor with an explicit scheduler and conditioning tensors. */
export function denoiseStableDiffusionLatents(options: StableDiffusionDenoiseOptions): MxArray {
  assertPositiveInteger("numInferenceSteps", options.numInferenceSteps);
  const [batchSize] = options.initialLatents.shape;
  if (options.initialLatents.shape.length !== 4 || batchSize === undefined) {
    throw new Error(
      `initialLatents must be NHWC latents, got ${formatShape(options.initialLatents.shape)}.`,
    );
  }
  assertConditioningBatch(options.conditioning, batchSize, "conditioning");

  let current = retainArray(options.initialLatents);
  try {
    const steps = schedulerSteps(options.scheduler, options.numInferenceSteps);
    for (let index = 0; index < steps.length; index += 1) {
      const step = steps[index];
      if (step === undefined) {
        throw new Error("denoiseStableDiffusionLatents: missing scheduler step.");
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
        latents: current,
      });
    }
    return current;
  } catch (error) {
    current.free();
    throw error;
  }
}

/** Decode Stable Diffusion latents into an NHWC image tensor in the 0..1 range. */
export function decodeStableDiffusionLatents(
  vae: StableDiffusionLatentDecoder,
  latents: MxArray,
): MxArray {
  using scaled = divide(latents, vae.scalingFactor);
  using decoded = vae.decode(scaled);
  using shifted = divide(decoded, 2);
  using normalized = add(shifted, 0.5);
  using clippedLow = maximum(normalized, 0);
  return minimum(clippedLow, 1);
}

/** Generate an image from supplied Stable Diffusion conditioning tensors. */
export function generateStableDiffusionImage(
  options: StableDiffusionImageGenerationOptions,
): MxArray {
  const initialLatentOptions: StableDiffusionInitialLatentOptions = {
    scheduler: options.scheduler,
    batchSize: options.batchSize,
    height: options.height,
    width: options.width,
    latentChannels: options.vae.latentChannels,
  };
  const vaeScaleFactor = resolveVaeScaleFactor(options);
  if (vaeScaleFactor !== undefined) {
    initialLatentOptions.vaeScaleFactor = vaeScaleFactor;
  }
  if (options.dtype !== undefined) {
    initialLatentOptions.dtype = options.dtype;
  }
  if (options.rngKey !== undefined) {
    initialLatentOptions.rngKey = options.rngKey;
  }
  using initialLatents = createStableDiffusionInitialLatents(initialLatentOptions);

  const denoiseOptions: StableDiffusionDenoiseOptions = {
    unet: options.unet,
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
  using denoised = denoiseStableDiffusionLatents(denoiseOptions);
  return decodeStableDiffusionLatents(options.vae, denoised);
}
