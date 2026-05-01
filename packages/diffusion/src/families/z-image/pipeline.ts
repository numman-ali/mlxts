/**
 * Z-Image tensor sampling contract over prepared caption embeddings.
 * @module
 */

import {
  add,
  type DType,
  divide,
  formatShape,
  full,
  type MxArray,
  maximum,
  minimum,
  multiply,
  mxEval,
  negative,
  retainArray,
  squeeze,
  transpose,
} from "@mlxts/core";

import type {
  FlowMatchEulerScheduler,
  FlowMatchEulerStep,
} from "../../schedulers/flow-match-euler";
import {
  createZImageInitialLatents,
  sliceZImageLatentBatchItem,
  type ZImageInitialLatentOptions,
  zImageLatentShape,
} from "./latents";
import type { ZImageDenoiserInput } from "./transformer";

/** Scheduler implementation supported by the Z-Image sampling loop. */
export type ZImageScheduler = FlowMatchEulerScheduler;

/** Conditional tensors consumed by the Z-Image transformer denoiser. */
export type ZImageConditioning = {
  captionFeatures: readonly MxArray[];
};

/** Conditional denoiser shape required by Z-Image sampling. */
export type ZImageDenoiser = {
  forward(input: ZImageDenoiserInput): MxArray;
};

/** VAE decoder shape required by Z-Image image decoding. */
export type ZImageLatentDecoder = {
  readonly scalingFactor: number;
  readonly shiftFactor: number;
  readonly latentChannels: number;
  readonly vaeScaleFactor?: number;
  decode(latents: MxArray): MxArray;
};

/** Options for denoising existing Z-Image NCHW latents. */
export type ZImageDenoiseOptions = {
  denoiser: ZImageDenoiser;
  scheduler: ZImageScheduler;
  initialLatents: MxArray;
  conditioning: ZImageConditioning;
  numInferenceSteps: number;
  evaluateEachStep?: boolean;
  onStep?: (event: ZImageDenoisingStepEvent) => void;
};

/** Step event emitted after a denoising update has produced the next latent. */
export type ZImageDenoisingStepEvent = {
  stepIndex: number;
  timestep: number;
  previousTimestep: number;
  sigma: number;
  nextSigma: number;
  latents: MxArray;
};

/** Options for complete Z-Image generation from supplied caption embeddings. */
export type ZImageGenerationOptions = Omit<ZImageDenoiseOptions, "initialLatents"> & {
  vae: ZImageLatentDecoder;
  batchSize: number;
  height: number;
  width: number;
  dtype?: DType;
  rngKey?: MxArray;
  vaeScaleFactor?: number;
};

function assertPositiveInteger(name: string, value: number): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
}

function resolveVaeScaleFactor(
  vaeScaleFactor: number | undefined,
  vae?: ZImageLatentDecoder,
): number {
  const resolved = vaeScaleFactor ?? vae?.vaeScaleFactor ?? 8;
  assertPositiveInteger("vaeScaleFactor", resolved);
  return resolved;
}

function assertZImageLatents(latents: MxArray): readonly [number, number, number, number] {
  const [batchSize, channels, height, width] = latents.shape;
  if (
    latents.shape.length !== 4 ||
    batchSize === undefined ||
    channels === undefined ||
    height === undefined ||
    width === undefined
  ) {
    throw new Error(
      `initialLatents must be NCHW Z-Image latents, got ${formatShape(latents.shape)}.`,
    );
  }
  assertPositiveInteger("batchSize", batchSize);
  assertPositiveInteger("channels", channels);
  assertPositiveInteger("height", height);
  assertPositiveInteger("width", width);
  return [batchSize, channels, height, width];
}

function assertConditioning(conditioning: ZImageConditioning, batchSize: number): void {
  if (conditioning.captionFeatures.length !== batchSize) {
    throw new Error(
      `conditioning.captionFeatures length must be ${batchSize}, got ${conditioning.captionFeatures.length}.`,
    );
  }
  if (batchSize !== 1) {
    throw new Error("Z-Image sampling currently supports batch size 1.");
  }
}

function makeDenoiserInput(
  options: ZImageDenoiseOptions,
  latents: MxArray,
  timestep: MxArray,
): ZImageDenoiserInput {
  using latent = sliceZImageLatentBatchItem(latents, 0);
  return {
    latents: [retainArray(latent)],
    captionFeatures: options.conditioning.captionFeatures,
    timestep,
  };
}

function freeDenoiserInput(input: ZImageDenoiserInput): void {
  for (const latent of input.latents) {
    latent.free();
  }
}

function predictVelocity(
  options: ZImageDenoiseOptions,
  latents: MxArray,
  step: FlowMatchEulerStep,
  batchSize: number,
): MxArray {
  using scaledLatents = options.scheduler.scaleModelInput(latents);
  using timestep = full([batchSize], (1000 - step.timestep) / 1000, scaledLatents.dtype);
  const input = makeDenoiserInput(options, scaledLatents, timestep);
  try {
    using prediction = options.denoiser.forward(input);
    using squeezed = squeeze(prediction, 2);
    return negative(squeezed);
  } finally {
    freeDenoiserInput(input);
  }
}

function denoiseStep(
  options: ZImageDenoiseOptions,
  latents: MxArray,
  step: FlowMatchEulerStep,
  batchSize: number,
): MxArray {
  using prediction = predictVelocity(options, latents, step, batchSize);
  return options.scheduler.step(prediction, latents, step);
}

/** Denoise NCHW Z-Image latents with prepared caption tensors. */
export function denoiseZImageLatents(options: ZImageDenoiseOptions): MxArray {
  assertPositiveInteger("numInferenceSteps", options.numInferenceSteps);
  const [batchSize, , latentHeight, latentWidth] = assertZImageLatents(options.initialLatents);
  assertConditioning(options.conditioning, batchSize);
  const imageSequenceLength = (latentHeight / 2) * (latentWidth / 2);

  let current = retainArray(options.initialLatents);
  try {
    const steps = options.scheduler.timesteps(options.numInferenceSteps, {
      imageSequenceLength,
    });
    for (let index = 0; index < steps.length; index += 1) {
      const step = steps[index];
      if (step === undefined) {
        throw new Error("denoiseZImageLatents: missing scheduler step.");
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

/** Decode NCHW Z-Image latents into an NHWC image tensor in the 0..1 range. */
export function decodeZImageLatents(vae: ZImageLatentDecoder, latents: MxArray): MxArray {
  using nhwc = transpose(latents, [0, 2, 3, 1]);
  using scaled = divide(nhwc, vae.scalingFactor);
  using shiftedLatents = add(scaled, vae.shiftFactor);
  using decoded = vae.decode(shiftedLatents);
  using shiftedImage = add(decoded, 1);
  using normalized = multiply(shiftedImage, 0.5);
  using clippedLow = maximum(normalized, 0);
  return minimum(clippedLow, 1);
}

/** Generate an image from supplied Z-Image caption conditioning tensors. */
export function generateZImage(options: ZImageGenerationOptions): MxArray {
  const vaeScaleFactor = resolveVaeScaleFactor(options.vaeScaleFactor, options.vae);
  const initialLatentOptions: ZImageInitialLatentOptions = {
    scheduler: options.scheduler,
    batchSize: options.batchSize,
    height: options.height,
    width: options.width,
    latentChannels: options.vae.latentChannels,
    vaeScaleFactor,
  };
  if (options.dtype !== undefined) {
    initialLatentOptions.dtype = options.dtype;
  }
  if (options.rngKey !== undefined) {
    initialLatentOptions.rngKey = options.rngKey;
  }
  using initialLatents = createZImageInitialLatents(initialLatentOptions);
  zImageLatentShape(initialLatentOptions);

  const denoiseOptions: ZImageDenoiseOptions = {
    denoiser: options.denoiser,
    scheduler: options.scheduler,
    initialLatents,
    conditioning: options.conditioning,
    numInferenceSteps: options.numInferenceSteps,
  };
  if (options.evaluateEachStep !== undefined) {
    denoiseOptions.evaluateEachStep = options.evaluateEachStep;
  }
  if (options.onStep !== undefined) {
    denoiseOptions.onStep = options.onStep;
  }

  using denoised = denoiseZImageLatents(denoiseOptions);
  return decodeZImageLatents(options.vae, denoised);
}
