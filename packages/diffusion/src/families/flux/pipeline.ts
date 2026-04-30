/**
 * FLUX tensor sampling contract over prepared conditioning tensors.
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
  retainArray,
  zeros,
} from "@mlxts/core";

import type {
  FlowMatchEulerScheduler,
  FlowMatchEulerStep,
} from "../../schedulers/flow-match-euler";
import {
  createFluxLatentImageIds,
  fluxPackedLatentShape,
  packFluxLatents,
  unpackFluxLatents,
} from "./latents";

/** Scheduler implementation supported by the FLUX sampling loop. */
export type FluxScheduler = FlowMatchEulerScheduler;

/** Conditional tensors consumed by the FLUX transformer denoiser. */
export type FluxConditioning = {
  encoderHiddenStates: MxArray;
  pooledProjections: MxArray;
  textIds?: MxArray;
  guidance?: MxArray;
};

/** Denoiser input names match the FLUX transformer semantics. */
export type FluxDenoiserInput = {
  hiddenStates: MxArray;
  imageIds: MxArray;
  encoderHiddenStates: MxArray;
  textIds: MxArray;
  pooledProjections: MxArray;
  timestep: MxArray;
  guidance?: MxArray;
};

/** Conditional denoiser shape required by FLUX sampling. */
export type FluxDenoiser = {
  forward(input: FluxDenoiserInput): MxArray;
};

/** VAE decoder shape required by FLUX image decoding. */
export type FluxLatentDecoder = {
  readonly scalingFactor: number;
  readonly shiftFactor: number;
  readonly latentChannels: number;
  readonly vaeScaleFactor?: number;
  decode(latents: MxArray): MxArray;
};

/** Shape options for initial FLUX latent sampling. */
export type FluxInitialLatentOptions = {
  scheduler: FluxScheduler;
  batchSize: number;
  height: number;
  width: number;
  latentChannels: number;
  vaeScaleFactor?: number;
  dtype?: DType;
  rngKey?: MxArray;
};

/** Options for denoising existing packed FLUX latents. */
export type FluxDenoiseOptions = {
  denoiser: FluxDenoiser;
  scheduler: FluxScheduler;
  initialLatents: MxArray;
  latentHeight: number;
  latentWidth: number;
  conditioning: FluxConditioning;
  imageIds?: MxArray;
  numInferenceSteps: number;
  evaluateEachStep?: boolean;
  onStep?: (event: FluxDenoisingStepEvent) => void;
};

/** Step event emitted after a denoising update has produced the next latent. */
export type FluxDenoisingStepEvent = {
  stepIndex: number;
  timestep: number;
  previousTimestep: number;
  sigma: number;
  nextSigma: number;
  latents: MxArray;
};

/** Options for complete FLUX image generation from supplied conditioning. */
export type FluxImageGenerationOptions = Omit<
  FluxDenoiseOptions,
  "initialLatents" | "latentHeight" | "latentWidth"
> & {
  vae: FluxLatentDecoder;
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

function assertEvenPositiveInteger(name: string, value: number): void {
  assertPositiveInteger(name, value);
  if (value % 2 !== 0) {
    throw new Error(`${name} must be divisible by 2.`);
  }
}

function resolveVaeScaleFactor(
  vaeScaleFactor: number | undefined,
  vae?: FluxLatentDecoder,
): number {
  const resolved = vaeScaleFactor ?? vae?.vaeScaleFactor ?? 8;
  assertPositiveInteger("vaeScaleFactor", resolved);
  return resolved;
}

function assertPackedFluxLatents(
  latents: MxArray,
  latentHeight: number,
  latentWidth: number,
): readonly [number, number, number] {
  const [batchSize, sequenceLength, packedChannels] = latents.shape;
  if (
    latents.shape.length !== 3 ||
    batchSize === undefined ||
    sequenceLength === undefined ||
    packedChannels === undefined
  ) {
    throw new Error(
      `initialLatents must be packed FLUX latents, got ${formatShape(latents.shape)}.`,
    );
  }
  assertPositiveInteger("batchSize", batchSize);
  assertEvenPositiveInteger("latentHeight", latentHeight);
  assertEvenPositiveInteger("latentWidth", latentWidth);
  if (packedChannels % 4 !== 0) {
    throw new Error("initialLatents packed channel dimension must be divisible by 4.");
  }
  const expectedShape = fluxPackedLatentShape(
    batchSize,
    latentHeight,
    latentWidth,
    packedChannels / 4,
  );
  if (sequenceLength !== expectedShape[1]) {
    throw new Error(
      `initialLatents sequence length must be ${expectedShape[1]}, got ${sequenceLength}.`,
    );
  }
  return [batchSize, sequenceLength, packedChannels];
}

function assertIdsShape(ids: MxArray, name: string, expectedLength: number): void {
  const [length, columns] = ids.shape;
  if (ids.shape.length !== 2 || length !== expectedLength || columns !== 3) {
    throw new Error(
      `${name} must have shape [${expectedLength}, 3], got ${formatShape(ids.shape)}.`,
    );
  }
}

function assertConditioning(conditioning: FluxConditioning, batchSize: number): number {
  const [encoderBatch, textLength] = conditioning.encoderHiddenStates.shape;
  if (
    conditioning.encoderHiddenStates.shape.length !== 3 ||
    encoderBatch !== batchSize ||
    textLength === undefined
  ) {
    throw new Error(
      `conditioning.encoderHiddenStates must have batch ${batchSize}, got ${formatShape(
        conditioning.encoderHiddenStates.shape,
      )}.`,
    );
  }
  const [pooledBatch] = conditioning.pooledProjections.shape;
  if (conditioning.pooledProjections.shape.length !== 2 || pooledBatch !== batchSize) {
    throw new Error(
      `conditioning.pooledProjections must have batch ${batchSize}, got ${formatShape(
        conditioning.pooledProjections.shape,
      )}.`,
    );
  }
  if (conditioning.textIds !== undefined) {
    assertIdsShape(conditioning.textIds, "conditioning.textIds", textLength);
  }
  if (
    conditioning.guidance !== undefined &&
    (conditioning.guidance.shape.length !== 1 || conditioning.guidance.shape[0] !== batchSize)
  ) {
    throw new Error(
      `conditioning.guidance must have batch ${batchSize}, got ${formatShape(
        conditioning.guidance.shape,
      )}.`,
    );
  }
  return textLength;
}

function makeDenoiserInput(
  options: FluxDenoiseOptions,
  hiddenStates: MxArray,
  imageIds: MxArray,
  textIds: MxArray,
  timestep: MxArray,
): FluxDenoiserInput {
  const input: FluxDenoiserInput = {
    hiddenStates,
    imageIds,
    encoderHiddenStates: options.conditioning.encoderHiddenStates,
    textIds,
    pooledProjections: options.conditioning.pooledProjections,
    timestep,
  };
  if (options.conditioning.guidance !== undefined) {
    input.guidance = options.conditioning.guidance;
  }
  return input;
}

function predictVelocity(
  options: FluxDenoiseOptions,
  latents: MxArray,
  imageIds: MxArray,
  textIds: MxArray,
  step: FlowMatchEulerStep,
  batchSize: number,
): MxArray {
  using scaledLatents = options.scheduler.scaleModelInput(latents);
  using timestep = full([batchSize], step.sigma, scaledLatents.dtype);
  return options.denoiser.forward(
    makeDenoiserInput(options, scaledLatents, imageIds, textIds, timestep),
  );
}

function denoiseStep(
  options: FluxDenoiseOptions,
  latents: MxArray,
  imageIds: MxArray,
  textIds: MxArray,
  step: FlowMatchEulerStep,
  batchSize: number,
): MxArray {
  using prediction = predictVelocity(options, latents, imageIds, textIds, step, batchSize);
  return options.scheduler.step(prediction, latents, step);
}

/** Return the NHWC latent shape for an image size and FLUX VAE scale factor. */
export function fluxLatentShape(
  options: Omit<FluxInitialLatentOptions, "scheduler" | "dtype" | "rngKey">,
): readonly [number, number, number, number] {
  assertPositiveInteger("batchSize", options.batchSize);
  assertPositiveInteger("height", options.height);
  assertPositiveInteger("width", options.width);
  assertPositiveInteger("latentChannels", options.latentChannels);
  const vaeScaleFactor = resolveVaeScaleFactor(options.vaeScaleFactor);
  if (options.height % vaeScaleFactor !== 0 || options.width % vaeScaleFactor !== 0) {
    throw new Error("height and width must be divisible by vaeScaleFactor.");
  }
  const latentHeight = options.height / vaeScaleFactor;
  const latentWidth = options.width / vaeScaleFactor;
  assertEvenPositiveInteger("latentHeight", latentHeight);
  assertEvenPositiveInteger("latentWidth", latentWidth);
  return [options.batchSize, latentHeight, latentWidth, options.latentChannels];
}

/** Create packed FLUX initial noise latents for text-to-image sampling. */
export function createFluxInitialLatents(options: FluxInitialLatentOptions): MxArray {
  const shape = fluxLatentShape(options);
  using latents = options.scheduler.samplePrior(
    [...shape],
    options.dtype ?? "float32",
    options.rngKey,
  );
  return packFluxLatents(latents);
}

/** Create FLUX text position ids for prepared text embeddings. */
export function createFluxTextIds(textSequenceLength: number, dtype: DType = "int32"): MxArray {
  assertPositiveInteger("textSequenceLength", textSequenceLength);
  return zeros([textSequenceLength, 3], dtype);
}

/** Denoise packed FLUX latents with prepared conditioning tensors. */
export function denoiseFluxLatents(options: FluxDenoiseOptions): MxArray {
  assertPositiveInteger("numInferenceSteps", options.numInferenceSteps);
  const [batchSize, sequenceLength] = assertPackedFluxLatents(
    options.initialLatents,
    options.latentHeight,
    options.latentWidth,
  );
  const textLength = assertConditioning(options.conditioning, batchSize);
  const imageIds =
    options.imageIds === undefined
      ? createFluxLatentImageIds(options.latentHeight / 2, options.latentWidth / 2)
      : retainArray(options.imageIds);
  const textIds =
    options.conditioning.textIds === undefined
      ? createFluxTextIds(textLength)
      : retainArray(options.conditioning.textIds);

  try {
    assertIdsShape(imageIds, "imageIds", sequenceLength);
    assertIdsShape(textIds, "conditioning.textIds", textLength);

    let current = retainArray(options.initialLatents);
    try {
      const steps = options.scheduler.timesteps(options.numInferenceSteps, {
        imageSequenceLength: sequenceLength,
      });
      for (let index = 0; index < steps.length; index += 1) {
        const step = steps[index];
        if (step === undefined) {
          throw new Error("denoiseFluxLatents: missing scheduler step.");
        }
        const next = denoiseStep(options, current, imageIds, textIds, step, batchSize);
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
  } finally {
    imageIds.free();
    textIds.free();
  }
}

/** Decode packed FLUX latents into an NHWC image tensor in the 0..1 range. */
export function decodeFluxLatents(
  vae: FluxLatentDecoder,
  latents: MxArray,
  latentHeight: number,
  latentWidth: number,
): MxArray {
  using unpacked = unpackFluxLatents(latents, latentHeight, latentWidth);
  using scaled = divide(unpacked, vae.scalingFactor);
  using shiftedLatents = add(scaled, vae.shiftFactor);
  using decoded = vae.decode(shiftedLatents);
  using shiftedImage = add(decoded, 1);
  using normalized = multiply(shiftedImage, 0.5);
  using clippedLow = maximum(normalized, 0);
  return minimum(clippedLow, 1);
}

/** Generate an image from supplied FLUX conditioning tensors. */
export function generateFluxImage(options: FluxImageGenerationOptions): MxArray {
  const vaeScaleFactor = resolveVaeScaleFactor(options.vaeScaleFactor, options.vae);
  const initialLatentOptions: FluxInitialLatentOptions = {
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
  using initialLatents = createFluxInitialLatents(initialLatentOptions);
  const [, latentHeight, latentWidth] = fluxLatentShape(initialLatentOptions);

  const denoiseOptions: FluxDenoiseOptions = {
    denoiser: options.denoiser,
    scheduler: options.scheduler,
    initialLatents,
    latentHeight,
    latentWidth,
    conditioning: options.conditioning,
    numInferenceSteps: options.numInferenceSteps,
  };
  if (options.imageIds !== undefined) {
    denoiseOptions.imageIds = options.imageIds;
  }
  if (options.evaluateEachStep !== undefined) {
    denoiseOptions.evaluateEachStep = options.evaluateEachStep;
  }
  if (options.onStep !== undefined) {
    denoiseOptions.onStep = options.onStep;
  }

  using denoised = denoiseFluxLatents(denoiseOptions);
  return decodeFluxLatents(options.vae, denoised, latentHeight, latentWidth);
}
