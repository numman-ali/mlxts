import {
  add,
  type DType,
  formatShape,
  full,
  type MxArray,
  multiply,
  mxEval,
  retainArray,
  subtract,
} from "@mlxts/core";

import type {
  FlowMatchEulerScheduler,
  FlowMatchEulerStep,
} from "../../schedulers/flow-match-euler";
import { decodeFlux2KleinLatents, type Flux2KleinLatentDecoder } from "./decoding";
import {
  createFlux2InitialLatents,
  createFlux2LatentIds,
  createFlux2TextIds,
  type Flux2InitialLatentOptions,
  flux2LatentMapShape,
} from "./latents";

/** Scheduler implementation supported by the FLUX.2 Klein sampling loop. */
export type Flux2KleinScheduler = FlowMatchEulerScheduler;

/** Prepared text conditioning consumed by the FLUX.2 Klein transformer denoiser. */
export type Flux2KleinConditioning = {
  promptEmbeds: MxArray;
  textIds?: MxArray;
  negativePromptEmbeds?: MxArray;
  negativeTextIds?: MxArray;
  guidanceScale?: number;
};

/** Denoiser input names match Diffusers FLUX.2 transformer semantics. */
export type Flux2KleinDenoiserInput = {
  hiddenStates: MxArray;
  encoderHiddenStates: MxArray;
  timestep: MxArray;
  guidance?: MxArray;
  imageIds: MxArray;
  textIds: MxArray;
};

/** Conditional denoiser shape required by FLUX.2 Klein sampling. */
export type Flux2KleinDenoiser = {
  forward(input: Flux2KleinDenoiserInput): MxArray;
};

/** Options for denoising existing packed FLUX.2 Klein latents. */
export type Flux2KleinDenoiseOptions = {
  denoiser: Flux2KleinDenoiser;
  scheduler: Flux2KleinScheduler;
  initialLatents: MxArray;
  packedHeight: number;
  packedWidth: number;
  conditioning: Flux2KleinConditioning;
  numInferenceSteps: number;
  isDistilled?: boolean;
  sigmas?: readonly number[];
  evaluateEachStep?: boolean;
  onStep?: (event: Flux2KleinDenoisingStepEvent) => void;
};

/** Step event emitted after a denoising update has produced the next packed latent. */
export type Flux2KleinDenoisingStepEvent = {
  stepIndex: number;
  timestep: number;
  previousTimestep: number;
  sigma: number;
  nextSigma: number;
  latents: MxArray;
};

/** Options for complete FLUX.2 Klein generation from supplied prompt embeddings. */
export type Flux2KleinImageGenerationOptions = Omit<
  Flux2KleinDenoiseOptions,
  "initialLatents" | "packedHeight" | "packedWidth"
> & {
  vae: Flux2KleinLatentDecoder;
  batchSize: number;
  height: number;
  width: number;
  dtype?: DType;
  rngKey?: MxArray;
  vaeScaleFactor?: number;
  patchSize?: number;
};

const FLUX2_MU_A1 = 8.73809524e-5;
const FLUX2_MU_B1 = 1.89833333;
const FLUX2_MU_A2 = 0.00016927;
const FLUX2_MU_B2 = 0.45666666;

function assertPositiveInteger(name: string, value: number): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
}

function assertPositiveFinite(name: string, value: number): void {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${name} must be a positive finite number.`);
  }
}

function guidanceScale(conditioning: Flux2KleinConditioning): number {
  const scale = conditioning.guidanceScale ?? 1;
  assertPositiveFinite("conditioning.guidanceScale", scale);
  return scale;
}

function usesClassifierFreeGuidance(
  conditioning: Flux2KleinConditioning,
  isDistilled: boolean,
): boolean {
  return guidanceScale(conditioning) > 1 && !isDistilled;
}

function assertPackedFlux2Latents(
  latents: MxArray,
  packedHeight: number,
  packedWidth: number,
): readonly [number, number, number] {
  assertPositiveInteger("packedHeight", packedHeight);
  assertPositiveInteger("packedWidth", packedWidth);
  const [batchSize, sequenceLength, channels] = latents.shape;
  if (
    latents.shape.length !== 3 ||
    batchSize === undefined ||
    sequenceLength === undefined ||
    channels === undefined
  ) {
    throw new Error(
      `initialLatents must be packed FLUX.2 latents, got ${formatShape(latents.shape)}.`,
    );
  }
  assertPositiveInteger("batchSize", batchSize);
  assertPositiveInteger("sequenceLength", sequenceLength);
  assertPositiveInteger("channels", channels);
  const expectedLength = packedHeight * packedWidth;
  if (sequenceLength !== expectedLength) {
    throw new Error(
      `initialLatents sequence length must be ${expectedLength}, got ${sequenceLength}.`,
    );
  }
  return [batchSize, sequenceLength, channels];
}

function assertIdsShape(ids: MxArray, name: string, expectedLength: number): void {
  const [length, axes] = ids.shape;
  if (ids.shape.length !== 2 || length !== expectedLength || axes !== 4) {
    throw new Error(
      `${name} must have shape [${expectedLength}, 4], got ${formatShape(ids.shape)}.`,
    );
  }
}

function assertConditioning(
  conditioning: Flux2KleinConditioning,
  batchSize: number,
  isDistilled: boolean,
): { textLength: number; negativeTextLength: number | null } {
  const [promptBatch, textLength, hiddenSize] = conditioning.promptEmbeds.shape;
  if (
    conditioning.promptEmbeds.shape.length !== 3 ||
    promptBatch !== batchSize ||
    textLength === undefined ||
    hiddenSize === undefined
  ) {
    throw new Error(
      `conditioning.promptEmbeds must have batch ${batchSize}, got ${formatShape(
        conditioning.promptEmbeds.shape,
      )}.`,
    );
  }
  if (conditioning.textIds !== undefined) {
    assertIdsShape(conditioning.textIds, "conditioning.textIds", textLength);
  }
  if (!usesClassifierFreeGuidance(conditioning, isDistilled)) {
    return { textLength, negativeTextLength: null };
  }
  if (conditioning.negativePromptEmbeds === undefined) {
    throw new Error(
      "conditioning.negativePromptEmbeds is required when FLUX.2 classifier-free guidance is enabled.",
    );
  }
  const [negativeBatch, negativeTextLength, negativeHiddenSize] =
    conditioning.negativePromptEmbeds.shape;
  if (
    conditioning.negativePromptEmbeds.shape.length !== 3 ||
    negativeBatch !== batchSize ||
    negativeTextLength === undefined ||
    negativeHiddenSize !== hiddenSize
  ) {
    throw new Error(
      `conditioning.negativePromptEmbeds must have shape [${batchSize}, length, ${hiddenSize}], got ${formatShape(
        conditioning.negativePromptEmbeds.shape,
      )}.`,
    );
  }
  if (conditioning.negativeTextIds !== undefined) {
    assertIdsShape(
      conditioning.negativeTextIds,
      "conditioning.negativeTextIds",
      negativeTextLength,
    );
  }
  return { textLength, negativeTextLength };
}

function makeDenoiserInput(
  hiddenStates: MxArray,
  encoderHiddenStates: MxArray,
  timestep: MxArray,
  imageIds: MxArray,
  textIds: MxArray,
): Flux2KleinDenoiserInput {
  return {
    hiddenStates,
    encoderHiddenStates,
    timestep,
    imageIds,
    textIds,
  };
}

function predictFlux2KleinVelocity(
  options: Flux2KleinDenoiseOptions,
  latents: MxArray,
  imageIds: MxArray,
  textIds: MxArray,
  negativeTextIds: MxArray | null,
  step: FlowMatchEulerStep,
  batchSize: number,
): MxArray {
  using scaledLatents = options.scheduler.scaleModelInput(latents);
  using timestep = full(
    [batchSize],
    step.timestep / options.scheduler.maxTimestep,
    scaledLatents.dtype,
  );
  using conditionalPrediction = options.denoiser.forward(
    makeDenoiserInput(
      scaledLatents,
      options.conditioning.promptEmbeds,
      timestep,
      imageIds,
      textIds,
    ),
  );
  if (!usesClassifierFreeGuidance(options.conditioning, options.isDistilled ?? false)) {
    return retainArray(conditionalPrediction);
  }
  if (options.conditioning.negativePromptEmbeds === undefined || negativeTextIds === null) {
    throw new Error("predictFlux2KleinVelocity: missing negative FLUX.2 conditioning.");
  }
  using negativePrediction = options.denoiser.forward(
    makeDenoiserInput(
      scaledLatents,
      options.conditioning.negativePromptEmbeds,
      timestep,
      imageIds,
      negativeTextIds,
    ),
  );
  using predictionDelta = subtract(conditionalPrediction, negativePrediction);
  using scaledDelta = multiply(predictionDelta, guidanceScale(options.conditioning));
  return add(negativePrediction, scaledDelta);
}

function denoiseStep(
  options: Flux2KleinDenoiseOptions,
  latents: MxArray,
  imageIds: MxArray,
  textIds: MxArray,
  negativeTextIds: MxArray | null,
  step: FlowMatchEulerStep,
  batchSize: number,
): MxArray {
  using prediction = predictFlux2KleinVelocity(
    options,
    latents,
    imageIds,
    textIds,
    negativeTextIds,
    step,
    batchSize,
  );
  return options.scheduler.step(prediction, latents, step);
}

function createDenoiseIds(options: {
  packedHeight: number;
  packedWidth: number;
  textLength: number;
  negativeTextLength: number | null;
  conditioning: Flux2KleinConditioning;
}): {
  imageIds: MxArray;
  textIds: MxArray;
  negativeTextIds: MxArray | null;
} {
  const imageIds = createFlux2LatentIds(options.packedHeight, options.packedWidth);
  const textIds =
    options.conditioning.textIds === undefined
      ? createFlux2TextIds(options.textLength)
      : retainArray(options.conditioning.textIds);
  const negativeTextIds =
    options.negativeTextLength === null
      ? null
      : options.conditioning.negativeTextIds === undefined
        ? createFlux2TextIds(options.negativeTextLength)
        : retainArray(options.conditioning.negativeTextIds);
  return { imageIds, textIds, negativeTextIds };
}

function flux2KleinTimesteps(
  scheduler: Flux2KleinScheduler,
  numInferenceSteps: number,
  sequenceLength: number,
  sigmas?: readonly number[],
): readonly FlowMatchEulerStep[] {
  const timestepOptions = {
    mu: computeFlux2KleinEmpiricalMu(sequenceLength, numInferenceSteps),
  };
  if (sigmas === undefined) {
    return scheduler.timesteps(numInferenceSteps, timestepOptions);
  }
  return scheduler.timesteps(numInferenceSteps, { ...timestepOptions, sigmas });
}

/** FLUX.2 Klein empirical dynamic-shift value used by Diffusers. */
export function computeFlux2KleinEmpiricalMu(
  imageSequenceLength: number,
  numInferenceSteps: number,
): number {
  assertPositiveInteger("imageSequenceLength", imageSequenceLength);
  assertPositiveInteger("numInferenceSteps", numInferenceSteps);
  if (imageSequenceLength > 4300) {
    return FLUX2_MU_A2 * imageSequenceLength + FLUX2_MU_B2;
  }
  const mu200 = FLUX2_MU_A2 * imageSequenceLength + FLUX2_MU_B2;
  const mu10 = FLUX2_MU_A1 * imageSequenceLength + FLUX2_MU_B1;
  const slope = (mu200 - mu10) / 190;
  const intercept = mu200 - 200 * slope;
  return slope * numInferenceSteps + intercept;
}

/** Denoise packed FLUX.2 Klein latents with prepared prompt embedding tensors. */
export function denoiseFlux2KleinLatents(options: Flux2KleinDenoiseOptions): MxArray {
  assertPositiveInteger("numInferenceSteps", options.numInferenceSteps);
  const [batchSize, sequenceLength] = assertPackedFlux2Latents(
    options.initialLatents,
    options.packedHeight,
    options.packedWidth,
  );
  const { textLength, negativeTextLength } = assertConditioning(
    options.conditioning,
    batchSize,
    options.isDistilled ?? false,
  );

  const ids = createDenoiseIds({
    packedHeight: options.packedHeight,
    packedWidth: options.packedWidth,
    textLength,
    negativeTextLength,
    conditioning: options.conditioning,
  });

  try {
    assertIdsShape(ids.imageIds, "imageIds", sequenceLength);
    assertIdsShape(ids.textIds, "conditioning.textIds", textLength);
    if (ids.negativeTextIds !== null && negativeTextLength !== null) {
      assertIdsShape(ids.negativeTextIds, "conditioning.negativeTextIds", negativeTextLength);
    }

    let current = retainArray(options.initialLatents);
    try {
      const steps = flux2KleinTimesteps(
        options.scheduler,
        options.numInferenceSteps,
        sequenceLength,
        options.sigmas,
      );
      for (let index = 0; index < steps.length; index += 1) {
        const step = steps[index];
        if (step === undefined) {
          throw new Error("denoiseFlux2KleinLatents: missing scheduler step.");
        }
        const next = denoiseStep(
          options,
          current,
          ids.imageIds,
          ids.textIds,
          ids.negativeTextIds,
          step,
          batchSize,
        );
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
    ids.imageIds.free();
    ids.textIds.free();
    ids.negativeTextIds?.free();
  }
}

/** Generate an image from supplied FLUX.2 Klein prompt embeddings. */
export function generateFlux2KleinImage(options: Flux2KleinImageGenerationOptions): MxArray {
  const vaeScaleFactor = options.vaeScaleFactor ?? options.vae.vaeScaleFactor ?? 8;
  const patchSize = options.patchSize ?? options.vae.patchSize ?? 2;
  const initialLatentOptions: Flux2InitialLatentOptions = {
    scheduler: options.scheduler,
    batchSize: options.batchSize,
    height: options.height,
    width: options.width,
    latentChannels: options.vae.latentChannels,
    vaeScaleFactor,
    patchSize,
  };
  if (options.dtype !== undefined) {
    initialLatentOptions.dtype = options.dtype;
  }
  if (options.rngKey !== undefined) {
    initialLatentOptions.rngKey = options.rngKey;
  }
  using initialLatents = createFlux2InitialLatents(initialLatentOptions);
  const [, , packedHeight, packedWidth] = flux2LatentMapShape(
    options.batchSize,
    options.height,
    options.width,
    options.vae.latentChannels,
    vaeScaleFactor,
    patchSize,
  );

  const denoiseOptions: Flux2KleinDenoiseOptions = {
    denoiser: options.denoiser,
    scheduler: options.scheduler,
    initialLatents,
    packedHeight,
    packedWidth,
    conditioning: options.conditioning,
    numInferenceSteps: options.numInferenceSteps,
  };
  if (options.isDistilled !== undefined) {
    denoiseOptions.isDistilled = options.isDistilled;
  }
  if (options.sigmas !== undefined) {
    denoiseOptions.sigmas = options.sigmas;
  }
  if (options.evaluateEachStep !== undefined) {
    denoiseOptions.evaluateEachStep = options.evaluateEachStep;
  }
  if (options.onStep !== undefined) {
    denoiseOptions.onStep = options.onStep;
  }

  using denoised = denoiseFlux2KleinLatents(denoiseOptions);
  return decodeFlux2KleinLatents(options.vae, denoised, packedHeight, packedWidth);
}
