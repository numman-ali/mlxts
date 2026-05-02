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
  slice,
  sqrt,
  squeeze,
  subtract,
  sum,
  transpose,
} from "@mlxts/core";

import type {
  FlowMatchEulerScheduler,
  FlowMatchEulerStep,
} from "../../schedulers/flow-match-euler";
import { qwenImageLatentStatsTensor, validateQwenImageLatentStats } from "./latent-stats";
import {
  createQwenImageInitialLatents,
  type QwenImageInitialLatentOptions,
  type QwenImageRopeImageShape,
  qwenImageLatentShape,
  qwenImageRopeImageShape,
  unpackQwenImageLatents,
} from "./latents";
import type { QwenImageDenoiserInput } from "./transformer";

/** Scheduler implementation supported by the Qwen-Image sampling loop. */
export type QwenImageScheduler = FlowMatchEulerScheduler;

/** Conditional tensors consumed by the Qwen-Image transformer denoiser. */
export type QwenImageConditioning = {
  promptEmbeds: MxArray;
  promptEmbedsMask?: MxArray;
  negativePromptEmbeds?: MxArray;
  negativePromptEmbedsMask?: MxArray;
  trueCfgScale?: number;
};

/** Conditional denoiser shape required by Qwen-Image sampling. */
export type QwenImageDenoiser = {
  forward(input: QwenImageDenoiserInput): MxArray;
};

/** VAE decoder surface required by Qwen-Image latent image decoding. */
export type QwenImageLatentDecoder = {
  readonly latentChannels: number;
  readonly latentsMean: readonly number[];
  readonly latentsStd: readonly number[];
  readonly spatialCompressionRatio?: number;
  decodeRaw(latents: MxArray): MxArray;
};

/** Options for denoising existing packed Qwen-Image latents. */
export type QwenImageDenoiseOptions = {
  denoiser: QwenImageDenoiser;
  scheduler: QwenImageScheduler;
  initialLatents: MxArray;
  imageShape: QwenImageRopeImageShape;
  conditioning: QwenImageConditioning;
  numInferenceSteps: number;
  evaluateEachStep?: boolean;
  onStep?: (event: QwenImageDenoisingStepEvent) => void;
};

/** Step event emitted after a denoising update has produced the next packed latent. */
export type QwenImageDenoisingStepEvent = {
  stepIndex: number;
  timestep: number;
  previousTimestep: number;
  sigma: number;
  nextSigma: number;
  latents: MxArray;
};

/** Options for complete Qwen-Image generation from supplied prompt embeddings. */
export type QwenImageGenerationOptions = Omit<
  QwenImageDenoiseOptions,
  "initialLatents" | "imageShape"
> & {
  vae: QwenImageLatentDecoder;
  batchSize: number;
  height: number;
  width: number;
  dtype?: DType;
  rngKey?: MxArray;
  vaeScaleFactor?: number;
  patchSize?: number;
};

function assertPositiveInteger(name: string, value: number): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer.`);
  }
}

function assertPackedQwenImageLatents(
  latents: MxArray,
  imageShape: QwenImageRopeImageShape,
): readonly [number, number, number] {
  for (const [index, value] of imageShape.entries()) {
    assertPositiveInteger(`imageShape[${index}]`, value);
  }
  const [batchSize, sequenceLength, packedChannels] = latents.shape;
  if (
    latents.shape.length !== 3 ||
    batchSize === undefined ||
    sequenceLength === undefined ||
    packedChannels === undefined
  ) {
    throw new Error(
      `initialLatents must be packed Qwen-Image latents, got ${formatShape(latents.shape)}.`,
    );
  }
  assertPositiveInteger("batchSize", batchSize);
  assertPositiveInteger("sequenceLength", sequenceLength);
  assertPositiveInteger("packedChannels", packedChannels);
  const expectedLength = imageShape[0] * imageShape[1] * imageShape[2];
  if (sequenceLength !== expectedLength) {
    throw new Error(
      `initialLatents sequence length must match imageShape product ${expectedLength}, got ${sequenceLength}.`,
    );
  }
  return [batchSize, sequenceLength, packedChannels];
}

function assertConditioning(
  conditioning: QwenImageConditioning,
  batchSize: number,
): readonly [number, number] {
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
  assertPromptMask(
    conditioning.promptEmbedsMask,
    "conditioning.promptEmbedsMask",
    batchSize,
    textLength,
  );
  if (conditioning.negativePromptEmbeds !== undefined) {
    const negativeTextLength = assertMatchingPromptEmbeds(
      conditioning.negativePromptEmbeds,
      "conditioning.negativePromptEmbeds",
      batchSize,
      hiddenSize,
    );
    assertPromptMask(
      conditioning.negativePromptEmbedsMask,
      "conditioning.negativePromptEmbedsMask",
      batchSize,
      negativeTextLength,
    );
  } else if (conditioning.negativePromptEmbedsMask !== undefined) {
    throw new Error(
      "conditioning.negativePromptEmbedsMask requires conditioning.negativePromptEmbeds.",
    );
  }
  const trueCfgScale = conditioning.trueCfgScale ?? 1;
  if (!Number.isFinite(trueCfgScale) || trueCfgScale <= 0) {
    throw new Error("conditioning.trueCfgScale must be a positive finite number.");
  }
  return [textLength, hiddenSize];
}

function assertMatchingPromptEmbeds(
  embeds: MxArray,
  name: string,
  batchSize: number,
  hiddenSize: number,
): number {
  const [embedBatch, textLength, embedHiddenSize] = embeds.shape;
  if (
    embeds.shape.length !== 3 ||
    embedBatch !== batchSize ||
    textLength === undefined ||
    embedHiddenSize !== hiddenSize
  ) {
    throw new Error(
      `${name} must have shape [${batchSize}, length, ${hiddenSize}], got ${formatShape(
        embeds.shape,
      )}.`,
    );
  }
  return textLength;
}

function assertPromptMask(
  mask: MxArray | undefined,
  name: string,
  batchSize: number,
  textLength: number,
): void {
  if (mask === undefined) {
    return;
  }
  if (mask.shape.length !== 2 || mask.shape[0] !== batchSize || mask.shape[1] !== textLength) {
    throw new Error(
      `${name} must have shape [${batchSize}, ${textLength}], got ${formatShape(mask.shape)}.`,
    );
  }
}

function makeDenoiserInput(
  hiddenStates: MxArray,
  encoderHiddenStates: MxArray,
  encoderHiddenStatesMask: MxArray | undefined,
  timestep: MxArray,
  imageShape: QwenImageRopeImageShape,
): QwenImageDenoiserInput {
  const input: QwenImageDenoiserInput = {
    hiddenStates,
    encoderHiddenStates,
    timestep,
    imageShape,
  };
  if (encoderHiddenStatesMask !== undefined) {
    input.encoderHiddenStatesMask = encoderHiddenStatesMask;
  }
  return input;
}

function l2NormAlongLastAxis(x: MxArray): MxArray {
  using squared = multiply(x, x);
  using reduced = sum(squared, -1, true);
  return sqrt(reduced);
}

function rescaleGuidanceToConditionalNorm(
  prediction: MxArray,
  conditionalPrediction: MxArray,
): MxArray {
  using conditionalNorm = l2NormAlongLastAxis(conditionalPrediction);
  using predictionNorm = l2NormAlongLastAxis(prediction);
  using scale = divide(conditionalNorm, predictionNorm);
  return multiply(prediction, scale);
}

function predictQwenImageVelocity(
  options: QwenImageDenoiseOptions,
  latents: MxArray,
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
      options.conditioning.promptEmbedsMask,
      timestep,
      options.imageShape,
    ),
  );

  const trueCfgScale = options.conditioning.trueCfgScale ?? 1;
  if (trueCfgScale <= 1 || options.conditioning.negativePromptEmbeds === undefined) {
    return retainArray(conditionalPrediction);
  }

  using negativePrediction = options.denoiser.forward(
    makeDenoiserInput(
      scaledLatents,
      options.conditioning.negativePromptEmbeds,
      options.conditioning.negativePromptEmbedsMask,
      timestep,
      options.imageShape,
    ),
  );
  using predictionDelta = subtract(conditionalPrediction, negativePrediction);
  using scaledDelta = multiply(predictionDelta, trueCfgScale);
  using guided = add(negativePrediction, scaledDelta);
  return rescaleGuidanceToConditionalNorm(guided, conditionalPrediction);
}

function denoiseStep(
  options: QwenImageDenoiseOptions,
  latents: MxArray,
  step: FlowMatchEulerStep,
  batchSize: number,
): MxArray {
  using prediction = predictQwenImageVelocity(options, latents, step, batchSize);
  return options.scheduler.step(prediction, latents, step);
}

function expectSingleFrameSample(sample: MxArray): readonly [number, number, number, number] {
  const [batch, channels, frames, height, width] = sample.shape;
  if (
    sample.shape.length !== 5 ||
    batch === undefined ||
    channels === undefined ||
    frames === undefined ||
    height === undefined ||
    width === undefined
  ) {
    throw new Error(
      `decodeQwenImageLatents: expected decoded NCFHW sample, got ${formatShape(sample.shape)}.`,
    );
  }
  if (frames !== 1) {
    throw new Error(`decodeQwenImageLatents: expected one decoded frame, got ${frames}.`);
  }
  return [batch, channels, height, width];
}

/** Denoise packed Qwen-Image latents with prepared prompt embedding tensors. */
export function denoiseQwenImageLatents(options: QwenImageDenoiseOptions): MxArray {
  assertPositiveInteger("numInferenceSteps", options.numInferenceSteps);
  const [batchSize, sequenceLength] = assertPackedQwenImageLatents(
    options.initialLatents,
    options.imageShape,
  );
  assertConditioning(options.conditioning, batchSize);

  let current = retainArray(options.initialLatents);
  try {
    const steps = options.scheduler.timesteps(options.numInferenceSteps, {
      imageSequenceLength: sequenceLength,
    });
    for (let index = 0; index < steps.length; index += 1) {
      const step = steps[index];
      if (step === undefined) {
        throw new Error("denoiseQwenImageLatents: missing scheduler step.");
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

/** Decode packed Qwen-Image latents into an NHWC image tensor in the `0..1` range. */
export function decodeQwenImageLatents(
  vae: QwenImageLatentDecoder,
  packedLatents: MxArray,
  latentHeight: number,
  latentWidth: number,
  patchSize = 2,
): MxArray {
  validateQwenImageLatentStats("decodeQwenImageLatents", vae);

  using unpacked = unpackQwenImageLatents(packedLatents, latentHeight, latentWidth, patchSize);
  using std = qwenImageLatentStatsTensor(vae.latentsStd, vae.latentChannels, unpacked.dtype);
  using mean = qwenImageLatentStatsTensor(vae.latentsMean, vae.latentChannels, unpacked.dtype);
  using scaled = multiply(unpacked, std);
  using shifted = add(scaled, mean);
  using decoded = vae.decodeRaw(shifted);
  const [batch, channels, height, width] = expectSingleFrameSample(decoded);
  using firstFrame = slice(decoded, [0, 0, 0, 0, 0], [batch, channels, 1, height, width]);
  using imageNchw = squeeze(firstFrame, 2);
  using imageNhwc = transpose(imageNchw, [0, 2, 3, 1]);
  using positive = add(imageNhwc, 1);
  using normalized = divide(positive, 2);
  using lowerClamped = maximum(normalized, 0);
  return minimum(lowerClamped, 1);
}

/** Generate an image from supplied Qwen text conditioning tensors. */
export function generateQwenImage(options: QwenImageGenerationOptions): MxArray {
  const vaeScaleFactor = options.vaeScaleFactor ?? options.vae.spatialCompressionRatio ?? 8;
  const initialLatentOptions: QwenImageInitialLatentOptions = {
    scheduler: options.scheduler,
    batchSize: options.batchSize,
    height: options.height,
    width: options.width,
    latentChannels: options.vae.latentChannels,
    vaeScaleFactor,
  };
  if (options.patchSize !== undefined) {
    initialLatentOptions.patchSize = options.patchSize;
  }
  if (options.dtype !== undefined) {
    initialLatentOptions.dtype = options.dtype;
  }
  if (options.rngKey !== undefined) {
    initialLatentOptions.rngKey = options.rngKey;
  }
  using initialLatents = createQwenImageInitialLatents(initialLatentOptions);
  const [, , , latentHeight, latentWidth] = qwenImageLatentShape(initialLatentOptions);

  const denoiseOptions: QwenImageDenoiseOptions = {
    denoiser: options.denoiser,
    scheduler: options.scheduler,
    initialLatents,
    imageShape: qwenImageRopeImageShape(initialLatentOptions),
    conditioning: options.conditioning,
    numInferenceSteps: options.numInferenceSteps,
  };
  if (options.evaluateEachStep !== undefined) {
    denoiseOptions.evaluateEachStep = options.evaluateEachStep;
  }
  if (options.onStep !== undefined) {
    denoiseOptions.onStep = options.onStep;
  }

  using denoised = denoiseQwenImageLatents(denoiseOptions);
  return decodeQwenImageLatents(
    options.vae,
    denoised,
    latentHeight,
    latentWidth,
    options.patchSize,
  );
}
