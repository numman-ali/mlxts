import {
  add,
  concatenate,
  formatShape,
  full,
  type MxArray,
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
import type { LtxVideoRopeInterpolationScale } from "./embeddings";
import { ltxVideoPackedLatentShape } from "./latents";

/** Scheduler implementation supported by the LTX-Video sampling loop. */
export type LtxVideoScheduler = FlowMatchEulerScheduler;

/** Prepared text conditioning tensors consumed by the LTX-Video transformer. */
export type LtxVideoConditioning = {
  promptEmbeds: MxArray;
  promptAttentionMask: MxArray;
  negativePromptEmbeds?: MxArray;
  negativePromptAttentionMask?: MxArray;
};

/** Denoiser input names match Diffusers LTX-Video transformer semantics. */
export type LtxVideoDenoiserInput = {
  hiddenStates: MxArray;
  encoderHiddenStates: MxArray;
  timestep: MxArray;
  encoderAttentionMask?: MxArray;
  numFrames: number;
  height: number;
  width: number;
  ropeInterpolationScale: LtxVideoRopeInterpolationScale;
};

/** Conditional denoiser shape required by LTX-Video sampling. */
export type LtxVideoDenoiser = {
  forward(input: LtxVideoDenoiserInput): MxArray;
};

/** Options for denoising existing packed LTX-Video latents. */
export type LtxVideoDenoiseOptions = {
  denoiser: LtxVideoDenoiser;
  scheduler: LtxVideoScheduler;
  initialLatents: MxArray;
  latentFrames: number;
  latentHeight: number;
  latentWidth: number;
  conditioning: LtxVideoConditioning;
  numInferenceSteps: number;
  patchSize?: number;
  patchSizeT?: number;
  guidanceScale?: number;
  sigmas?: readonly number[];
  vaeSpatialCompressionRatio?: number;
  vaeTemporalCompressionRatio?: number;
  frameRate?: number;
  evaluateEachStep?: boolean;
  onStep?: (event: LtxVideoDenoisingStepEvent) => void;
};

/** Step event emitted after a denoising update has produced the next packed latent. */
export type LtxVideoDenoisingStepEvent = {
  stepIndex: number;
  timestep: number;
  previousTimestep: number;
  sigma: number;
  nextSigma: number;
  latents: MxArray;
};

type OwnedLtxVideoConditioning = {
  encoderHiddenStates: MxArray;
  encoderAttentionMask?: MxArray;
};

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

function assertNonNegativeFinite(name: string, value: number): void {
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`${name} must be a finite non-negative number.`);
  }
}

function resolvePositiveInteger(value: number | undefined, fallback: number, name: string): number {
  const resolved = value ?? fallback;
  assertPositiveInteger(name, resolved);
  return resolved;
}

function resolvePositiveFinite(value: number | undefined, fallback: number, name: string): number {
  const resolved = value ?? fallback;
  assertPositiveFinite(name, resolved);
  return resolved;
}

function resolveGuidanceScale(value: number | undefined): number {
  const scale = value ?? 1;
  assertNonNegativeFinite("guidanceScale", scale);
  return scale;
}

function usesClassifierFreeGuidance(options: LtxVideoDenoiseOptions): boolean {
  return resolveGuidanceScale(options.guidanceScale) > 1;
}

function assertPackedLtxVideoLatents(
  latents: MxArray,
  latentFrames: number,
  latentHeight: number,
  latentWidth: number,
  patchSize: number,
  patchSizeT: number,
): readonly [number, number, number] {
  const [batchSize, sequenceLength, channels] = latents.shape;
  if (
    latents.shape.length !== 3 ||
    batchSize === undefined ||
    sequenceLength === undefined ||
    channels === undefined
  ) {
    throw new Error(
      `initialLatents must be packed LTX-Video latents, got ${formatShape(latents.shape)}.`,
    );
  }
  assertPositiveInteger("batchSize", batchSize);
  assertPositiveInteger("sequenceLength", sequenceLength);
  assertPositiveInteger("channels", channels);
  const expectedLength = ltxVideoPackedLatentShape(
    batchSize,
    latentFrames,
    latentHeight,
    latentWidth,
    1,
    patchSize,
    patchSizeT,
  )[1];
  if (sequenceLength !== expectedLength) {
    throw new Error(
      `initialLatents sequence length must be ${expectedLength}, got ${sequenceLength}.`,
    );
  }
  return [batchSize, sequenceLength, channels];
}

function assertPromptMask(
  mask: MxArray,
  name: string,
  batchSize: number,
  textLength: number,
): void {
  const [maskBatch, maskLength] = mask.shape;
  if (mask.shape.length !== 2 || maskBatch !== batchSize || maskLength !== textLength) {
    throw new Error(
      `${name} must have shape [${batchSize}, ${textLength}], got ${formatShape(mask.shape)}.`,
    );
  }
}

function assertMatchingPromptShape(positive: MxArray, negative: MxArray, owner: string): void {
  if (positive.shape.length !== negative.shape.length) {
    throw new Error(`${owner} rank must match conditioning.promptEmbeds.`);
  }
  for (let index = 1; index < positive.shape.length; index += 1) {
    if (positive.shape[index] !== negative.shape[index]) {
      throw new Error(`${owner} shape must match conditioning.promptEmbeds.`);
    }
  }
}

function assertConditioning(
  conditioning: LtxVideoConditioning,
  batchSize: number,
  needsClassifierFreeGuidance: boolean,
): void {
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
  if (conditioning.promptAttentionMask === undefined) {
    throw new Error("conditioning.promptAttentionMask is required for LTX-Video denoising.");
  }
  assertPromptMask(
    conditioning.promptAttentionMask,
    "conditioning.promptAttentionMask",
    batchSize,
    textLength,
  );
  if (!needsClassifierFreeGuidance) {
    return;
  }
  if (conditioning.negativePromptEmbeds === undefined) {
    throw new Error(
      "conditioning.negativePromptEmbeds is required when LTX-Video classifier-free guidance is enabled.",
    );
  }
  const [negativeBatch] = conditioning.negativePromptEmbeds.shape;
  if (negativeBatch !== batchSize) {
    throw new Error(
      `conditioning.negativePromptEmbeds must have batch ${batchSize}, got ${formatShape(
        conditioning.negativePromptEmbeds.shape,
      )}.`,
    );
  }
  assertMatchingPromptShape(
    conditioning.promptEmbeds,
    conditioning.negativePromptEmbeds,
    "conditioning.negativePromptEmbeds",
  );
  if (conditioning.negativePromptAttentionMask === undefined) {
    throw new Error(
      "conditioning.negativePromptAttentionMask is required when guided attention masks are used.",
    );
  }
  assertPromptMask(
    conditioning.negativePromptAttentionMask,
    "conditioning.negativePromptAttentionMask",
    batchSize,
    textLength,
  );
}

function disposeOwnedConditioning(conditioning: OwnedLtxVideoConditioning): void {
  conditioning.encoderHiddenStates.free();
  conditioning.encoderAttentionMask?.free();
}

function makeGuidedConditioning(conditioning: LtxVideoConditioning): OwnedLtxVideoConditioning {
  if (conditioning.negativePromptEmbeds === undefined) {
    throw new Error("makeGuidedConditioning: missing negative prompt embeddings.");
  }
  const encoderHiddenStates = concatenate(
    [conditioning.negativePromptEmbeds, conditioning.promptEmbeds],
    0,
  );
  if (conditioning.promptAttentionMask === undefined) {
    return { encoderHiddenStates };
  }
  if (conditioning.negativePromptAttentionMask === undefined) {
    encoderHiddenStates.free();
    throw new Error("makeGuidedConditioning: missing negative prompt attention mask.");
  }
  try {
    return {
      encoderHiddenStates,
      encoderAttentionMask: concatenate(
        [conditioning.negativePromptAttentionMask, conditioning.promptAttentionMask],
        0,
      ),
    };
  } catch (error) {
    encoderHiddenStates.free();
    throw error;
  }
}

function ltxVideoRopeInterpolationScale(
  options: Pick<
    LtxVideoDenoiseOptions,
    "vaeTemporalCompressionRatio" | "vaeSpatialCompressionRatio" | "frameRate"
  >,
): LtxVideoRopeInterpolationScale {
  const temporalRatio = resolvePositiveInteger(
    options.vaeTemporalCompressionRatio,
    8,
    "vaeTemporalCompressionRatio",
  );
  const spatialRatio = resolvePositiveInteger(
    options.vaeSpatialCompressionRatio,
    32,
    "vaeSpatialCompressionRatio",
  );
  const frameRate = resolvePositiveFinite(options.frameRate, 24, "frameRate");
  return [temporalRatio / frameRate, spatialRatio, spatialRatio];
}

function makeDenoiserInput(
  hiddenStates: MxArray,
  encoderHiddenStates: MxArray,
  timestep: MxArray,
  options: LtxVideoDenoiseOptions,
  ropeInterpolationScale: LtxVideoRopeInterpolationScale,
  encoderAttentionMask?: MxArray,
): LtxVideoDenoiserInput {
  const input: LtxVideoDenoiserInput = {
    hiddenStates,
    encoderHiddenStates,
    timestep,
    numFrames: options.latentFrames,
    height: options.latentHeight,
    width: options.latentWidth,
    ropeInterpolationScale,
  };
  if (encoderAttentionMask !== undefined) {
    input.encoderAttentionMask = encoderAttentionMask;
  }
  return input;
}

function predictLtxVideoVelocity(
  options: LtxVideoDenoiseOptions,
  latents: MxArray,
  step: FlowMatchEulerStep,
  batchSize: number,
  ropeInterpolationScale: LtxVideoRopeInterpolationScale,
): MxArray {
  using scaledLatents = options.scheduler.scaleModelInput(latents);
  const guidanceScale = resolveGuidanceScale(options.guidanceScale);
  if (guidanceScale <= 1) {
    using timestep = full([batchSize], step.timestep, scaledLatents.dtype);
    return options.denoiser.forward(
      makeDenoiserInput(
        scaledLatents,
        options.conditioning.promptEmbeds,
        timestep,
        options,
        ropeInterpolationScale,
        options.conditioning.promptAttentionMask,
      ),
    );
  }
  const guidedConditioning = makeGuidedConditioning(options.conditioning);
  try {
    using guidedLatents = concatenate([scaledLatents, scaledLatents], 0);
    using timestep = full([batchSize * 2], step.timestep, scaledLatents.dtype);
    using prediction = options.denoiser.forward(
      makeDenoiserInput(
        guidedLatents,
        guidedConditioning.encoderHiddenStates,
        timestep,
        options,
        ropeInterpolationScale,
        guidedConditioning.encoderAttentionMask,
      ),
    );
    return applyLtxVideoClassifierFreeGuidance(prediction, guidanceScale);
  } finally {
    disposeOwnedConditioning(guidedConditioning);
  }
}

function denoiseStep(
  options: LtxVideoDenoiseOptions,
  latents: MxArray,
  step: FlowMatchEulerStep,
  batchSize: number,
  ropeInterpolationScale: LtxVideoRopeInterpolationScale,
): MxArray {
  using prediction = predictLtxVideoVelocity(
    options,
    latents,
    step,
    batchSize,
    ropeInterpolationScale,
  );
  return options.scheduler.step(prediction, latents, step);
}

function ltxVideoTimesteps(options: LtxVideoDenoiseOptions): readonly FlowMatchEulerStep[] {
  const timestepOptions = {
    imageSequenceLength: options.latentFrames * options.latentHeight * options.latentWidth,
  };
  if (options.sigmas === undefined) {
    return options.scheduler.timesteps(options.numInferenceSteps, timestepOptions);
  }
  return options.scheduler.timesteps(options.numInferenceSteps, {
    ...timestepOptions,
    sigmas: options.sigmas,
  });
}

/** Apply LTX-Video classifier-free guidance to a paired prediction batch. */
export function applyLtxVideoClassifierFreeGuidance(
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

/** Denoise packed LTX-Video latents with prepared prompt embedding tensors. */
export function denoiseLtxVideoLatents(options: LtxVideoDenoiseOptions): MxArray {
  assertPositiveInteger("numInferenceSteps", options.numInferenceSteps);
  assertPositiveInteger("latentFrames", options.latentFrames);
  assertPositiveInteger("latentHeight", options.latentHeight);
  assertPositiveInteger("latentWidth", options.latentWidth);
  const patchSize = resolvePositiveInteger(options.patchSize, 1, "patchSize");
  const patchSizeT = resolvePositiveInteger(options.patchSizeT, 1, "patchSizeT");
  const [batchSize] = assertPackedLtxVideoLatents(
    options.initialLatents,
    options.latentFrames,
    options.latentHeight,
    options.latentWidth,
    patchSize,
    patchSizeT,
  );
  assertConditioning(options.conditioning, batchSize, usesClassifierFreeGuidance(options));
  const ropeInterpolationScale = ltxVideoRopeInterpolationScale(options);

  let current = retainArray(options.initialLatents);
  try {
    const steps = ltxVideoTimesteps(options);
    for (let index = 0; index < steps.length; index += 1) {
      const step = steps[index];
      if (step === undefined) {
        throw new Error("denoiseLtxVideoLatents: missing scheduler step.");
      }
      const next = denoiseStep(options, current, step, batchSize, ropeInterpolationScale);
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
