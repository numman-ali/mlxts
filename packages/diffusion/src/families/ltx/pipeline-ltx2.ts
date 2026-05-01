import {
  add,
  concatenate,
  full,
  type MxArray,
  multiply,
  mxEval,
  repeat,
  retainArray,
  split,
  subtract,
} from "@mlxts/core";

import type { FlowMatchEulerStep } from "../../schedulers/flow-match-euler";
import { createLtx2AudioCoords, createLtx2VideoCoords } from "./embeddings";
import type {
  Ltx2Conditioning,
  Ltx2DenoiseOptions,
  Ltx2DenoiseResult,
  Ltx2DenoiserInput,
  Ltx2DenoiserOutput,
  Ltx2Scheduler,
} from "./pipeline-ltx2-types";

export type {
  Ltx2Conditioning,
  Ltx2DenoiseOptions,
  Ltx2DenoiseResult,
  Ltx2Denoiser,
  Ltx2DenoiserInput,
  Ltx2DenoiserOutput,
  Ltx2DenoisingStepEvent,
  Ltx2Scheduler,
} from "./pipeline-ltx2-types";

import {
  type ResolvedLtx2DenoiseShape,
  resolveDenoiseShape,
  resolveGuidanceScale,
  resolvePositiveFinite,
} from "./pipeline-ltx2-validation";

type OwnedLtx2Conditioning = {
  encoderHiddenStates: MxArray;
  audioEncoderHiddenStates: MxArray;
  encoderAttentionMask: MxArray;
};

function makeGuidedConditioning(conditioning: Ltx2Conditioning): OwnedLtx2Conditioning {
  if (
    conditioning.negativePromptEmbeds === undefined ||
    conditioning.negativeAudioPromptEmbeds === undefined ||
    conditioning.negativePromptAttentionMask === undefined
  ) {
    throw new Error("makeGuidedConditioning: missing negative LTX-2 conditioning.");
  }
  const encoderHiddenStates = concatenate(
    [conditioning.negativePromptEmbeds, conditioning.promptEmbeds],
    0,
  );
  try {
    const audioEncoderHiddenStates = concatenate(
      [conditioning.negativeAudioPromptEmbeds, conditioning.audioPromptEmbeds],
      0,
    );
    try {
      const encoderAttentionMask = concatenate(
        [conditioning.negativePromptAttentionMask, conditioning.promptAttentionMask],
        0,
      );
      return { encoderHiddenStates, audioEncoderHiddenStates, encoderAttentionMask };
    } catch (error) {
      audioEncoderHiddenStates.free();
      throw error;
    }
  } catch (error) {
    encoderHiddenStates.free();
    throw error;
  }
}

function disposeOwnedConditioning(conditioning: OwnedLtx2Conditioning): void {
  conditioning.encoderHiddenStates.free();
  conditioning.audioEncoderHiddenStates.free();
  conditioning.encoderAttentionMask.free();
}

function repeatedCoords(coords: MxArray, needsClassifierFreeGuidance: boolean): MxArray {
  if (!needsClassifierFreeGuidance) {
    return coords;
  }
  try {
    return repeat(coords, 2, 0);
  } finally {
    coords.free();
  }
}

function createVideoCoords(
  options: Ltx2DenoiseOptions,
  batchSize: number,
  needsCfg: boolean,
): MxArray {
  const coordinateOptions = {
    batchSize,
    latentFrames: options.latentFrames,
    latentHeight: options.latentHeight,
    latentWidth: options.latentWidth,
  };
  if (options.patchSize !== undefined) {
    Object.assign(coordinateOptions, { patchSize: options.patchSize });
  }
  if (options.patchSizeT !== undefined) {
    Object.assign(coordinateOptions, { patchSizeT: options.patchSizeT });
  }
  if (options.vaeScaleFactors !== undefined) {
    Object.assign(coordinateOptions, { vaeScaleFactors: options.vaeScaleFactors });
  }
  if (options.causalOffset !== undefined) {
    Object.assign(coordinateOptions, { causalOffset: options.causalOffset });
  }
  if (options.frameRate !== undefined) {
    Object.assign(coordinateOptions, { frameRate: options.frameRate });
  }
  const coords = createLtx2VideoCoords(coordinateOptions);
  return repeatedCoords(coords, needsCfg);
}

function createAudioCoords(
  options: Ltx2DenoiseOptions,
  batchSize: number,
  needsCfg: boolean,
): MxArray {
  const coordinateOptions = {
    batchSize,
    audioLatentFrames: options.audioLatentFrames,
  };
  if (options.audioPatchSizeT !== undefined) {
    Object.assign(coordinateOptions, { patchSizeT: options.audioPatchSizeT });
  }
  if (options.audioScaleFactor !== undefined) {
    Object.assign(coordinateOptions, { audioScaleFactor: options.audioScaleFactor });
  }
  if (options.causalOffset !== undefined) {
    Object.assign(coordinateOptions, { causalOffset: options.causalOffset });
  }
  if (options.audioHopLength !== undefined) {
    Object.assign(coordinateOptions, { hopLength: options.audioHopLength });
  }
  if (options.audioSamplingRate !== undefined) {
    Object.assign(coordinateOptions, { samplingRate: options.audioSamplingRate });
  }
  const coords = createLtx2AudioCoords(coordinateOptions);
  return repeatedCoords(coords, needsCfg);
}

function makeDenoiserInput(input: {
  videoLatents: MxArray;
  audioLatents: MxArray;
  encoderHiddenStates: MxArray;
  audioEncoderHiddenStates: MxArray;
  timestep: MxArray;
  encoderAttentionMask: MxArray;
  options: Ltx2DenoiseOptions;
  videoCoords: MxArray;
  audioCoords: MxArray;
}): Ltx2DenoiserInput {
  return {
    hiddenStates: input.videoLatents,
    audioHiddenStates: input.audioLatents,
    encoderHiddenStates: input.encoderHiddenStates,
    audioEncoderHiddenStates: input.audioEncoderHiddenStates,
    timestep: input.timestep,
    sigma: input.timestep,
    encoderAttentionMask: input.encoderAttentionMask,
    audioEncoderAttentionMask: input.encoderAttentionMask,
    numFrames: input.options.latentFrames,
    height: input.options.latentHeight,
    width: input.options.latentWidth,
    fps: resolvePositiveFinite(input.options.frameRate, 24, "frameRate"),
    audioNumFrames: input.options.audioLatentFrames,
    videoCoords: input.videoCoords,
    audioCoords: input.audioCoords,
    useCrossTimestep: input.options.useCrossTimestep ?? false,
  };
}

function predictDenoisedWithCfg(
  scheduler: Ltx2Scheduler,
  sample: MxArray,
  prediction: MxArray,
  step: FlowMatchEulerStep,
  guidanceScale: number,
  needsClassifierFreeGuidance: boolean,
): MxArray {
  if (!needsClassifierFreeGuidance) {
    return scheduler.predictDenoised(prediction, sample, step);
  }
  const parts = split(prediction, 2, 0);
  try {
    const unconditional = parts[0];
    const conditional = parts[1];
    if (parts.length !== 2 || unconditional === undefined || conditional === undefined) {
      throw new Error("LTX-2 classifier-free guidance prediction must split into two batches.");
    }
    using unconditionalDenoised = scheduler.predictDenoised(unconditional, sample, step);
    using conditionalDenoised = scheduler.predictDenoised(conditional, sample, step);
    using delta = subtract(conditionalDenoised, unconditionalDenoised);
    using scaledDelta = multiply(delta, guidanceScale - 1);
    return add(conditionalDenoised, scaledDelta);
  } finally {
    for (const part of parts) {
      part.free();
    }
  }
}

function velocityFromDenoised(
  sample: MxArray,
  denoised: MxArray,
  step: FlowMatchEulerStep,
): MxArray {
  using residual = subtract(sample, denoised);
  return multiply(residual, 1 / step.sigma);
}

function predictLtx2Velocities(
  options: Ltx2DenoiseOptions,
  videoLatents: MxArray,
  audioLatents: MxArray,
  videoStep: FlowMatchEulerStep,
  audioStep: FlowMatchEulerStep,
  videoCoords: MxArray,
  audioCoords: MxArray,
  batchSize: number,
  needsCfg: boolean,
): Ltx2DenoiserOutput {
  const videoScale = resolveGuidanceScale(options.guidanceScale, "guidanceScale");
  const audioScale = resolveGuidanceScale(
    options.audioGuidanceScale ?? videoScale,
    "audioGuidanceScale",
  );
  const guidedConditioning = needsCfg ? makeGuidedConditioning(options.conditioning) : null;
  try {
    using scaledVideoLatents = options.scheduler.scaleModelInput(videoLatents);
    using scaledAudioLatents = options.scheduler.scaleModelInput(audioLatents);
    using denoiserVideoLatents = needsCfg
      ? concatenate([scaledVideoLatents, scaledVideoLatents], 0)
      : retainArray(scaledVideoLatents);
    using denoiserAudioLatents = needsCfg
      ? concatenate([scaledAudioLatents, scaledAudioLatents], 0)
      : retainArray(scaledAudioLatents);
    using timestep = full(
      [needsCfg ? batchSize * 2 : batchSize],
      videoStep.timestep,
      denoiserVideoLatents.dtype,
    );
    const output = options.denoiser.forward(
      makeDenoiserInput({
        videoLatents: denoiserVideoLatents,
        audioLatents: denoiserAudioLatents,
        encoderHiddenStates:
          guidedConditioning?.encoderHiddenStates ?? options.conditioning.promptEmbeds,
        audioEncoderHiddenStates:
          guidedConditioning?.audioEncoderHiddenStates ?? options.conditioning.audioPromptEmbeds,
        timestep,
        encoderAttentionMask:
          guidedConditioning?.encoderAttentionMask ?? options.conditioning.promptAttentionMask,
        options,
        videoCoords,
        audioCoords,
      }),
    );
    try {
      using videoDenoised = predictDenoisedWithCfg(
        options.scheduler,
        videoLatents,
        output.video,
        videoStep,
        videoScale,
        needsCfg,
      );
      using audioDenoised = predictDenoisedWithCfg(
        options.scheduler,
        audioLatents,
        output.audio,
        audioStep,
        audioScale,
        needsCfg,
      );
      return {
        video: velocityFromDenoised(videoLatents, videoDenoised, videoStep),
        audio: velocityFromDenoised(audioLatents, audioDenoised, audioStep),
      };
    } finally {
      output.video.free();
      output.audio.free();
    }
  } finally {
    if (guidedConditioning !== null) {
      disposeOwnedConditioning(guidedConditioning);
    }
  }
}

function ltx2Timesteps(
  scheduler: Ltx2Scheduler,
  options: Ltx2DenoiseOptions,
  imageSequenceLength: number,
): readonly FlowMatchEulerStep[] {
  const timestepOptions = { imageSequenceLength };
  if (options.mu !== undefined) {
    Object.assign(timestepOptions, { mu: options.mu });
  }
  if (options.sigmas !== undefined) {
    Object.assign(timestepOptions, { sigmas: options.sigmas });
  }
  return scheduler.timesteps(options.numInferenceSteps, timestepOptions);
}

function denoiseLtx2Loop(
  options: Ltx2DenoiseOptions,
  shape: ResolvedLtx2DenoiseShape,
  videoCoords: MxArray,
  audioCoords: MxArray,
): Ltx2DenoiseResult {
  let currentVideo = retainArray(options.initialVideoLatents);
  let currentAudio = retainArray(options.initialAudioLatents);
  try {
    const steps = ltx2Timesteps(options.scheduler, options, shape.videoLength);
    for (let index = 0; index < steps.length; index += 1) {
      const step = steps[index];
      if (step === undefined) {
        throw new Error("denoiseLtx2Latents: missing scheduler step.");
      }
      const prediction = predictLtx2Velocities(
        options,
        currentVideo,
        currentAudio,
        step,
        step,
        videoCoords,
        audioCoords,
        shape.batchSize,
        shape.needsClassifierFreeGuidance,
      );
      try {
        const nextVideo = options.scheduler.step(prediction.video, currentVideo, step);
        try {
          const nextAudio = options.scheduler.step(prediction.audio, currentAudio, step);
          currentVideo.free();
          currentAudio.free();
          currentVideo = nextVideo;
          currentAudio = nextAudio;
        } catch (error) {
          nextVideo.free();
          throw error;
        }
      } finally {
        prediction.video.free();
        prediction.audio.free();
      }
      if (options.evaluateEachStep ?? true) {
        mxEval(currentVideo, currentAudio);
      }
      options.onStep?.({
        stepIndex: index,
        timestep: step.timestep,
        previousTimestep: step.previousTimestep,
        sigma: step.sigma,
        nextSigma: step.nextSigma,
        audioTimestep: step.timestep,
        audioPreviousTimestep: step.previousTimestep,
        audioSigma: step.sigma,
        audioNextSigma: step.nextSigma,
        videoLatents: currentVideo,
        audioLatents: currentAudio,
      });
    }
    return { videoLatents: currentVideo, audioLatents: currentAudio };
  } catch (error) {
    currentVideo.free();
    currentAudio.free();
    throw error;
  }
}

/** Denoise packed LTX-2 audio-video latents with prepared connector tensors. */
export function denoiseLtx2Latents(options: Ltx2DenoiseOptions): Ltx2DenoiseResult {
  const shape = resolveDenoiseShape(options);
  const videoCoords = createVideoCoords(
    options,
    shape.batchSize,
    shape.needsClassifierFreeGuidance,
  );
  try {
    const audioCoords = createAudioCoords(
      options,
      shape.batchSize,
      shape.needsClassifierFreeGuidance,
    );
    try {
      return denoiseLtx2Loop(options, shape, videoCoords, audioCoords);
    } finally {
      audioCoords.free();
    }
  } finally {
    videoCoords.free();
  }
}
