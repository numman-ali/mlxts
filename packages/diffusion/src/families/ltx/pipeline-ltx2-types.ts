import type { MxArray } from "@mlxts/core";

import type { FlowMatchEulerScheduler } from "../../schedulers/flow-match-euler";

/** Scheduler implementation supported by the prepared LTX-2 audio-video sampling loop. */
export type Ltx2Scheduler = FlowMatchEulerScheduler;

/** Prepared connector tensors consumed by the LTX-2 audio-video transformer. */
export type Ltx2Conditioning = {
  promptEmbeds: MxArray;
  audioPromptEmbeds: MxArray;
  promptAttentionMask: MxArray;
  negativePromptEmbeds?: MxArray;
  negativeAudioPromptEmbeds?: MxArray;
  negativePromptAttentionMask?: MxArray;
};

/** Denoiser input names match Diffusers LTX-2 transformer semantics. */
export type Ltx2DenoiserInput = {
  hiddenStates: MxArray;
  audioHiddenStates: MxArray;
  encoderHiddenStates: MxArray;
  audioEncoderHiddenStates: MxArray;
  timestep: MxArray;
  sigma: MxArray;
  encoderAttentionMask: MxArray;
  audioEncoderAttentionMask: MxArray;
  numFrames: number;
  height: number;
  width: number;
  fps: number;
  audioNumFrames: number;
  videoCoords: MxArray;
  audioCoords: MxArray;
  useCrossTimestep: boolean;
};

/** Paired video/audio denoiser output required by LTX-2 sampling. */
export type Ltx2DenoiserOutput = {
  video: MxArray;
  audio: MxArray;
};

/** Conditional denoiser shape required by LTX-2 audio-video sampling. */
export type Ltx2Denoiser = {
  forward(input: Ltx2DenoiserInput): Ltx2DenoiserOutput;
};

/** Options for denoising existing packed LTX-2 video and audio latents. */
export type Ltx2DenoiseOptions = {
  denoiser: Ltx2Denoiser;
  scheduler: Ltx2Scheduler;
  initialVideoLatents: MxArray;
  initialAudioLatents: MxArray;
  latentFrames: number;
  latentHeight: number;
  latentWidth: number;
  audioLatentFrames: number;
  audioLatentMelBins: number;
  conditioning: Ltx2Conditioning;
  numInferenceSteps: number;
  patchSize?: number;
  patchSizeT?: number;
  audioPatchSize?: number;
  audioPatchSizeT?: number;
  frameRate?: number;
  vaeScaleFactors?: readonly [temporal: number, height: number, width: number];
  audioScaleFactor?: number;
  audioHopLength?: number;
  audioSamplingRate?: number;
  causalOffset?: number;
  guidanceScale?: number;
  audioGuidanceScale?: number;
  guidanceRescale?: number;
  audioGuidanceRescale?: number;
  stgScale?: number;
  audioStgScale?: number;
  modalityScale?: number;
  audioModalityScale?: number;
  spatioTemporalGuidanceBlocks?: readonly number[];
  sigmas?: readonly number[];
  mu?: number;
  useCrossTimestep?: boolean;
  evaluateEachStep?: boolean;
  onStep?: (event: Ltx2DenoisingStepEvent) => void;
};

/** Step event emitted after both LTX-2 modalities produce their next packed latents. */
export type Ltx2DenoisingStepEvent = {
  stepIndex: number;
  timestep: number;
  previousTimestep: number;
  sigma: number;
  nextSigma: number;
  audioTimestep: number;
  audioPreviousTimestep: number;
  audioSigma: number;
  audioNextSigma: number;
  videoLatents: MxArray;
  audioLatents: MxArray;
};

/** Denoised packed LTX-2 audio-video latents. */
export type Ltx2DenoiseResult = {
  videoLatents: MxArray;
  audioLatents: MxArray;
};
