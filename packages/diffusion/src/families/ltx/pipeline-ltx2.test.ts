import { describe, expect, test } from "bun:test";
import { concatenate, full, MxArray, mxEval, zeros } from "@mlxts/core";

import { FlowMatchEulerScheduler } from "../../schedulers/flow-match-euler";
import { denoiseLtx2Latents, type Ltx2Denoiser, type Ltx2DenoiserInput } from "./pipeline-ltx2";

type DenoiserCall = {
  videoShape: readonly number[];
  audioShape: readonly number[];
  encoderShape: readonly number[];
  audioEncoderShape: readonly number[];
  maskShape: readonly number[];
  timestepValues: readonly number[];
  sigmaValues: readonly number[];
  encoderValues: readonly number[];
  audioEncoderValues: readonly number[];
  maskValues: readonly number[];
  videoCoordsShape: readonly number[];
  audioCoordsShape: readonly number[];
  videoCoordsValues: readonly number[];
  audioCoordsValues: readonly number[];
  numFrames: number;
  audioNumFrames: number;
  fps: number;
  useCrossTimestep: boolean;
};

class RecordingLtx2Denoiser implements Ltx2Denoiser {
  readonly calls: DenoiserCall[] = [];
  readonly videoPredictionValue: number;
  readonly audioPredictionValue: number;

  constructor(videoPredictionValue = 0, audioPredictionValue = 0) {
    this.videoPredictionValue = videoPredictionValue;
    this.audioPredictionValue = audioPredictionValue;
  }

  forward(input: Ltx2DenoiserInput) {
    this.calls.push(recordCall(input));
    return {
      video: full(
        [...input.hiddenStates.shape],
        this.videoPredictionValue,
        input.hiddenStates.dtype,
      ),
      audio: full(
        [...input.audioHiddenStates.shape],
        this.audioPredictionValue,
        input.audioHiddenStates.dtype,
      ),
    };
  }
}

class GuidedLtx2Denoiser implements Ltx2Denoiser {
  readonly calls: DenoiserCall[] = [];

  forward(input: Ltx2DenoiserInput) {
    this.calls.push(recordCall(input));
    const [, videoSequenceLength, videoChannels] = input.hiddenStates.shape;
    const [, audioSequenceLength, audioChannels] = input.audioHiddenStates.shape;
    if (
      videoSequenceLength === undefined ||
      videoChannels === undefined ||
      audioSequenceLength === undefined ||
      audioChannels === undefined
    ) {
      throw new Error("GuidedLtx2Denoiser requires packed video and audio latents.");
    }
    using videoUnconditional = full(
      [1, videoSequenceLength, videoChannels],
      1,
      input.hiddenStates.dtype,
    );
    using videoConditional = full(
      [1, videoSequenceLength, videoChannels],
      2,
      input.hiddenStates.dtype,
    );
    using audioUnconditional = full(
      [1, audioSequenceLength, audioChannels],
      2,
      input.audioHiddenStates.dtype,
    );
    using audioConditional = full(
      [1, audioSequenceLength, audioChannels],
      4,
      input.audioHiddenStates.dtype,
    );
    return {
      video: concatenate([videoUnconditional, videoConditional], 0),
      audio: concatenate([audioUnconditional, audioConditional], 0),
    };
  }
}

class ThrowingLtx2Denoiser implements Ltx2Denoiser {
  forward(): never {
    throw new Error("ltx2 denoiser failed");
  }
}

function recordCall(input: Ltx2DenoiserInput): DenoiserCall {
  return {
    videoShape: [...input.hiddenStates.shape],
    audioShape: [...input.audioHiddenStates.shape],
    encoderShape: [...input.encoderHiddenStates.shape],
    audioEncoderShape: [...input.audioEncoderHiddenStates.shape],
    maskShape: [...input.encoderAttentionMask.shape],
    timestepValues: Array.from(input.timestep.toTypedArray()),
    sigmaValues: Array.from(input.sigma.toTypedArray()),
    encoderValues: Array.from(input.encoderHiddenStates.toTypedArray()),
    audioEncoderValues: Array.from(input.audioEncoderHiddenStates.toTypedArray()),
    maskValues: Array.from(input.encoderAttentionMask.toTypedArray()),
    videoCoordsShape: [...input.videoCoords.shape],
    audioCoordsShape: [...input.audioCoords.shape],
    videoCoordsValues: Array.from(input.videoCoords.toTypedArray()),
    audioCoordsValues: Array.from(input.audioCoords.toTypedArray()),
    numFrames: input.numFrames,
    audioNumFrames: input.audioNumFrames,
    fps: input.fps,
    useCrossTimestep: input.useCrossTimestep,
  };
}

function expectCloseList(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBeCloseTo(expected[index] ?? Number.NaN, 5);
  }
}

describe("LTX-2 prepared audio-video sampling", () => {
  test("denoises video and audio latents with prepared connector tensors", () => {
    const scheduler = new FlowMatchEulerScheduler({ numTrainTimesteps: 100, shift: 1 });
    const denoiser = new RecordingLtx2Denoiser();
    const expectedSteps = scheduler.timesteps(2, { imageSequenceLength: 8, sigmas: [1, 0.5] });
    const seenSteps: number[] = [];
    const seenVideoTimesteps: number[] = [];
    const seenAudioTimesteps: number[] = [];
    using initialVideoLatents = zeros([1, 8, 8]);
    using initialAudioLatents = zeros([1, 4, 2]);
    using promptEmbeds = zeros([1, 3, 5]);
    using audioPromptEmbeds = zeros([1, 3, 7]);
    using promptAttentionMask = MxArray.fromData([1, 1, 0], [1, 3], "int32");

    const result = denoiseLtx2Latents({
      denoiser,
      scheduler,
      initialVideoLatents,
      initialAudioLatents,
      latentFrames: 2,
      latentHeight: 4,
      latentWidth: 4,
      audioLatentFrames: 4,
      audioLatentMelBins: 1,
      conditioning: { promptEmbeds, audioPromptEmbeds, promptAttentionMask },
      numInferenceSteps: 2,
      patchSize: 2,
      patchSizeT: 1,
      vaeScaleFactors: [8, 32, 32],
      causalOffset: 1,
      frameRate: 12,
      audioPatchSizeT: 1,
      audioScaleFactor: 4,
      audioHopLength: 160,
      audioSamplingRate: 16000,
      sigmas: [1, 0.5],
      mu: 0,
      evaluateEachStep: false,
      onStep: (event) => {
        seenSteps.push(event.stepIndex);
        seenVideoTimesteps.push(event.timestep);
        seenAudioTimesteps.push(event.audioTimestep);
      },
    });
    try {
      mxEval(result.videoLatents, result.audioLatents);
      expect(result.videoLatents.shape).toEqual([1, 8, 8]);
      expect(result.audioLatents.shape).toEqual([1, 4, 2]);
      expect(denoiser.calls).toHaveLength(2);
      expect(denoiser.calls[0]?.videoShape).toEqual([1, 8, 8]);
      expect(denoiser.calls[0]?.audioShape).toEqual([1, 4, 2]);
      expect(denoiser.calls[0]?.encoderShape).toEqual([1, 3, 5]);
      expect(denoiser.calls[0]?.audioEncoderShape).toEqual([1, 3, 7]);
      expect(denoiser.calls[0]?.maskShape).toEqual([1, 3]);
      expect(denoiser.calls[0]?.videoCoordsShape).toEqual([1, 3, 8, 2]);
      expect(denoiser.calls[0]?.audioCoordsShape).toEqual([1, 1, 4, 2]);
      expectCloseList(denoiser.calls[0]?.timestepValues ?? [], [expectedSteps[0]?.timestep ?? -1]);
      expectCloseList(denoiser.calls[1]?.timestepValues ?? [], [expectedSteps[1]?.timestep ?? -1]);
      expectCloseList(denoiser.calls[0]?.sigmaValues ?? [], [expectedSteps[0]?.timestep ?? -1]);
      expect(denoiser.calls[0]?.numFrames).toBe(2);
      expect(denoiser.calls[0]?.audioNumFrames).toBe(4);
      expect(denoiser.calls[0]?.fps).toBe(12);
      expect(denoiser.calls[0]?.useCrossTimestep).toBe(false);
      expect(seenSteps).toEqual([0, 1]);
      expectCloseList(seenVideoTimesteps, [
        expectedSteps[0]?.timestep ?? -1,
        expectedSteps[1]?.timestep ?? -1,
      ]);
      expectCloseList(seenAudioTimesteps, [
        expectedSteps[0]?.timestep ?? -1,
        expectedSteps[1]?.timestep ?? -1,
      ]);
    } finally {
      result.videoLatents.free();
      result.audioLatents.free();
    }
  });

  test("applies per-modality classifier-free guidance in denoised space", () => {
    const scheduler = new FlowMatchEulerScheduler({ shift: 1 });
    const denoiser = new GuidedLtx2Denoiser();
    using initialVideoLatents = zeros([1, 2, 1]);
    using initialAudioLatents = zeros([1, 3, 1]);
    using promptEmbeds = full([1, 2, 3], 10);
    using audioPromptEmbeds = full([1, 2, 4], 20);
    using negativePromptEmbeds = full([1, 2, 3], -10);
    using negativeAudioPromptEmbeds = full([1, 2, 4], -20);
    using promptAttentionMask = MxArray.fromData([1, 0], [1, 2], "int32");
    using negativePromptAttentionMask = MxArray.fromData([0, 1], [1, 2], "int32");

    const result = denoiseLtx2Latents({
      denoiser,
      scheduler,
      initialVideoLatents,
      initialAudioLatents,
      latentFrames: 1,
      latentHeight: 1,
      latentWidth: 2,
      audioLatentFrames: 3,
      audioLatentMelBins: 1,
      conditioning: {
        promptEmbeds,
        audioPromptEmbeds,
        promptAttentionMask,
        negativePromptEmbeds,
        negativeAudioPromptEmbeds,
        negativePromptAttentionMask,
      },
      guidanceScale: 3,
      audioGuidanceScale: 5,
      numInferenceSteps: 1,
      evaluateEachStep: false,
    });
    try {
      mxEval(result.videoLatents, result.audioLatents);
      expect(denoiser.calls).toHaveLength(1);
      expect(denoiser.calls[0]?.videoShape).toEqual([2, 2, 1]);
      expect(denoiser.calls[0]?.audioShape).toEqual([2, 3, 1]);
      expect(denoiser.calls[0]?.encoderShape).toEqual([2, 2, 3]);
      expect(denoiser.calls[0]?.audioEncoderShape).toEqual([2, 2, 4]);
      expect(denoiser.calls[0]?.maskShape).toEqual([2, 2]);
      expect(denoiser.calls[0]?.videoCoordsShape).toEqual([2, 3, 2, 2]);
      expect(denoiser.calls[0]?.audioCoordsShape).toEqual([2, 1, 3, 2]);
      expectCloseList(
        denoiser.calls[0]?.encoderValues ?? [],
        [-10, -10, -10, -10, -10, -10, 10, 10, 10, 10, 10, 10],
      );
      expectCloseList(
        denoiser.calls[0]?.audioEncoderValues ?? [],
        [-20, -20, -20, -20, -20, -20, -20, -20, 20, 20, 20, 20, 20, 20, 20, 20],
      );
      expectCloseList(denoiser.calls[0]?.maskValues ?? [], [0, 1, 1, 0]);
      expectCloseList(denoiser.calls[0]?.timestepValues ?? [], [1000, 1000]);
      expectCloseList(result.videoLatents.toTypedArray(), [-4, -4]);
      expectCloseList(result.audioLatents.toTypedArray(), [-12, -12, -12]);
    } finally {
      result.videoLatents.free();
      result.audioLatents.free();
    }
  });

  test("rejects malformed requests before calling the denoiser", () => {
    const scheduler = new FlowMatchEulerScheduler();
    const denoiser = new RecordingLtx2Denoiser();
    using initialVideoLatents = zeros([1, 2, 1]);
    using wrongVideoLength = zeros([1, 3, 1]);
    using initialAudioLatents = zeros([1, 3, 1]);
    using promptEmbeds = zeros([1, 2, 3]);
    using audioPromptEmbeds = zeros([1, 2, 4]);
    using promptAttentionMask = MxArray.fromData([1, 1], [1, 2], "int32");

    expect(() =>
      denoiseLtx2Latents({
        denoiser,
        scheduler,
        initialVideoLatents: wrongVideoLength,
        initialAudioLatents,
        latentFrames: 1,
        latentHeight: 1,
        latentWidth: 2,
        audioLatentFrames: 3,
        audioLatentMelBins: 1,
        conditioning: { promptEmbeds, audioPromptEmbeds, promptAttentionMask },
        numInferenceSteps: 1,
      }),
    ).toThrow("initialVideoLatents sequence length");
    expect(() =>
      denoiseLtx2Latents({
        denoiser,
        scheduler,
        initialVideoLatents,
        initialAudioLatents,
        latentFrames: 1,
        latentHeight: 1,
        latentWidth: 2,
        audioLatentFrames: 3,
        audioLatentMelBins: 1,
        conditioning: { promptEmbeds, audioPromptEmbeds, promptAttentionMask },
        guidanceScale: 2,
        numInferenceSteps: 1,
      }),
    ).toThrow("negativePromptEmbeds");
    expect(() =>
      denoiseLtx2Latents({
        denoiser,
        scheduler,
        initialVideoLatents,
        initialAudioLatents,
        latentFrames: 1,
        latentHeight: 1,
        latentWidth: 2,
        audioLatentFrames: 3,
        audioLatentMelBins: 1,
        conditioning: { promptEmbeds, audioPromptEmbeds, promptAttentionMask },
        stgScale: 1,
        numInferenceSteps: 1,
      }),
    ).toThrow("STG");
    expect(() =>
      denoiseLtx2Latents({
        denoiser,
        scheduler,
        initialVideoLatents,
        initialAudioLatents,
        latentFrames: 1,
        latentHeight: 1,
        latentWidth: 2,
        audioLatentFrames: 3,
        audioLatentMelBins: 1,
        conditioning: { promptEmbeds, audioPromptEmbeds, promptAttentionMask },
        modalityScale: 2,
        numInferenceSteps: 1,
      }),
    ).toThrow("modality isolation");
    expect(() =>
      denoiseLtx2Latents({
        denoiser,
        scheduler,
        initialVideoLatents,
        initialAudioLatents,
        latentFrames: 1,
        latentHeight: 1,
        latentWidth: 2,
        audioLatentFrames: 3,
        audioLatentMelBins: 1,
        conditioning: { promptEmbeds, audioPromptEmbeds, promptAttentionMask },
        guidanceRescale: 0.7,
        numInferenceSteps: 1,
      }),
    ).toThrow("guidance rescale");
    expect(denoiser.calls).toHaveLength(0);
  });

  test("disposes retained denoising state when the denoiser fails", () => {
    const scheduler = new FlowMatchEulerScheduler();
    using initialVideoLatents = zeros([1, 2, 1]);
    using initialAudioLatents = zeros([1, 3, 1]);
    using promptEmbeds = zeros([1, 2, 3]);
    using audioPromptEmbeds = zeros([1, 2, 4]);
    using promptAttentionMask = MxArray.fromData([1, 1], [1, 2], "int32");

    expect(() =>
      denoiseLtx2Latents({
        denoiser: new ThrowingLtx2Denoiser(),
        scheduler,
        initialVideoLatents,
        initialAudioLatents,
        latentFrames: 1,
        latentHeight: 1,
        latentWidth: 2,
        audioLatentFrames: 3,
        audioLatentMelBins: 1,
        conditioning: { promptEmbeds, audioPromptEmbeds, promptAttentionMask },
        numInferenceSteps: 1,
      }),
    ).toThrow("ltx2 denoiser failed");
  });
});
