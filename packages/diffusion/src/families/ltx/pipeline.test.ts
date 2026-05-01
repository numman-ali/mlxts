import { describe, expect, test } from "bun:test";
import { concatenate, full, MxArray, mxEval, zeros } from "@mlxts/core";

import { FlowMatchEulerScheduler } from "../../schedulers/flow-match-euler";
import { ltxVideoPackedLatentShape } from "./latents";
import {
  applyLtxVideoClassifierFreeGuidance,
  denoiseLtxVideoLatents,
  type LtxVideoDenoiser,
  type LtxVideoDenoiserInput,
} from "./pipeline";

type DenoiserCall = {
  hiddenShape: readonly number[];
  encoderShape: readonly number[];
  maskShape: readonly number[] | null;
  timestepValues: readonly number[];
  encoderValues: readonly number[];
  maskValues: readonly number[] | null;
  numFrames: number;
  height: number;
  width: number;
  ropeInterpolationScale: readonly number[];
};

class RecordingLtxVideoDenoiser implements LtxVideoDenoiser {
  readonly calls: DenoiserCall[] = [];
  readonly predictionValue: number;

  constructor(predictionValue = 0) {
    this.predictionValue = predictionValue;
  }

  forward(input: LtxVideoDenoiserInput): MxArray {
    const call: DenoiserCall = {
      hiddenShape: [...input.hiddenStates.shape],
      encoderShape: [...input.encoderHiddenStates.shape],
      maskShape:
        input.encoderAttentionMask === undefined ? null : [...input.encoderAttentionMask.shape],
      timestepValues: Array.from(input.timestep.toTypedArray()),
      encoderValues: Array.from(input.encoderHiddenStates.toTypedArray()),
      maskValues:
        input.encoderAttentionMask === undefined
          ? null
          : Array.from(input.encoderAttentionMask.toTypedArray()),
      numFrames: input.numFrames,
      height: input.height,
      width: input.width,
      ropeInterpolationScale: [...input.ropeInterpolationScale],
    };
    this.calls.push(call);
    return full([...input.hiddenStates.shape], this.predictionValue, input.hiddenStates.dtype);
  }
}

class GuidedLtxVideoDenoiser implements LtxVideoDenoiser {
  readonly calls: DenoiserCall[] = [];

  forward(input: LtxVideoDenoiserInput): MxArray {
    this.calls.push({
      hiddenShape: [...input.hiddenStates.shape],
      encoderShape: [...input.encoderHiddenStates.shape],
      maskShape:
        input.encoderAttentionMask === undefined ? null : [...input.encoderAttentionMask.shape],
      timestepValues: Array.from(input.timestep.toTypedArray()),
      encoderValues: Array.from(input.encoderHiddenStates.toTypedArray()),
      maskValues:
        input.encoderAttentionMask === undefined
          ? null
          : Array.from(input.encoderAttentionMask.toTypedArray()),
      numFrames: input.numFrames,
      height: input.height,
      width: input.width,
      ropeInterpolationScale: [...input.ropeInterpolationScale],
    });
    const [, sequenceLength, channels] = input.hiddenStates.shape;
    if (sequenceLength === undefined || channels === undefined) {
      throw new Error("GuidedLtxVideoDenoiser: hiddenStates must be packed LTX latents.");
    }
    using unconditional = full([1, sequenceLength, channels], 1, input.hiddenStates.dtype);
    using conditional = full([1, sequenceLength, channels], 2, input.hiddenStates.dtype);
    return concatenate([unconditional, conditional], 0);
  }
}

class ThrowingLtxVideoDenoiser implements LtxVideoDenoiser {
  forward(): MxArray {
    throw new Error("denoiser failed");
  }
}

function expectCloseList(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBeCloseTo(expected[index] ?? Number.NaN, 5);
  }
}

describe("LTX-Video prepared-embedding sampling", () => {
  test("denoises with raw timesteps and unpatched video length dynamic shift", () => {
    const scheduler = new FlowMatchEulerScheduler({
      numTrainTimesteps: 100,
      useDynamicShifting: true,
      baseImageSeqLen: 1,
      maxImageSeqLen: 33,
      baseShift: 0.5,
      maxShift: 1.5,
      timeShiftType: "linear",
    });
    const denoiser = new RecordingLtxVideoDenoiser(0);
    const [batchSize, sequenceLength, channels] = ltxVideoPackedLatentShape(1, 2, 4, 4, 2, 2, 1);
    const expectedSteps = scheduler.timesteps(2, {
      imageSequenceLength: 2 * 4 * 4,
      sigmas: [1, 0.5],
    });
    using initialLatents = zeros([batchSize, sequenceLength, channels]);
    using promptEmbeds = zeros([1, 3, 5]);
    using promptAttentionMask = MxArray.fromData([1, 1, 0], [1, 3], "int32");
    const seenSteps: number[] = [];

    using latents = denoiseLtxVideoLatents({
      denoiser,
      scheduler,
      initialLatents,
      latentFrames: 2,
      latentHeight: 4,
      latentWidth: 4,
      patchSize: 2,
      patchSizeT: 1,
      conditioning: { promptEmbeds, promptAttentionMask },
      numInferenceSteps: 2,
      sigmas: [1, 0.5],
      evaluateEachStep: false,
      onStep: (event) => {
        seenSteps.push(event.stepIndex);
      },
    });

    mxEval(latents);
    expect(latents.shape).toEqual([1, 8, 8]);
    expect(denoiser.calls).toHaveLength(2);
    expect(denoiser.calls[0]?.hiddenShape).toEqual([1, 8, 8]);
    expect(denoiser.calls[0]?.encoderShape).toEqual([1, 3, 5]);
    expect(denoiser.calls[0]?.maskShape).toEqual([1, 3]);
    expectCloseList(denoiser.calls[0]?.timestepValues ?? [], [expectedSteps[0]?.timestep ?? -1]);
    expectCloseList(denoiser.calls[1]?.timestepValues ?? [], [expectedSteps[1]?.timestep ?? -1]);
    expect(denoiser.calls[0]?.numFrames).toBe(2);
    expect(denoiser.calls[0]?.height).toBe(4);
    expect(denoiser.calls[0]?.width).toBe(4);
    expectCloseList(denoiser.calls[0]?.ropeInterpolationScale ?? [], [8 / 24, 32, 32]);
    expect(seenSteps).toEqual([0, 1]);
  });

  test("batches classifier-free guidance negative first with attention masks", () => {
    const scheduler = new FlowMatchEulerScheduler({ shift: 1 });
    const denoiser = new GuidedLtxVideoDenoiser();
    using initialLatents = zeros([1, 2, 1]);
    using promptEmbeds = full([1, 2, 3], 10);
    using negativePromptEmbeds = full([1, 2, 3], -10);
    using promptAttentionMask = MxArray.fromData([1, 0], [1, 2], "int32");
    using negativePromptAttentionMask = MxArray.fromData([0, 1], [1, 2], "int32");

    using latents = denoiseLtxVideoLatents({
      denoiser,
      scheduler,
      initialLatents,
      latentFrames: 1,
      latentHeight: 1,
      latentWidth: 2,
      conditioning: {
        promptEmbeds,
        promptAttentionMask,
        negativePromptEmbeds,
        negativePromptAttentionMask,
      },
      guidanceScale: 3,
      numInferenceSteps: 1,
      evaluateEachStep: false,
    });

    mxEval(latents);
    expect(denoiser.calls).toHaveLength(1);
    expect(denoiser.calls[0]?.hiddenShape).toEqual([2, 2, 1]);
    expect(denoiser.calls[0]?.encoderShape).toEqual([2, 2, 3]);
    expect(denoiser.calls[0]?.maskShape).toEqual([2, 2]);
    expectCloseList(
      denoiser.calls[0]?.encoderValues ?? [],
      [-10, -10, -10, -10, -10, -10, 10, 10, 10, 10, 10, 10],
    );
    expectCloseList(denoiser.calls[0]?.maskValues ?? [], [0, 1, 1, 0]);
    expectCloseList(denoiser.calls[0]?.timestepValues ?? [], [1000, 1000]);
    expectCloseList(latents.toTypedArray(), [-4, -4]);
  });

  test("applies classifier-free guidance to paired predictions", () => {
    using prediction = MxArray.fromData([1, 2, 3, 4], [2, 2, 1]);

    using guided = applyLtxVideoClassifierFreeGuidance(prediction, 2);

    expectCloseList(guided.toTypedArray(), [5, 6]);
  });

  test("rejects malformed denoising requests before calling the denoiser", () => {
    const scheduler = new FlowMatchEulerScheduler();
    const denoiser = new RecordingLtxVideoDenoiser();
    using rankTwoLatents = zeros([1, 4]);
    using wrongLengthLatents = zeros([1, 3, 1]);
    using initialLatents = zeros([1, 2, 1]);
    using promptEmbeds = zeros([1, 2, 3]);
    using promptAttentionMask = MxArray.fromData([1, 1], [1, 2], "int32");
    using badPromptAttentionMask = MxArray.fromData([1, 1, 0], [1, 3], "int32");
    using negativePromptEmbeds = zeros([1, 2, 3]);

    expect(() =>
      denoiseLtxVideoLatents({
        denoiser,
        scheduler,
        initialLatents: rankTwoLatents,
        latentFrames: 1,
        latentHeight: 1,
        latentWidth: 2,
        conditioning: { promptEmbeds, promptAttentionMask },
        numInferenceSteps: 1,
      }),
    ).toThrow("packed LTX-Video latents");
    expect(() =>
      denoiseLtxVideoLatents({
        denoiser,
        scheduler,
        initialLatents: wrongLengthLatents,
        latentFrames: 1,
        latentHeight: 1,
        latentWidth: 2,
        conditioning: { promptEmbeds, promptAttentionMask },
        numInferenceSteps: 1,
      }),
    ).toThrow("sequence length");
    expect(() =>
      denoiseLtxVideoLatents({
        denoiser,
        scheduler,
        initialLatents,
        latentFrames: 1,
        latentHeight: 1,
        latentWidth: 2,
        conditioning: { promptEmbeds, promptAttentionMask: badPromptAttentionMask },
        numInferenceSteps: 1,
      }),
    ).toThrow("promptAttentionMask");
    expect(() =>
      denoiseLtxVideoLatents({
        denoiser,
        scheduler,
        initialLatents,
        latentFrames: 1,
        latentHeight: 1,
        latentWidth: 2,
        conditioning: { promptEmbeds, promptAttentionMask },
        guidanceScale: 2,
        numInferenceSteps: 1,
      }),
    ).toThrow("negativePromptEmbeds");
    expect(() =>
      denoiseLtxVideoLatents({
        denoiser,
        scheduler,
        initialLatents,
        latentFrames: 1,
        latentHeight: 1,
        latentWidth: 2,
        conditioning: { promptEmbeds, promptAttentionMask, negativePromptEmbeds },
        guidanceScale: 2,
        numInferenceSteps: 1,
      }),
    ).toThrow("negativePromptAttentionMask");
    expect(denoiser.calls).toHaveLength(0);
  });

  test("disposes retained denoising state when the denoiser fails", () => {
    const scheduler = new FlowMatchEulerScheduler();
    using initialLatents = zeros([1, 2, 1]);
    using promptEmbeds = zeros([1, 2, 3]);
    using promptAttentionMask = MxArray.fromData([1, 1], [1, 2], "int32");

    expect(() =>
      denoiseLtxVideoLatents({
        denoiser: new ThrowingLtxVideoDenoiser(),
        scheduler,
        initialLatents,
        latentFrames: 1,
        latentHeight: 1,
        latentWidth: 2,
        conditioning: { promptEmbeds, promptAttentionMask },
        numInferenceSteps: 1,
      }),
    ).toThrow("denoiser failed");
  });
});
