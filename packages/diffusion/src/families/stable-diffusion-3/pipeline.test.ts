import { describe, expect, test } from "bun:test";
import { add, array, concatenate, full, type MxArray, mxEval, zeros } from "@mlxts/core";

import { FlowMatchEulerScheduler } from "../../schedulers/flow-match-euler";
import {
  applyStableDiffusion3ClassifierFreeGuidance,
  decodeStableDiffusion3Latents,
  denoiseStableDiffusion3Latents,
  generateStableDiffusion3Image,
  type StableDiffusion3Denoiser,
  type StableDiffusion3LatentDecoder,
} from "./pipeline";
import type { StableDiffusion3DenoiserInput } from "./transformer";

type DenoiserCall = {
  hiddenShape: readonly number[];
  encoderShape: readonly number[];
  pooledShape: readonly number[];
  timestepValues: readonly number[];
};

class RecordingStableDiffusion3Denoiser implements StableDiffusion3Denoiser {
  readonly calls: DenoiserCall[] = [];
  readonly predictionValue: number;

  constructor(predictionValue: number) {
    this.predictionValue = predictionValue;
  }

  forward(input: StableDiffusion3DenoiserInput): MxArray {
    this.calls.push({
      hiddenShape: [...input.hiddenStates.shape],
      encoderShape: [...input.encoderHiddenStates.shape],
      pooledShape: [...input.pooledProjections.shape],
      timestepValues: Array.from(input.timestep.toTypedArray()),
    });
    return full([...input.hiddenStates.shape], this.predictionValue, input.hiddenStates.dtype);
  }
}

class GuidedStableDiffusion3Denoiser implements StableDiffusion3Denoiser {
  readonly calls: DenoiserCall[] = [];

  forward(input: StableDiffusion3DenoiserInput): MxArray {
    this.calls.push({
      hiddenShape: [...input.hiddenStates.shape],
      encoderShape: [...input.encoderHiddenStates.shape],
      pooledShape: [...input.pooledProjections.shape],
      timestepValues: Array.from(input.timestep.toTypedArray()),
    });
    const [, height, width, channels] = input.hiddenStates.shape;
    if (height === undefined || width === undefined || channels === undefined) {
      throw new Error("GuidedStableDiffusion3Denoiser: hiddenStates must be NHWC.");
    }
    using unconditional = full([1, height, width, channels], 1, input.hiddenStates.dtype);
    using conditional = full([1, height, width, channels], 2, input.hiddenStates.dtype);
    return concatenate([unconditional, conditional], 0);
  }
}

class RecordingStableDiffusion3Decoder implements StableDiffusion3LatentDecoder {
  readonly scalingFactor = 2;
  readonly shiftFactor = 0.5;
  readonly latentChannels = 1;
  readonly inputs: number[][] = [];

  decode(latents: MxArray): MxArray {
    mxEval(latents);
    this.inputs.push(Array.from(latents.toTypedArray()));
    return add(latents, 0);
  }
}

function expectCloseList(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBeCloseTo(expected[index] ?? Number.NaN, 6);
  }
}

describe("Stable Diffusion 3 prepared-embedding sampling", () => {
  test("denoises with raw FlowMatch timestep values", () => {
    const scheduler = new FlowMatchEulerScheduler({ shift: 1 });
    const denoiser = new RecordingStableDiffusion3Denoiser(1);
    using initialLatents = zeros([1, 2, 2, 1]);
    using encoderHiddenStates = zeros([1, 3, 6]);
    using pooledProjections = zeros([1, 5]);
    const steps: number[] = [];

    using latents = denoiseStableDiffusion3Latents({
      denoiser,
      scheduler,
      initialLatents,
      conditioning: { encoderHiddenStates, pooledProjections },
      numInferenceSteps: 2,
      evaluateEachStep: false,
      onStep: (event) => {
        steps.push(event.stepIndex);
      },
    });

    mxEval(latents);
    expect(latents.shape).toEqual([1, 2, 2, 1]);
    expectCloseList(latents.toTypedArray(), [-1, -1, -1, -1]);
    expect(denoiser.calls).toHaveLength(2);
    expect(denoiser.calls[0]?.hiddenShape).toEqual([1, 2, 2, 1]);
    expect(denoiser.calls[0]?.encoderShape).toEqual([1, 3, 6]);
    expect(denoiser.calls[0]?.pooledShape).toEqual([1, 5]);
    expectCloseList(denoiser.calls[0]?.timestepValues ?? [], [1000]);
    expectCloseList(denoiser.calls[1]?.timestepValues ?? [], [500]);
    expect(steps).toEqual([0, 1]);
  });

  test("batches classifier-free guidance over prepared text tensors", () => {
    const scheduler = new FlowMatchEulerScheduler({ shift: 1 });
    const denoiser = new GuidedStableDiffusion3Denoiser();
    using initialLatents = zeros([1, 2, 2, 1]);
    using encoderHiddenStates = zeros([1, 3, 6]);
    using pooledProjections = zeros([1, 5]);
    using negativeEncoderHiddenStates = full([1, 3, 6], -1);
    using negativePooledProjections = full([1, 5], -1);

    using latents = denoiseStableDiffusion3Latents({
      denoiser,
      scheduler,
      initialLatents,
      conditioning: { encoderHiddenStates, pooledProjections },
      negativeConditioning: {
        encoderHiddenStates: negativeEncoderHiddenStates,
        pooledProjections: negativePooledProjections,
      },
      guidanceScale: 3,
      numInferenceSteps: 1,
      evaluateEachStep: false,
    });

    mxEval(latents);
    expect(denoiser.calls).toHaveLength(1);
    expect(denoiser.calls[0]?.hiddenShape).toEqual([2, 2, 2, 1]);
    expect(denoiser.calls[0]?.encoderShape).toEqual([2, 3, 6]);
    expect(denoiser.calls[0]?.pooledShape).toEqual([2, 5]);
    expectCloseList(latents.toTypedArray(), [-4, -4, -4, -4]);
  });

  test("decodes with SD3 scaling and shift factors", () => {
    const decoder = new RecordingStableDiffusion3Decoder();
    using latents = array([[[[1], [-1]]]], "float32");

    using decoded = decodeStableDiffusion3Latents(decoder, latents);

    expect(decoder.inputs).toHaveLength(1);
    expectCloseList(decoder.inputs[0] ?? [], [1, 0]);
    expectCloseList(decoded.toTypedArray(), [1, 0.5]);
  });

  test("generates an image from prepared conditioning tensors", () => {
    const scheduler = new FlowMatchEulerScheduler({ shift: 1 });
    const denoiser = new RecordingStableDiffusion3Denoiser(0);
    const decoder = new RecordingStableDiffusion3Decoder();
    using encoderHiddenStates = zeros([1, 2, 6]);
    using pooledProjections = zeros([1, 5]);
    const steps: number[] = [];

    using image = generateStableDiffusion3Image({
      denoiser,
      scheduler,
      vae: decoder,
      batchSize: 1,
      height: 8,
      width: 8,
      conditioning: { encoderHiddenStates, pooledProjections },
      numInferenceSteps: 1,
      evaluateEachStep: false,
      onStep: (event) => {
        steps.push(event.stepIndex);
      },
    });

    mxEval(image);
    expect(image.shape).toEqual([1, 1, 1, 1]);
    expect(denoiser.calls).toHaveLength(1);
    expect(decoder.inputs).toHaveLength(1);
    expect(steps).toEqual([0]);
    expect(Array.from(image.toTypedArray()).every(Number.isFinite)).toBe(true);
  });

  test("applies classifier-free guidance to paired predictions", () => {
    using prediction = array([[[[1]]], [[[3]]]], "float32");

    using guided = applyStableDiffusion3ClassifierFreeGuidance(prediction, 2);

    expectCloseList(guided.toTypedArray(), [5]);
  });

  test("rejects malformed denoising requests before producing latents", () => {
    const scheduler = new FlowMatchEulerScheduler({ shift: 1 });
    const denoiser = new RecordingStableDiffusion3Denoiser(1);
    using initialLatents = zeros([1, 2, 2, 1]);
    using badInitialLatents = zeros([1, 2, 2]);
    using encoderHiddenStates = zeros([1, 3, 6]);
    using pooledProjections = zeros([1, 5]);
    using wrongBatchEncoderHiddenStates = zeros([2, 3, 6]);
    using wrongBatchPooledProjections = zeros([2, 5]);
    using negativeEncoderHiddenStates = zeros([1, 4, 6]);
    using negativePooledProjections = zeros([1, 5]);

    expect(() =>
      denoiseStableDiffusion3Latents({
        denoiser,
        scheduler,
        initialLatents: badInitialLatents,
        conditioning: { encoderHiddenStates, pooledProjections },
        numInferenceSteps: 1,
      }),
    ).toThrow("initialLatents");
    expect(() =>
      denoiseStableDiffusion3Latents({
        denoiser,
        scheduler,
        initialLatents,
        conditioning: { encoderHiddenStates: wrongBatchEncoderHiddenStates, pooledProjections },
        numInferenceSteps: 1,
      }),
    ).toThrow("encoderHiddenStates");
    expect(() =>
      denoiseStableDiffusion3Latents({
        denoiser,
        scheduler,
        initialLatents,
        conditioning: { encoderHiddenStates, pooledProjections: wrongBatchPooledProjections },
        numInferenceSteps: 1,
      }),
    ).toThrow("pooledProjections");
    expect(() =>
      denoiseStableDiffusion3Latents({
        denoiser,
        scheduler,
        initialLatents,
        conditioning: { encoderHiddenStates, pooledProjections },
        guidanceScale: -1,
        numInferenceSteps: 1,
      }),
    ).toThrow("guidanceScale");
    expect(() =>
      denoiseStableDiffusion3Latents({
        denoiser,
        scheduler,
        initialLatents,
        conditioning: { encoderHiddenStates, pooledProjections },
        guidanceScale: 2,
        numInferenceSteps: 1,
      }),
    ).toThrow("negativeConditioning");
    expect(() =>
      denoiseStableDiffusion3Latents({
        denoiser,
        scheduler,
        initialLatents,
        conditioning: { encoderHiddenStates, pooledProjections },
        negativeConditioning: {
          encoderHiddenStates: negativeEncoderHiddenStates,
          pooledProjections: negativePooledProjections,
        },
        guidanceScale: 2,
        numInferenceSteps: 1,
      }),
    ).toThrow("shape must match conditioning");
  });
});
