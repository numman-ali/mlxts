import { describe, expect, test } from "bun:test";
import { array, type MxArray, multiply, zeros } from "@mlxts/core";

import { DDIMScheduler } from "../../schedulers/ddim";
import { EulerScheduler } from "../../schedulers/euler";
import {
  applyStableDiffusionClassifierFreeGuidance,
  createStableDiffusionInitialLatents,
  decodeStableDiffusionLatents,
  denoiseStableDiffusionLatents,
  generateStableDiffusionImage,
  type StableDiffusionDenoiser,
  type StableDiffusionLatentDecoder,
  stableDiffusionLatentShape,
} from "./pipeline";
import type { StableDiffusionUNetForwardOptions } from "./unet";

type DenoiserCall = {
  latentShape: readonly number[];
  encoderShape: readonly number[];
  textEmbedsShape?: readonly number[];
  timeIdsShape?: readonly number[];
  timestep: number | MxArray;
};

class RecordingDenoiser implements StableDiffusionDenoiser {
  readonly calls: DenoiserCall[] = [];

  forward(
    x: MxArray,
    timestep: number | MxArray,
    encoderHiddenStates: MxArray,
    options?: StableDiffusionUNetForwardOptions,
  ): MxArray {
    const call: DenoiserCall = {
      latentShape: [...x.shape],
      encoderShape: [...encoderHiddenStates.shape],
      timestep,
    };
    if (options?.textTime !== undefined) {
      call.textEmbedsShape = [...options.textTime.textEmbeds.shape];
      call.timeIdsShape = [...options.textTime.timeIds.shape];
    }
    this.calls.push(call);
    return zeros([...x.shape], x.dtype);
  }
}

class TestLatentDecoder implements StableDiffusionLatentDecoder {
  readonly scalingFactor = 2;
  readonly latentChannels = 1;
  readonly vaeScaleFactor = 4;

  decode(latents: MxArray): MxArray {
    return multiply(latents, 2);
  }
}

function expectTensorClose(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    const actualValue = actual[index];
    const expectedValue = expected[index];
    if (actualValue === undefined || expectedValue === undefined) {
      throw new Error("expectTensorClose: missing value.");
    }
    expect(actualValue).toBeCloseTo(expectedValue, 5);
  }
}

describe("Stable Diffusion pipeline assembly", () => {
  test("computes NHWC latent shape and samples scheduler-scaled initial latents", () => {
    const scheduler = new DDIMScheduler({ numTrainTimesteps: 4 });

    expect(
      stableDiffusionLatentShape({
        batchSize: 2,
        height: 64,
        width: 32,
        latentChannels: 4,
      }),
    ).toEqual([2, 8, 4, 4]);
    expect(() =>
      stableDiffusionLatentShape({
        batchSize: 1,
        height: 65,
        width: 64,
        latentChannels: 4,
      }),
    ).toThrow("divisible");

    using latents = createStableDiffusionInitialLatents({
      scheduler,
      batchSize: 2,
      height: 64,
      width: 32,
      latentChannels: 4,
      dtype: "float16",
    });

    expect(latents.shape).toEqual([2, 8, 4, 4]);
    expect(latents.dtype).toBe("float16");
  });

  test("applies classifier-free guidance with negative then positive prediction order", () => {
    using prediction = array([1, 3], "float32");

    using guided = applyStableDiffusionClassifierFreeGuidance(prediction, 2);

    guided.eval();
    expectTensorClose(guided.toTypedArray(), [5]);
  });

  test("denoising batches latents and conditioning for classifier-free guidance", () => {
    const scheduler = new EulerScheduler({
      betaSchedule: "linear",
      betaStart: 0.1,
      betaEnd: 0.2,
      numTrainTimesteps: 2,
    });
    const unet = new RecordingDenoiser();
    using initialLatents = zeros([1, 1, 1, 1]);
    using positiveStates = array([[[3]]], "float32");
    using negativeStates = array([[[1]]], "float32");
    using positiveTextEmbeds = zeros([1, 2]);
    using negativeTextEmbeds = zeros([1, 2]);
    using positiveTimeIds = zeros([1, 2]);
    using negativeTimeIds = zeros([1, 2]);

    using latents = denoiseStableDiffusionLatents({
      unet,
      scheduler,
      initialLatents,
      conditioning: {
        encoderHiddenStates: positiveStates,
        textTime: { textEmbeds: positiveTextEmbeds, timeIds: positiveTimeIds },
      },
      negativeConditioning: {
        encoderHiddenStates: negativeStates,
        textTime: { textEmbeds: negativeTextEmbeds, timeIds: negativeTimeIds },
      },
      guidanceScale: 2,
      numInferenceSteps: 1,
    });

    expect(latents.shape).toEqual([1, 1, 1, 1]);
    expect(unet.calls).toHaveLength(1);
    expect(unet.calls[0]?.latentShape).toEqual([2, 1, 1, 1]);
    expect(unet.calls[0]?.encoderShape).toEqual([2, 1, 1]);
    expect(unet.calls[0]?.textEmbedsShape).toEqual([2, 2]);
    expect(unet.calls[0]?.timeIdsShape).toEqual([2, 2]);
  });

  test("denoising requires negative conditioning when guidance is active", () => {
    const scheduler = new DDIMScheduler({
      betaSchedule: "linear",
      betaStart: 0.1,
      betaEnd: 0.2,
      numTrainTimesteps: 2,
    });
    const unet = new RecordingDenoiser();
    using initialLatents = zeros([1, 1, 1, 1]);
    using positiveStates = array([[[3]]], "float32");

    expect(() =>
      denoiseStableDiffusionLatents({
        unet,
        scheduler,
        initialLatents,
        conditioning: { encoderHiddenStates: positiveStates },
        guidanceScale: 7.5,
        numInferenceSteps: 1,
      }),
    ).toThrow("negativeConditioning");
  });

  test("decodes latents through VAE scaling and 0..1 image normalization", () => {
    const vae = new TestLatentDecoder();
    using latents = array([[[[2, -4]]]], "float32");

    using image = decodeStableDiffusionLatents(vae, latents);

    image.eval();
    expect(image.shape).toEqual([1, 1, 1, 2]);
    expectTensorClose(image.toTypedArray(), [1, 0]);
  });

  test("generates an image through initial noise, denoising, and VAE decode", () => {
    const scheduler = new DDIMScheduler({
      betaSchedule: "linear",
      betaStart: 0.1,
      betaEnd: 0.2,
      numTrainTimesteps: 2,
    });
    const unet = new RecordingDenoiser();
    const vae = new TestLatentDecoder();
    using states = array([[[0]]], "float32");

    using image = generateStableDiffusionImage({
      unet,
      vae,
      scheduler,
      batchSize: 1,
      height: 4,
      width: 4,
      conditioning: { encoderHiddenStates: states },
      numInferenceSteps: 1,
      evaluateEachStep: false,
    });

    expect(image.shape).toEqual([1, 1, 1, 1]);
    expect(unet.calls).toHaveLength(1);
  });
});
