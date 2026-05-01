import { describe, expect, test } from "bun:test";
import { add, array, full, MxArray, mxEval, random, zeros } from "@mlxts/core";

import { FlowMatchEulerScheduler } from "../../schedulers/flow-match-euler";
import { packQwenImageLatents } from "./latents";
import {
  decodeQwenImageLatents,
  denoiseQwenImageLatents,
  generateQwenImage,
  type QwenImageDenoiser,
  type QwenImageLatentDecoder,
} from "./pipeline";
import type { QwenImageDenoiserInput } from "./transformer";

type DenoiserCall = {
  hiddenShape: readonly number[];
  encoderShape: readonly number[];
  timestepShape: readonly number[];
  timestepValues: readonly number[];
  imageShape: readonly number[];
  maskShape?: readonly number[];
};

class RecordingQwenImageDenoiser implements QwenImageDenoiser {
  readonly calls: DenoiserCall[] = [];
  readonly predictionValues: number[];

  constructor(predictionValues: readonly number[] = [0]) {
    this.predictionValues = [...predictionValues];
  }

  forward(input: QwenImageDenoiserInput): MxArray {
    const call: DenoiserCall = {
      hiddenShape: [...input.hiddenStates.shape],
      encoderShape: [...input.encoderHiddenStates.shape],
      timestepShape: [...input.timestep.shape],
      timestepValues: Array.from(input.timestep.toTypedArray()),
      imageShape: [...input.imageShape],
    };
    if (input.encoderHiddenStatesMask !== undefined) {
      call.maskShape = [...input.encoderHiddenStatesMask.shape];
    }
    this.calls.push(call);
    const value =
      this.predictionValues[Math.min(this.calls.length - 1, this.predictionValues.length - 1)] ?? 0;
    return full([...input.hiddenStates.shape], value, input.hiddenStates.dtype);
  }
}

class ThrowingQwenImageDenoiser implements QwenImageDenoiser {
  forward(): MxArray {
    throw new Error("denoiser failed");
  }
}

class RecordingQwenImageDecoder implements QwenImageLatentDecoder {
  readonly latentChannels = 2;
  readonly latentsMean: readonly number[];
  readonly latentsStd: readonly number[];
  readonly inputs: number[][] = [];
  readonly mode: "identity" | "zeros";

  constructor(
    options: {
      latentsMean?: readonly number[];
      latentsStd?: readonly number[];
      mode?: "identity" | "zeros";
    } = {},
  ) {
    this.latentsMean = options.latentsMean ?? [0, 0];
    this.latentsStd = options.latentsStd ?? [1, 1];
    this.mode = options.mode ?? "identity";
  }

  decodeRaw(latents: MxArray): MxArray {
    mxEval(latents);
    this.inputs.push(Array.from(latents.toTypedArray()));
    if (this.mode === "identity") {
      return add(latents, 0);
    }
    return zeros([...latents.shape], latents.dtype);
  }
}

function expectCloseList(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBeCloseTo(expected[index] ?? Number.NaN, 6);
  }
}

describe("Qwen-Image latent decoding", () => {
  test("denoises prepared embeddings with FlowMatch timesteps normalized for the transformer", () => {
    const scheduler = new FlowMatchEulerScheduler({ shift: 1 });
    const denoiser = new RecordingQwenImageDenoiser([1]);
    using initialLatents = zeros([1, 2, 4]);
    using promptEmbeds = zeros([1, 3, 8]);
    const steps: number[] = [];

    using latents = denoiseQwenImageLatents({
      denoiser,
      scheduler,
      initialLatents,
      imageShape: [1, 1, 2],
      conditioning: { promptEmbeds },
      numInferenceSteps: 2,
      evaluateEachStep: false,
      onStep: (event) => {
        steps.push(event.stepIndex);
      },
    });

    mxEval(latents);
    expect(latents.shape).toEqual([1, 2, 4]);
    expectCloseList(latents.toTypedArray(), [-1, -1, -1, -1, -1, -1, -1, -1]);
    expect(denoiser.calls).toHaveLength(2);
    expect(denoiser.calls[0]?.hiddenShape).toEqual([1, 2, 4]);
    expect(denoiser.calls[0]?.encoderShape).toEqual([1, 3, 8]);
    expect(denoiser.calls[0]?.timestepShape).toEqual([1]);
    expectCloseList(denoiser.calls[0]?.timestepValues ?? [], [1]);
    expectCloseList(denoiser.calls[1]?.timestepValues ?? [], [0.5]);
    expect(denoiser.calls[0]?.imageShape).toEqual([1, 1, 2]);
    expect(steps).toEqual([0, 1]);
  });

  test("runs true CFG as separate conditional and negative denoiser calls", () => {
    const scheduler = new FlowMatchEulerScheduler({ shift: 1 });
    const denoiser = new RecordingQwenImageDenoiser([2, 1]);
    using initialLatents = zeros([1, 2, 4]);
    using promptEmbeds = full([1, 2, 8], 1);
    using promptMask = array([[1, 1]], "bool");
    using negativePromptEmbeds = full([1, 3, 8], -1);
    using negativePromptMask = array([[1, 0, 1]], "bool");

    using latents = denoiseQwenImageLatents({
      denoiser,
      scheduler,
      initialLatents,
      imageShape: [1, 1, 2],
      conditioning: {
        promptEmbeds,
        promptEmbedsMask: promptMask,
        negativePromptEmbeds,
        negativePromptEmbedsMask: negativePromptMask,
        trueCfgScale: 3,
      },
      numInferenceSteps: 1,
      evaluateEachStep: false,
    });

    mxEval(latents);
    expect(denoiser.calls).toHaveLength(2);
    expect(denoiser.calls[0]?.encoderShape).toEqual([1, 2, 8]);
    expect(denoiser.calls[0]?.maskShape).toEqual([1, 2]);
    expect(denoiser.calls[1]?.encoderShape).toEqual([1, 3, 8]);
    expect(denoiser.calls[1]?.maskShape).toEqual([1, 3]);
    expectCloseList(latents.toTypedArray(), [-2, -2, -2, -2, -2, -2, -2, -2]);
  });

  test("does not run negative prompt denoising when true CFG is disabled", () => {
    const scheduler = new FlowMatchEulerScheduler({ shift: 1 });
    const denoiser = new RecordingQwenImageDenoiser([3, 1]);
    using initialLatents = zeros([1, 2, 4]);
    using promptEmbeds = zeros([1, 2, 8]);
    using negativePromptEmbeds = zeros([1, 2, 8]);

    using latents = denoiseQwenImageLatents({
      denoiser,
      scheduler,
      initialLatents,
      imageShape: [1, 1, 2],
      conditioning: { promptEmbeds, negativePromptEmbeds, trueCfgScale: 1 },
      numInferenceSteps: 1,
      evaluateEachStep: false,
    });

    mxEval(latents);
    expect(denoiser.calls).toHaveLength(1);
    expectCloseList(latents.toTypedArray(), [-3, -3, -3, -3, -3, -3, -3, -3]);
  });

  test("rejects malformed prepared denoising tensors before calling the denoiser", () => {
    const scheduler = new FlowMatchEulerScheduler();
    const denoiser = new RecordingQwenImageDenoiser();
    using rankTwoLatents = zeros([1, 4]);
    using initialLatents = zeros([1, 2, 4]);
    using promptEmbeds = zeros([1, 3, 8]);
    using wrongPromptBatch = zeros([2, 3, 8]);
    using wrongMask = zeros([1, 2], "bool");
    using negativeMask = zeros([1, 3], "bool");
    using wrongNegativeHidden = zeros([1, 3, 7]);

    expect(() =>
      denoiseQwenImageLatents({
        denoiser,
        scheduler,
        initialLatents: rankTwoLatents,
        imageShape: [1, 1, 2],
        conditioning: { promptEmbeds },
        numInferenceSteps: 1,
      }),
    ).toThrow("packed Qwen-Image");
    expect(() =>
      denoiseQwenImageLatents({
        denoiser,
        scheduler,
        initialLatents,
        imageShape: [1, 1, 3],
        conditioning: { promptEmbeds },
        numInferenceSteps: 1,
      }),
    ).toThrow("imageShape product");
    expect(() =>
      denoiseQwenImageLatents({
        denoiser,
        scheduler,
        initialLatents,
        imageShape: [1, 1, 2],
        conditioning: { promptEmbeds: wrongPromptBatch },
        numInferenceSteps: 1,
      }),
    ).toThrow("promptEmbeds");
    expect(() =>
      denoiseQwenImageLatents({
        denoiser,
        scheduler,
        initialLatents,
        imageShape: [1, 1, 2],
        conditioning: { promptEmbeds, promptEmbedsMask: wrongMask },
        numInferenceSteps: 1,
      }),
    ).toThrow("promptEmbedsMask");
    expect(() =>
      denoiseQwenImageLatents({
        denoiser,
        scheduler,
        initialLatents,
        imageShape: [1, 1, 2],
        conditioning: { promptEmbeds, negativePromptEmbedsMask: negativeMask },
        numInferenceSteps: 1,
      }),
    ).toThrow("negativePromptEmbeds");
    expect(() =>
      denoiseQwenImageLatents({
        denoiser,
        scheduler,
        initialLatents,
        imageShape: [1, 1, 2],
        conditioning: { promptEmbeds, negativePromptEmbeds: wrongNegativeHidden },
        numInferenceSteps: 1,
      }),
    ).toThrow("negativePromptEmbeds");
    expect(() =>
      denoiseQwenImageLatents({
        denoiser,
        scheduler,
        initialLatents,
        imageShape: [1, 1, 2],
        conditioning: { promptEmbeds, trueCfgScale: 0 },
        numInferenceSteps: 1,
      }),
    ).toThrow("trueCfgScale");
    expect(denoiser.calls).toHaveLength(0);
  });

  test("disposes retained denoising state when the denoiser fails", () => {
    const scheduler = new FlowMatchEulerScheduler();
    using initialLatents = zeros([1, 2, 4]);
    using promptEmbeds = zeros([1, 3, 8]);

    expect(() =>
      denoiseQwenImageLatents({
        denoiser: new ThrowingQwenImageDenoiser(),
        scheduler,
        initialLatents,
        imageShape: [1, 1, 2],
        conditioning: { promptEmbeds },
        numInferenceSteps: 1,
      }),
    ).toThrow("denoiser failed");
  });

  test("generation assembles packed denoising and Qwen VAE decode", () => {
    const scheduler = new FlowMatchEulerScheduler();
    const denoiser = new RecordingQwenImageDenoiser([0]);
    const decoder = new RecordingQwenImageDecoder();
    using promptEmbeds = zeros([1, 2, 8]);
    using rngKey = random.key(0);

    using image = generateQwenImage({
      denoiser,
      scheduler,
      vae: decoder,
      batchSize: 1,
      height: 16,
      width: 16,
      conditioning: { promptEmbeds },
      numInferenceSteps: 1,
      dtype: "float32",
      rngKey,
    });

    mxEval(image);
    expect(image.shape).toEqual([1, 2, 2, 2]);
    expect(denoiser.calls).toHaveLength(1);
    expect(denoiser.calls[0]?.hiddenShape).toEqual([1, 1, 8]);
    expect(denoiser.calls[0]?.imageShape).toEqual([1, 1, 1]);
    expect(decoder.inputs).toHaveLength(1);
  });

  test("decodes packed latents through single-frame VAE output and NHWC postprocess", () => {
    const decoder = new RecordingQwenImageDecoder();
    using latents = MxArray.fromData([-1, -0.5, 0, 0.5, 0.25, -0.25, 0.75, -0.75], [1, 2, 1, 2, 2]);
    using packed = packQwenImageLatents(latents);
    using decoded = decodeQwenImageLatents(decoder, packed, 2, 2);

    mxEval(decoded);
    expect(decoded.shape).toEqual([1, 2, 2, 2]);
    expectCloseList(decoded.toTypedArray(), [0, 0.625, 0.25, 0.375, 0.5, 0.875, 0.75, 0.125]);
  });

  test("applies Qwen-Image per-channel mean and std before raw VAE decode", () => {
    const decoder = new RecordingQwenImageDecoder({
      latentsMean: [1, 10],
      latentsStd: [2, 3],
      mode: "zeros",
    });
    using latents = MxArray.fromData([0.5, 1, 1.5, 2, 3, 4, 5, 6], [1, 2, 1, 2, 2]);
    using packed = packQwenImageLatents(latents);
    using decoded = decodeQwenImageLatents(decoder, packed, 2, 2);

    mxEval(decoded);
    expect(decoded.shape).toEqual([1, 2, 2, 2]);
    expectCloseList(decoder.inputs[0] ?? [], [2, 3, 4, 5, 19, 22, 25, 28]);
    expectCloseList(decoded.toTypedArray(), [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
  });

  test("rejects VAE latent statistics that do not match latent channels", () => {
    const decoder = new RecordingQwenImageDecoder({ latentsMean: [0], latentsStd: [1] });
    using latents = MxArray.fromData([0, 0, 0, 0, 0, 0, 0, 0], [1, 2, 1, 2, 2]);
    using packed = packQwenImageLatents(latents);

    expect(() => decodeQwenImageLatents(decoder, packed, 2, 2)).toThrow("mean/std");
  });
});
