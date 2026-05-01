import { describe, expect, test } from "bun:test";
import { add, full, MxArray, mxEval, random, zeros } from "@mlxts/core";

import { FlowMatchEulerScheduler } from "../../schedulers/flow-match-euler";
import { decodeFlux2KleinLatents, type Flux2KleinLatentDecoder } from "./decoding";
import {
  computeFlux2KleinEmpiricalMu,
  denoiseFlux2KleinLatents,
  type Flux2KleinDenoiser,
  type Flux2KleinDenoiserInput,
  generateFlux2KleinImage,
} from "./pipeline";

type DenoiserCall = {
  hiddenShape: readonly number[];
  encoderShape: readonly number[];
  timestepShape: readonly number[];
  timestepValues: readonly number[];
  imageIdsShape: readonly number[];
  textIdsShape: readonly number[];
};

class RecordingFlux2KleinDenoiser implements Flux2KleinDenoiser {
  readonly calls: DenoiserCall[] = [];
  readonly predictionValues: number[];

  constructor(predictionValues: readonly number[] = [0]) {
    this.predictionValues = [...predictionValues];
  }

  forward(input: Flux2KleinDenoiserInput): MxArray {
    this.calls.push({
      hiddenShape: [...input.hiddenStates.shape],
      encoderShape: [...input.encoderHiddenStates.shape],
      timestepShape: [...input.timestep.shape],
      timestepValues: Array.from(input.timestep.toTypedArray()),
      imageIdsShape: [...input.imageIds.shape],
      textIdsShape: [...input.textIds.shape],
    });
    const value =
      this.predictionValues[Math.min(this.calls.length - 1, this.predictionValues.length - 1)] ?? 0;
    return full([...input.hiddenStates.shape], value, input.hiddenStates.dtype);
  }
}

class ThrowingFlux2KleinDenoiser implements Flux2KleinDenoiser {
  forward(): MxArray {
    throw new Error("denoiser failed");
  }
}

class RecordingFlux2KleinDecoder implements Flux2KleinLatentDecoder {
  readonly latentChannels = 1;
  readonly batchNormMean: readonly number[];
  readonly batchNormVar: readonly number[];
  readonly batchNormEps: number;
  readonly inputs: number[][] = [];
  readonly mode: "identity" | "zeros";

  constructor(
    options: {
      batchNormMean?: readonly number[];
      batchNormVar?: readonly number[];
      batchNormEps?: number;
      mode?: "identity" | "zeros";
    } = {},
  ) {
    this.batchNormMean = options.batchNormMean ?? [0, 0, 0, 0];
    this.batchNormVar = options.batchNormVar ?? [1, 1, 1, 1];
    this.batchNormEps = options.batchNormEps ?? 1e-12;
    this.mode = options.mode ?? "identity";
  }

  decode(latents: MxArray): MxArray {
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

describe("FLUX.2 Klein empirical mu", () => {
  test("matches the Diffusers thresholded empirical formula", () => {
    expect(computeFlux2KleinEmpiricalMu(1024, 10)).toBeCloseTo(
      8.73809524e-5 * 1024 + 1.89833333,
      6,
    );
    expect(computeFlux2KleinEmpiricalMu(1024, 200)).toBeCloseTo(0.00016927 * 1024 + 0.45666666, 6);
    expect(computeFlux2KleinEmpiricalMu(4301, 4)).toBeCloseTo(0.00016927 * 4301 + 0.45666666, 6);
    expect(() => computeFlux2KleinEmpiricalMu(0, 4)).toThrow("imageSequenceLength");
  });
});

describe("FLUX.2 Klein prepared-embedding sampling", () => {
  test("denoises prepared embeddings with normalized timesteps and 4-axis ids", () => {
    const scheduler = new FlowMatchEulerScheduler({ shift: 1 });
    const denoiser = new RecordingFlux2KleinDenoiser([1]);
    using initialLatents = zeros([1, 2, 4]);
    using promptEmbeds = zeros([1, 3, 8]);
    const steps: number[] = [];

    using latents = denoiseFlux2KleinLatents({
      denoiser,
      scheduler,
      initialLatents,
      packedHeight: 1,
      packedWidth: 2,
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
    expect(denoiser.calls[0]?.imageIdsShape).toEqual([2, 4]);
    expect(denoiser.calls[0]?.textIdsShape).toEqual([3, 4]);
    expect(steps).toEqual([0, 1]);
  });

  test("runs plain external CFG for non-distilled FLUX.2 Klein", () => {
    const scheduler = new FlowMatchEulerScheduler({ shift: 1 });
    const denoiser = new RecordingFlux2KleinDenoiser([2, 1]);
    using initialLatents = zeros([1, 2, 4]);
    using promptEmbeds = full([1, 2, 8], 1);
    using negativePromptEmbeds = full([1, 3, 8], -1);

    using latents = denoiseFlux2KleinLatents({
      denoiser,
      scheduler,
      initialLatents,
      packedHeight: 1,
      packedWidth: 2,
      conditioning: { promptEmbeds, negativePromptEmbeds, guidanceScale: 3 },
      numInferenceSteps: 1,
      evaluateEachStep: false,
    });

    mxEval(latents);
    expect(denoiser.calls).toHaveLength(2);
    expect(denoiser.calls[0]?.encoderShape).toEqual([1, 2, 8]);
    expect(denoiser.calls[0]?.textIdsShape).toEqual([2, 4]);
    expect(denoiser.calls[1]?.encoderShape).toEqual([1, 3, 8]);
    expect(denoiser.calls[1]?.textIdsShape).toEqual([3, 4]);
    expectCloseList(latents.toTypedArray(), [-4, -4, -4, -4, -4, -4, -4, -4]);
  });

  test("ignores external CFG for distilled FLUX.2 Klein checkpoints", () => {
    const scheduler = new FlowMatchEulerScheduler({ shift: 1 });
    const denoiser = new RecordingFlux2KleinDenoiser([2, 1]);
    using initialLatents = zeros([1, 2, 4]);
    using promptEmbeds = zeros([1, 2, 8]);
    using negativePromptEmbeds = zeros([1, 2, 8]);

    using latents = denoiseFlux2KleinLatents({
      denoiser,
      scheduler,
      initialLatents,
      packedHeight: 1,
      packedWidth: 2,
      conditioning: { promptEmbeds, negativePromptEmbeds, guidanceScale: 3 },
      numInferenceSteps: 1,
      isDistilled: true,
      evaluateEachStep: false,
    });

    mxEval(latents);
    expect(denoiser.calls).toHaveLength(1);
    expectCloseList(latents.toTypedArray(), [-2, -2, -2, -2, -2, -2, -2, -2]);
  });

  test("rejects malformed prepared denoising tensors before calling the denoiser", () => {
    const scheduler = new FlowMatchEulerScheduler();
    const denoiser = new RecordingFlux2KleinDenoiser();
    using rankTwoLatents = zeros([1, 4]);
    using initialLatents = zeros([1, 2, 4]);
    using promptEmbeds = zeros([1, 3, 8]);
    using wrongPromptBatch = zeros([2, 3, 8]);
    using badTextIds = zeros([2, 4], "int32");
    using wrongNegativeHidden = zeros([1, 3, 7]);

    expect(() =>
      denoiseFlux2KleinLatents({
        denoiser,
        scheduler,
        initialLatents: rankTwoLatents,
        packedHeight: 1,
        packedWidth: 2,
        conditioning: { promptEmbeds },
        numInferenceSteps: 1,
      }),
    ).toThrow("packed FLUX.2");
    expect(() =>
      denoiseFlux2KleinLatents({
        denoiser,
        scheduler,
        initialLatents,
        packedHeight: 1,
        packedWidth: 3,
        conditioning: { promptEmbeds },
        numInferenceSteps: 1,
      }),
    ).toThrow("sequence length");
    expect(() =>
      denoiseFlux2KleinLatents({
        denoiser,
        scheduler,
        initialLatents,
        packedHeight: 1,
        packedWidth: 2,
        conditioning: { promptEmbeds: wrongPromptBatch },
        numInferenceSteps: 1,
      }),
    ).toThrow("promptEmbeds");
    expect(() =>
      denoiseFlux2KleinLatents({
        denoiser,
        scheduler,
        initialLatents,
        packedHeight: 1,
        packedWidth: 2,
        conditioning: { promptEmbeds, textIds: badTextIds },
        numInferenceSteps: 1,
      }),
    ).toThrow("textIds");
    expect(() =>
      denoiseFlux2KleinLatents({
        denoiser,
        scheduler,
        initialLatents,
        packedHeight: 1,
        packedWidth: 2,
        conditioning: { promptEmbeds, guidanceScale: 2 },
        numInferenceSteps: 1,
      }),
    ).toThrow("negativePromptEmbeds");
    expect(() =>
      denoiseFlux2KleinLatents({
        denoiser,
        scheduler,
        initialLatents,
        packedHeight: 1,
        packedWidth: 2,
        conditioning: {
          promptEmbeds,
          negativePromptEmbeds: wrongNegativeHidden,
          guidanceScale: 2,
        },
        numInferenceSteps: 1,
      }),
    ).toThrow("negativePromptEmbeds");
    expect(denoiser.calls).toHaveLength(0);
  });

  test("disposes retained denoising state when the denoiser fails", () => {
    const scheduler = new FlowMatchEulerScheduler();
    using initialLatents = zeros([1, 2, 4]);
    using promptEmbeds = zeros([1, 3, 8]);

    expect(() =>
      denoiseFlux2KleinLatents({
        denoiser: new ThrowingFlux2KleinDenoiser(),
        scheduler,
        initialLatents,
        packedHeight: 1,
        packedWidth: 2,
        conditioning: { promptEmbeds },
        numInferenceSteps: 1,
      }),
    ).toThrow("denoiser failed");
  });

  test("applies FLUX.2 VAE batch-norm inverse before decode", () => {
    const decoder = new RecordingFlux2KleinDecoder({
      batchNormMean: [1, 10, 100, 1000],
      batchNormVar: [4, 9, 16, 25],
      batchNormEps: 1e-12,
      mode: "zeros",
    });
    using packedLatents = MxArray.fromData([0.5, 1, 1.5, 2], [1, 1, 4]);
    using decoded = decodeFlux2KleinLatents(decoder, packedLatents, 1, 1);

    mxEval(decoded);
    expect(decoded.shape).toEqual([1, 2, 2, 1]);
    expectCloseList(decoder.inputs[0] ?? [], [2, 13, 106, 1010]);
    expectCloseList(decoded.toTypedArray(), [0.5, 0.5, 0.5, 0.5]);
  });

  test("generation assembles initial latents, denoising, and decode", () => {
    const scheduler = new FlowMatchEulerScheduler();
    const denoiser = new RecordingFlux2KleinDenoiser([0]);
    const decoder = new RecordingFlux2KleinDecoder();
    using promptEmbeds = zeros([1, 2, 8]);
    using rngKey = random.key(0);

    using image = generateFlux2KleinImage({
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
    expect(image.shape).toEqual([1, 2, 2, 1]);
    expect(denoiser.calls).toHaveLength(1);
    expect(denoiser.calls[0]?.hiddenShape).toEqual([1, 1, 4]);
    expect(denoiser.calls[0]?.imageIdsShape).toEqual([1, 4]);
    expect(decoder.inputs).toHaveLength(1);
  });

  test("rejects VAE batch-norm statistics that do not match packed latent channels", () => {
    const decoder = new RecordingFlux2KleinDecoder({ batchNormMean: [0], batchNormVar: [1] });
    using packedLatents = MxArray.fromData([0, 0, 0, 0], [1, 1, 4]);

    expect(() => decodeFlux2KleinLatents(decoder, packedLatents, 1, 1)).toThrow("batch-norm");
  });
});
