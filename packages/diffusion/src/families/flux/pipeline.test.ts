import { describe, expect, test } from "bun:test";
import { array, type MxArray, random, zeros } from "@mlxts/core";

import { FlowMatchEulerScheduler } from "../../schedulers/flow-match-euler";
import {
  createFluxInitialLatents,
  decodeFluxLatents,
  denoiseFluxLatents,
  type FluxDenoiser,
  type FluxDenoiserInput,
  type FluxLatentDecoder,
  fluxLatentShape,
  generateFluxImage,
} from "./pipeline";

type DenoiserCall = {
  hiddenShape: readonly number[];
  imageIdsShape: readonly number[];
  encoderShape: readonly number[];
  textIdsShape: readonly number[];
  pooledShape: readonly number[];
  timestepShape: readonly number[];
  timestepValues: readonly number[];
  guidanceShape?: readonly number[];
};

class RecordingFluxDenoiser implements FluxDenoiser {
  readonly calls: DenoiserCall[] = [];

  forward(input: FluxDenoiserInput): MxArray {
    const call: DenoiserCall = {
      hiddenShape: [...input.hiddenStates.shape],
      imageIdsShape: [...input.imageIds.shape],
      encoderShape: [...input.encoderHiddenStates.shape],
      textIdsShape: [...input.textIds.shape],
      pooledShape: [...input.pooledProjections.shape],
      timestepShape: [...input.timestep.shape],
      timestepValues: Array.from(input.timestep.toTypedArray()),
    };
    if (input.guidance !== undefined) {
      call.guidanceShape = [...input.guidance.shape];
    }
    this.calls.push(call);
    return zeros([...input.hiddenStates.shape], input.hiddenStates.dtype);
  }
}

class RecordingFluxDecoder implements FluxLatentDecoder {
  readonly scalingFactor = 2;
  readonly shiftFactor = 0.25;
  readonly latentChannels = 1;
  readonly vaeScaleFactor = 2;
  readonly inputs: number[][] = [];

  decode(latents: MxArray): MxArray {
    this.inputs.push(Array.from(latents.toTypedArray()));
    return array(
      [
        [
          [[-2], [0]],
          [[2], [4]],
        ],
      ],
      "float32",
    );
  }
}

class ThrowingFluxDenoiser implements FluxDenoiser {
  forward(): MxArray {
    throw new Error("denoiser failed");
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

describe("Flux tensor pipeline contract", () => {
  test("computes latent shape and samples packed scheduler latents", () => {
    const scheduler = new FlowMatchEulerScheduler();

    expect(
      fluxLatentShape({
        batchSize: 2,
        height: 16,
        width: 8,
        latentChannels: 1,
        vaeScaleFactor: 2,
      }),
    ).toEqual([2, 8, 4, 1]);
    expect(() =>
      fluxLatentShape({
        batchSize: 1,
        height: 10,
        width: 8,
        latentChannels: 1,
        vaeScaleFactor: 4,
      }),
    ).toThrow("divisible");

    using packed = createFluxInitialLatents({
      scheduler,
      batchSize: 2,
      height: 16,
      width: 8,
      latentChannels: 1,
      vaeScaleFactor: 2,
      dtype: "float16",
    });

    expect(packed.shape).toEqual([2, 8, 4]);
    expect(packed.dtype).toBe("float16");
  });

  test("denoising uses packed-image sequence length and prepared conditioning tensors", () => {
    const scheduler = new FlowMatchEulerScheduler({
      numTrainTimesteps: 100,
      useDynamicShifting: true,
      baseImageSeqLen: 1,
      maxImageSeqLen: 9,
      baseShift: 0.5,
      maxShift: 1.5,
    });
    const expectedSteps = scheduler.timesteps(2, { imageSequenceLength: 4 });
    const denoiser = new RecordingFluxDenoiser();
    using initialLatents = zeros([1, 4, 4]);
    using encoderHiddenStates = zeros([1, 3, 5]);
    using pooledProjections = zeros([1, 7]);
    using guidance = array([3.5], "float32");

    using latents = denoiseFluxLatents({
      denoiser,
      scheduler,
      initialLatents,
      latentHeight: 4,
      latentWidth: 4,
      conditioning: { encoderHiddenStates, pooledProjections, guidance },
      numInferenceSteps: 2,
      evaluateEachStep: false,
    });

    expect(latents.shape).toEqual([1, 4, 4]);
    expect(denoiser.calls).toHaveLength(2);
    expect(denoiser.calls[0]?.hiddenShape).toEqual([1, 4, 4]);
    expect(denoiser.calls[0]?.imageIdsShape).toEqual([4, 3]);
    expect(denoiser.calls[0]?.encoderShape).toEqual([1, 3, 5]);
    expect(denoiser.calls[0]?.textIdsShape).toEqual([3, 3]);
    expect(denoiser.calls[0]?.pooledShape).toEqual([1, 7]);
    expect(denoiser.calls[0]?.timestepShape).toEqual([1]);
    expect(denoiser.calls[0]?.guidanceShape).toEqual([1]);
    expectTensorClose(denoiser.calls[0]?.timestepValues ?? [], [expectedSteps[0]?.sigma ?? -1]);
    expectTensorClose(denoiser.calls[1]?.timestepValues ?? [], [expectedSteps[1]?.sigma ?? -1]);
  });

  test("denoising rejects cache-shape mismatches before calling the denoiser", () => {
    const scheduler = new FlowMatchEulerScheduler();
    const denoiser = new RecordingFluxDenoiser();
    using initialLatents = zeros([1, 3, 4]);
    using encoderHiddenStates = zeros([1, 3, 5]);
    using pooledProjections = zeros([1, 7]);

    expect(() =>
      denoiseFluxLatents({
        denoiser,
        scheduler,
        initialLatents,
        latentHeight: 4,
        latentWidth: 4,
        conditioning: { encoderHiddenStates, pooledProjections },
        numInferenceSteps: 1,
      }),
    ).toThrow("sequence length");
    expect(denoiser.calls).toHaveLength(0);
  });

  test("denoising rejects malformed packed latents, ids, and conditioning tensors", () => {
    const scheduler = new FlowMatchEulerScheduler();
    const denoiser = new RecordingFluxDenoiser();
    using rankTwoLatents = zeros([1, 4]);
    using badPackedChannels = zeros([1, 4, 5]);
    using initialLatents = zeros([1, 4, 4]);
    using encoderHiddenStates = zeros([1, 3, 5]);
    using pooledProjections = zeros([1, 7]);
    using wrongImageIds = zeros([3, 3], "int32");
    using wrongTextIds = zeros([2, 3], "int32");
    using wrongEncoderBatch = zeros([2, 3, 5]);
    using wrongPooledBatch = zeros([2, 7]);
    using wrongGuidanceBatch = zeros([2]);

    expect(() =>
      fluxLatentShape({
        batchSize: 0,
        height: 8,
        width: 8,
        latentChannels: 1,
        vaeScaleFactor: 2,
      }),
    ).toThrow("batchSize");
    expect(() =>
      fluxLatentShape({
        batchSize: 1,
        height: 6,
        width: 8,
        latentChannels: 1,
        vaeScaleFactor: 2,
      }),
    ).toThrow("latentHeight");
    expect(() =>
      denoiseFluxLatents({
        denoiser,
        scheduler,
        initialLatents: rankTwoLatents,
        latentHeight: 4,
        latentWidth: 4,
        conditioning: { encoderHiddenStates, pooledProjections },
        numInferenceSteps: 1,
      }),
    ).toThrow("packed FLUX latents");
    expect(() =>
      denoiseFluxLatents({
        denoiser,
        scheduler,
        initialLatents: badPackedChannels,
        latentHeight: 4,
        latentWidth: 4,
        conditioning: { encoderHiddenStates, pooledProjections },
        numInferenceSteps: 1,
      }),
    ).toThrow("channel dimension");
    expect(() =>
      denoiseFluxLatents({
        denoiser,
        scheduler,
        initialLatents,
        latentHeight: 4,
        latentWidth: 4,
        imageIds: wrongImageIds,
        conditioning: { encoderHiddenStates, pooledProjections },
        numInferenceSteps: 1,
      }),
    ).toThrow("imageIds");
    expect(() =>
      denoiseFluxLatents({
        denoiser,
        scheduler,
        initialLatents,
        latentHeight: 4,
        latentWidth: 4,
        conditioning: { encoderHiddenStates, pooledProjections, textIds: wrongTextIds },
        numInferenceSteps: 1,
      }),
    ).toThrow("conditioning.textIds");
    expect(() =>
      denoiseFluxLatents({
        denoiser,
        scheduler,
        initialLatents,
        latentHeight: 4,
        latentWidth: 4,
        conditioning: { encoderHiddenStates: wrongEncoderBatch, pooledProjections },
        numInferenceSteps: 1,
      }),
    ).toThrow("encoderHiddenStates");
    expect(() =>
      denoiseFluxLatents({
        denoiser,
        scheduler,
        initialLatents,
        latentHeight: 4,
        latentWidth: 4,
        conditioning: { encoderHiddenStates, pooledProjections: wrongPooledBatch },
        numInferenceSteps: 1,
      }),
    ).toThrow("pooledProjections");
    expect(() =>
      denoiseFluxLatents({
        denoiser,
        scheduler,
        initialLatents,
        latentHeight: 4,
        latentWidth: 4,
        conditioning: {
          encoderHiddenStates,
          pooledProjections,
          guidance: wrongGuidanceBatch,
        },
        numInferenceSteps: 1,
      }),
    ).toThrow("guidance");
  });

  test("denoising disposes owned state when the denoiser fails", () => {
    const scheduler = new FlowMatchEulerScheduler();
    using initialLatents = zeros([1, 4, 4]);
    using encoderHiddenStates = zeros([1, 3, 5]);
    using pooledProjections = zeros([1, 7]);

    expect(() =>
      denoiseFluxLatents({
        denoiser: new ThrowingFluxDenoiser(),
        scheduler,
        initialLatents,
        latentHeight: 4,
        latentWidth: 4,
        conditioning: { encoderHiddenStates, pooledProjections },
        numInferenceSteps: 1,
      }),
    ).toThrow("denoiser failed");
  });

  test("decoding applies FLUX VAE scale and shift before image normalization", () => {
    const decoder = new RecordingFluxDecoder();
    using packed = array([[[2, 4, 6, 8]]], "float32");

    using image = decodeFluxLatents(decoder, packed, 2, 2);

    expect(decoder.inputs).toHaveLength(1);
    expectTensorClose(decoder.inputs[0] ?? [], [1.25, 2.25, 3.25, 4.25]);
    expect(image.shape).toEqual([1, 2, 2, 1]);
    expectTensorClose(image.toTypedArray(), [0, 0.5, 1, 1]);
  });

  test("generation assembles sampling and decode without classifier-free batch expansion", () => {
    const scheduler = new FlowMatchEulerScheduler();
    const denoiser = new RecordingFluxDenoiser();
    const decoder = new RecordingFluxDecoder();
    using encoderHiddenStates = zeros([1, 2, 5]);
    using pooledProjections = zeros([1, 7]);
    using imageIds = zeros([4, 3], "int32");
    using rngKey = random.key(0);
    const steps: number[] = [];

    using image = generateFluxImage({
      denoiser,
      scheduler,
      vae: decoder,
      batchSize: 1,
      height: 8,
      width: 8,
      imageIds,
      conditioning: { encoderHiddenStates, pooledProjections },
      numInferenceSteps: 1,
      dtype: "float32",
      rngKey,
      onStep: (event) => {
        steps.push(event.stepIndex);
      },
    });

    expect(image.shape).toEqual([1, 2, 2, 1]);
    expect(denoiser.calls).toHaveLength(1);
    expect(denoiser.calls[0]?.hiddenShape[0]).toBe(1);
    expect(steps).toEqual([0]);
  });
});
