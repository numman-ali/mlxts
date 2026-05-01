import { describe, expect, test } from "bun:test";
import { array, full, type MxArray, mxEval, treeFlatten, zeros } from "@mlxts/core";

import { FlowMatchEulerScheduler } from "../../schedulers/flow-match-euler";
import type { Ltx2VideoTransformerConfig } from "./config";
import { createLtx2AudioCoords, createLtx2VideoCoords } from "./embeddings";
import { denoiseLtx2Latents } from "./pipeline-ltx2";
import type { Ltx2DenoiserOutput } from "./pipeline-ltx2-types";
import { Ltx2VideoTransformer3DModel } from "./transformer-ltx2";

function tinyConfig(
  overrides: Partial<Ltx2VideoTransformerConfig> = {},
): Ltx2VideoTransformerConfig {
  return {
    inChannels: 4,
    outChannels: 4,
    patchSize: 1,
    patchSizeT: 1,
    numAttentionHeads: 2,
    attentionHeadDim: 4,
    hiddenSize: 8,
    crossAttentionDim: 8,
    vaeScaleFactors: [8, 32, 32],
    posEmbedMaxPos: 20,
    baseHeight: 32,
    baseWidth: 32,
    audioInChannels: 4,
    audioOutChannels: 4,
    audioPatchSize: 1,
    audioPatchSizeT: 1,
    audioNumAttentionHeads: 2,
    audioAttentionHeadDim: 4,
    audioHiddenSize: 8,
    audioCrossAttentionDim: 8,
    audioScaleFactor: 4,
    audioPosEmbedMaxPos: 20,
    audioSamplingRate: 16000,
    audioHopLength: 160,
    numLayers: 1,
    activationFn: "gelu-approximate",
    qkNorm: "rms_norm_across_heads",
    normElementwiseAffine: false,
    normEps: 1e-6,
    captionChannels: 6,
    attentionBias: true,
    attentionOutBias: true,
    ropeTheta: 10000,
    ropeDoublePrecision: true,
    causalOffset: 1,
    timestepScaleMultiplier: 1000,
    crossAttnTimestepScaleMultiplier: 1000,
    ropeType: "split",
    gatedAttn: false,
    crossAttnMod: false,
    audioGatedAttn: false,
    audioCrossAttnMod: false,
    usePromptEmbeddings: true,
    perturbedAttn: false,
    rawConfig: {},
    ...overrides,
  };
}

function disposeOutput(output: Ltx2DenoiserOutput): void {
  output.video.free();
  output.audio.free();
}

function expectFiniteValues(values: ArrayLike<number>): void {
  expect(Array.from(values).every(Number.isFinite)).toBe(true);
}

function runTinyForward(model: Ltx2VideoTransformer3DModel, mask: MxArray): Ltx2DenoiserOutput {
  using hiddenStates = array(
    [
      [
        [0.1, 0.2, 0.3, 0.4],
        [0.4, 0.3, 0.2, 0.1],
      ],
    ],
    "float32",
  );
  using audioHiddenStates = array(
    [
      [
        [0.2, 0.1, 0.4, 0.3],
        [0.5, 0.7, 0.6, 0.8],
      ],
    ],
    "float32",
  );
  using encoderHiddenStates = array(
    [
      [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
      ],
    ],
    "float32",
  );
  using audioEncoderHiddenStates = array(
    [
      [
        [0.3, 0.2, 0.1, 0.6, 0.5, 0.4],
        [0.4, 0.6, 0.8, 0.1, 0.3, 0.5],
        [1.2, 1.0, 0.8, 0.6, 0.4, 0.2],
      ],
    ],
    "float32",
  );
  using timestep = array([0.5], "float32");
  using videoCoords = createLtx2VideoCoords({
    batchSize: 1,
    latentFrames: 1,
    latentHeight: 1,
    latentWidth: 2,
    frameRate: 24,
  });
  using audioCoords = createLtx2AudioCoords({ batchSize: 1, audioLatentFrames: 2 });

  return model.forward({
    hiddenStates,
    audioHiddenStates,
    encoderHiddenStates,
    audioEncoderHiddenStates,
    timestep,
    sigma: timestep,
    encoderAttentionMask: mask,
    audioEncoderAttentionMask: mask,
    numFrames: 1,
    height: 1,
    width: 2,
    fps: 24,
    audioNumFrames: 2,
    videoCoords,
    audioCoords,
    useCrossTimestep: false,
  });
}

describe("Ltx2VideoTransformer3DModel", () => {
  test("constructs the LTX-2 transformer parameter tree", () => {
    using model = new Ltx2VideoTransformer3DModel(tinyConfig());
    const paths = treeFlatten(model.parameters()).map(([path]) => path.join("."));

    expect(paths).toContain("projIn.weight");
    expect(paths).toContain("audioProjIn.weight");
    expect(paths).toContain("captionProjection.linear1.weight");
    expect(paths).toContain("audioCaptionProjection.linear1.weight");
    expect(paths).toContain("avCrossAttnVideoScaleShift.linear.weight");
    expect(paths).toContain("transformerBlocks.0.attn1.toQ.weight");
    expect(paths).toContain("transformerBlocks.0.audioAttn1.toQ.weight");
    expect(paths).toContain("transformerBlocks.0.audioToVideoAttn.toK.weight");
    expect(paths).toContain("transformerBlocks.0.videoToAudioAttn.toV.weight");
    expect(paths).toContain("transformerBlocks.0.videoA2vCrossAttnScaleShiftTable");
    expect(paths).toContain("audioProjOut.weight");
    expect(paths.some((path) => path.startsWith("normOut"))).toBe(false);
  });

  test("runs a tiny split-RoPE video/audio denoising prediction", () => {
    using model = new Ltx2VideoTransformer3DModel(tinyConfig());
    model.eval();
    using mask = array([[1, 0, 1]], "bool");

    const output = runTinyForward(model, mask);
    try {
      mxEval(output.video, output.audio);
      expect(output.video.shape).toEqual([1, 2, 4]);
      expect(output.audio.shape).toEqual([1, 2, 4]);
      expectFiniteValues(output.video.toTypedArray());
      expectFiniteValues(output.audio.toTypedArray());
    } finally {
      disposeOutput(output);
    }
  });

  test("honors prompt attention masks in the text cross-attention path", () => {
    using model = new Ltx2VideoTransformer3DModel(tinyConfig());
    model.eval();
    using fullMask = array([[1, 1, 1]], "bool");
    using partialMask = array([[1, 0, 0]], "bool");

    const fullOutput = runTinyForward(model, fullMask);
    try {
      const partialOutput = runTinyForward(model, partialMask);
      try {
        mxEval(fullOutput.video, partialOutput.video);
        const fullValues = Array.from(fullOutput.video.toTypedArray());
        const partialValues = Array.from(partialOutput.video.toTypedArray());
        expect(fullValues.some((value, index) => value !== partialValues[index])).toBe(true);
      } finally {
        disposeOutput(partialOutput);
      }
    } finally {
      disposeOutput(fullOutput);
    }
  });

  test("plugs into the prepared LTX-2 denoising loop", () => {
    const scheduler = new FlowMatchEulerScheduler({ numTrainTimesteps: 100, shift: 1 });
    using model = new Ltx2VideoTransformer3DModel(tinyConfig());
    model.eval();
    using initialVideoLatents = zeros([1, 2, 4]);
    using initialAudioLatents = zeros([1, 2, 4]);
    using promptEmbeds = zeros([1, 3, 6]);
    using audioPromptEmbeds = zeros([1, 3, 6]);
    using promptAttentionMask = full([1, 3], 1, "bool");

    const result = denoiseLtx2Latents({
      denoiser: model,
      scheduler,
      initialVideoLatents,
      initialAudioLatents,
      latentFrames: 1,
      latentHeight: 1,
      latentWidth: 2,
      audioLatentFrames: 2,
      audioLatentMelBins: 1,
      conditioning: { promptEmbeds, audioPromptEmbeds, promptAttentionMask },
      numInferenceSteps: 1,
      sigmas: [1],
      mu: 0,
      evaluateEachStep: false,
    });
    try {
      mxEval(result.videoLatents, result.audioLatents);
      expect(result.videoLatents.shape).toEqual([1, 2, 4]);
      expect(result.audioLatents.shape).toEqual([1, 2, 4]);
    } finally {
      result.videoLatents.free();
      result.audioLatents.free();
    }
  });

  test("rejects unsupported LTX-2.3 and non-default runtime branches", () => {
    expect(
      () => new Ltx2VideoTransformer3DModel(tinyConfig({ usePromptEmbeddings: false })),
    ).toThrow("prepared prompt embeddings");
    expect(() => new Ltx2VideoTransformer3DModel(tinyConfig({ gatedAttn: true }))).toThrow(
      "gated attention",
    );
    expect(() => new Ltx2VideoTransformer3DModel(tinyConfig({ audioGatedAttn: true }))).toThrow(
      "gated attention",
    );
    expect(() => new Ltx2VideoTransformer3DModel(tinyConfig({ crossAttnMod: true }))).toThrow(
      "prompt cross-attention modulation",
    );
    expect(() => new Ltx2VideoTransformer3DModel(tinyConfig({ audioCrossAttnMod: true }))).toThrow(
      "prompt cross-attention modulation",
    );
    expect(() => new Ltx2VideoTransformer3DModel(tinyConfig({ perturbedAttn: true }))).toThrow(
      "perturbed attention",
    );
    expect(
      () => new Ltx2VideoTransformer3DModel(tinyConfig({ normElementwiseAffine: true })),
    ).toThrow("affine LTX-2 norms");
  });
});
