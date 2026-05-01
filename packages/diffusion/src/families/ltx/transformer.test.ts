import { describe, expect, test } from "bun:test";
import { array, asType, full, MxArray, mxEval, treeFlatten, zeros } from "@mlxts/core";

import { FlowMatchEulerScheduler } from "../../schedulers/flow-match-euler";
import { applyLtxVideoRotary } from "./attention";
import type { LtxVideoTransformerConfig } from "./config";
import { denoiseLtxVideoLatents } from "./pipeline";
import { LtxVideoTransformer3DModel } from "./transformer";

function tinyConfig(overrides: Partial<LtxVideoTransformerConfig> = {}): LtxVideoTransformerConfig {
  return {
    inChannels: 4,
    outChannels: 4,
    patchSize: 1,
    patchSizeT: 1,
    numAttentionHeads: 2,
    attentionHeadDim: 4,
    hiddenSize: 8,
    crossAttentionDim: 8,
    numLayers: 1,
    activationFn: "gelu-approximate",
    qkNorm: "rms_norm_across_heads",
    normElementwiseAffine: false,
    normEps: 1e-6,
    captionChannels: 6,
    attentionBias: true,
    attentionOutBias: true,
    rawConfig: {},
    ...overrides,
  };
}

function expectCloseList(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBeCloseTo(expected[index] ?? Number.NaN, 5);
  }
}

describe("LtxVideoTransformer3DModel", () => {
  test("constructs the LTX transformer parameter tree", () => {
    using model = new LtxVideoTransformer3DModel(tinyConfig());
    const paths = treeFlatten(model.parameters()).map(([path]) => path.join("."));

    expect(paths).toContain("projIn.weight");
    expect(paths).toContain("scaleShiftTable");
    expect(paths).toContain("timeEmbed.emb.timestepEmbedder.linear1.weight");
    expect(paths).toContain("timeEmbed.linear.weight");
    expect(paths).toContain("captionProjection.linear1.weight");
    expect(paths).toContain("transformerBlocks.0.attn1.normQ.weight");
    expect(paths).toContain("transformerBlocks.0.attn2.toK.weight");
    expect(paths).toContain("transformerBlocks.0.ff.linear1.weight");
    expect(paths).toContain("transformerBlocks.0.scaleShiftTable");
    expect(paths).toContain("projOut.weight");
    expect(paths.some((path) => path.startsWith("normOut"))).toBe(false);
  });

  test("applies classic LTX RoPE over interleaved pairs", () => {
    using x = MxArray.fromData([1, 2, 3, 4], [1, 1, 4], "float32");
    using cos = full([1, 1, 4], 0);
    using sin = full([1, 1, 4], 1);

    using rotated = applyLtxVideoRotary(x, { cos, sin });

    expectCloseList(rotated.toTypedArray(), [-2, 1, -4, 3]);
  });

  test("casts RoPE math back to the query dtype", () => {
    using x = MxArray.fromData([1, 2, 3, 4], [1, 1, 4], "float32");
    using x16 = asType(x, "float16");
    using cos = full([1, 1, 4], 1);
    using sin = zeros([1, 1, 4]);

    using output = applyLtxVideoRotary(x16, { cos, sin });

    expect(output.dtype).toBe("float16");
  });

  test("runs a tiny prepared LTX-Video denoising prediction", () => {
    using model = new LtxVideoTransformer3DModel(tinyConfig());
    model.eval();
    using hiddenStates = array(
      [
        [
          [0.1, 0.2, 0.3, 0.4],
          [0.4, 0.3, 0.2, 0.1],
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
    using timestep = array([0.5], "float32");
    using mask = array([[1, 0, 1]], "bool");
    using output = model.forward({
      hiddenStates,
      encoderHiddenStates,
      encoderAttentionMask: mask,
      timestep,
      numFrames: 1,
      height: 1,
      width: 2,
      ropeInterpolationScale: [8 / 24, 32, 32],
    });

    mxEval(output);
    expect(output.shape).toEqual([1, 2, 4]);
    expect(Array.from(output.toTypedArray()).every(Number.isFinite)).toBe(true);
  });

  test("plugs into the prepared-embedding LTX denoising loop", () => {
    using model = new LtxVideoTransformer3DModel(tinyConfig());
    model.eval();
    const scheduler = new FlowMatchEulerScheduler({ shift: 1 });
    using initialLatents = zeros([1, 2, 4]);
    using promptEmbeds = zeros([1, 3, 6]);
    using promptAttentionMask = MxArray.fromData([1, 0, 1], [1, 3], "int32");

    using latents = denoiseLtxVideoLatents({
      denoiser: model,
      scheduler,
      initialLatents,
      latentFrames: 1,
      latentHeight: 1,
      latentWidth: 2,
      conditioning: { promptEmbeds, promptAttentionMask },
      numInferenceSteps: 1,
      evaluateEachStep: false,
    });

    mxEval(latents);
    expect(latents.shape).toEqual([1, 2, 4]);
  });

  test("rejects unsupported LTX transformer config variants", () => {
    expect(() => new LtxVideoTransformer3DModel(tinyConfig({ patchSize: 2 }))).toThrow(
      "patch sizes",
    );
    expect(() => new LtxVideoTransformer3DModel(tinyConfig({ patchSizeT: 2 }))).toThrow(
      "patch sizes",
    );
    expect(
      () => new LtxVideoTransformer3DModel(tinyConfig({ normElementwiseAffine: true })),
    ).toThrow("affine RMSNorm");
    expect(() => new LtxVideoTransformer3DModel(tinyConfig({ crossAttentionDim: 4 }))).toThrow(
      "crossAttentionDim",
    );
  });

  test("validates prepared tensor geometry", () => {
    using model = new LtxVideoTransformer3DModel(tinyConfig());
    using hiddenStates = zeros([1, 2, 4]);
    using badChannels = zeros([1, 2, 3]);
    using encoderHiddenStates = zeros([1, 3, 6]);
    using badEncoder = zeros([1, 3, 5]);
    using timestep = array([0.5], "float32");
    using badTimestep = array([0.5, 0.25], "float32");
    using mask = zeros([1, 1, 3]);

    expect(() =>
      model.forward({
        hiddenStates: badChannels,
        encoderHiddenStates,
        timestep,
        numFrames: 1,
        height: 1,
        width: 2,
        ropeInterpolationScale: [8 / 24, 32, 32],
      }),
    ).toThrow("hiddenStates channels");
    expect(() =>
      model.forward({
        hiddenStates,
        encoderHiddenStates,
        timestep,
        numFrames: 1,
        height: 1,
        width: 3,
        ropeInterpolationScale: [8 / 24, 32, 32],
      }),
    ).toThrow("numFrames * height * width");
    expect(() =>
      model.forward({
        hiddenStates,
        encoderHiddenStates: badEncoder,
        timestep,
        numFrames: 1,
        height: 1,
        width: 2,
        ropeInterpolationScale: [8 / 24, 32, 32],
      }),
    ).toThrow("encoderHiddenStates");
    expect(() =>
      model.forward({
        hiddenStates,
        encoderHiddenStates,
        timestep: badTimestep,
        numFrames: 1,
        height: 1,
        width: 2,
        ropeInterpolationScale: [8 / 24, 32, 32],
      }),
    ).toThrow("timestep");
    expect(() =>
      model.forward({
        hiddenStates,
        encoderHiddenStates,
        encoderAttentionMask: mask,
        timestep,
        numFrames: 1,
        height: 1,
        width: 2,
        ropeInterpolationScale: [8 / 24, 32, 32],
      }),
    ).toThrow("encoderAttentionMask");
  });
});
