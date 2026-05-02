import { describe, expect, test } from "bun:test";
import { array, mxEval, treeFlatten } from "@mlxts/core";

import type { QwenImageTransformerConfig } from "./config";
import { QwenImageTransformer2DModel } from "./transformer";

function tinyConfig(
  overrides: Partial<QwenImageTransformerConfig> = {},
): QwenImageTransformerConfig {
  return {
    patchSize: 2,
    inChannels: 4,
    outChannels: 1,
    latentChannels: 1,
    packedLatentChannels: 4,
    numLayers: 1,
    attentionHeadDim: 6,
    numAttentionHeads: 2,
    hiddenSize: 12,
    jointAttentionDim: 8,
    guidanceEmbeds: false,
    axesDimsRope: [2, 2, 2],
    ropeTheta: 10000,
    zeroCondT: false,
    useAdditionalTCond: false,
    useLayer3dRope: false,
    rawConfig: {},
    ...overrides,
  };
}

describe("QwenImageTransformer2DModel", () => {
  test("constructs the Qwen-Image transformer parameter tree", () => {
    using model = new QwenImageTransformer2DModel(tinyConfig());
    const paths = treeFlatten(model.parameters()).map(([path]) => path.join("."));

    expect(paths).toContain("imgIn.weight");
    expect(paths).toContain("txtNorm.weight");
    expect(paths).toContain("timeTextEmbed.timestepEmbedder.linear1.weight");
    expect(paths).toContain("transformerBlocks.0.attn.toQ.weight");
    expect(paths).toContain("transformerBlocks.0.attn.addQProj.weight");
    expect(paths).toContain("transformerBlocks.0.imgMlp.linear1.weight");
    expect(paths).toContain("normOut.linear.weight");
    expect(paths).toContain("projOut.weight");
  });

  test("runs a tiny prepared Qwen-Image denoising prediction", () => {
    using model = new QwenImageTransformer2DModel(tinyConfig());
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
          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
          [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
          [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
        ],
      ],
      "float32",
    );
    using timestep = array([0.5], "float32");
    using output = model.forward({
      hiddenStates,
      encoderHiddenStates,
      timestep,
      imageShape: [1, 1, 2],
    });

    mxEval(output);
    expect(output.shape).toEqual([1, 2, 4]);
    expect(Array.from(output.toTypedArray()).every(Number.isFinite)).toBe(true);
  });

  test("accepts non-contiguous text masks without shortening RoPE length", () => {
    using model = new QwenImageTransformer2DModel(tinyConfig());
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
          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
          [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
          [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
        ],
      ],
      "float32",
    );
    using mask = array([[1, 0, 1]], "bool");
    using timestep = array([0.25], "float32");
    using output = model.forward({
      hiddenStates,
      encoderHiddenStates,
      encoderHiddenStatesMask: mask,
      timestep,
      imageShape: [1, 1, 2],
    });

    mxEval(output);
    expect(output.shape).toEqual([1, 2, 4]);
  });

  test("runs zero_cond_t with target and reference image segments", () => {
    using model = new QwenImageTransformer2DModel(tinyConfig({ zeroCondT: true }));
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
          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
          [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
          [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
        ],
      ],
      "float32",
    );
    using timestep = array([0.5], "float32");
    using output = model.forward({
      hiddenStates,
      encoderHiddenStates,
      timestep,
      imageShape: [1, 1, 1],
      imageShapes: [
        [1, 1, 1],
        [1, 1, 1],
      ],
    });

    mxEval(output);
    expect(output.shape).toEqual([1, 2, 4]);
    expect(Array.from(output.toTypedArray()).every(Number.isFinite)).toBe(true);
  });

  test("rejects unsupported Qwen-Image transformer variants deliberately", () => {
    expect(() => new QwenImageTransformer2DModel(tinyConfig({ guidanceEmbeds: true }))).toThrow(
      "guidance_embeds",
    );
    expect(() => new QwenImageTransformer2DModel(tinyConfig({ useAdditionalTCond: true }))).toThrow(
      "use_additional_t_cond",
    );
    expect(() => new QwenImageTransformer2DModel(tinyConfig({ useLayer3dRope: true }))).toThrow(
      "use_layer3d_rope",
    );
  });

  test("validates prepared tensor geometry", () => {
    using model = new QwenImageTransformer2DModel(tinyConfig());
    using hiddenStates = array([[[0.1, 0.2, 0.3, 0.4]]], "float32");
    using encoderHiddenStates = array([[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]], "float32");
    using timestep = array([0.5], "float32");

    expect(() =>
      model.forward({
        hiddenStates,
        encoderHiddenStates,
        timestep,
        imageShape: [1, 1, 2],
      }),
    ).toThrow("imageShape");
    expect(() =>
      model.forward({
        hiddenStates,
        encoderHiddenStates,
        timestep,
        imageShape: [1, 1, 1],
        imageShapes: [[1, 1, 2]],
      }),
    ).toThrow("first imageShapes");
  });
});
