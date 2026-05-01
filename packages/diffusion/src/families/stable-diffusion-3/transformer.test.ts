import { describe, expect, test } from "bun:test";
import { array, mxEval, treeFlatten, zeros } from "@mlxts/core";

import type { StableDiffusion3TransformerConfig } from "./config";
import { StableDiffusion3Transformer2DModel } from "./transformer";

function tinyStableDiffusion3Config(
  overrides: Partial<StableDiffusion3TransformerConfig> = {},
): StableDiffusion3TransformerConfig {
  return {
    sampleSize: 4,
    patchSize: 2,
    inChannels: 4,
    outChannels: 4,
    numLayers: 2,
    attentionHeadDim: 4,
    numAttentionHeads: 2,
    hiddenSize: 8,
    jointAttentionDim: 6,
    captionProjectionDim: 8,
    pooledProjectionDim: 5,
    posEmbedMaxSize: 4,
    dualAttentionLayers: [],
    qkNorm: null,
    rawConfig: {},
    ...overrides,
  };
}

describe("StableDiffusion3Transformer2DModel", () => {
  test("constructs the SD3 parameter tree with final context pre-only block", () => {
    using model = new StableDiffusion3Transformer2DModel(tinyStableDiffusion3Config());
    const paths = treeFlatten(model.parameters()).map(([path]) => path.join("."));

    expect(paths).toContain("posEmbed.projection.weight");
    expect(paths).toContain("timeTextEmbed.timestepLinear1.weight");
    expect(paths).toContain("contextEmbedder.weight");
    expect(paths).toContain("transformerBlocks.0.attention.toQ.weight");
    expect(paths).toContain("transformerBlocks.0.attention.toAddOut.weight");
    expect(paths).toContain("transformerBlocks.0.ffContext.linear1.weight");
    expect(paths).toContain("transformerBlocks.1.norm1Context.linear.weight");
    expect(paths).toContain("normOut.linear.weight");
    expect(paths).toContain("projOut.weight");
    expect(paths).not.toContain("transformerBlocks.1.attention.toAddOut.weight");
    expect(paths).not.toContain("transformerBlocks.1.ffContext.linear1.weight");
  });

  test("runs a tiny prepared SD3 denoising prediction", () => {
    using model = new StableDiffusion3Transformer2DModel(tinyStableDiffusion3Config());
    model.eval();
    using hiddenStates = zeros([1, 4, 4, 4]);
    using encoderHiddenStates = zeros([1, 3, 6]);
    using pooledProjections = zeros([1, 5]);
    using timestep = array([500], "float32");

    using output = model.forward({
      hiddenStates,
      encoderHiddenStates,
      pooledProjections,
      timestep,
    });

    mxEval(output);
    expect(output.shape).toEqual([1, 4, 4, 4]);
    expect(Array.from(output.toTypedArray()).every(Number.isFinite)).toBe(true);
  });

  test("runs the SD3.5 qk-norm and dual-attention path", () => {
    using model = new StableDiffusion3Transformer2DModel(
      tinyStableDiffusion3Config({
        dualAttentionLayers: [0],
        qkNorm: "rms_norm",
      }),
    );
    model.eval();
    using hiddenStates = zeros([1, 4, 4, 4]);
    using encoderHiddenStates = zeros([1, 2, 6]);
    using pooledProjections = zeros([1, 5]);
    using timestep = array([750], "float32");

    using output = model.forward({
      hiddenStates,
      encoderHiddenStates,
      pooledProjections,
      timestep,
    });
    const paths = treeFlatten(model.parameters()).map(([path]) => path.join("."));

    mxEval(output);
    expect(output.shape).toEqual([1, 4, 4, 4]);
    expect(paths).toContain("transformerBlocks.0.attention.normQ.weight");
    expect(paths).toContain("transformerBlocks.0.attention.normAddedQ.weight");
    expect(paths).toContain("transformerBlocks.0.attention2.toQ.weight");
    expect(paths).toContain("transformerBlocks.0.attention2.normQ.weight");
  });

  test("rejects malformed prepared tensors and unsupported configs", () => {
    using model = new StableDiffusion3Transformer2DModel(tinyStableDiffusion3Config());
    using hiddenStates = zeros([1, 4, 4, 4]);
    using badHiddenStates = zeros([1, 5, 4, 4]);
    using encoderHiddenStates = zeros([1, 3, 6]);
    using badEncoderHiddenStates = zeros([1, 3, 7]);
    using pooledProjections = zeros([1, 5]);
    using badPooledProjections = zeros([1, 4]);
    using timestep = array([500], "float32");
    using badTimestep = array([500, 250], "float32");

    expect(() =>
      model.forward({
        hiddenStates: badHiddenStates,
        encoderHiddenStates,
        pooledProjections,
        timestep,
      }),
    ).toThrow("patchSize");
    expect(() =>
      model.forward({
        hiddenStates,
        encoderHiddenStates: badEncoderHiddenStates,
        pooledProjections,
        timestep,
      }),
    ).toThrow("encoderHiddenStates");
    expect(() =>
      model.forward({
        hiddenStates,
        encoderHiddenStates,
        pooledProjections: badPooledProjections,
        timestep,
      }),
    ).toThrow("pooledProjections");
    expect(() =>
      model.forward({
        hiddenStates,
        encoderHiddenStates,
        pooledProjections,
        timestep: badTimestep,
      }),
    ).toThrow("timestep");
    expect(
      () =>
        new StableDiffusion3Transformer2DModel(
          tinyStableDiffusion3Config({ captionProjectionDim: 12 }),
        ),
    ).toThrow("captionProjectionDim");
  });
});
