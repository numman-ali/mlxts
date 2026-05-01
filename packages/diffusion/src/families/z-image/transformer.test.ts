import { describe, expect, test } from "bun:test";
import { MxArray, mxEval } from "@mlxts/core";

import type { ZImageTransformerConfig } from "./config";
import { ZImageTransformer2DModel } from "./transformer";

function tinyZImageConfig(
  overrides: Partial<ZImageTransformerConfig> = {},
): ZImageTransformerConfig {
  return {
    patchGeometries: [{ patchSize: 2, framePatchSize: 1, packedLatentChannels: 16 }],
    inChannels: 4,
    outChannels: 4,
    hiddenSize: 16,
    numLayers: 1,
    numRefinerLayers: 1,
    numAttentionHeads: 2,
    numKeyValueHeads: 2,
    attentionHeadDim: 8,
    normEps: 1e-5,
    qkNorm: true,
    captionFeatureDim: 8,
    siglipFeatureDim: null,
    ropeTheta: 256,
    timestepScale: 1000,
    sequenceMultiple: 32,
    latentPadDim: 64,
    axesDims: [4, 2, 2],
    axesLens: [128, 32, 32],
    rawConfig: {},
    ...overrides,
  };
}

describe("ZImageTransformer2DModel", () => {
  test("runs the base batch-1 denoising path", () => {
    using model = new ZImageTransformer2DModel(tinyZImageConfig());
    using latent = MxArray.fromData(
      Array.from({ length: 16 }, (_, index) => index / 16),
      [4, 1, 2, 2],
    );
    using caption = MxArray.fromData(
      Array.from({ length: 24 }, (_, index) => index / 24),
      [3, 8],
    );
    using timestep = MxArray.fromData([0.5], [1]);

    using output = model.forward({
      latents: [latent],
      captionFeatures: [caption],
      timestep,
    });

    mxEval(output);
    expect(output.shape).toEqual([1, 4, 1, 2, 2]);
  });

  test("rejects unsupported batch sizes and bad RoPE axes", () => {
    using model = new ZImageTransformer2DModel(tinyZImageConfig());
    using latent = MxArray.fromData(
      Array.from({ length: 16 }, (_, index) => index / 16),
      [4, 1, 2, 2],
    );
    using caption = MxArray.fromData(
      Array.from({ length: 24 }, (_, index) => index / 24),
      [3, 8],
    );
    using timestep = MxArray.fromData([0.5, 0.5], [2]);

    expect(() =>
      model.forward({
        latents: [latent, latent],
        captionFeatures: [caption, caption],
        timestep,
      }),
    ).toThrow("batch size 1");
    expect(
      () =>
        new ZImageTransformer2DModel(
          tinyZImageConfig({ siglipFeatureDim: null, axesDims: [2, 2, 2] }),
        ),
    ).toThrow("axesDims");
  });
});
