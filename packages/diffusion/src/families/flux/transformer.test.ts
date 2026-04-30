import { describe, expect, test } from "bun:test";
import { array, ones, zeros } from "@mlxts/core";

import type { FluxTransformerConfig } from "./config";
import { FluxEmbedND, fluxTimestepEmbedding } from "./embeddings";
import { FluxTransformer2DModel } from "./transformer";

function tinyFluxConfig(overrides?: Partial<FluxTransformerConfig>): FluxTransformerConfig {
  return {
    patchSize: 1,
    inChannels: 4,
    latentChannels: 1,
    outChannels: 4,
    numLayers: 1,
    numSingleLayers: 1,
    attentionHeadDim: 6,
    numAttentionHeads: 2,
    hiddenSize: 12,
    mlpRatio: 2,
    jointAttentionDim: 5,
    pooledProjectionDim: 7,
    guidanceEmbeds: false,
    axesDimsRope: [2, 2, 2],
    ropeTheta: 10000,
    qkvBias: true,
    rawConfig: {},
    ...overrides,
  };
}

describe("fluxTimestepEmbedding", () => {
  test("creates cosine-sine scalar embeddings", () => {
    using timesteps = array([0, 1], "float32");
    using embedding = fluxTimestepEmbedding(timesteps, 6);

    expect(embedding.shape).toEqual([2, 6]);
    expect(embedding.dtype).toBe("float32");
  });
});

describe("FluxEmbedND", () => {
  test("builds rope matrices from text and image ids", () => {
    using ids = array(
      [
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 1],
      ],
      "float32",
    );
    using embedder = new FluxEmbedND(6, 10000, [2, 2, 2]);
    using rope = embedder.embed(ids, "float32");

    expect(rope.shape).toEqual([1, 1, 3, 3, 2, 2]);
  });
});

describe("FluxTransformer2DModel", () => {
  test("runs a tiny double and single stream denoiser path", () => {
    using model = new FluxTransformer2DModel(tinyFluxConfig());
    using hiddenStates = ones([1, 2, 4]);
    using imageIds = array(
      [
        [0, 0, 0],
        [0, 0, 1],
      ],
      "float32",
    );
    using encoderHiddenStates = ones([1, 3, 5]);
    using textIds = zeros([3, 3]);
    using pooledProjections = ones([1, 7]);
    using timestep = array([0.25], "float32");

    using output = model.forward({
      hiddenStates,
      imageIds,
      encoderHiddenStates,
      textIds,
      pooledProjections,
      timestep,
    });

    expect(output.shape).toEqual([1, 2, 4]);
  });

  test("requires guidance for guidance-embedded configs", () => {
    using model = new FluxTransformer2DModel(tinyFluxConfig({ guidanceEmbeds: true }));
    using hiddenStates = ones([1, 1, 4]);
    using imageIds = array([[0, 0, 0]], "float32");
    using encoderHiddenStates = ones([1, 1, 5]);
    using textIds = zeros([1, 3]);
    using pooledProjections = ones([1, 7]);
    using timestep = array([0.25], "float32");

    expect(() =>
      model.forward({
        hiddenStates,
        imageIds,
        encoderHiddenStates,
        textIds,
        pooledProjections,
        timestep,
      }),
    ).toThrow("guidance is required");
  });
});
