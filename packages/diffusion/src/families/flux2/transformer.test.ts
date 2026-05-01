import { describe, expect, test } from "bun:test";
import { array, mxEval, treeFlatten } from "@mlxts/core";

import type { Flux2KleinTransformerConfig } from "./config";
import { Flux2PosEmbed, flux2TimestepEmbedding } from "./embeddings";
import { Flux2KleinTransformer2DModel } from "./transformer";

function tinyFlux2Config(
  overrides: Partial<Flux2KleinTransformerConfig> = {},
): Flux2KleinTransformerConfig {
  return {
    patchSize: 1,
    inChannels: 4,
    latentChannels: 1,
    outChannels: 4,
    numLayers: 1,
    numSingleLayers: 1,
    attentionHeadDim: 8,
    numAttentionHeads: 1,
    hiddenSize: 8,
    mlpRatio: 2,
    jointAttentionDim: 12,
    timestepGuidanceChannels: 8,
    axesDimsRope: [2, 2, 2, 2],
    ropeTheta: 2000,
    normEps: 1e-6,
    guidanceEmbeds: true,
    rawConfig: {},
    ...overrides,
  };
}

describe("flux2TimestepEmbedding", () => {
  test("creates cosine-sine scalar embeddings", () => {
    using timesteps = array([0, 1], "float32");
    using embedding = flux2TimestepEmbedding(timesteps, 8);

    expect(embedding.shape).toEqual([2, 8]);
    expect(embedding.dtype).toBe("float32");
  });
});

describe("Flux2PosEmbed", () => {
  test("builds four-axis rope matrices from text and image ids", () => {
    using ids = array(
      [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 1],
      ],
      "float32",
    );
    using embedder = new Flux2PosEmbed(8, 2000, [2, 2, 2, 2]);
    using rope = embedder.embed(ids, "float32");

    expect(rope.shape).toEqual([1, 1, 4, 4, 2, 2]);
  });

  test("rejects non-FLUX.2 ids", () => {
    using ids = array([[0, 0, 0]], "float32");
    using embedder = new Flux2PosEmbed(8, 2000, [2, 2, 2, 2]);

    expect(() => embedder.embed(ids, "float32")).toThrow("ids shape [length, 4]");
  });
});

describe("Flux2KleinTransformer2DModel", () => {
  test("constructs the FLUX.2 Klein transformer parameter tree", () => {
    using model = new Flux2KleinTransformer2DModel(tinyFlux2Config());
    const paths = treeFlatten(model.parameters()).map(([path]) => path.join("."));

    expect(paths).toContain("timeGuidanceEmbed.timestepEmbedder.linear1.weight");
    expect(paths).toContain("timeGuidanceEmbed.guidanceEmbedder.linear2.weight");
    expect(paths).toContain("doubleStreamModulationImg.linear.weight");
    expect(paths).toContain("xEmbedder.weight");
    expect(paths).toContain("contextEmbedder.weight");
    expect(paths).toContain("transformerBlocks.0.attn.toQ.weight");
    expect(paths).toContain("transformerBlocks.0.attn.addQProj.weight");
    expect(paths).toContain("singleTransformerBlocks.0.attn.toQkvMlpProj.weight");
    expect(paths).toContain("normOut.linear.weight");
    expect(paths).toContain("projOut.weight");
    expect(paths.some((path) => path.endsWith(".bias"))).toBe(false);
  });

  test("runs a tiny prepared FLUX.2 Klein denoising prediction", () => {
    using model = new Flux2KleinTransformer2DModel(tinyFlux2Config());
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
          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
          [1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
          [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4],
        ],
      ],
      "float32",
    );
    using timestep = array([0.5], "float32");
    using guidance = array([0.25], "float32");
    using imageIds = array(
      [
        [0, 0, 0, 0],
        [0, 0, 1, 0],
      ],
      "float32",
    );
    using textIds = array(
      [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 2],
      ],
      "float32",
    );

    using output = model.forward({
      hiddenStates,
      encoderHiddenStates,
      timestep,
      guidance,
      imageIds,
      textIds,
    });

    mxEval(output);
    expect(output.shape).toEqual([1, 2, 4]);
    expect(Array.from(output.toTypedArray()).every(Number.isFinite)).toBe(true);
  });

  test("accepts omitted guidance when guidance embedder exists", () => {
    using model = new Flux2KleinTransformer2DModel(tinyFlux2Config());
    model.eval();
    using hiddenStates = array([[[0.1, 0.2, 0.3, 0.4]]], "float32");
    using encoderHiddenStates = array(
      [[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]]],
      "float32",
    );
    using timestep = array([0.5], "float32");
    using imageIds = array([[0, 0, 0, 0]], "float32");
    using textIds = array([[0, 0, 0, 0]], "float32");

    using output = model.forward({
      hiddenStates,
      encoderHiddenStates,
      timestep,
      imageIds,
      textIds,
    });

    mxEval(output);
    expect(output.shape).toEqual([1, 1, 4]);
  });

  test("validates prepared tensor geometry", () => {
    using model = new Flux2KleinTransformer2DModel(tinyFlux2Config());
    using hiddenStates = array([[[0.1, 0.2, 0.3, 0.4]]], "float32");
    using encoderHiddenStates = array(
      [[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]]],
      "float32",
    );
    using timestep = array([0.5], "float32");
    using imageIds = array(
      [
        [0, 0, 0, 0],
        [0, 0, 1, 0],
      ],
      "float32",
    );
    using textIds = array([[0, 0, 0, 0]], "float32");

    expect(() =>
      model.forward({
        hiddenStates,
        encoderHiddenStates,
        timestep,
        imageIds,
        textIds,
      }),
    ).toThrow("ids lengths");
  });

  test("rejects invalid transformer configs", () => {
    expect(() => new Flux2KleinTransformer2DModel(tinyFlux2Config({ hiddenSize: 10 }))).toThrow(
      "hiddenSize",
    );
    expect(
      () => new Flux2KleinTransformer2DModel(tinyFlux2Config({ axesDimsRope: [2, 2, 2, 4] })),
    ).toThrow("axesDimsRope");
    expect(() => new Flux2KleinTransformer2DModel(tinyFlux2Config({ mlpRatio: 0 }))).toThrow(
      "mlpRatio",
    );
  });
});
