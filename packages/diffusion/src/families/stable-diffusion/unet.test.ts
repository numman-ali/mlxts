import { describe, expect, test } from "bun:test";
import { array, type MxArray, treeFlatten, zeros } from "@mlxts/core";

import type { StableDiffusionUNetConfig } from "./config";
import { StableDiffusionUNet2DConditionModel } from "./unet";
import { StableDiffusionUNetDownBlock2d, StableDiffusionUNetUpBlock2d } from "./unet-blocks";
import { StableDiffusionSinusoidalTimesteps } from "./unet-embeddings";

function tinyUNetConfig(
  overrides: Partial<StableDiffusionUNetConfig> = {},
): StableDiffusionUNetConfig {
  return {
    inChannels: 4,
    outChannels: 4,
    convInKernel: 3,
    convOutKernel: 3,
    blockOutChannels: [4],
    layersPerBlock: [1],
    midBlockLayers: 2,
    transformerLayersPerBlock: [1],
    numAttentionHeads: [2],
    crossAttentionDim: [6],
    normNumGroups: 2,
    normEps: 0.00001,
    downBlockTypes: ["CrossAttnDownBlock2D"],
    upBlockTypes: ["CrossAttnUpBlock2D"],
    additionEmbedType: null,
    additionTimeEmbedDim: null,
    projectionClassEmbeddingsInputDim: null,
    useLinearProjection: false,
    upcastAttention: false,
    flipSinToCos: true,
    freqShift: 0,
    rawConfig: {},
    ...overrides,
  };
}

function disposeAll(values: readonly MxArray[]): void {
  for (const value of values) {
    value.free();
  }
}

describe("StableDiffusionUNet2DConditionModel", () => {
  test("runs a tiny conditional UNet forward over NHWC latent tensors", () => {
    using model = new StableDiffusionUNet2DConditionModel(tinyUNetConfig());
    using latents = zeros([1, 4, 4, 4]);
    using encoderHiddenStates = zeros([1, 3, 6]);

    using output = model.forward(latents, 1, encoderHiddenStates);

    expect(output.shape).toEqual([1, 4, 4, 4]);
  });

  test("parameter tree exposes Diffusers-mappable UNet module paths", () => {
    using model = new StableDiffusionUNet2DConditionModel(tinyUNetConfig());

    const paths = treeFlatten(model.parameters()).map(([path]) => path.join("."));

    expect(paths).toContain("convIn.weight");
    expect(paths).toContain("timeEmbedding.linear1.weight");
    expect(paths).toContain("downBlocks.0.resnets.0.timeEmbeddingProjection.weight");
    expect(paths).toContain("downBlocks.0.attentions.0.projectionIn.weight");
    expect(paths).toContain(
      "downBlocks.0.attentions.0.transformerBlocks.0.attention1.queryProjection.weight",
    );
    expect(paths).toContain(
      "downBlocks.0.attentions.0.transformerBlocks.0.feedForward.projectionIn.weight",
    );
    expect(paths).toContain("midBlock.resnetIn.conv1.weight");
    expect(paths).toContain(
      "midBlock.attention.transformerBlocks.0.attention2.keyProjection.weight",
    );
    expect(paths).toContain("upBlocks.0.resnets.0.timeEmbeddingProjection.weight");
    expect(paths).toContain("convOut.weight");
  });

  test("attention projection bias layout follows Diffusers checkpoints", () => {
    using model = new StableDiffusionUNet2DConditionModel(tinyUNetConfig());

    const paths = treeFlatten(model.parameters()).map(([path]) => path.join("."));

    expect(paths).not.toContain(
      "downBlocks.0.attentions.0.transformerBlocks.0.attention1.queryProjection.bias",
    );
    expect(paths).not.toContain(
      "downBlocks.0.attentions.0.transformerBlocks.0.attention1.keyProjection.bias",
    );
    expect(paths).not.toContain(
      "downBlocks.0.attentions.0.transformerBlocks.0.attention1.valueProjection.bias",
    );
    expect(paths).toContain(
      "downBlocks.0.attentions.0.transformerBlocks.0.attention1.outputProjection.bias",
    );
    expect(paths).not.toContain(
      "downBlocks.0.attentions.0.transformerBlocks.0.attention2.queryProjection.bias",
    );
    expect(paths).not.toContain(
      "downBlocks.0.attentions.0.transformerBlocks.0.attention2.keyProjection.bias",
    );
    expect(paths).not.toContain(
      "downBlocks.0.attentions.0.transformerBlocks.0.attention2.valueProjection.bias",
    );
    expect(paths).toContain(
      "downBlocks.0.attentions.0.transformerBlocks.0.attention2.outputProjection.bias",
    );
  });

  test("projection shape follows useLinearProjection checkpoint semantics", () => {
    using convProjectionModel = new StableDiffusionUNet2DConditionModel(
      tinyUNetConfig({ useLinearProjection: false }),
    );
    expect(convProjectionModel.downBlocks[0]?.attentions?.[0]?.projectionIn.weight.shape).toEqual([
      4, 1, 1, 4,
    ]);

    using linearProjectionModel = new StableDiffusionUNet2DConditionModel(
      tinyUNetConfig({ useLinearProjection: true }),
    );
    expect(linearProjectionModel.downBlocks[0]?.attentions?.[0]?.projectionIn.weight.shape).toEqual(
      [4, 4],
    );
  });

  test("up block resnet channels follow Diffusers previous-output semantics", () => {
    using model = new StableDiffusionUNet2DConditionModel(
      tinyUNetConfig({
        blockOutChannels: [4, 8, 16],
        layersPerBlock: [1, 1, 1],
        transformerLayersPerBlock: [1, 1, 1],
        numAttentionHeads: [2, 2, 4],
        crossAttentionDim: [6, 6, 6],
        downBlockTypes: ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
        upBlockTypes: ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
      }),
    );

    expect(model.upBlocks[1]?.resnets[0]?.conv1.weight.shape).toEqual([8, 3, 3, 24]);
    expect(model.upBlocks[2]?.resnets[0]?.conv1.weight.shape).toEqual([4, 3, 3, 12]);
  });

  test("downsample uses UNet stride-2 padding instead of VAE bottom-right pre-padding", () => {
    using block = new StableDiffusionUNetDownBlock2d({
      inChannels: 4,
      outChannels: 4,
      timeEmbedDims: 16,
      layers: 1,
      transformerLayers: 1,
      numHeads: 2,
      crossAttentionDims: 6,
      normGroups: 2,
      normEps: 0.00001,
      addDownsample: true,
      addCrossAttention: false,
      useLinearProjection: false,
    });
    using hidden = zeros([1, 5, 5, 4]);
    using timeEmbedding = zeros([1, 16]);
    using encoderHiddenStates = zeros([1, 2, 6]);

    const result = block.run(hidden, timeEmbedding, encoderHiddenStates);
    try {
      expect(result.hidden.shape).toEqual([1, 3, 3, 4]);
      expect(result.residuals.map((value) => value.shape)).toEqual([
        [1, 5, 5, 4],
        [1, 3, 3, 4],
      ]);
    } finally {
      result.hidden.free();
      disposeAll(result.residuals);
    }
  });

  test("up blocks consume residuals by channel-last concatenation", () => {
    using block = new StableDiffusionUNetUpBlock2d({
      inChannels: 4,
      outChannels: 4,
      previousOutputChannels: 4,
      timeEmbedDims: 16,
      layers: 2,
      transformerLayers: 1,
      numHeads: 2,
      crossAttentionDims: 6,
      normGroups: 2,
      normEps: 0.00001,
      addUpsample: false,
      addCrossAttention: false,
      useLinearProjection: false,
    });
    using hidden = zeros([1, 2, 2, 4]);
    using timeEmbedding = zeros([1, 16]);
    using encoderHiddenStates = zeros([1, 2, 6]);
    const residuals = [zeros([1, 2, 2, 4]), zeros([1, 2, 2, 4])];

    using output = block.run(hidden, residuals, timeEmbedding, encoderHiddenStates);

    expect(output.shape).toEqual([1, 2, 2, 4]);
    expect(residuals).toHaveLength(0);
  });

  test("sinusoidal timesteps match the Diffusers positional formula", () => {
    using timesteps = new StableDiffusionSinusoidalTimesteps(4, true, 0);
    using timestepValues = array([1], "float32");

    using output = timesteps.forward(timestepValues, 1, "float32");

    const values = output.toList() as number[][];
    expect(values[0]?.[0]).toBeCloseTo(Math.cos(1), 5);
    expect(values[0]?.[1]).toBeCloseTo(Math.cos(0.01), 5);
    expect(values[0]?.[2]).toBeCloseTo(Math.sin(1), 5);
    expect(values[0]?.[3]).toBeCloseTo(Math.sin(0.01), 5);
  });

  test("SDXL text-time conditioning is explicit", () => {
    const config = tinyUNetConfig({
      additionEmbedType: "text_time",
      additionTimeEmbedDim: 4,
      projectionClassEmbeddingsInputDim: 14,
      useLinearProjection: true,
    });
    using model = new StableDiffusionUNet2DConditionModel(config);
    using latents = zeros([1, 2, 2, 4]);
    using encoderHiddenStates = zeros([1, 3, 6]);

    expect(() => model.forward(latents, 1, encoderHiddenStates)).toThrow("text-time");

    using textEmbeds = zeros([1, 6]);
    using timeIds = zeros([1, 2]);
    using output = model.forward(latents, 1, encoderHiddenStates, {
      textTime: { textEmbeds, timeIds },
    });

    expect(output.shape).toEqual([1, 2, 2, 4]);
    expect(model.addEmbedding?.linear1.weight.shape).toEqual([16, 14]);
  });

  test("construction rejects unsupported upcast attention", () => {
    expect(
      () =>
        new StableDiffusionUNet2DConditionModel(
          tinyUNetConfig({
            upcastAttention: true,
          }),
        ),
    ).toThrow("upcastAttention");
  });
});
