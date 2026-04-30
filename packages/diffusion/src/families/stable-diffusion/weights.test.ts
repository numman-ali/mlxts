import { describe, expect, test } from "bun:test";
import { type MxArray, type ParameterTree, saveSafetensors, treeFlatten, zeros } from "@mlxts/core";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { DiffusionMissingWeightsError, DiffusionWeightMismatchError } from "../../errors";
import type { DiffusionSnapshotComponent } from "../../pretrained/snapshot-manifest";
import { loadDiffusionSnapshotManifest } from "../../pretrained/snapshot-manifest";
import { StableDiffusionAutoencoderKL } from "./autoencoder";
import type { StableDiffusionAutoencoderConfig, StableDiffusionUNetConfig } from "./config";
import { StableDiffusionUNet2DConditionModel } from "./unet";
import {
  loadStableDiffusionAutoencoderFromSnapshot,
  loadStableDiffusionAutoencoderWeights,
  loadStableDiffusionUNetFromSnapshot,
  loadStableDiffusionUNetWeights,
  stableDiffusionAutoencoderWeightPath,
  stableDiffusionUNetWeightPath,
  transformStableDiffusionAutoencoderWeight,
  transformStableDiffusionUNetWeight,
} from "./weights";

async function withTempDirectory<T>(
  prefix: string,
  run: (directory: string) => Promise<T>,
): Promise<T> {
  const directory = mkdtempSync(join(tmpdir(), prefix));
  try {
    return await run(directory);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

function writeJson(path: string, payload: unknown): void {
  writeFileSync(path, JSON.stringify(payload));
}

function tinyVaeConfig(): StableDiffusionAutoencoderConfig {
  return {
    inChannels: 3,
    outChannels: 3,
    latentChannels: 2,
    latentChannelsOut: 4,
    blockOutChannels: [4, 8],
    layersPerBlock: 1,
    normNumGroups: 4,
    scalingFactor: 0.18215,
    downBlockTypes: ["DownEncoderBlock2D", "DownEncoderBlock2D"],
    upBlockTypes: ["UpDecoderBlock2D", "UpDecoderBlock2D"],
    forceUpcast: true,
    rawConfig: {},
  };
}

function diffusersVaeConfig(): Record<string, unknown> {
  return {
    _class_name: "AutoencoderKL",
    in_channels: 3,
    out_channels: 3,
    down_block_types: ["DownEncoderBlock2D", "DownEncoderBlock2D"],
    up_block_types: ["UpDecoderBlock2D", "UpDecoderBlock2D"],
    block_out_channels: [4, 8],
    layers_per_block: 1,
    latent_channels: 2,
    norm_num_groups: 4,
    scaling_factor: 0.18215,
    force_upcast: true,
  };
}

function tinyUNetConfig(
  overrides: Partial<StableDiffusionUNetConfig> = {},
): StableDiffusionUNetConfig {
  return {
    sampleSize: 8,
    inChannels: 4,
    outChannels: 4,
    convInKernel: 3,
    convOutKernel: 3,
    blockOutChannels: [8, 8],
    layersPerBlock: [1, 1],
    midBlockLayers: 2,
    transformerLayersPerBlock: [1, 1],
    numAttentionHeads: [2, 2],
    crossAttentionDim: [4, 4],
    normNumGroups: 4,
    normEps: 0.00001,
    downBlockTypes: ["CrossAttnDownBlock2D", "DownBlock2D"],
    upBlockTypes: ["UpBlock2D", "CrossAttnUpBlock2D"],
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

function diffusersUNetConfig(): Record<string, unknown> {
  return {
    _class_name: "UNet2DConditionModel",
    sample_size: 8,
    in_channels: 4,
    out_channels: 4,
    down_block_types: ["CrossAttnDownBlock2D", "DownBlock2D"],
    up_block_types: ["UpBlock2D", "CrossAttnUpBlock2D"],
    block_out_channels: [8, 8],
    layers_per_block: 1,
    attention_head_dim: 2,
    cross_attention_dim: 4,
    norm_num_groups: 4,
    norm_eps: 0.00001,
    flip_sin_to_cos: true,
    freq_shift: 0,
    use_linear_projection: false,
    upcast_attention: false,
  };
}

function writeSnapshotMetadata(directory: string): void {
  writeJson(join(directory, "model_index.json"), {
    _class_name: "StableDiffusionPipeline",
    vae: ["diffusers", "AutoencoderKL"],
    text_encoder: ["transformers", "CLIPTextModel"],
    tokenizer: ["transformers", "CLIPTokenizer"],
    unet: ["diffusers", "UNet2DConditionModel"],
    scheduler: ["diffusers", "DDIMScheduler"],
    safety_checker: [null, null],
    requires_safety_checker: false,
  });

  const vaeDirectory = join(directory, "vae");
  mkdirSync(vaeDirectory, { recursive: true });
  writeJson(join(vaeDirectory, "config.json"), diffusersVaeConfig());
  writeFileSync(join(vaeDirectory, "diffusion_pytorch_model.safetensors"), "");

  const textEncoderDirectory = join(directory, "text_encoder");
  mkdirSync(textEncoderDirectory, { recursive: true });
  writeJson(join(textEncoderDirectory, "config.json"), { hidden_size: 4 });
  writeFileSync(join(textEncoderDirectory, "model.safetensors"), "");

  const tokenizerDirectory = join(directory, "tokenizer");
  mkdirSync(tokenizerDirectory, { recursive: true });
  writeJson(join(tokenizerDirectory, "vocab.json"), {});
  writeFileSync(join(tokenizerDirectory, "merges.txt"), "");

  const unetDirectory = join(directory, "unet");
  mkdirSync(unetDirectory, { recursive: true });
  writeJson(join(unetDirectory, "config.json"), diffusersUNetConfig());
  writeFileSync(join(unetDirectory, "diffusion_pytorch_model.safetensors"), "");

  const schedulerDirectory = join(directory, "scheduler");
  mkdirSync(schedulerDirectory, { recursive: true });
  writeJson(join(schedulerDirectory, "scheduler_config.json"), {
    _class_name: "DDIMScheduler",
    beta_schedule: "linear",
    beta_start: 0.1,
    beta_end: 0.2,
    num_train_timesteps: 4,
  });
}

function vaeComponent(weightPaths: readonly string[]): DiffusionSnapshotComponent {
  return {
    name: "vae",
    role: "vae",
    library: "diffusers",
    className: "AutoencoderKL",
    enabled: true,
    optional: false,
    subfolder: "vae",
    metadataPaths: [],
    weightPaths,
  };
}

function unetComponent(weightPaths: readonly string[]): DiffusionSnapshotComponent {
  return {
    name: "unet",
    role: "backbone",
    library: "diffusers",
    className: "UNet2DConditionModel",
    enabled: true,
    optional: false,
    subfolder: "unet",
    metadataPaths: [],
    weightPaths,
  };
}

function checkpointNameForPath(path: string): string {
  return path
    .replaceAll("downBlocks", "down_blocks")
    .replaceAll("upBlocks", "up_blocks")
    .replaceAll("downsample.conv", "downsamplers.0.conv")
    .replaceAll("upsample.conv", "upsamplers.0.conv")
    .replaceAll("midBlock.resnetIn", "mid_block.resnets.0")
    .replaceAll("midBlock.attention", "mid_block.attentions.0")
    .replaceAll("midBlock.resnetOut", "mid_block.resnets.1")
    .replaceAll("queryProjection", "to_q")
    .replaceAll("keyProjection", "to_k")
    .replaceAll("valueProjection", "to_v")
    .replaceAll("outputProjection", "to_out.0")
    .replaceAll("groupNorm", "group_norm")
    .replaceAll("convNormOut", "conv_norm_out")
    .replaceAll("convShortcut", "conv_shortcut")
    .replaceAll("convIn", "conv_in")
    .replaceAll("convOut", "conv_out")
    .replaceAll("postQuantConv", "post_quant_conv")
    .replaceAll("quantConv", "quant_conv");
}

function unetCheckpointNameForPath(path: string): string {
  return path
    .replaceAll("downBlocks", "down_blocks")
    .replaceAll("upBlocks", "up_blocks")
    .replaceAll("downsample", "downsamplers.0.conv")
    .replaceAll("upsample", "upsamplers.0.conv")
    .replaceAll("midBlock.resnetIn", "mid_block.resnets.0")
    .replaceAll("midBlock.attention", "mid_block.attentions.0")
    .replaceAll("midBlock.resnetOut", "mid_block.resnets.1")
    .replaceAll("timeEmbeddingProjection", "time_emb_proj")
    .replaceAll("timeEmbedding.linear1", "time_embedding.linear_1")
    .replaceAll("timeEmbedding.linear2", "time_embedding.linear_2")
    .replaceAll("addEmbedding.linear1", "add_embedding.linear_1")
    .replaceAll("addEmbedding.linear2", "add_embedding.linear_2")
    .replaceAll("convNormOut", "conv_norm_out")
    .replaceAll("convIn", "conv_in")
    .replaceAll("convOut", "conv_out")
    .replaceAll("convShortcut", "conv_shortcut")
    .replaceAll("transformerBlocks", "transformer_blocks")
    .replaceAll("attention1", "attn1")
    .replaceAll("attention2", "attn2")
    .replaceAll("queryProjection", "to_q")
    .replaceAll("keyProjection", "to_k")
    .replaceAll("valueProjection", "to_v")
    .replaceAll("outputProjection", "to_out.0")
    .replaceAll("feedForward.projectionIn", "ff.net.0.proj")
    .replaceAll("feedForward.projectionOut", "ff.net.2")
    .replaceAll("projectionIn", "proj_in")
    .replaceAll("projectionOut", "proj_out");
}

function checkpointShapeFor(path: string, tensor: MxArray): number[] {
  const [outputChannels, kernelHeight, kernelWidth, inputChannels] = tensor.shape;
  if (
    path.endsWith(".weight") &&
    tensor.shape.length === 4 &&
    outputChannels !== undefined &&
    kernelHeight !== undefined &&
    kernelWidth !== undefined &&
    inputChannels !== undefined
  ) {
    return [outputChannels, inputChannels, kernelHeight, kernelWidth];
  }
  return [...tensor.shape];
}

function checkpointTensorsForParameters(
  parameters: ParameterTree,
  options: { omitPath?: string; extra?: Record<string, MxArray>; mismatchPath?: string } = {},
): Record<string, MxArray> {
  const tensors: Record<string, MxArray> = {};
  for (const [path, parameter] of treeFlatten(parameters)) {
    const internalPath = path.join(".");
    if (internalPath === options.omitPath) {
      continue;
    }
    const checkpointName = checkpointNameForPath(internalPath);
    const shape =
      internalPath === options.mismatchPath ? [1] : checkpointShapeFor(internalPath, parameter);
    tensors[checkpointName] = zeros(shape);
  }
  return { ...tensors, ...(options.extra ?? {}) };
}

function checkpointTensorsForUNetParameters(
  parameters: ParameterTree,
  options: { omitPath?: string; extra?: Record<string, MxArray>; mismatchPath?: string } = {},
): Record<string, MxArray> {
  const tensors: Record<string, MxArray> = {};
  for (const [path, parameter] of treeFlatten(parameters)) {
    const internalPath = path.join(".");
    if (internalPath === options.omitPath) {
      continue;
    }
    const checkpointName = unetCheckpointNameForPath(internalPath);
    const shape =
      internalPath === options.mismatchPath ? [1] : checkpointShapeFor(internalPath, parameter);
    tensors[checkpointName] = zeros(shape);
  }
  return { ...tensors, ...(options.extra ?? {}) };
}

function freeTensors(tensors: Record<string, MxArray>): void {
  for (const tensor of Object.values(tensors)) {
    tensor.free();
  }
}

async function writeCheckpoint(path: string, tensors: Record<string, MxArray>): Promise<void> {
  try {
    await saveSafetensors(tensors, path);
  } finally {
    freeTensors(tensors);
  }
}

function splitIndexedShards(entries: Record<string, MxArray>): {
  firstShard: Record<string, MxArray>;
  secondShard: Record<string, MxArray>;
  weightMap: Record<string, string>;
} {
  const names = Object.keys(entries).toSorted((left, right) => left.localeCompare(right));
  const midpoint = Math.ceil(names.length / 2);
  const firstShard: Record<string, MxArray> = {};
  const secondShard: Record<string, MxArray> = {};
  const weightMap: Record<string, string> = {};

  for (const [index, name] of names.entries()) {
    const tensor = entries[name];
    if (tensor === undefined) {
      throw new Error(`splitIndexedShards: missing tensor for ${name}.`);
    }
    const shardName =
      index < midpoint
        ? "diffusion_pytorch_model-00001-of-00002.safetensors"
        : "diffusion_pytorch_model-00002-of-00002.safetensors";
    const shard = index < midpoint ? firstShard : secondShard;
    shard[name] = tensor;
    weightMap[name] = shardName;
  }

  return { firstShard, secondShard, weightMap };
}

describe("Stable Diffusion VAE weight loading", () => {
  test("maps Diffusers VAE names onto the package parameter tree", () => {
    expect(stableDiffusionAutoencoderWeightPath("encoder.conv_in.weight")).toBe(
      "encoder.convIn.weight",
    );
    expect(
      stableDiffusionAutoencoderWeightPath("encoder.down_blocks.1.downsamplers.0.conv.bias"),
    ).toBe("encoder.downBlocks.1.downsample.conv.bias");
    expect(
      stableDiffusionAutoencoderWeightPath("decoder.mid_block.attentions.0.to_out.0.weight"),
    ).toBe("decoder.midBlock.attention.outputProjection.weight");
    expect(
      stableDiffusionAutoencoderWeightPath("decoder.up_blocks.1.resnets.0.conv_shortcut.bias"),
    ).toBe("decoder.upBlocks.1.resnets.0.convShortcut.bias");
    expect(stableDiffusionAutoencoderWeightPath("quant_conv.weight")).toBe("quantConv.weight");
    expect(stableDiffusionAutoencoderWeightPath("post_quant_conv.bias")).toBe("postQuantConv.bias");
  });

  test("transforms Conv2d weights without squeezing VAE 1x1 projections", () => {
    using convolution = zeros([7, 2, 3, 5]);
    using transformed = transformStableDiffusionAutoencoderWeight(
      "decoder.conv_out.weight",
      "decoder.convOut.weight",
      convolution,
    );
    expect(transformed.shape).toEqual([7, 3, 5, 2]);

    using projection = zeros([4, 8, 1, 1]);
    using transformedProjection = transformStableDiffusionAutoencoderWeight(
      "quant_conv.weight",
      "quantConv.weight",
      projection,
    );
    expect(transformedProjection.shape).toEqual([4, 1, 1, 8]);

    using linear = zeros([7, 2]);
    const unchanged = transformStableDiffusionAutoencoderWeight(
      "encoder.mid_block.attentions.0.to_q.weight",
      "encoder.midBlock.attention.queryProjection.weight",
      linear,
    );
    expect(unchanged).toBe(linear);
  });

  test("loads a complete generated VAE safetensors shard", async () => {
    await withTempDirectory("mlxts-sd-vae-weights-", async (directory) => {
      using source = new StableDiffusionAutoencoderKL(tinyVaeConfig());
      const checkpointPath = join(directory, "diffusion_pytorch_model.safetensors");
      await writeCheckpoint(checkpointPath, checkpointTensorsForParameters(source.parameters()));
      using target = new StableDiffusionAutoencoderKL(tinyVaeConfig());

      const result = await loadStableDiffusionAutoencoderWeights(
        target,
        vaeComponent([checkpointPath]),
      );

      expect(result.shardCount).toBe(1);
      expect(result.unexpectedWeights).toEqual([]);
      expect(result.assignedPaths).toEqual(
        treeFlatten(target.parameters())
          .map(([path]) => path.join("."))
          .toSorted((left, right) => left.localeCompare(right)),
      );
      expect(target.quantConv.weight.shape).toEqual([4, 1, 1, 4]);
      expect(target.decoder.upBlocks[1]?.resnets[0]?.convShortcut?.weight.shape).toEqual([
        4, 1, 1, 8,
      ]);
    });
  });

  test("loads VAE shards from a Diffusers safetensors index", async () => {
    await withTempDirectory("mlxts-sd-vae-index-", async (directory) => {
      using source = new StableDiffusionAutoencoderKL(tinyVaeConfig());
      const entries = checkpointTensorsForParameters(source.parameters());
      const { firstShard, secondShard, weightMap } = splitIndexedShards(entries);

      await saveSafetensors(
        firstShard,
        join(directory, "diffusion_pytorch_model-00001-of-00002.safetensors"),
      );
      await saveSafetensors(
        secondShard,
        join(directory, "diffusion_pytorch_model-00002-of-00002.safetensors"),
      );
      freeTensors(entries);
      const indexPath = join(directory, "diffusion_pytorch_model.safetensors.index.json");
      writeJson(indexPath, { weight_map: weightMap });

      using target = new StableDiffusionAutoencoderKL(tinyVaeConfig());
      const result = await loadStableDiffusionAutoencoderWeights(target, vaeComponent([indexPath]));

      expect(result.shardCount).toBe(2);
      expect(result.unexpectedWeights).toEqual([]);
    });
  });

  test("loads a VAE directly from an inspected Stable Diffusion snapshot manifest", async () => {
    await withTempDirectory("mlxts-sd-vae-manifest-", async (directory) => {
      writeSnapshotMetadata(directory);
      using source = new StableDiffusionAutoencoderKL(tinyVaeConfig());
      await writeCheckpoint(
        join(directory, "vae", "diffusion_pytorch_model.safetensors"),
        checkpointTensorsForParameters(source.parameters()),
      );

      const manifest = await loadDiffusionSnapshotManifest(directory);
      using model = await loadStableDiffusionAutoencoderFromSnapshot(manifest);

      expect(model.scalingFactor).toBe(0.18215);
      expect(model.latentChannels).toBe(2);
      expect(model.encoder.convIn.weight.shape).toEqual([4, 3, 3, 3]);
    });
  });

  test("rejects missing, mismatched, and strict unexpected VAE weights", async () => {
    await withTempDirectory("mlxts-sd-vae-invalid-", async (directory) => {
      using source = new StableDiffusionAutoencoderKL(tinyVaeConfig());
      const missingPath = join(directory, "missing.safetensors");
      await writeCheckpoint(
        missingPath,
        checkpointTensorsForParameters(source.parameters(), {
          omitPath: "quantConv.bias",
        }),
      );
      using missingTarget = new StableDiffusionAutoencoderKL(tinyVaeConfig());
      await expect(
        loadStableDiffusionAutoencoderWeights(missingTarget, vaeComponent([missingPath])),
      ).rejects.toThrow(DiffusionMissingWeightsError);

      const mismatchPath = join(directory, "mismatch.safetensors");
      await writeCheckpoint(
        mismatchPath,
        checkpointTensorsForParameters(source.parameters(), {
          mismatchPath: "encoder.convOut.weight",
        }),
      );
      using mismatchTarget = new StableDiffusionAutoencoderKL(tinyVaeConfig());
      await expect(
        loadStableDiffusionAutoencoderWeights(mismatchTarget, vaeComponent([mismatchPath])),
      ).rejects.toThrow(DiffusionWeightMismatchError);

      const unexpectedPath = join(directory, "unexpected.safetensors");
      await writeCheckpoint(
        unexpectedPath,
        checkpointTensorsForParameters(source.parameters(), {
          extra: { "decoder.extra.weight": zeros([1]) },
        }),
      );
      using unexpectedTarget = new StableDiffusionAutoencoderKL(tinyVaeConfig());
      await expect(
        loadStableDiffusionAutoencoderWeights(unexpectedTarget, vaeComponent([unexpectedPath]), {
          strictUnexpectedWeights: true,
        }),
      ).rejects.toThrow("unexpected unmapped weights");
    });
  });
});

describe("Stable Diffusion UNet weight loading", () => {
  test("maps Diffusers UNet names onto the package parameter tree", () => {
    expect(stableDiffusionUNetWeightPath("conv_in.weight")).toBe("convIn.weight");
    expect(stableDiffusionUNetWeightPath("time_embedding.linear_1.bias")).toBe(
      "timeEmbedding.linear1.bias",
    );
    expect(stableDiffusionUNetWeightPath("add_embedding.linear_2.weight")).toBe(
      "addEmbedding.linear2.weight",
    );
    expect(stableDiffusionUNetWeightPath("down_blocks.0.resnets.0.time_emb_proj.weight")).toBe(
      "downBlocks.0.resnets.0.timeEmbeddingProjection.weight",
    );
    expect(stableDiffusionUNetWeightPath("down_blocks.0.downsamplers.0.conv.bias")).toBe(
      "downBlocks.0.downsample.bias",
    );
    expect(
      stableDiffusionUNetWeightPath(
        "mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0.weight",
      ),
    ).toBe("midBlock.attention.transformerBlocks.0.attention2.outputProjection.weight");
    expect(stableDiffusionUNetWeightPath("up_blocks.1.attentions.0.proj_out.weight")).toBe(
      "upBlocks.1.attentions.0.projectionOut.weight",
    );
    expect(
      stableDiffusionUNetWeightPath(
        "up_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj.bias",
      ),
    ).toBe("upBlocks.1.attentions.0.transformerBlocks.0.feedForward.projectionIn.bias");
  });

  test("transforms UNet Conv2d weights and leaves Linear-shaped tensors intact", () => {
    using convolution = zeros([7, 2, 3, 5]);
    using transformed = transformStableDiffusionUNetWeight(
      "conv_in.weight",
      "convIn.weight",
      convolution,
    );
    expect(transformed.shape).toEqual([7, 3, 5, 2]);

    using transformerProjection = zeros([8, 8, 1, 1]);
    using transformedProjection = transformStableDiffusionUNetWeight(
      "down_blocks.0.attentions.0.proj_in.weight",
      "downBlocks.0.attentions.0.projectionIn.weight",
      transformerProjection,
    );
    expect(transformedProjection.shape).toEqual([8, 1, 1, 8]);

    using linear = zeros([16, 8]);
    const unchanged = transformStableDiffusionUNetWeight(
      "down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj.weight",
      "downBlocks.0.attentions.0.transformerBlocks.0.feedForward.projectionIn.weight",
      linear,
    );
    expect(unchanged).toBe(linear);
  });

  test("loads complete generated UNet safetensors shards", async () => {
    await withTempDirectory("mlxts-sd-unet-weights-", async (directory) => {
      using source = new StableDiffusionUNet2DConditionModel(tinyUNetConfig());
      const checkpointPath = join(directory, "diffusion_pytorch_model.safetensors");
      await writeCheckpoint(
        checkpointPath,
        checkpointTensorsForUNetParameters(source.parameters()),
      );
      using target = new StableDiffusionUNet2DConditionModel(tinyUNetConfig());

      const result = await loadStableDiffusionUNetWeights(target, unetComponent([checkpointPath]));

      expect(result.shardCount).toBe(1);
      expect(result.unexpectedWeights).toEqual([]);
      expect(result.assignedPaths).toEqual(
        treeFlatten(target.parameters())
          .map(([path]) => path.join("."))
          .toSorted((left, right) => left.localeCompare(right)),
      );
      expect(target.convIn.weight.shape).toEqual([8, 3, 3, 4]);
      expect(target.downBlocks[0]?.attentions?.[0]?.projectionIn.weight.shape).toEqual([
        8, 1, 1, 8,
      ]);
    });
  });

  test("loads linear-projection UNet weights without Conv2d projection coercion", async () => {
    await withTempDirectory("mlxts-sd-unet-linear-weights-", async (directory) => {
      const config = tinyUNetConfig({ useLinearProjection: true });
      using source = new StableDiffusionUNet2DConditionModel(config);
      const checkpointPath = join(directory, "diffusion_pytorch_model.safetensors");
      await writeCheckpoint(
        checkpointPath,
        checkpointTensorsForUNetParameters(source.parameters()),
      );
      using target = new StableDiffusionUNet2DConditionModel(config);

      const result = await loadStableDiffusionUNetWeights(target, unetComponent([checkpointPath]));

      expect(result.unexpectedWeights).toEqual([]);
      expect(target.downBlocks[0]?.attentions?.[0]?.projectionIn.weight.shape).toEqual([8, 8]);
    });
  });

  test("loads SDXL text-time UNet addition embedding weights", async () => {
    await withTempDirectory("mlxts-sd-unet-text-time-weights-", async (directory) => {
      const config = tinyUNetConfig({
        additionEmbedType: "text_time",
        additionTimeEmbedDim: 4,
        projectionClassEmbeddingsInputDim: 12,
      });
      using source = new StableDiffusionUNet2DConditionModel(config);
      const checkpointPath = join(directory, "diffusion_pytorch_model.safetensors");
      await writeCheckpoint(
        checkpointPath,
        checkpointTensorsForUNetParameters(source.parameters()),
      );
      using target = new StableDiffusionUNet2DConditionModel(config);

      const result = await loadStableDiffusionUNetWeights(target, unetComponent([checkpointPath]));

      expect(result.unexpectedWeights).toEqual([]);
      expect(target.addEmbedding?.linear1.weight.shape).toEqual([32, 12]);
      expect(target.addEmbedding?.linear2.weight.shape).toEqual([32, 32]);
    });
  });

  test("loads UNet shards from a Diffusers safetensors index", async () => {
    await withTempDirectory("mlxts-sd-unet-index-", async (directory) => {
      using source = new StableDiffusionUNet2DConditionModel(tinyUNetConfig());
      const entries = checkpointTensorsForUNetParameters(source.parameters());
      const { firstShard, secondShard, weightMap } = splitIndexedShards(entries);

      await saveSafetensors(
        firstShard,
        join(directory, "diffusion_pytorch_model-00001-of-00002.safetensors"),
      );
      await saveSafetensors(
        secondShard,
        join(directory, "diffusion_pytorch_model-00002-of-00002.safetensors"),
      );
      freeTensors(entries);
      const indexPath = join(directory, "diffusion_pytorch_model.safetensors.index.json");
      writeJson(indexPath, { weight_map: weightMap });

      using target = new StableDiffusionUNet2DConditionModel(tinyUNetConfig());
      const result = await loadStableDiffusionUNetWeights(target, unetComponent([indexPath]));

      expect(result.shardCount).toBe(2);
      expect(result.unexpectedWeights).toEqual([]);
    });
  });

  test("loads a UNet directly from an inspected Stable Diffusion snapshot manifest", async () => {
    await withTempDirectory("mlxts-sd-unet-manifest-", async (directory) => {
      writeSnapshotMetadata(directory);
      using source = new StableDiffusionUNet2DConditionModel(tinyUNetConfig());
      await writeCheckpoint(
        join(directory, "unet", "diffusion_pytorch_model.safetensors"),
        checkpointTensorsForUNetParameters(source.parameters()),
      );

      const manifest = await loadDiffusionSnapshotManifest(directory);
      using model = await loadStableDiffusionUNetFromSnapshot(manifest);

      expect(model.convIn.weight.shape).toEqual([8, 3, 3, 4]);
      expect(
        model.midBlock.attention.transformerBlocks[0]?.attention2.keyProjection.weight.shape,
      ).toEqual([8, 4]);
    });
  });

  test("rejects missing, mismatched, and strict unexpected UNet weights", async () => {
    await withTempDirectory("mlxts-sd-unet-invalid-", async (directory) => {
      using source = new StableDiffusionUNet2DConditionModel(tinyUNetConfig());
      const missingPath = join(directory, "missing.safetensors");
      await writeCheckpoint(
        missingPath,
        checkpointTensorsForUNetParameters(source.parameters(), {
          omitPath: "convOut.bias",
        }),
      );
      using missingTarget = new StableDiffusionUNet2DConditionModel(tinyUNetConfig());
      await expect(
        loadStableDiffusionUNetWeights(missingTarget, unetComponent([missingPath])),
      ).rejects.toThrow(DiffusionMissingWeightsError);

      const mismatchPath = join(directory, "mismatch.safetensors");
      await writeCheckpoint(
        mismatchPath,
        checkpointTensorsForUNetParameters(source.parameters(), {
          mismatchPath: "downBlocks.0.attentions.0.projectionIn.weight",
        }),
      );
      using mismatchTarget = new StableDiffusionUNet2DConditionModel(tinyUNetConfig());
      await expect(
        loadStableDiffusionUNetWeights(mismatchTarget, unetComponent([mismatchPath])),
      ).rejects.toThrow(DiffusionWeightMismatchError);

      const unexpectedPath = join(directory, "unexpected.safetensors");
      await writeCheckpoint(
        unexpectedPath,
        checkpointTensorsForUNetParameters(source.parameters(), {
          extra: { "unet.extra.weight": zeros([1]) },
        }),
      );
      using unexpectedTarget = new StableDiffusionUNet2DConditionModel(tinyUNetConfig());
      await expect(
        loadStableDiffusionUNetWeights(unexpectedTarget, unetComponent([unexpectedPath]), {
          strictUnexpectedWeights: true,
        }),
      ).rejects.toThrow("unexpected unmapped weights");
    });
  });
});
