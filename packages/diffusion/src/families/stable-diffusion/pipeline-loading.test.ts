import { describe, expect, test } from "bun:test";
import { type MxArray, type ParameterTree, saveSafetensors, treeFlatten, zeros } from "@mlxts/core";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { DiffusionMissingWeightsError } from "../../errors";
import { DDIMScheduler } from "../../schedulers/ddim";
import { StableDiffusionAutoencoderKL } from "./autoencoder";
import type { StableDiffusionAutoencoderConfig, StableDiffusionUNetConfig } from "./config";
import { loadStableDiffusionPipelineFromSnapshot } from "./pipeline-loading";
import { StableDiffusionUNet2DConditionModel } from "./unet";

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
    latentChannels: 4,
    latentChannelsOut: 8,
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

function tinyUNetConfig(): StableDiffusionUNetConfig {
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
  };
}

function writeSnapshotMetadata(
  directory: string,
  options: {
    safetyChecker?:
      | readonly [null, null]
      | readonly ["stable_diffusion", "StableDiffusionSafetyChecker"];
    requiresSafetyChecker?: boolean;
  } = {},
): void {
  writeJson(join(directory, "model_index.json"), {
    _class_name: "StableDiffusionPipeline",
    vae: ["diffusers", "AutoencoderKL"],
    text_encoder: ["transformers", "CLIPTextModel"],
    tokenizer: ["transformers", "CLIPTokenizer"],
    unet: ["diffusers", "UNet2DConditionModel"],
    scheduler: ["diffusers", "DDIMScheduler"],
    safety_checker: options.safetyChecker ?? [null, null],
    requires_safety_checker: options.requiresSafetyChecker ?? false,
  });

  const vaeDirectory = join(directory, "vae");
  mkdirSync(vaeDirectory, { recursive: true });
  writeJson(join(vaeDirectory, "config.json"), {
    _class_name: "AutoencoderKL",
    in_channels: 3,
    out_channels: 3,
    down_block_types: ["DownEncoderBlock2D", "DownEncoderBlock2D"],
    up_block_types: ["UpDecoderBlock2D", "UpDecoderBlock2D"],
    block_out_channels: [4, 8],
    layers_per_block: 1,
    latent_channels: 4,
    norm_num_groups: 4,
    scaling_factor: 0.18215,
    force_upcast: true,
  });
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
  writeJson(join(unetDirectory, "config.json"), {
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
  });
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

  if (options.safetyChecker?.[0] === "stable_diffusion") {
    const safetyDirectory = join(directory, "safety_checker");
    mkdirSync(safetyDirectory, { recursive: true });
    writeJson(join(safetyDirectory, "config.json"), { hidden_size: 4 });
    writeFileSync(join(safetyDirectory, "model.safetensors"), "");
  }
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

function checkpointTensorsForParameters(parameters: ParameterTree): Record<string, MxArray> {
  const tensors: Record<string, MxArray> = {};
  for (const [path, parameter] of treeFlatten(parameters)) {
    const internalPath = path.join(".");
    tensors[checkpointNameForPath(internalPath)] = zeros(
      checkpointShapeFor(internalPath, parameter),
    );
  }
  return tensors;
}

function checkpointTensorsForUNetParameters(parameters: ParameterTree): Record<string, MxArray> {
  const tensors: Record<string, MxArray> = {};
  for (const [path, parameter] of treeFlatten(parameters)) {
    const internalPath = path.join(".");
    tensors[unetCheckpointNameForPath(internalPath)] = zeros(
      checkpointShapeFor(internalPath, parameter),
    );
  }
  return tensors;
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

describe("Stable Diffusion pipeline loading", () => {
  test("loads VAE, UNet, scheduler, and parsed configs as one disposable bundle", async () => {
    await withTempDirectory("mlxts-sd-pipeline-load-", async (directory) => {
      writeSnapshotMetadata(directory);
      using vaeSource = new StableDiffusionAutoencoderKL(tinyVaeConfig());
      await writeCheckpoint(
        join(directory, "vae", "diffusion_pytorch_model.safetensors"),
        checkpointTensorsForParameters(vaeSource.parameters()),
      );
      using unetSource = new StableDiffusionUNet2DConditionModel(tinyUNetConfig());
      await writeCheckpoint(
        join(directory, "unet", "diffusion_pytorch_model.safetensors"),
        checkpointTensorsForUNetParameters(unetSource.parameters()),
      );

      using bundle = await loadStableDiffusionPipelineFromSnapshot(directory);

      expect(bundle.snapshotDirectory).toBe(directory);
      expect(bundle.configs.pipelineKind).toBe("stable-diffusion");
      expect(bundle.scheduler).toBeInstanceOf(DDIMScheduler);
      expect(bundle.vae.latentChannels).toBe(4);
      expect(bundle.vae.vaeScaleFactor).toBe(2);
      expect(bundle.unet.convIn.weight.shape).toEqual([8, 3, 3, 4]);
      expect(bundle.manifest.components.map((component) => component.name)).toContain("scheduler");
      using conditioning = zeros([1, 1, 4]);
      using image = bundle.generateImage({
        batchSize: 1,
        height: 16,
        width: 16,
        conditioning: { encoderHiddenStates: conditioning },
        numInferenceSteps: 1,
        evaluateEachStep: false,
      });
      expect(image.shape).toEqual([1, 16, 16, 3]);
    });
  });

  test("surfaces component load failures without returning a partial bundle", async () => {
    await withTempDirectory("mlxts-sd-pipeline-load-invalid-", async (directory) => {
      writeSnapshotMetadata(directory);
      using vaeSource = new StableDiffusionAutoencoderKL(tinyVaeConfig());
      await writeCheckpoint(
        join(directory, "vae", "diffusion_pytorch_model.safetensors"),
        checkpointTensorsForParameters(vaeSource.parameters()),
      );
      await writeCheckpoint(join(directory, "unet", "diffusion_pytorch_model.safetensors"), {});

      await expect(loadStableDiffusionPipelineFromSnapshot(directory)).rejects.toThrow(
        DiffusionMissingWeightsError,
      );
    });
  });

  test("rejects required or enabled safety checker semantics", async () => {
    await withTempDirectory("mlxts-sd-pipeline-load-safety-", async (directory) => {
      writeSnapshotMetadata(directory, { requiresSafetyChecker: true });

      await expect(loadStableDiffusionPipelineFromSnapshot(directory)).rejects.toThrow(
        "requires_safety_checker=false",
      );
    });

    await withTempDirectory("mlxts-sd-pipeline-load-safety-enabled-", async (directory) => {
      writeSnapshotMetadata(directory, {
        safetyChecker: ["stable_diffusion", "StableDiffusionSafetyChecker"],
      });

      await expect(loadStableDiffusionPipelineFromSnapshot(directory)).rejects.toThrow(
        "enabled safety_checker",
      );
    });
  });

  test("guards thin runtime methods after disposal", async () => {
    await withTempDirectory("mlxts-sd-pipeline-load-disposed-", async (directory) => {
      writeSnapshotMetadata(directory);
      using vaeSource = new StableDiffusionAutoencoderKL(tinyVaeConfig());
      await writeCheckpoint(
        join(directory, "vae", "diffusion_pytorch_model.safetensors"),
        checkpointTensorsForParameters(vaeSource.parameters()),
      );
      using unetSource = new StableDiffusionUNet2DConditionModel(tinyUNetConfig());
      await writeCheckpoint(
        join(directory, "unet", "diffusion_pytorch_model.safetensors"),
        checkpointTensorsForUNetParameters(unetSource.parameters()),
      );
      const bundle = await loadStableDiffusionPipelineFromSnapshot(directory);
      bundle[Symbol.dispose]();
      using conditioning = zeros([1, 1, 4]);

      expect(() =>
        bundle.generateImage({
          batchSize: 1,
          height: 16,
          width: 16,
          conditioning: { encoderHiddenStates: conditioning },
          numInferenceSteps: 1,
        }),
      ).toThrow("disposed");
      bundle[Symbol.dispose]();
    });
  });
});
