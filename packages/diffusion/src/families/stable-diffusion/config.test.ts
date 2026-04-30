import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { DiffusionConfigError } from "../../errors";
import type { DiffusionSnapshotManifest } from "../../pretrained/snapshot-manifest";
import { loadDiffusionSnapshotManifest } from "../../pretrained/snapshot-manifest";

import {
  loadStableDiffusionComponentConfigs,
  parseStableDiffusionAutoencoderConfig,
  parseStableDiffusionUNetConfig,
} from "./config";

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

function writeComponentFiles(
  snapshot: string,
  component: string,
  config: unknown,
  weightFile?: string,
): void {
  const directory = join(snapshot, component);
  mkdirSync(directory, { recursive: true });
  writeJson(join(directory, "config.json"), config);
  if (weightFile !== undefined) {
    writeFileSync(join(directory, weightFile), "");
  }
}

function vaeConfig(): Record<string, unknown> {
  return {
    _class_name: "AutoencoderKL",
    in_channels: 3,
    out_channels: 3,
    down_block_types: [
      "DownEncoderBlock2D",
      "DownEncoderBlock2D",
      "DownEncoderBlock2D",
      "DownEncoderBlock2D",
    ],
    up_block_types: [
      "UpDecoderBlock2D",
      "UpDecoderBlock2D",
      "UpDecoderBlock2D",
      "UpDecoderBlock2D",
    ],
    block_out_channels: [128, 256, 512, 512],
    layers_per_block: 2,
    latent_channels: 4,
    norm_num_groups: 32,
    sample_size: 512,
    scaling_factor: 0.18215,
    force_upcast: true,
  };
}

function unetConfig(): Record<string, unknown> {
  return {
    _class_name: "UNet2DConditionModel",
    sample_size: 64,
    in_channels: 4,
    out_channels: 4,
    down_block_types: [
      "CrossAttnDownBlock2D",
      "CrossAttnDownBlock2D",
      "CrossAttnDownBlock2D",
      "DownBlock2D",
    ],
    up_block_types: ["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
    block_out_channels: [320, 640, 1280, 1280],
    layers_per_block: 2,
    attention_head_dim: 8,
    cross_attention_dim: 768,
    norm_num_groups: 32,
    norm_eps: 0.00001,
    flip_sin_to_cos: true,
    freq_shift: 0,
    use_linear_projection: false,
    upcast_attention: false,
  };
}

function sdxlUnetConfig(): Record<string, unknown> {
  return {
    ...unetConfig(),
    sample_size: [128, 128],
    layers_per_block: [2, 2, 2, 2],
    transformer_layers_per_block: [1, 2, 10, 10],
    attention_head_dim: [5, 10, 20, 20],
    cross_attention_dim: [2048, 2048, 2048, 2048],
    addition_embed_type: "text_time",
    addition_time_embed_dim: 256,
    projection_class_embeddings_input_dim: 2816,
    use_linear_projection: true,
  };
}

function writeStableDiffusionSnapshot(snapshot: string, unet: Record<string, unknown>): void {
  writeJson(join(snapshot, "model_index.json"), {
    _class_name: "StableDiffusionPipeline",
    vae: ["diffusers", "AutoencoderKL"],
    text_encoder: ["transformers", "CLIPTextModel"],
    tokenizer: ["transformers", "CLIPTokenizer"],
    unet: ["diffusers", "UNet2DConditionModel"],
    scheduler: ["diffusers", "DDIMScheduler"],
    safety_checker: [null, null],
    requires_safety_checker: false,
  });
  writeComponentFiles(snapshot, "vae", vaeConfig(), "diffusion_pytorch_model.safetensors");
  writeComponentFiles(snapshot, "text_encoder", { hidden_size: 768 }, "model.safetensors");
  const tokenizerDirectory = join(snapshot, "tokenizer");
  mkdirSync(tokenizerDirectory, { recursive: true });
  writeJson(join(tokenizerDirectory, "vocab.json"), {});
  writeFileSync(join(tokenizerDirectory, "merges.txt"), "");
  writeComponentFiles(snapshot, "unet", unet, "diffusion_pytorch_model.safetensors");
  const schedulerDirectory = join(snapshot, "scheduler");
  mkdirSync(schedulerDirectory, { recursive: true });
  writeJson(join(schedulerDirectory, "scheduler_config.json"), {
    _class_name: "DDIMScheduler",
    beta_schedule: "linear",
    beta_start: 0.1,
    beta_end: 0.2,
    num_train_timesteps: 4,
  });
}

describe("Stable Diffusion component config parsing", () => {
  test("parses AutoencoderKL config into package-owned shape", () => {
    const parsed = parseStableDiffusionAutoencoderConfig(vaeConfig());

    expect(parsed.inChannels).toBe(3);
    expect(parsed.outChannels).toBe(3);
    expect(parsed.latentChannels).toBe(4);
    expect(parsed.latentChannelsOut).toBe(8);
    expect(parsed.blockOutChannels).toEqual([128, 256, 512, 512]);
    expect(parsed.downBlockTypes).toEqual([
      "DownEncoderBlock2D",
      "DownEncoderBlock2D",
      "DownEncoderBlock2D",
      "DownEncoderBlock2D",
    ]);
    expect(parsed.scalingFactor).toBe(0.18215);
    expect(parsed.forceUpcast).toBe(true);
  });

  test("parses UNet2DConditionModel config and normalizes scalar block fields", () => {
    const parsed = parseStableDiffusionUNetConfig(unetConfig());

    expect(parsed.sampleSize).toBe(64);
    expect(parsed.blockOutChannels).toEqual([320, 640, 1280, 1280]);
    expect(parsed.layersPerBlock).toEqual([2, 2, 2, 2]);
    expect(parsed.transformerLayersPerBlock).toEqual([1, 1, 1, 1]);
    expect(parsed.numAttentionHeads).toEqual([8, 8, 8, 8]);
    expect(parsed.crossAttentionDim).toEqual([768, 768, 768, 768]);
    expect(parsed.additionEmbedType).toBeNull();
    expect(parsed.useLinearProjection).toBe(false);
  });

  test("parses SDXL text-time conditioning fields without constructing a model", () => {
    const parsed = parseStableDiffusionUNetConfig(sdxlUnetConfig());

    expect(parsed.sampleSize).toEqual([128, 128]);
    expect(parsed.transformerLayersPerBlock).toEqual([1, 2, 10, 10]);
    expect(parsed.numAttentionHeads).toEqual([5, 10, 20, 20]);
    expect(parsed.crossAttentionDim).toEqual([2048, 2048, 2048, 2048]);
    expect(parsed.additionEmbedType).toBe("text_time");
    expect(parsed.additionTimeEmbedDim).toBe(256);
    expect(parsed.projectionClassEmbeddingsInputDim).toBe(2816);
    expect(parsed.useLinearProjection).toBe(true);
  });

  test("loads VAE and UNet configs from an inspected local snapshot manifest", async () => {
    await withTempDirectory("mlxts-stable-diffusion-config-", async (directory) => {
      writeStableDiffusionSnapshot(directory, unetConfig());

      const manifest = await loadDiffusionSnapshotManifest(directory);
      const configs = await loadStableDiffusionComponentConfigs(manifest);

      expect(configs.pipelineKind).toBe("stable-diffusion");
      expect(configs.vae.latentChannels).toBe(4);
      expect(configs.unet.inChannels).toBe(4);
      expect(configs.unet.crossAttentionDim).toEqual([768, 768, 768, 768]);
    });
  });

  test("rejects unsupported VAE semantics that would alter latent math", () => {
    expect(() =>
      parseStableDiffusionAutoencoderConfig({
        ...vaeConfig(),
        down_block_types: [
          "AttnDownEncoderBlock2D",
          "DownEncoderBlock2D",
          "DownEncoderBlock2D",
          "DownEncoderBlock2D",
        ],
      }),
    ).toThrow("down_block_types");
    expect(() =>
      parseStableDiffusionAutoencoderConfig({
        ...vaeConfig(),
        shift_factor: 0.1,
      }),
    ).toThrow("shift_factor");
    expect(() =>
      parseStableDiffusionAutoencoderConfig({
        ...vaeConfig(),
        use_quant_conv: false,
      }),
    ).toThrow("use_quant_conv");
  });

  test("rejects unsupported UNet semantics that need explicit model work", () => {
    expect(() =>
      parseStableDiffusionUNetConfig({
        ...unetConfig(),
        transformer_layers_per_block: [[1], [1], [1], [1]],
      }),
    ).toThrow("transformer_layers_per_block");
    expect(() =>
      parseStableDiffusionUNetConfig({
        ...unetConfig(),
        addition_embed_type: "text_time",
      }),
    ).toThrow("text_time");
    expect(() =>
      parseStableDiffusionUNetConfig({
        ...unetConfig(),
        class_embed_type: "timestep",
      }),
    ).toThrow("class_embed_type");
    expect(() =>
      parseStableDiffusionUNetConfig({
        ...unetConfig(),
        mid_block_type: "UNetMidBlock2DSimpleCrossAttn",
      }),
    ).toThrow("mid_block_type");
    expect(() =>
      parseStableDiffusionUNetConfig({
        ...unetConfig(),
        attention_type: "gated",
      }),
    ).toThrow("attention_type");
    expect(() =>
      parseStableDiffusionUNetConfig({
        ...unetConfig(),
        dropout: 0.1,
      }),
    ).toThrow("dropout");
    expect(() =>
      parseStableDiffusionUNetConfig({
        ...unetConfig(),
        downsample_padding: 0,
      }),
    ).toThrow("downsample_padding");
    expect(() =>
      parseStableDiffusionUNetConfig({
        ...unetConfig(),
        mid_block_scale_factor: 0.5,
      }),
    ).toThrow("mid_block_scale_factor");
    expect(() =>
      parseStableDiffusionUNetConfig({
        ...unetConfig(),
        time_embedding_act_fn: "silu",
      }),
    ).toThrow("time_embedding_act_fn");
    expect(() =>
      parseStableDiffusionUNetConfig({
        ...unetConfig(),
        timestep_post_act: "silu",
      }),
    ).toThrow("timestep_post_act");
  });

  test("rejects malformed scalar and repeated config fields", () => {
    expect(() => parseStableDiffusionAutoencoderConfig(null)).toThrow("JSON object");
    expect(() =>
      parseStableDiffusionAutoencoderConfig({
        ...vaeConfig(),
        _class_name: "AutoencoderTiny",
      }),
    ).toThrow("_class_name");
    expect(() =>
      parseStableDiffusionAutoencoderConfig({
        ...vaeConfig(),
        block_out_channels: [],
      }),
    ).toThrow("block_out_channels");
    expect(() =>
      parseStableDiffusionAutoencoderConfig({
        ...vaeConfig(),
        block_out_channels: [128, "wide", 512, 512],
      }),
    ).toThrow("block_out_channels");
    expect(() =>
      parseStableDiffusionAutoencoderConfig({
        ...vaeConfig(),
        sample_size: [64, "wide"],
      }),
    ).toThrow("sample_size");
    expect(() =>
      parseStableDiffusionAutoencoderConfig({
        ...vaeConfig(),
        scaling_factor: "0.18215",
      }),
    ).toThrow("scaling_factor");
    expect(() =>
      parseStableDiffusionAutoencoderConfig({
        ...vaeConfig(),
        force_upcast: "true",
      }),
    ).toThrow("force_upcast");
    expect(() =>
      parseStableDiffusionAutoencoderConfig({
        ...vaeConfig(),
        up_block_types: [
          "UpDecoderBlock2D",
          "UpDecoderBlock2D",
          "UpDecoderBlock2D",
          "AttnUpDecoderBlock2D",
        ],
      }),
    ).toThrow("up_block_types");

    expect(() =>
      parseStableDiffusionUNetConfig({
        ...unetConfig(),
        conv_in_kernel: 0,
      }),
    ).toThrow("conv_in_kernel");
    expect(() =>
      parseStableDiffusionUNetConfig({
        ...unetConfig(),
        layers_per_block: [2, 2],
      }),
    ).toThrow("layers_per_block");
    expect(() =>
      parseStableDiffusionUNetConfig({
        ...unetConfig(),
        layers_per_block: [2, "bad", 2, 2],
      }),
    ).toThrow("layers_per_block");
    expect(() =>
      parseStableDiffusionUNetConfig({
        ...unetConfig(),
        projection_class_embeddings_input_dim: "2816",
      }),
    ).toThrow("projection_class_embeddings_input_dim");
    expect(() =>
      parseStableDiffusionUNetConfig({
        ...unetConfig(),
        only_cross_attention: true,
      }),
    ).toThrow("only_cross_attention");
    expect(() =>
      parseStableDiffusionUNetConfig({
        ...unetConfig(),
        only_cross_attention: [false, true, false, false],
      }),
    ).toThrow("only_cross_attention");
    expect(() =>
      parseStableDiffusionUNetConfig({
        ...unetConfig(),
        center_input_sample: "false",
      }),
    ).toThrow("center_input_sample");
    expect(() =>
      parseStableDiffusionUNetConfig({
        ...unetConfig(),
        addition_embed_type: "text_image",
      }),
    ).toThrow("addition_embed_type");
    expect(() =>
      parseStableDiffusionUNetConfig({
        ...unetConfig(),
        down_block_types: [
          "CrossAttnDownBlock2D",
          "CrossAttnDownBlock2D",
          "CrossAttnDownBlock2D",
          "SkipDownBlock2D",
        ],
      }),
    ).toThrow("down_block_types");
    expect(() =>
      parseStableDiffusionUNetConfig({
        ...unetConfig(),
        up_block_types: ["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "SkipUpBlock2D"],
      }),
    ).toThrow("up_block_types");
    expect(() =>
      parseStableDiffusionUNetConfig({
        ...unetConfig(),
        resnet_out_scale_factor: 0.5,
      }),
    ).toThrow("resnet_out_scale_factor");

    const parsed = parseStableDiffusionUNetConfig({
      ...unetConfig(),
      num_attention_heads: [5, 10, 20, 20],
      transformer_layers_per_block: undefined,
      sample_size: undefined,
    });
    expect(parsed.numAttentionHeads).toEqual([5, 10, 20, 20]);
    expect(parsed.transformerLayersPerBlock).toEqual([1, 1, 1, 1]);
    expect(parsed.sampleSize).toBeUndefined();
  });

  test("rejects non-Stable-Diffusion manifests and malformed config files", async () => {
    await withTempDirectory("mlxts-stable-diffusion-flux-config-", async (directory) => {
      writeJson(join(directory, "model_index.json"), {
        _class_name: "FluxPipeline",
        transformer: ["diffusers", "FluxTransformer2DModel"],
        vae: ["diffusers", "AutoencoderKL"],
        scheduler: ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        text_encoder: ["transformers", "CLIPTextModel"],
        tokenizer: ["transformers", "CLIPTokenizer"],
        text_encoder_2: ["transformers", "T5EncoderModel"],
        tokenizer_2: ["transformers", "T5TokenizerFast"],
      });
      const manifest: DiffusionSnapshotManifest = {
        directory,
        modelIndexPath: join(directory, "model_index.json"),
        modelIndex: {
          kind: "flux",
          className: "FluxPipeline",
          components: [],
          pipelineConfig: {},
          rawConfig: {},
        },
        schedulerConfig: {
          kind: "euler",
          className: "EulerDiscreteScheduler",
          config: {},
        },
        components: [],
      };

      await expect(loadStableDiffusionComponentConfigs(manifest)).rejects.toThrow(
        DiffusionConfigError,
      );
    });

    await withTempDirectory("mlxts-stable-diffusion-invalid-config-", async (directory) => {
      writeStableDiffusionSnapshot(directory, unetConfig());
      writeFileSync(join(directory, "unet", "config.json"), "{");
      const manifest = await loadDiffusionSnapshotManifest(directory);
      await expect(loadStableDiffusionComponentConfigs(manifest)).rejects.toThrow("valid JSON");
    });

    await withTempDirectory("mlxts-stable-diffusion-missing-config-", async (directory) => {
      writeStableDiffusionSnapshot(directory, unetConfig());
      const manifest = await loadDiffusionSnapshotManifest(directory);
      rmSync(join(directory, "unet", "config.json"), { force: true });
      await expect(loadStableDiffusionComponentConfigs(manifest)).rejects.toThrow("unet");
    });
  });
});
