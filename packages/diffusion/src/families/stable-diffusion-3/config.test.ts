import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { DiffusionConfigError } from "../../errors";
import { loadDiffusionSnapshotManifest } from "../../pretrained/snapshot-manifest";

import {
  loadStableDiffusion3ComponentConfigs,
  parseStableDiffusion3AutoencoderConfig,
  parseStableDiffusion3TransformerConfig,
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
  metadataFiles: readonly string[],
  weightFiles: readonly string[] = [],
): void {
  const directory = join(snapshot, component);
  mkdirSync(directory, { recursive: true });
  for (const file of metadataFiles) {
    writeFileSync(join(directory, file), file.endsWith(".json") ? "{}" : "");
  }
  for (const file of weightFiles) {
    writeFileSync(join(directory, file), "");
  }
}

function sd3TransformerConfig(): Record<string, unknown> {
  return {
    _class_name: "SD3Transformer2DModel",
    _diffusers_version: "0.30.0.dev0",
    attention_head_dim: 8,
    caption_projection_dim: 32,
    in_channels: 4,
    joint_attention_dim: 32,
    num_attention_heads: 4,
    num_layers: 1,
    out_channels: 4,
    patch_size: 1,
    pooled_projection_dim: 64,
    pos_embed_max_size: 96,
    sample_size: 32,
  };
}

function sd35TransformerConfig(): Record<string, unknown> {
  return {
    _class_name: "SD3Transformer2DModel",
    attention_head_dim: 64,
    caption_projection_dim: 2432,
    dual_attention_layers: [0, 1, 2],
    in_channels: 16,
    joint_attention_dim: 4096,
    num_attention_heads: 38,
    num_layers: 38,
    out_channels: null,
    patch_size: 2,
    pooled_projection_dim: 2048,
    pos_embed_max_size: 192,
    qk_norm: "rms_norm",
    sample_size: 128,
  };
}

function sd3VaeConfig(): Record<string, unknown> {
  return {
    _class_name: "AutoencoderKL",
    _diffusers_version: "0.30.0.dev0",
    act_fn: "silu",
    block_out_channels: [4],
    down_block_types: ["DownEncoderBlock2D"],
    force_upcast: true,
    in_channels: 3,
    latent_channels: 4,
    latents_mean: null,
    latents_std: null,
    layers_per_block: 1,
    norm_num_groups: 1,
    out_channels: 3,
    sample_size: 32,
    scaling_factor: 1.5035,
    shift_factor: 0.0609,
    up_block_types: ["UpDecoderBlock2D"],
    use_post_quant_conv: false,
    use_quant_conv: false,
  };
}

function writeStableDiffusion3Snapshot(
  snapshot: string,
  transformerConfig: Record<string, unknown> = sd3TransformerConfig(),
  vaeConfig: Record<string, unknown> = sd3VaeConfig(),
): void {
  writeJson(join(snapshot, "model_index.json"), {
    _class_name: "StableDiffusion3Pipeline",
    _diffusers_version: "0.30.0.dev0",
    scheduler: ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    text_encoder: ["transformers", "CLIPTextModelWithProjection"],
    text_encoder_2: ["transformers", "CLIPTextModelWithProjection"],
    text_encoder_3: ["transformers", "T5EncoderModel"],
    tokenizer: ["transformers", "CLIPTokenizer"],
    tokenizer_2: ["transformers", "CLIPTokenizer"],
    tokenizer_3: ["transformers", "T5TokenizerFast"],
    transformer: ["diffusers", "SD3Transformer2DModel"],
    vae: ["diffusers", "AutoencoderKL"],
  });
  writeComponentFiles(
    snapshot,
    "transformer",
    ["config.json"],
    ["diffusion_pytorch_model.safetensors"],
  );
  writeJson(join(snapshot, "transformer", "config.json"), transformerConfig);
  writeComponentFiles(snapshot, "vae", ["config.json"], ["diffusion_pytorch_model.safetensors"]);
  writeJson(join(snapshot, "vae", "config.json"), vaeConfig);
  writeComponentFiles(snapshot, "scheduler", ["scheduler_config.json"]);
  writeJson(join(snapshot, "scheduler", "scheduler_config.json"), {
    _class_name: "FlowMatchEulerDiscreteScheduler",
    num_train_timesteps: 1000,
    shift: 1,
  });
  writeComponentFiles(snapshot, "text_encoder", ["config.json"], ["model.safetensors"]);
  writeComponentFiles(snapshot, "text_encoder_2", ["config.json"], ["model.safetensors"]);
  writeComponentFiles(snapshot, "text_encoder_3", ["config.json"], ["model.safetensors"]);
  writeComponentFiles(snapshot, "tokenizer", ["vocab.json", "merges.txt"]);
  writeComponentFiles(snapshot, "tokenizer_2", ["vocab.json", "merges.txt"]);
  writeComponentFiles(snapshot, "tokenizer_3", ["spiece.model"]);
}

describe("Stable Diffusion 3 component config parsing", () => {
  test("parses SD3 transformer config into package-owned shape", () => {
    const parsed = parseStableDiffusion3TransformerConfig(sd3TransformerConfig());

    expect(parsed.sampleSize).toBe(32);
    expect(parsed.patchSize).toBe(1);
    expect(parsed.inChannels).toBe(4);
    expect(parsed.outChannels).toBe(4);
    expect(parsed.numLayers).toBe(1);
    expect(parsed.attentionHeadDim).toBe(8);
    expect(parsed.numAttentionHeads).toBe(4);
    expect(parsed.hiddenSize).toBe(32);
    expect(parsed.jointAttentionDim).toBe(32);
    expect(parsed.captionProjectionDim).toBe(32);
    expect(parsed.pooledProjectionDim).toBe(64);
    expect(parsed.posEmbedMaxSize).toBe(96);
    expect(parsed.dualAttentionLayers).toEqual([]);
    expect(parsed.qkNorm).toBeNull();
  });

  test("parses SD3.5 dual-attention transformer config fields", () => {
    const parsed = parseStableDiffusion3TransformerConfig(sd35TransformerConfig());

    expect(parsed.outChannels).toBe(16);
    expect(parsed.hiddenSize).toBe(2432);
    expect(parsed.captionProjectionDim).toBe(2432);
    expect(parsed.dualAttentionLayers).toEqual([0, 1, 2]);
    expect(parsed.qkNorm).toBe("rms_norm");
  });

  test("parses SD3 AutoencoderKL config into package-owned shape", () => {
    const parsed = parseStableDiffusion3AutoencoderConfig(sd3VaeConfig());

    expect(parsed.inChannels).toBe(3);
    expect(parsed.outChannels).toBe(3);
    expect(parsed.latentChannels).toBe(4);
    expect(parsed.latentChannelsOut).toBe(8);
    expect(parsed.useQuantConv).toBe(false);
    expect(parsed.usePostQuantConv).toBe(false);
    expect(parsed.blockOutChannels).toEqual([4]);
    expect(parsed.layersPerBlock).toBe(1);
    expect(parsed.normNumGroups).toBe(1);
    expect(parsed.scalingFactor).toBe(1.5035);
    expect(parsed.shiftFactor).toBe(0.0609);
    expect(parsed.sampleSize).toBe(32);
    expect(parsed.vaeScaleFactor).toBe(1);
  });

  test("loads SD3 configs from an inspected snapshot manifest", async () => {
    await withTempDirectory("mlxts-sd3-config-", async (directory) => {
      writeStableDiffusion3Snapshot(directory);

      const manifest = await loadDiffusionSnapshotManifest(directory);
      const configs = await loadStableDiffusion3ComponentConfigs(manifest);

      expect(manifest.modelIndex.kind).toBe("stable-diffusion-3");
      expect(configs.pipelineKind).toBe("stable-diffusion-3");
      expect(configs.vae.latentChannels).toBe(4);
      expect(configs.transformer.hiddenSize).toBe(32);
      expect(
        manifest.components.find((component) => component.name === "tokenizer_3")?.metadataPaths,
      ).toEqual([join(directory, "tokenizer_3", "spiece.model")]);
    });
  });

  test("rejects snapshots whose VAE latent channels do not match the transformer", async () => {
    await withTempDirectory("mlxts-sd3-mismatch-", async (directory) => {
      writeStableDiffusion3Snapshot(directory, {
        ...sd3TransformerConfig(),
        in_channels: 8,
      });

      const manifest = await loadDiffusionSnapshotManifest(directory);
      await expect(loadStableDiffusion3ComponentConfigs(manifest)).rejects.toThrow(
        "latent channels",
      );
    });
  });

  test("rejects unsupported SD3 transformer config shapes", () => {
    expect(() => parseStableDiffusion3TransformerConfig(null)).toThrow("JSON object");
    expect(() =>
      parseStableDiffusion3TransformerConfig({
        ...sd3TransformerConfig(),
        _class_name: "FluxTransformer2DModel",
      }),
    ).toThrow(DiffusionConfigError);
    expect(() =>
      parseStableDiffusion3TransformerConfig({
        ...sd3TransformerConfig(),
        dual_attention_layers: [1],
      }),
    ).toThrow("less than num_layers");
    expect(() =>
      parseStableDiffusion3TransformerConfig({
        ...sd3TransformerConfig(),
        dual_attention_layers: [0, 0],
      }),
    ).toThrow("duplicates");
    expect(() =>
      parseStableDiffusion3TransformerConfig({
        ...sd3TransformerConfig(),
        qk_norm: "layer_norm",
      }),
    ).toThrow("qk_norm");
    expect(() =>
      parseStableDiffusion3TransformerConfig({
        ...sd3TransformerConfig(),
        axes_dims_rope: [16, 56, 56],
      }),
    ).toThrow("axes_dims_rope");
  });

  test("rejects unsupported SD3 VAE config shapes", () => {
    expect(() => parseStableDiffusion3AutoencoderConfig([])).toThrow("JSON object");
    expect(() =>
      parseStableDiffusion3AutoencoderConfig({
        ...sd3VaeConfig(),
        _class_name: "AutoencoderKLFlux2",
      }),
    ).toThrow(DiffusionConfigError);
    expect(() =>
      parseStableDiffusion3AutoencoderConfig({
        ...sd3VaeConfig(),
        act_fn: "gelu",
      }),
    ).toThrow("act_fn");
    expect(() =>
      parseStableDiffusion3AutoencoderConfig({
        ...sd3VaeConfig(),
        latents_mean: [0],
      }),
    ).toThrow("latents_mean");
    expect(() =>
      parseStableDiffusion3AutoencoderConfig({
        ...sd3VaeConfig(),
        down_block_types: ["CrossAttnDownBlock2D"],
      }),
    ).toThrow("down_block_types");
  });
});
