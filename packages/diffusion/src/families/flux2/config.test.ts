import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { DiffusionConfigError } from "../../errors";
import { loadDiffusionSnapshotManifest } from "../../pretrained/snapshot-manifest";

import {
  loadFlux2KleinComponentConfigs,
  parseFlux2KleinAutoencoderConfig,
  parseFlux2KleinTransformerConfig,
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
  configFiles: readonly string[],
  weightFiles: readonly string[] = [],
): void {
  const directory = join(snapshot, component);
  mkdirSync(directory, { recursive: true });
  for (const file of configFiles) {
    writeFileSync(join(directory, file), file.endsWith(".json") ? "{}" : "");
  }
  for (const file of weightFiles) {
    writeFileSync(join(directory, file), "");
  }
}

function flux2TransformerConfig(): Record<string, unknown> {
  return {
    _class_name: "Flux2Transformer2DModel",
    _diffusers_version: "0.37.0.dev0",
    _name_or_path: "/snapshots/flux2-klein/transformer",
    attention_head_dim: 128,
    axes_dims_rope: [32, 32, 32, 32],
    eps: 1e-6,
    guidance_embeds: false,
    in_channels: 128,
    joint_attention_dim: 7680,
    mlp_ratio: 3,
    num_attention_heads: 24,
    num_layers: 5,
    num_single_layers: 20,
    out_channels: null,
    patch_size: 1,
    rope_theta: 2000,
    timestep_guidance_channels: 256,
  };
}

function flux2VaeConfig(): Record<string, unknown> {
  return {
    _class_name: "AutoencoderKLFlux2",
    _diffusers_version: "0.37.0.dev0",
    _name_or_path: "black-forest-labs/FLUX.2-dev",
    act_fn: "silu",
    batch_norm_eps: 0.0001,
    batch_norm_momentum: 0.1,
    block_out_channels: [128, 256, 512, 512],
    down_block_types: [
      "DownEncoderBlock2D",
      "DownEncoderBlock2D",
      "DownEncoderBlock2D",
      "DownEncoderBlock2D",
    ],
    force_upcast: true,
    in_channels: 3,
    latent_channels: 32,
    layers_per_block: 2,
    mid_block_add_attention: true,
    norm_num_groups: 32,
    out_channels: 3,
    patch_size: [2, 2],
    sample_size: 1024,
    up_block_types: [
      "UpDecoderBlock2D",
      "UpDecoderBlock2D",
      "UpDecoderBlock2D",
      "UpDecoderBlock2D",
    ],
    use_post_quant_conv: true,
    use_quant_conv: true,
  };
}

function writeFlux2KleinSnapshot(
  snapshot: string,
  transformerConfig: Record<string, unknown> = flux2TransformerConfig(),
  vaeConfig: Record<string, unknown> = flux2VaeConfig(),
): void {
  writeJson(join(snapshot, "model_index.json"), {
    _class_name: "Flux2KleinPipeline",
    _diffusers_version: "0.37.0.dev0",
    is_distilled: true,
    scheduler: ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    text_encoder: ["transformers", "Qwen3ForCausalLM"],
    tokenizer: ["transformers", "Qwen2TokenizerFast"],
    transformer: ["diffusers", "Flux2Transformer2DModel"],
    vae: ["diffusers", "AutoencoderKLFlux2"],
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
    base_image_seq_len: 256,
    base_shift: 0.5,
    max_image_seq_len: 4096,
    max_shift: 1.15,
    num_train_timesteps: 1000,
    shift: 3,
    time_shift_type: "exponential",
    use_dynamic_shifting: true,
  });
  writeComponentFiles(snapshot, "text_encoder", ["config.json"], ["model.safetensors"]);
  writeComponentFiles(snapshot, "tokenizer", ["tokenizer.json"]);
}

describe("FLUX.2 Klein component config parsing", () => {
  test("parses FLUX.2 Klein transformer config into package-owned shape", () => {
    const parsed = parseFlux2KleinTransformerConfig(flux2TransformerConfig());

    expect(parsed.patchSize).toBe(1);
    expect(parsed.inChannels).toBe(128);
    expect(parsed.latentChannels).toBe(32);
    expect(parsed.outChannels).toBe(128);
    expect(parsed.numLayers).toBe(5);
    expect(parsed.numSingleLayers).toBe(20);
    expect(parsed.attentionHeadDim).toBe(128);
    expect(parsed.numAttentionHeads).toBe(24);
    expect(parsed.hiddenSize).toBe(3072);
    expect(parsed.mlpRatio).toBe(3);
    expect(parsed.jointAttentionDim).toBe(7680);
    expect(parsed.timestepGuidanceChannels).toBe(256);
    expect(parsed.axesDimsRope).toEqual([32, 32, 32, 32]);
    expect(parsed.ropeTheta).toBe(2000);
    expect(parsed.normEps).toBe(1e-6);
    expect(parsed.guidanceEmbeds).toBe(false);
  });

  test("parses FLUX.2 Klein AutoencoderKLFlux2 config into package-owned shape", () => {
    const parsed = parseFlux2KleinAutoencoderConfig(flux2VaeConfig());

    expect(parsed.inChannels).toBe(3);
    expect(parsed.outChannels).toBe(3);
    expect(parsed.latentChannels).toBe(32);
    expect(parsed.latentChannelsOut).toBe(64);
    expect(parsed.packedLatentChannels).toBe(128);
    expect(parsed.useQuantConv).toBe(true);
    expect(parsed.usePostQuantConv).toBe(true);
    expect(parsed.blockOutChannels).toEqual([128, 256, 512, 512]);
    expect(parsed.decoderBlockOutChannels).toBeNull();
    expect(parsed.layersPerBlock).toBe(2);
    expect(parsed.normNumGroups).toBe(32);
    expect(parsed.forceUpcast).toBe(true);
    expect(parsed.midBlockAddAttention).toBe(true);
    expect(parsed.batchNormEps).toBe(0.0001);
    expect(parsed.batchNormMomentum).toBe(0.1);
    expect(parsed.patchSize).toEqual([2, 2]);
    expect(parsed.sampleSize).toBe(1024);
    expect(parsed.vaeScaleFactor).toBe(8);
  });

  test("loads FLUX.2 Klein configs from an inspected snapshot manifest", async () => {
    await withTempDirectory("mlxts-flux2-klein-config-", async (directory) => {
      writeFlux2KleinSnapshot(directory);

      const manifest = await loadDiffusionSnapshotManifest(directory);
      const configs = await loadFlux2KleinComponentConfigs(manifest);

      expect(manifest.modelIndex.kind).toBe("flux2-klein");
      expect(configs.pipelineKind).toBe("flux2-klein");
      expect(configs.isDistilled).toBe(true);
      expect(configs.vae.patchSize).toEqual([2, 2]);
      expect(configs.transformer.hiddenSize).toBe(3072);
      expect(configs.transformer.guidanceEmbeds).toBe(false);
    });
  });

  test("rejects snapshots whose VAE packed latent channels do not match the transformer", async () => {
    await withTempDirectory("mlxts-flux2-klein-mismatch-", async (directory) => {
      writeFlux2KleinSnapshot(directory, {
        ...flux2TransformerConfig(),
        in_channels: 64,
      });

      const manifest = await loadDiffusionSnapshotManifest(directory);
      await expect(loadFlux2KleinComponentConfigs(manifest)).rejects.toThrow(
        "packed latent channels",
      );
    });
  });

  test("rejects unsupported FLUX.2 Klein transformer config shapes", () => {
    expect(() => parseFlux2KleinTransformerConfig(null)).toThrow("JSON object");
    expect(() =>
      parseFlux2KleinTransformerConfig({
        ...flux2TransformerConfig(),
        _class_name: "FluxTransformer2DModel",
      }),
    ).toThrow(DiffusionConfigError);
    expect(() =>
      parseFlux2KleinTransformerConfig({
        ...flux2TransformerConfig(),
        axes_dims_rope: [32, 32, 32],
      }),
    ).toThrow("axes_dims_rope");
    expect(() =>
      parseFlux2KleinTransformerConfig({
        ...flux2TransformerConfig(),
        axes_dims_rope: [32, 32, 32, 16],
      }),
    ).toThrow("sum");
    expect(() =>
      parseFlux2KleinTransformerConfig({
        ...flux2TransformerConfig(),
        in_channels: 130,
      }),
    ).toThrow("divisible");
    expect(() =>
      parseFlux2KleinTransformerConfig({
        ...flux2TransformerConfig(),
        guidance_embeds: "false",
      }),
    ).toThrow("guidance_embeds");
    expect(() =>
      parseFlux2KleinTransformerConfig({
        ...flux2TransformerConfig(),
        pooled_projection_dim: 768,
      }),
    ).toThrow("pooled_projection_dim");
  });

  test("rejects unsupported FLUX.2 Klein VAE config shapes", () => {
    expect(() => parseFlux2KleinAutoencoderConfig([])).toThrow("JSON object");
    expect(() =>
      parseFlux2KleinAutoencoderConfig({
        ...flux2VaeConfig(),
        _class_name: "AutoencoderKL",
      }),
    ).toThrow(DiffusionConfigError);
    expect(() =>
      parseFlux2KleinAutoencoderConfig({
        ...flux2VaeConfig(),
        act_fn: "gelu",
      }),
    ).toThrow("act_fn");
    expect(() =>
      parseFlux2KleinAutoencoderConfig({
        ...flux2VaeConfig(),
        patch_size: [1, 1],
      }),
    ).toThrow("patch_size");
    expect(() =>
      parseFlux2KleinAutoencoderConfig({
        ...flux2VaeConfig(),
        decoder_block_out_channels: [128, 256],
      }),
    ).toThrow("decoder_block_out_channels");
    expect(() =>
      parseFlux2KleinAutoencoderConfig({
        ...flux2VaeConfig(),
        block_out_channels: [128, 256, 512],
      }),
    ).toThrow("down_block_types");
    expect(() =>
      parseFlux2KleinAutoencoderConfig({
        ...flux2VaeConfig(),
        batch_norm_eps: 0,
      }),
    ).toThrow("batch_norm_eps");
    expect(() =>
      parseFlux2KleinAutoencoderConfig({
        ...flux2VaeConfig(),
        scaling_factor: 1,
      }),
    ).toThrow("scaling_factor");
  });
});
