import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { DiffusionConfigError } from "../../errors";
import { loadDiffusionSnapshotManifest } from "../../pretrained/snapshot-manifest";

import {
  loadQwenImageComponentConfigs,
  parseQwenImageAutoencoderConfig,
  parseQwenImageTransformerConfig,
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
    writeJson(join(directory, file), {});
  }
  for (const file of weightFiles) {
    writeFileSync(join(directory, file), "");
  }
}

function qwenImageTransformerConfig(): Record<string, unknown> {
  return {
    _class_name: "QwenImageTransformer2DModel",
    _diffusers_version: "0.36.0.dev0",
    attention_head_dim: 128,
    axes_dims_rope: [16, 56, 56],
    guidance_embeds: false,
    in_channels: 64,
    joint_attention_dim: 3584,
    num_attention_heads: 24,
    num_layers: 60,
    out_channels: 16,
    patch_size: 2,
    pooled_projection_dim: 768,
    zero_cond_t: true,
  };
}

function qwenImageVaeConfig(): Record<string, unknown> {
  return {
    _class_name: "AutoencoderKLQwenImage",
    _diffusers_version: "0.36.0.dev0",
    attn_scales: [],
    base_dim: 96,
    dim_mult: [1, 2, 4, 4],
    dropout: 0,
    latents_mean: [
      -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517,
      -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
    ],
    latents_std: [
      2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652,
      1.5579, 1.6382, 1.1253, 2.8251, 1.916,
    ],
    num_res_blocks: 2,
    temperal_downsample: [false, true, true],
    z_dim: 16,
  };
}

function writeQwenImageSnapshot(snapshot: string): void {
  writeJson(join(snapshot, "model_index.json"), {
    _class_name: "QwenImagePipeline",
    _diffusers_version: "0.36.0.dev0",
    transformer: ["diffusers", "QwenImageTransformer2DModel"],
    vae: ["diffusers", "AutoencoderKLQwenImage"],
    scheduler: ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    text_encoder: ["transformers", "Qwen2_5_VLForConditionalGeneration"],
    tokenizer: ["transformers", "Qwen2Tokenizer"],
  });
  writeComponentFiles(
    snapshot,
    "transformer",
    ["config.json"],
    ["diffusion_pytorch_model.safetensors"],
  );
  writeJson(join(snapshot, "transformer", "config.json"), qwenImageTransformerConfig());
  writeComponentFiles(snapshot, "vae", ["config.json"], ["diffusion_pytorch_model.safetensors"]);
  writeJson(join(snapshot, "vae", "config.json"), qwenImageVaeConfig());
  writeComponentFiles(snapshot, "scheduler", ["scheduler_config.json"]);
  writeJson(join(snapshot, "scheduler", "scheduler_config.json"), {
    _class_name: "FlowMatchEulerDiscreteScheduler",
    shift_terminal: 0.02,
  });
  writeComponentFiles(snapshot, "text_encoder", ["config.json"], ["model.safetensors"]);
  writeComponentFiles(snapshot, "tokenizer", ["tokenizer.json"]);
}

describe("Qwen-Image component config parsing", () => {
  test("parses Qwen-Image transformer config into package-owned shape", () => {
    const parsed = parseQwenImageTransformerConfig(qwenImageTransformerConfig());

    expect(parsed.patchSize).toBe(2);
    expect(parsed.inChannels).toBe(64);
    expect(parsed.outChannels).toBe(16);
    expect(parsed.latentChannels).toBe(16);
    expect(parsed.packedLatentChannels).toBe(64);
    expect(parsed.numLayers).toBe(60);
    expect(parsed.attentionHeadDim).toBe(128);
    expect(parsed.numAttentionHeads).toBe(24);
    expect(parsed.hiddenSize).toBe(3072);
    expect(parsed.jointAttentionDim).toBe(3584);
    expect(parsed.guidanceEmbeds).toBe(false);
    expect(parsed.axesDimsRope).toEqual([16, 56, 56]);
    expect(parsed.ropeTheta).toBe(10000);
    expect(parsed.zeroCondT).toBe(true);
    expect(parsed.useAdditionalTCond).toBe(false);
    expect(parsed.useLayer3dRope).toBe(false);
  });

  test("parses Qwen-Image 3D causal VAE config", () => {
    const parsed = parseQwenImageAutoencoderConfig(qwenImageVaeConfig());

    expect(parsed.baseDim).toBe(96);
    expect(parsed.latentChannels).toBe(16);
    expect(parsed.latentChannelsOut).toBe(32);
    expect(parsed.dimMultipliers).toEqual([1, 2, 4, 4]);
    expect(parsed.numResBlocks).toBe(2);
    expect(parsed.attentionScales).toEqual([]);
    expect(parsed.temporalDownsample).toEqual([false, true, true]);
    expect(parsed.temporalUpsample).toEqual([true, true, false]);
    expect(parsed.dropout).toBe(0);
    expect(parsed.inputChannels).toBe(3);
    expect(parsed.latentsMean).toHaveLength(16);
    expect(parsed.latentsStd).toHaveLength(16);
    expect(parsed.spatialCompressionRatio).toBe(8);
  });

  test("loads Qwen-Image configs from an inspected snapshot manifest", async () => {
    await withTempDirectory("mlxts-qwen-image-config-", async (directory) => {
      writeQwenImageSnapshot(directory);

      const manifest = await loadDiffusionSnapshotManifest(directory);
      const configs = await loadQwenImageComponentConfigs(manifest);

      expect(configs.pipelineKind).toBe("qwen-image");
      expect(configs.transformer.jointAttentionDim).toBe(3584);
      expect(configs.vae.spatialCompressionRatio).toBe(8);
    });
  });

  test("rejects unsupported Qwen-Image transformer config shapes", () => {
    expect(() => parseQwenImageTransformerConfig([])).toThrow("JSON object");
    expect(() =>
      parseQwenImageTransformerConfig({
        ...qwenImageTransformerConfig(),
        _class_name: "FluxTransformer2DModel",
      }),
    ).toThrow(DiffusionConfigError);
    expect(() =>
      parseQwenImageTransformerConfig({
        ...qwenImageTransformerConfig(),
        axes_dims_rope: [16, 56],
      }),
    ).toThrow("axes_dims_rope");
    expect(() =>
      parseQwenImageTransformerConfig({
        ...qwenImageTransformerConfig(),
        attention_head_dim: 64,
      }),
    ).toThrow("axes_dims_rope");
    expect(() =>
      parseQwenImageTransformerConfig({
        ...qwenImageTransformerConfig(),
        in_channels: 32,
      }),
    ).toThrow("in_channels");
  });

  test("rejects unsupported Qwen-Image VAE config shapes", () => {
    expect(() =>
      parseQwenImageAutoencoderConfig({
        ...qwenImageVaeConfig(),
        _class_name: "AutoencoderKL",
      }),
    ).toThrow(DiffusionConfigError);
    expect(() =>
      parseQwenImageAutoencoderConfig({
        ...qwenImageVaeConfig(),
        temperal_downsample: [true, true],
      }),
    ).toThrow("temperal_downsample");
    expect(() =>
      parseQwenImageAutoencoderConfig({
        ...qwenImageVaeConfig(),
        latents_mean: [0],
      }),
    ).toThrow("latents_mean");
    expect(() =>
      parseQwenImageAutoencoderConfig({
        ...qwenImageVaeConfig(),
        dropout: "0",
      }),
    ).toThrow("dropout");
  });

  test("rejects loading Qwen-Image configs from other pipeline manifests", async () => {
    await expect(
      loadQwenImageComponentConfigs({
        directory: "/tmp/noop",
        modelIndexPath: "/tmp/noop/model_index.json",
        modelIndex: {
          kind: "flux",
          className: "FluxPipeline",
          components: [],
          pipelineConfig: {},
          rawConfig: {},
        },
        schedulerConfig: {
          kind: "flow-match-euler",
          className: "FlowMatchEulerDiscreteScheduler",
          config: {},
        },
        components: [],
      }),
    ).rejects.toThrow("Qwen-Image");
  });
});
