import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { DiffusionConfigError } from "../../errors";
import {
  type DiffusionSnapshotManifest,
  loadDiffusionSnapshotManifest,
} from "../../pretrained/snapshot-manifest";

import {
  loadZImageComponentConfigs,
  parseZImageAutoencoderConfig,
  parseZImageTransformerConfig,
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

function zImageTransformerConfig(): Record<string, unknown> {
  return {
    _class_name: "ZImageTransformer2DModel",
    _diffusers_version: "0.36.0.dev0",
    all_f_patch_size: [1],
    all_patch_size: [2],
    axes_dims: [32, 48, 48],
    axes_lens: [1536, 512, 512],
    cap_feat_dim: 2560,
    dim: 3840,
    in_channels: 16,
    n_heads: 30,
    n_kv_heads: 30,
    n_layers: 30,
    n_refiner_layers: 2,
    norm_eps: 1e-5,
    qk_norm: true,
    rope_theta: 256,
    t_scale: 1000,
  };
}

function zImageVaeConfig(): Record<string, unknown> {
  return {
    _class_name: "AutoencoderKL",
    _diffusers_version: "0.36.0.dev0",
    act_fn: "silu",
    block_out_channels: [128, 256, 512, 512],
    down_block_types: [
      "DownEncoderBlock2D",
      "DownEncoderBlock2D",
      "DownEncoderBlock2D",
      "DownEncoderBlock2D",
    ],
    force_upcast: true,
    in_channels: 3,
    latent_channels: 16,
    latents_mean: null,
    latents_std: null,
    layers_per_block: 2,
    mid_block_add_attention: true,
    norm_num_groups: 32,
    out_channels: 3,
    sample_size: 1024,
    scaling_factor: 0.3611,
    shift_factor: 0.1159,
    up_block_types: [
      "UpDecoderBlock2D",
      "UpDecoderBlock2D",
      "UpDecoderBlock2D",
      "UpDecoderBlock2D",
    ],
    use_post_quant_conv: false,
    use_quant_conv: false,
  };
}

function writeZImageSnapshot(snapshot: string): void {
  writeJson(join(snapshot, "model_index.json"), {
    _class_name: "ZImagePipeline",
    _diffusers_version: "0.36.0.dev0",
    transformer: ["diffusers", "ZImageTransformer2DModel"],
    vae: ["diffusers", "AutoencoderKL"],
    scheduler: ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    text_encoder: ["transformers", "Qwen3Model"],
    tokenizer: ["transformers", "Qwen2Tokenizer"],
  });
  writeComponentFiles(
    snapshot,
    "transformer",
    ["config.json"],
    ["diffusion_pytorch_model.safetensors"],
  );
  writeJson(join(snapshot, "transformer", "config.json"), zImageTransformerConfig());
  writeComponentFiles(snapshot, "vae", ["config.json"], ["diffusion_pytorch_model.safetensors"]);
  writeJson(join(snapshot, "vae", "config.json"), zImageVaeConfig());
  writeComponentFiles(snapshot, "scheduler", ["scheduler_config.json"]);
  writeJson(join(snapshot, "scheduler", "scheduler_config.json"), {
    _class_name: "FlowMatchEulerDiscreteScheduler",
    num_train_timesteps: 1000,
    shift: 3,
    use_dynamic_shifting: false,
  });
  writeComponentFiles(snapshot, "text_encoder", ["config.json"], ["model.safetensors"]);
  writeComponentFiles(snapshot, "tokenizer", ["tokenizer.json"]);
}

function zImageManifest(
  directory: string,
  components: DiffusionSnapshotManifest["components"],
): DiffusionSnapshotManifest {
  return {
    directory,
    modelIndexPath: join(directory, "model_index.json"),
    modelIndex: {
      kind: "z-image",
      className: "ZImagePipeline",
      components,
      pipelineConfig: {},
      rawConfig: {},
    },
    schedulerConfig: {
      kind: "flow-match-euler",
      className: "FlowMatchEulerDiscreteScheduler",
      config: {},
    },
    components,
  };
}

describe("Z-Image component config parsing", () => {
  test("parses Z-Image transformer config into package-owned shape", () => {
    const parsed = parseZImageTransformerConfig(zImageTransformerConfig());

    expect(parsed.patchGeometries).toEqual([
      { patchSize: 2, framePatchSize: 1, packedLatentChannels: 64 },
    ]);
    expect(parsed.inChannels).toBe(16);
    expect(parsed.outChannels).toBe(16);
    expect(parsed.hiddenSize).toBe(3840);
    expect(parsed.numLayers).toBe(30);
    expect(parsed.numRefinerLayers).toBe(2);
    expect(parsed.numAttentionHeads).toBe(30);
    expect(parsed.numKeyValueHeads).toBe(30);
    expect(parsed.attentionHeadDim).toBe(128);
    expect(parsed.normEps).toBe(1e-5);
    expect(parsed.qkNorm).toBe(true);
    expect(parsed.captionFeatureDim).toBe(2560);
    expect(parsed.siglipFeatureDim).toBeNull();
    expect(parsed.ropeTheta).toBe(256);
    expect(parsed.timestepScale).toBe(1000);
    expect(parsed.sequenceMultiple).toBe(32);
    expect(parsed.latentPadDim).toBe(64);
    expect(parsed.axesDims).toEqual([32, 48, 48]);
    expect(parsed.axesLens).toEqual([1536, 512, 512]);
  });

  test("parses Z-Image AutoencoderKL config", () => {
    const parsed = parseZImageAutoencoderConfig(zImageVaeConfig());

    expect(parsed.latentChannels).toBe(16);
    expect(parsed.blockOutChannels).toEqual([128, 256, 512, 512]);
    expect(parsed.scalingFactor).toBe(0.3611);
    expect(parsed.shiftFactor).toBe(0.1159);
    expect(parsed.vaeScaleFactor).toBe(8);
  });

  test("loads Z-Image configs from an inspected snapshot manifest", async () => {
    await withTempDirectory("mlxts-z-image-config-", async (directory) => {
      writeZImageSnapshot(directory);

      const manifest = await loadDiffusionSnapshotManifest(directory);
      const configs = await loadZImageComponentConfigs(manifest);

      expect(configs.pipelineKind).toBe("z-image");
      expect(configs.transformer.captionFeatureDim).toBe(2560);
      expect(configs.vae.shiftFactor).toBe(0.1159);
    });
  });

  test("rejects unsupported Z-Image transformer config shapes", () => {
    expect(() => parseZImageTransformerConfig([])).toThrow("JSON object");
    expect(() =>
      parseZImageTransformerConfig({
        ...zImageTransformerConfig(),
        in_channels: "16",
      }),
    ).toThrow("in_channels");
    expect(() =>
      parseZImageTransformerConfig({
        ...zImageTransformerConfig(),
        _class_name: "FluxTransformer2DModel",
      }),
    ).toThrow(DiffusionConfigError);
    expect(() =>
      parseZImageTransformerConfig({
        ...zImageTransformerConfig(),
        dim: 3841,
      }),
    ).toThrow("dim");
    expect(() =>
      parseZImageTransformerConfig({
        ...zImageTransformerConfig(),
        axes_dims: [32, 48, 46],
      }),
    ).toThrow("axes_dims");
    expect(() =>
      parseZImageTransformerConfig({
        ...zImageTransformerConfig(),
        axes_dims: [32, 48],
      }),
    ).toThrow("axes_dims");
    expect(() =>
      parseZImageTransformerConfig({
        ...zImageTransformerConfig(),
        axes_dims: [31, 49, 48],
      }),
    ).toThrow("axes_dims");
    expect(() =>
      parseZImageTransformerConfig({
        ...zImageTransformerConfig(),
        axes_dims: [32, "48", 48],
      }),
    ).toThrow("axes_dims");
    expect(() =>
      parseZImageTransformerConfig({
        ...zImageTransformerConfig(),
        axes_lens: [1536, 512],
      }),
    ).toThrow("axes_lens");
    expect(() =>
      parseZImageTransformerConfig({
        ...zImageTransformerConfig(),
        axes_lens: [1536, "512", 512],
      }),
    ).toThrow("axes_lens");
    expect(() =>
      parseZImageTransformerConfig({
        ...zImageTransformerConfig(),
        n_kv_heads: 7,
      }),
    ).toThrow("n_heads");
    expect(() =>
      parseZImageTransformerConfig({
        ...zImageTransformerConfig(),
        norm_eps: "1e-5",
      }),
    ).toThrow("norm_eps");
    expect(() =>
      parseZImageTransformerConfig({
        ...zImageTransformerConfig(),
        qk_norm: "true",
      }),
    ).toThrow("qk_norm");
    expect(() =>
      parseZImageTransformerConfig({
        ...zImageTransformerConfig(),
        rope_theta: "256",
      }),
    ).toThrow("rope_theta");
    expect(() =>
      parseZImageTransformerConfig({
        ...zImageTransformerConfig(),
        siglip_feat_dim: 1152,
      }),
    ).toThrow("siglip_feat_dim");
    expect(() =>
      parseZImageTransformerConfig({
        ...zImageTransformerConfig(),
        all_patch_size: [2, 4],
        all_f_patch_size: [1],
      }),
    ).toThrow("all_patch_size");
    expect(() =>
      parseZImageTransformerConfig({
        ...zImageTransformerConfig(),
        all_patch_size: [],
      }),
    ).toThrow("all_patch_size");
    expect(() =>
      parseZImageTransformerConfig({
        ...zImageTransformerConfig(),
        all_patch_size: [2, "4"],
        all_f_patch_size: [1, 1],
      }),
    ).toThrow("all_patch_size");
  });

  test("rejects unsupported Z-Image AutoencoderKL config shapes", () => {
    expect(() =>
      parseZImageAutoencoderConfig({
        ...zImageVaeConfig(),
        _class_name: "AutoencoderTiny",
      }),
    ).toThrow(DiffusionConfigError);
    expect(() =>
      parseZImageAutoencoderConfig({
        ...zImageVaeConfig(),
        latents_mean: [0],
      }),
    ).toThrow("latents_mean");
  });

  test("rejects loading Z-Image configs from other pipeline manifests", async () => {
    await expect(
      loadZImageComponentConfigs({
        directory: "/tmp/noop",
        modelIndexPath: "/tmp/noop/model_index.json",
        modelIndex: {
          kind: "qwen-image",
          className: "QwenImagePipeline",
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
    ).rejects.toThrow("Z-Image");
  });

  test("rejects manifests missing Z-Image component config paths", async () => {
    await expect(
      loadZImageComponentConfigs(
        zImageManifest("/tmp/noop", [
          {
            name: "transformer",
            role: "backbone",
            library: "diffusers",
            className: "ZImageTransformer2DModel",
            enabled: true,
            optional: false,
            subfolder: "transformer",
            metadataPaths: ["/tmp/noop/transformer/config.json"],
            weightPaths: ["/tmp/noop/transformer/diffusion_pytorch_model.safetensors"],
          },
          {
            name: "vae",
            role: "vae",
            library: "diffusers",
            className: "AutoencoderKL",
            enabled: true,
            optional: false,
            subfolder: "vae",
            metadataPaths: [],
            weightPaths: ["/tmp/noop/vae/diffusion_pytorch_model.safetensors"],
          },
        ]),
      ),
    ).rejects.toThrow("vae/config.json");
  });

  test("rejects malformed Z-Image component config JSON", async () => {
    await withTempDirectory("mlxts-z-image-malformed-", async (directory) => {
      const transformerConfigPath = join(directory, "transformer", "config.json");
      const vaeConfigPath = join(directory, "vae", "config.json");
      mkdirSync(join(directory, "transformer"), { recursive: true });
      mkdirSync(join(directory, "vae"), { recursive: true });
      writeFileSync(transformerConfigPath, "{");
      writeJson(vaeConfigPath, zImageVaeConfig());

      await expect(
        loadZImageComponentConfigs(
          zImageManifest(directory, [
            {
              name: "transformer",
              role: "backbone",
              library: "diffusers",
              className: "ZImageTransformer2DModel",
              enabled: true,
              optional: false,
              subfolder: "transformer",
              metadataPaths: [transformerConfigPath],
              weightPaths: [join(directory, "transformer", "diffusion_pytorch_model.safetensors")],
            },
            {
              name: "vae",
              role: "vae",
              library: "diffusers",
              className: "AutoencoderKL",
              enabled: true,
              optional: false,
              subfolder: "vae",
              metadataPaths: [vaeConfigPath],
              weightPaths: [join(directory, "vae", "diffusion_pytorch_model.safetensors")],
            },
          ]),
        ),
      ).rejects.toThrow("valid JSON");
    });
  });
});
