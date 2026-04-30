import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { DiffusionConfigError } from "../../errors";
import { loadDiffusionSnapshotManifest } from "../../pretrained/snapshot-manifest";

import { loadFluxComponentConfigs, parseFluxTransformerConfig } from "./config";

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

function fluxTransformerConfig(): Record<string, unknown> {
  return {
    _class_name: "FluxTransformer2DModel",
    _diffusers_version: "0.34.0.dev0",
    attention_head_dim: 128,
    axes_dims_rope: [16, 56, 56],
    guidance_embeds: true,
    in_channels: 64,
    joint_attention_dim: 4096,
    num_attention_heads: 24,
    num_layers: 19,
    num_single_layers: 38,
    out_channels: null,
    patch_size: 1,
    pooled_projection_dim: 768,
  };
}

function writeFluxSnapshot(snapshot: string, transformerConfig: Record<string, unknown>): void {
  writeJson(join(snapshot, "model_index.json"), {
    _class_name: "FluxPipeline",
    transformer: ["diffusers", "FluxTransformer2DModel"],
    vae: ["diffusers", "AutoencoderKL"],
    scheduler: ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    text_encoder: ["transformers", "CLIPTextModel"],
    tokenizer: ["transformers", "CLIPTokenizer"],
    text_encoder_2: ["transformers", "T5EncoderModel"],
    tokenizer_2: ["transformers", "T5TokenizerFast"],
  });
  writeComponentFiles(
    snapshot,
    "transformer",
    ["config.json"],
    ["diffusion_pytorch_model.safetensors"],
  );
  writeJson(join(snapshot, "transformer", "config.json"), transformerConfig);
  writeComponentFiles(snapshot, "vae", ["config.json"], ["diffusion_pytorch_model.safetensors"]);
  writeComponentFiles(snapshot, "scheduler", ["scheduler_config.json"]);
  writeJson(join(snapshot, "scheduler", "scheduler_config.json"), {
    _class_name: "FlowMatchEulerDiscreteScheduler",
    shift: 1,
    use_dynamic_shifting: false,
  });
  writeComponentFiles(snapshot, "text_encoder", ["config.json"], ["model.safetensors"]);
  writeComponentFiles(snapshot, "tokenizer", ["vocab.json", "merges.txt"]);
  writeComponentFiles(snapshot, "text_encoder_2", ["config.json"], ["model.safetensors"]);
  writeComponentFiles(snapshot, "tokenizer_2", ["spiece.model"]);
}

describe("Flux component config parsing", () => {
  test("parses FLUX.1 transformer config into package-owned shape", () => {
    const parsed = parseFluxTransformerConfig(fluxTransformerConfig());

    expect(parsed.patchSize).toBe(1);
    expect(parsed.inChannels).toBe(64);
    expect(parsed.latentChannels).toBe(16);
    expect(parsed.outChannels).toBe(64);
    expect(parsed.numLayers).toBe(19);
    expect(parsed.numSingleLayers).toBe(38);
    expect(parsed.attentionHeadDim).toBe(128);
    expect(parsed.numAttentionHeads).toBe(24);
    expect(parsed.hiddenSize).toBe(3072);
    expect(parsed.mlpRatio).toBe(4);
    expect(parsed.jointAttentionDim).toBe(4096);
    expect(parsed.pooledProjectionDim).toBe(768);
    expect(parsed.guidanceEmbeds).toBe(true);
    expect(parsed.axesDimsRope).toEqual([16, 56, 56]);
    expect(parsed.ropeTheta).toBe(10000);
    expect(parsed.qkvBias).toBe(true);
  });

  test("loads FLUX.1 transformer config from an inspected snapshot manifest", async () => {
    await withTempDirectory("mlxts-flux-config-", async (directory) => {
      writeFluxSnapshot(directory, fluxTransformerConfig());

      const manifest = await loadDiffusionSnapshotManifest(directory);
      const configs = await loadFluxComponentConfigs(manifest);

      expect(configs.pipelineKind).toBe("flux");
      expect(configs.transformer.hiddenSize).toBe(3072);
      expect(configs.transformer.guidanceEmbeds).toBe(true);
    });
  });

  test("rejects unsupported Flux transformer config shapes", () => {
    expect(() =>
      parseFluxTransformerConfig({
        ...fluxTransformerConfig(),
        _class_name: "Flux2Transformer2DModel",
      }),
    ).toThrow(DiffusionConfigError);
    expect(() =>
      parseFluxTransformerConfig({
        ...fluxTransformerConfig(),
        axes_dims_rope: [32, 32, 32, 32],
      }),
    ).toThrow("axes_dims_rope");
    expect(() =>
      parseFluxTransformerConfig({
        ...fluxTransformerConfig(),
        axes_dims_rope: [16, 48, 56],
      }),
    ).toThrow("FLUX.1 axes");
    expect(() =>
      parseFluxTransformerConfig({
        ...fluxTransformerConfig(),
        guidance_embeds: "true",
      }),
    ).toThrow("guidance_embeds");
    expect(() =>
      parseFluxTransformerConfig({
        ...fluxTransformerConfig(),
        patch_size: 2,
      }),
    ).toThrow("patch_size");
    expect(() =>
      parseFluxTransformerConfig({
        ...fluxTransformerConfig(),
        out_channels: 32,
      }),
    ).toThrow("out_channels");
    expect(() =>
      parseFluxTransformerConfig({
        ...fluxTransformerConfig(),
        mlp_ratio: 4,
      }),
    ).toThrow("mlp_ratio");
  });
});
