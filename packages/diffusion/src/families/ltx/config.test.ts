import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { DiffusionConfigError } from "../../errors";
import { loadDiffusionSnapshotManifest } from "../../pretrained/snapshot-manifest";

import {
  loadLtxComponentConfigs,
  parseLtx2AudioAutoencoderConfig,
  parseLtx2TextConnectorsConfig,
  parseLtx2VideoAutoencoderConfig,
  parseLtx2VideoTransformerConfig,
  parseLtx2VocoderConfig,
  parseLtxVideoAutoencoderConfig,
  parseLtxVideoTransformerConfig,
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

function writeComponent(
  snapshot: string,
  component: string,
  configName: string,
  payload: unknown,
  needsWeights = true,
): void {
  const directory = join(snapshot, component);
  mkdirSync(directory, { recursive: true });
  writeJson(join(directory, configName), payload);
  if (needsWeights) {
    writeFileSync(join(directory, "diffusion_pytorch_model.safetensors"), "");
  }
}

function writeTokenizer(snapshot: string): void {
  const directory = join(snapshot, "tokenizer");
  mkdirSync(directory, { recursive: true });
  writeFileSync(join(directory, "tokenizer.json"), "{}");
}

function writeScheduler(snapshot: string): void {
  writeComponent(
    snapshot,
    "scheduler",
    "scheduler_config.json",
    {
      _class_name: "FlowMatchEulerDiscreteScheduler",
      base_image_seq_len: 256,
      base_shift: 0.5,
      max_image_seq_len: 4096,
      max_shift: 1.15,
      num_train_timesteps: 1000,
      shift: 3,
      time_shift_type: "exponential",
      use_dynamic_shifting: true,
    },
    false,
  );
}

function ltxTransformerConfig(): Record<string, unknown> {
  return {
    _class_name: "LTXVideoTransformer3DModel",
    _diffusers_version: "0.32.0.dev0",
    activation_fn: "gelu-approximate",
    attention_bias: true,
    attention_head_dim: 64,
    attention_out_bias: true,
    caption_channels: 4096,
    cross_attention_dim: 2048,
    in_channels: 128,
    norm_elementwise_affine: false,
    norm_eps: 1e-6,
    num_attention_heads: 32,
    num_layers: 28,
    out_channels: 128,
    patch_size: 1,
    patch_size_t: 1,
    qk_norm: "rms_norm_across_heads",
  };
}

function ltxVaeConfig(): Record<string, unknown> {
  return {
    _class_name: "AutoencoderKLLTXVideo",
    _diffusers_version: "0.32.0.dev0",
    block_out_channels: [128, 256, 512, 512],
    decoder_causal: false,
    encoder_causal: true,
    in_channels: 3,
    latent_channels: 128,
    layers_per_block: [4, 3, 3, 3, 4],
    out_channels: 3,
    patch_size: 4,
    patch_size_t: 1,
    resnet_norm_eps: 1e-6,
    scaling_factor: 1.0,
    spatio_temporal_scaling: [true, true, true, false],
  };
}

function ltx2TransformerConfig(): Record<string, unknown> {
  return {
    _class_name: "LTX2VideoTransformer3DModel",
    _diffusers_version: "0.37.0.dev0",
    activation_fn: "gelu-approximate",
    attention_bias: true,
    attention_head_dim: 128,
    attention_out_bias: true,
    audio_attention_head_dim: 64,
    audio_cross_attention_dim: 2048,
    audio_hop_length: 160,
    audio_in_channels: 128,
    audio_num_attention_heads: 32,
    audio_out_channels: 128,
    audio_patch_size: 1,
    audio_patch_size_t: 1,
    audio_pos_embed_max_pos: 20,
    audio_sampling_rate: 16000,
    audio_scale_factor: 4,
    base_height: 2048,
    base_width: 2048,
    caption_channels: 3840,
    causal_offset: 1,
    cross_attention_dim: 4096,
    cross_attn_timestep_scale_multiplier: 1000,
    in_channels: 128,
    norm_elementwise_affine: false,
    norm_eps: 1e-6,
    num_attention_heads: 32,
    num_layers: 48,
    out_channels: 128,
    patch_size: 1,
    patch_size_t: 1,
    pos_embed_max_pos: 20,
    qk_norm: "rms_norm_across_heads",
    rope_double_precision: true,
    rope_theta: 10000,
    rope_type: "split",
    timestep_scale_multiplier: 1000,
    vae_scale_factors: [8, 32, 32],
  };
}

function ltx2VideoVaeConfig(): Record<string, unknown> {
  return {
    _class_name: "AutoencoderKLLTX2Video",
    _diffusers_version: "0.37.0.dev0",
    block_out_channels: [256, 512, 1024, 2048],
    decoder_block_out_channels: [256, 512, 1024],
    decoder_causal: false,
    decoder_inject_noise: [false, false, false, false],
    decoder_layers_per_block: [5, 5, 5, 5],
    decoder_spatial_padding_mode: "reflect",
    decoder_spatio_temporal_scaling: [true, true, true],
    down_block_types: [
      "LTX2VideoDownBlock3D",
      "LTX2VideoDownBlock3D",
      "LTX2VideoDownBlock3D",
      "LTX2VideoDownBlock3D",
    ],
    downsample_type: ["spatial", "temporal", "spatiotemporal", "spatiotemporal"],
    encoder_causal: true,
    encoder_spatial_padding_mode: "zeros",
    in_channels: 3,
    latent_channels: 128,
    layers_per_block: [4, 6, 6, 2, 2],
    out_channels: 3,
    patch_size: 4,
    patch_size_t: 1,
    resnet_norm_eps: 1e-6,
    scaling_factor: 1.0,
    spatial_compression_ratio: 32,
    spatio_temporal_scaling: [true, true, true, true],
    temporal_compression_ratio: 8,
    timestep_conditioning: false,
    upsample_factor: [2, 2, 2],
    upsample_residual: [true, true, true],
  };
}

function ltx2AudioVaeConfig(): Record<string, unknown> {
  return {
    _class_name: "AutoencoderKLLTX2Audio",
    _diffusers_version: "0.37.0.dev0",
    attn_resolutions: null,
    base_channels: 128,
    causality_axis: "height",
    ch_mult: [1, 2, 4],
    double_z: true,
    dropout: 0,
    in_channels: 2,
    is_causal: true,
    latent_channels: 8,
    mel_bins: 64,
    mel_hop_length: 160,
    mid_block_add_attention: false,
    norm_type: "pixel",
    num_res_blocks: 2,
    output_channels: 2,
    resolution: 256,
    sample_rate: 16000,
  };
}

function ltx2ConnectorsConfig(): Record<string, unknown> {
  return {
    _class_name: "LTX2TextConnectors",
    _diffusers_version: "0.37.0.dev0",
    audio_connector_attention_head_dim: 128,
    audio_connector_num_attention_heads: 30,
    audio_connector_num_layers: 2,
    audio_connector_num_learnable_registers: 128,
    caption_channels: 3840,
    causal_temporal_positioning: false,
    connector_rope_base_seq_len: 4096,
    rope_double_precision: true,
    rope_theta: 10000,
    rope_type: "split",
    text_proj_in_factor: 49,
    video_connector_attention_head_dim: 128,
    video_connector_num_attention_heads: 30,
    video_connector_num_layers: 2,
    video_connector_num_learnable_registers: 128,
  };
}

function ltx2VocoderConfig(): Record<string, unknown> {
  return {
    _class_name: "LTX2Vocoder",
    _diffusers_version: "0.37.0.dev0",
    hidden_channels: 1024,
    in_channels: 128,
    leaky_relu_negative_slope: 0.1,
    out_channels: 2,
    output_sampling_rate: 24000,
    resnet_dilations: [
      [1, 3, 5],
      [1, 3, 5],
      [1, 3, 5],
    ],
    resnet_kernel_sizes: [3, 7, 11],
    upsample_factors: [6, 5, 2, 2, 2],
    upsample_kernel_sizes: [16, 15, 8, 4, 4],
  };
}

function writeLtxVideoSnapshot(snapshot: string): void {
  writeJson(join(snapshot, "model_index.json"), {
    _class_name: "LTXPipeline",
    _diffusers_version: "0.32.0.dev0",
    scheduler: ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    text_encoder: ["transformers", "T5EncoderModel"],
    tokenizer: ["transformers", "T5Tokenizer"],
    transformer: ["diffusers", "LTXVideoTransformer3DModel"],
    vae: ["diffusers", "AutoencoderKLLTXVideo"],
  });
  writeComponent(snapshot, "transformer", "config.json", ltxTransformerConfig());
  writeComponent(snapshot, "vae", "config.json", ltxVaeConfig());
  writeScheduler(snapshot);
  writeComponent(snapshot, "text_encoder", "config.json", {}, true);
  writeTokenizer(snapshot);
}

function writeLtx2Snapshot(
  snapshot: string,
  transformerConfig: Record<string, unknown> = ltx2TransformerConfig(),
  videoVaeConfig: Record<string, unknown> = ltx2VideoVaeConfig(),
  audioVaeConfig: Record<string, unknown> = ltx2AudioVaeConfig(),
  connectorsConfig: Record<string, unknown> = ltx2ConnectorsConfig(),
  vocoderConfig: Record<string, unknown> = ltx2VocoderConfig(),
): void {
  writeJson(join(snapshot, "model_index.json"), {
    _class_name: "LTX2Pipeline",
    _diffusers_version: "0.37.0.dev0",
    audio_vae: ["diffusers", "AutoencoderKLLTX2Audio"],
    connectors: ["ltx2", "LTX2TextConnectors"],
    scheduler: ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    text_encoder: ["transformers", "Gemma3ForConditionalGeneration"],
    tokenizer: ["transformers", "GemmaTokenizerFast"],
    transformer: ["diffusers", "LTX2VideoTransformer3DModel"],
    vae: ["diffusers", "AutoencoderKLLTX2Video"],
    vocoder: ["ltx2", "LTX2Vocoder"],
  });
  writeComponent(snapshot, "transformer", "config.json", transformerConfig);
  writeComponent(snapshot, "vae", "config.json", videoVaeConfig);
  writeComponent(snapshot, "audio_vae", "config.json", audioVaeConfig);
  writeComponent(snapshot, "connectors", "config.json", connectorsConfig);
  writeComponent(snapshot, "vocoder", "config.json", vocoderConfig);
  writeScheduler(snapshot);
  writeComponent(snapshot, "text_encoder", "config.json", {}, true);
  writeTokenizer(snapshot);
}

describe("LTX component config parsing", () => {
  test("parses LTX-Video transformer and VAE configs into package-owned shape", () => {
    const transformer = parseLtxVideoTransformerConfig(ltxTransformerConfig());
    const vae = parseLtxVideoAutoencoderConfig(ltxVaeConfig());

    expect(transformer.hiddenSize).toBe(2048);
    expect(transformer.qkNorm).toBe("rms_norm_across_heads");
    expect(vae.latentChannels).toBe(128);
    expect(vae.layersPerBlock).toEqual([4, 3, 3, 3, 4]);
    expect(vae.spatioTemporalScaling).toEqual([true, true, true, false]);
  });

  test("parses LTX-2 component configs into package-owned shapes", () => {
    const transformer = parseLtx2VideoTransformerConfig(ltx2TransformerConfig());
    const vae = parseLtx2VideoAutoencoderConfig(ltx2VideoVaeConfig());
    const audioVae = parseLtx2AudioAutoencoderConfig(ltx2AudioVaeConfig());
    const connectors = parseLtx2TextConnectorsConfig(ltx2ConnectorsConfig());
    const vocoder = parseLtx2VocoderConfig(ltx2VocoderConfig());

    expect(transformer.hiddenSize).toBe(4096);
    expect(transformer.audioHiddenSize).toBe(2048);
    expect(transformer.vaeScaleFactors).toEqual([8, 32, 32]);
    expect(vae.temporalCompressionRatio).toBe(8);
    expect(vae.spatialCompressionRatio).toBe(32);
    expect(audioVae.sampleRate).toBe(16000);
    expect(audioVae.packedFeatureSize).toBe(128);
    expect(connectors.textEncoderDim).toBe(188160);
    expect(vocoder.totalUpsampleFactor).toBe(240);
  });

  test("loads LTX-Video configs from an inspected snapshot manifest", async () => {
    await withTempDirectory("mlxts-ltx-video-config-", async (directory) => {
      writeLtxVideoSnapshot(directory);

      const manifest = await loadDiffusionSnapshotManifest(directory);
      const configs = await loadLtxComponentConfigs(manifest);

      expect(configs.pipelineKind).toBe("ltx-video");
      if (configs.pipelineKind === "ltx-video") {
        expect(configs.transformer.numLayers).toBe(28);
        expect(configs.vae.decoderCausal).toBe(false);
      }
    });
  });

  test("loads LTX-2 configs from an inspected snapshot manifest", async () => {
    await withTempDirectory("mlxts-ltx2-config-", async (directory) => {
      writeLtx2Snapshot(directory);

      const manifest = await loadDiffusionSnapshotManifest(directory);
      const configs = await loadLtxComponentConfigs(manifest);

      expect(configs.pipelineKind).toBe("ltx2");
      if (configs.pipelineKind === "ltx2") {
        expect(configs.transformer.ropeType).toBe("split");
        expect(configs.audioVae.melHopLength).toBe(160);
        expect(configs.connectors.captionChannels).toBe(3840);
        expect(configs.vocoder.outputSamplingRate).toBe(24000);
      }
    });
  });

  test("rejects LTX-2 component configs whose cross-component dimensions disagree", async () => {
    await withTempDirectory("mlxts-ltx2-mismatch-", async (directory) => {
      writeLtx2Snapshot(directory, ltx2TransformerConfig(), {
        ...ltx2VideoVaeConfig(),
        temporal_compression_ratio: 4,
      });

      const manifest = await loadDiffusionSnapshotManifest(directory);
      await expect(loadLtxComponentConfigs(manifest)).rejects.toThrow("vae_scale_factors");
    });

    await withTempDirectory("mlxts-ltx2-audio-mismatch-", async (directory) => {
      writeLtx2Snapshot(
        directory,
        {
          ...ltx2TransformerConfig(),
          audio_in_channels: 64,
        },
        ltx2VideoVaeConfig(),
      );

      const manifest = await loadDiffusionSnapshotManifest(directory);
      await expect(loadLtxComponentConfigs(manifest)).rejects.toThrow("packed feature size");
    });
  });

  test("rejects unsupported LTX config variants", async () => {
    expect(() =>
      parseLtxVideoTransformerConfig({
        ...ltxTransformerConfig(),
        qk_norm: "layer_norm",
      }),
    ).toThrow(DiffusionConfigError);
    expect(() =>
      parseLtx2VocoderConfig({
        ...ltx2VocoderConfig(),
        resnet_dilations: [[1, 3, 5]],
      }),
    ).toThrow("resnet_dilations");

    await withTempDirectory("mlxts-ltx-unsupported-", async (directory) => {
      writeJson(join(directory, "model_index.json"), {
        _class_name: "QwenImagePipeline",
        scheduler: ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        text_encoder: ["transformers", "Qwen2_5_VLForConditionalGeneration"],
        tokenizer: ["transformers", "Qwen2TokenizerFast"],
        transformer: ["diffusers", "QwenImageTransformer2DModel"],
        vae: ["diffusers", "AutoencoderKLQwenImage"],
      });
      writeScheduler(directory);
      writeComponent(directory, "text_encoder", "config.json", {}, true);
      writeTokenizer(directory);
      writeComponent(directory, "transformer", "config.json", {}, true);
      writeComponent(directory, "vae", "config.json", {}, true);

      const manifest = await loadDiffusionSnapshotManifest(directory);
      await expect(loadLtxComponentConfigs(manifest)).rejects.toThrow("do not support qwen-image");
    });
  });
});
