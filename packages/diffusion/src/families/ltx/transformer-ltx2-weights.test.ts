import { describe, expect, test } from "bun:test";
import { type MxArray, type ParameterTree, saveSafetensors, treeFlatten, zeros } from "@mlxts/core";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { DiffusionMissingWeightsError, DiffusionWeightMismatchError } from "../../errors";
import {
  type DiffusionSnapshotComponent,
  loadDiffusionSnapshotManifest,
} from "../../pretrained/snapshot-manifest";
import type { Ltx2VideoTransformerConfig } from "./config";
import { Ltx2VideoTransformer3DModel } from "./transformer-ltx2";
import {
  loadLtx2VideoTransformerFromSnapshot,
  loadLtx2VideoTransformerWeights,
  ltx2VideoTransformerWeightPath,
} from "./transformer-ltx2-weights";

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

function tinyConfig(
  overrides: Partial<Ltx2VideoTransformerConfig> = {},
): Ltx2VideoTransformerConfig {
  return {
    inChannels: 4,
    outChannels: 4,
    patchSize: 1,
    patchSizeT: 1,
    numAttentionHeads: 2,
    attentionHeadDim: 4,
    hiddenSize: 8,
    crossAttentionDim: 8,
    vaeScaleFactors: [8, 32, 32],
    posEmbedMaxPos: 20,
    baseHeight: 32,
    baseWidth: 32,
    audioInChannels: 4,
    audioOutChannels: 4,
    audioPatchSize: 1,
    audioPatchSizeT: 1,
    audioNumAttentionHeads: 2,
    audioAttentionHeadDim: 4,
    audioHiddenSize: 8,
    audioCrossAttentionDim: 8,
    audioScaleFactor: 4,
    audioPosEmbedMaxPos: 20,
    audioSamplingRate: 16000,
    audioHopLength: 160,
    numLayers: 1,
    activationFn: "gelu-approximate",
    qkNorm: "rms_norm_across_heads",
    normElementwiseAffine: false,
    normEps: 1e-6,
    captionChannels: 6,
    attentionBias: true,
    attentionOutBias: true,
    ropeTheta: 10000,
    ropeDoublePrecision: true,
    causalOffset: 1,
    timestepScaleMultiplier: 1000,
    crossAttnTimestepScaleMultiplier: 1000,
    ropeType: "split",
    gatedAttn: false,
    crossAttnMod: false,
    audioGatedAttn: false,
    audioCrossAttnMod: false,
    usePromptEmbeddings: true,
    perturbedAttn: false,
    rawConfig: {},
    ...overrides,
  };
}

function transformerComponent(weightPaths: readonly string[]): DiffusionSnapshotComponent {
  return {
    name: "transformer",
    role: "backbone",
    library: "diffusers",
    className: "LTX2VideoTransformer3DModel",
    enabled: true,
    optional: false,
    subfolder: "transformer",
    metadataPaths: [],
    weightPaths,
  };
}

const TOP_LEVEL_CHECKPOINT_PATHS = new Map<string, string>([
  ["projIn.weight", "proj_in.weight"],
  ["projIn.bias", "proj_in.bias"],
  ["audioProjIn.weight", "audio_proj_in.weight"],
  ["audioProjIn.bias", "audio_proj_in.bias"],
  ["scaleShiftTable", "scale_shift_table"],
  ["audioScaleShiftTable", "audio_scale_shift_table"],
  ["projOut.weight", "proj_out.weight"],
  ["projOut.bias", "proj_out.bias"],
  ["audioProjOut.weight", "audio_proj_out.weight"],
  ["audioProjOut.bias", "audio_proj_out.bias"],
  ["captionProjection.linear1.weight", "caption_projection.linear_1.weight"],
  ["captionProjection.linear1.bias", "caption_projection.linear_1.bias"],
  ["captionProjection.linear2.weight", "caption_projection.linear_2.weight"],
  ["captionProjection.linear2.bias", "caption_projection.linear_2.bias"],
  ["audioCaptionProjection.linear1.weight", "audio_caption_projection.linear_1.weight"],
  ["audioCaptionProjection.linear1.bias", "audio_caption_projection.linear_1.bias"],
  ["audioCaptionProjection.linear2.weight", "audio_caption_projection.linear_2.weight"],
  ["audioCaptionProjection.linear2.bias", "audio_caption_projection.linear_2.bias"],
]);

const EMBEDDER_CHECKPOINT_PREFIXES = new Map<string, string>([
  ["timeEmbed", "time_embed"],
  ["audioTimeEmbed", "audio_time_embed"],
  ["avCrossAttnVideoScaleShift", "av_cross_attn_video_scale_shift"],
  ["avCrossAttnAudioScaleShift", "av_cross_attn_audio_scale_shift"],
  ["avCrossAttnVideoA2vGate", "av_cross_attn_video_a2v_gate"],
  ["avCrossAttnAudioV2aGate", "av_cross_attn_audio_v2a_gate"],
]);

const ATTENTION_CHECKPOINT_NAMES = new Map<string, string>([
  ["attn1", "attn1"],
  ["audioAttn1", "audio_attn1"],
  ["attn2", "attn2"],
  ["audioAttn2", "audio_attn2"],
  ["audioToVideoAttn", "audio_to_video_attn"],
  ["videoToAudioAttn", "video_to_audio_attn"],
]);

const ATTENTION_CHECKPOINT_PATHS = new Map<string, string>([
  ["toQ", "to_q"],
  ["toK", "to_k"],
  ["toV", "to_v"],
  ["toOut", "to_out.0"],
]);

function embedderCheckpointName(path: string): string | null {
  const match = path.match(/^([^.]+)\.(linear|emb\.timestepEmbedder\.(linear[12]))\.(.+)$/);
  if (match === null) {
    return null;
  }
  const [, prefix, local, linearName, leaf] = match;
  if (prefix === undefined || local === undefined || leaf === undefined) {
    throw new Error("checkpointNameForLtx2TransformerPath: malformed embedder path.");
  }
  const checkpointPrefix = EMBEDDER_CHECKPOINT_PREFIXES.get(prefix);
  if (checkpointPrefix === undefined) {
    return null;
  }
  if (local === "linear") {
    return `${checkpointPrefix}.linear.${leaf}`;
  }
  if (linearName === undefined) {
    throw new Error("checkpointNameForLtx2TransformerPath: malformed timestep embedder path.");
  }
  const checkpointLinear = linearName === "linear1" ? "linear_1" : "linear_2";
  return `${checkpointPrefix}.emb.timestep_embedder.${checkpointLinear}.${leaf}`;
}

function attentionCheckpointName(path: string): string | null {
  const match = path.match(/^transformerBlocks\.(\d+)\.([^.]+)\.([^.]+)\.(.+)$/);
  if (match === null) {
    return null;
  }
  const [, index, attentionName, local, leaf] = match;
  if (
    index === undefined ||
    attentionName === undefined ||
    local === undefined ||
    leaf === undefined
  ) {
    throw new Error("checkpointNameForLtx2TransformerPath: malformed attention path.");
  }
  const checkpointAttentionName = ATTENTION_CHECKPOINT_NAMES.get(attentionName);
  const checkpointLocal = ATTENTION_CHECKPOINT_PATHS.get(local);
  if (checkpointAttentionName === undefined || checkpointLocal === undefined) {
    return null;
  }
  return `transformer_blocks.${index}.${checkpointAttentionName}.${checkpointLocal}.${leaf}`;
}

function feedForwardCheckpointName(path: string): string | null {
  const match = path.match(/^transformerBlocks\.(\d+)\.(ff|audioFf)\.(linear[12])\.(.+)$/);
  if (match === null) {
    return null;
  }
  const [, index, name, layer, leaf] = match;
  if (index === undefined || name === undefined || layer === undefined || leaf === undefined) {
    throw new Error("checkpointNameForLtx2TransformerPath: malformed feed-forward path.");
  }
  const checkpointName = name === "ff" ? "ff" : "audio_ff";
  const checkpointLayer = layer === "linear1" ? "net.0.proj" : "net.2";
  return `transformer_blocks.${index}.${checkpointName}.${checkpointLayer}.${leaf}`;
}

function blockTableCheckpointName(path: string): string | null {
  const table = path.match(/^transformerBlocks\.(\d+)\.([^.]+)$/);
  if (table === null) {
    return null;
  }
  const [, index, local] = table;
  if (index === undefined || local === undefined) {
    throw new Error("checkpointNameForLtx2TransformerPath: malformed table path.");
  }
  const checkpointLocal =
    local === "scaleShiftTable"
      ? "scale_shift_table"
      : local === "audioScaleShiftTable"
        ? "audio_scale_shift_table"
        : local === "videoA2vCrossAttnScaleShiftTable"
          ? "video_a2v_cross_attn_scale_shift_table"
          : local === "audioA2vCrossAttnScaleShiftTable"
            ? "audio_a2v_cross_attn_scale_shift_table"
            : null;
  return checkpointLocal === null ? null : `transformer_blocks.${index}.${checkpointLocal}`;
}

function checkpointNameForLtx2TransformerPath(path: string): string {
  const mapped =
    TOP_LEVEL_CHECKPOINT_PATHS.get(path) ??
    embedderCheckpointName(path) ??
    attentionCheckpointName(path) ??
    feedForwardCheckpointName(path) ??
    blockTableCheckpointName(path);
  if (mapped !== undefined && mapped !== null) {
    return mapped;
  }
  throw new Error(`checkpointNameForLtx2TransformerPath: unmapped path ${path}.`);
}

function checkpointTensorsForLtx2Transformer(
  parameters: ParameterTree,
  options: { omitPath?: string; mismatchPath?: string; extra?: Record<string, MxArray> } = {},
): Record<string, MxArray> {
  const tensors: Record<string, MxArray> = {};
  for (const [path, parameter] of treeFlatten(parameters)) {
    const internalPath = path.join(".");
    if (internalPath === options.omitPath) {
      continue;
    }
    const checkpointName = checkpointNameForLtx2TransformerPath(internalPath);
    const shape = internalPath === options.mismatchPath ? [1] : [...parameter.shape];
    tensors[checkpointName] = zeros(shape);
  }
  return { ...tensors, ...(options.extra ?? {}) };
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

function writeJson(path: string, payload: unknown): void {
  writeFileSync(path, JSON.stringify(payload));
}

function writeComponent(snapshot: string, name: string, config: Record<string, unknown>): void {
  const directory = join(snapshot, name);
  mkdirSync(directory, { recursive: true });
  writeJson(join(directory, "config.json"), config);
  writeFileSync(join(directory, "diffusion_pytorch_model.safetensors"), "");
}

function ltx2TransformerRawConfig(config = tinyConfig()): Record<string, unknown> {
  return {
    _class_name: "LTX2VideoTransformer3DModel",
    activation_fn: config.activationFn,
    attention_bias: config.attentionBias,
    attention_head_dim: config.attentionHeadDim,
    attention_out_bias: config.attentionOutBias,
    audio_attention_head_dim: config.audioAttentionHeadDim,
    audio_cross_attention_dim: config.audioCrossAttentionDim,
    audio_hop_length: config.audioHopLength,
    audio_in_channels: config.audioInChannels,
    audio_num_attention_heads: config.audioNumAttentionHeads,
    audio_out_channels: config.audioOutChannels,
    audio_patch_size: config.audioPatchSize,
    audio_patch_size_t: config.audioPatchSizeT,
    audio_pos_embed_max_pos: config.audioPosEmbedMaxPos,
    audio_sampling_rate: config.audioSamplingRate,
    audio_scale_factor: config.audioScaleFactor,
    base_height: config.baseHeight,
    base_width: config.baseWidth,
    caption_channels: config.captionChannels,
    causal_offset: config.causalOffset,
    cross_attention_dim: config.crossAttentionDim,
    cross_attn_timestep_scale_multiplier: config.crossAttnTimestepScaleMultiplier,
    in_channels: config.inChannels,
    norm_elementwise_affine: config.normElementwiseAffine,
    norm_eps: config.normEps,
    num_attention_heads: config.numAttentionHeads,
    num_layers: config.numLayers,
    out_channels: config.outChannels,
    patch_size: config.patchSize,
    patch_size_t: config.patchSizeT,
    pos_embed_max_pos: config.posEmbedMaxPos,
    qk_norm: config.qkNorm,
    rope_double_precision: config.ropeDoublePrecision,
    rope_theta: config.ropeTheta,
    rope_type: config.ropeType,
    timestep_scale_multiplier: config.timestepScaleMultiplier,
    vae_scale_factors: config.vaeScaleFactors,
  };
}

function writeLtx2SnapshotMetadata(directory: string): void {
  const config = tinyConfig();
  writeJson(join(directory, "model_index.json"), {
    _class_name: "LTX2Pipeline",
    audio_vae: ["diffusers", "AutoencoderKLLTX2Audio"],
    connectors: ["ltx2", "LTX2TextConnectors"],
    scheduler: ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    text_encoder: ["transformers", "Gemma3ForConditionalGeneration"],
    tokenizer: ["transformers", "GemmaTokenizerFast"],
    transformer: ["diffusers", "LTX2VideoTransformer3DModel"],
    vae: ["diffusers", "AutoencoderKLLTX2Video"],
    vocoder: ["ltx2", "LTX2Vocoder"],
  });
  writeComponent(directory, "transformer", ltx2TransformerRawConfig(config));
  writeComponent(directory, "vae", {
    _class_name: "AutoencoderKLLTX2Video",
    latent_channels: config.inChannels,
  });
  writeComponent(directory, "audio_vae", {
    _class_name: "AutoencoderKLLTX2Audio",
    base_channels: config.audioInChannels,
    latent_channels: config.audioInChannels,
    mel_bins: 4,
    mel_hop_length: config.audioHopLength,
    output_channels: 1,
    sample_rate: config.audioSamplingRate,
  });
  writeComponent(directory, "connectors", {
    _class_name: "LTX2TextConnectors",
    caption_channels: config.captionChannels,
    rope_type: config.ropeType,
  });
  writeComponent(directory, "vocoder", {
    _class_name: "LTX2Vocoder",
    in_channels: config.audioOutChannels,
  });
  writeComponent(directory, "text_encoder", {});
  const scheduler = join(directory, "scheduler");
  mkdirSync(scheduler, { recursive: true });
  writeJson(join(scheduler, "scheduler_config.json"), {
    _class_name: "FlowMatchEulerDiscreteScheduler",
  });
  const tokenizer = join(directory, "tokenizer");
  mkdirSync(tokenizer, { recursive: true });
  writeFileSync(join(tokenizer, "tokenizer.json"), "{}");
}

describe("LTX-2 transformer weight mapping", () => {
  test("maps top-level Diffusers parameters", () => {
    expect(ltx2VideoTransformerWeightPath("proj_in.weight")).toBe("projIn.weight");
    expect(ltx2VideoTransformerWeightPath("audio_proj_in.bias")).toBe("audioProjIn.bias");
    expect(ltx2VideoTransformerWeightPath("scale_shift_table")).toBe("scaleShiftTable");
    expect(ltx2VideoTransformerWeightPath("audio_scale_shift_table")).toBe("audioScaleShiftTable");
    expect(ltx2VideoTransformerWeightPath("caption_projection.linear_2.bias")).toBe(
      "captionProjection.linear2.bias",
    );
    expect(ltx2VideoTransformerWeightPath("audio_caption_projection.linear_1.weight")).toBe(
      "audioCaptionProjection.linear1.weight",
    );
    expect(ltx2VideoTransformerWeightPath("proj_out.weight")).toBe("projOut.weight");
    expect(ltx2VideoTransformerWeightPath("audio_proj_out.bias")).toBe("audioProjOut.bias");
  });

  test("maps AdaLayerNormSingle and block parameters", () => {
    expect(ltx2VideoTransformerWeightPath("time_embed.linear.weight")).toBe(
      "timeEmbed.linear.weight",
    );
    expect(
      ltx2VideoTransformerWeightPath("audio_time_embed.emb.timestep_embedder.linear_2.bias"),
    ).toBe("audioTimeEmbed.emb.timestepEmbedder.linear2.bias");
    expect(ltx2VideoTransformerWeightPath("av_cross_attn_video_a2v_gate.linear.bias")).toBe(
      "avCrossAttnVideoA2vGate.linear.bias",
    );
    expect(ltx2VideoTransformerWeightPath("av_cross_attn_video_scale_shift.linear.weight")).toBe(
      "avCrossAttnVideoScaleShift.linear.weight",
    );
    expect(ltx2VideoTransformerWeightPath("av_cross_attn_audio_scale_shift.linear.bias")).toBe(
      "avCrossAttnAudioScaleShift.linear.bias",
    );
    expect(
      ltx2VideoTransformerWeightPath(
        "av_cross_attn_audio_v2a_gate.emb.timestep_embedder.linear_1.weight",
      ),
    ).toBe("avCrossAttnAudioV2aGate.emb.timestepEmbedder.linear1.weight");
    expect(ltx2VideoTransformerWeightPath("transformer_blocks.3.audio_attn1.to_q.weight")).toBe(
      "transformerBlocks.3.audioAttn1.toQ.weight",
    );
    expect(
      ltx2VideoTransformerWeightPath("transformer_blocks.3.audio_to_video_attn.to_out.0.bias"),
    ).toBe("transformerBlocks.3.audioToVideoAttn.toOut.bias");
    expect(ltx2VideoTransformerWeightPath("transformer_blocks.3.audio_ff.net.0.proj.weight")).toBe(
      "transformerBlocks.3.audioFf.linear1.weight",
    );
    expect(ltx2VideoTransformerWeightPath("transformer_blocks.3.ff.net.2.bias")).toBe(
      "transformerBlocks.3.ff.linear2.bias",
    );
    expect(
      ltx2VideoTransformerWeightPath("transformer_blocks.3.video_a2v_cross_attn_scale_shift_table"),
    ).toBe("transformerBlocks.3.videoA2vCrossAttnScaleShiftTable");
  });

  test("does not map unsupported LTX-2.3, affine norm, gated, dropout, and unknown paths", () => {
    expect(ltx2VideoTransformerWeightPath("transformer_blocks.3.attn1.norm_q.weight")).toBeNull();
    expect(
      ltx2VideoTransformerWeightPath("transformer_blocks.3.attn1.to_gate_logits.bias"),
    ).toBeNull();
    expect(ltx2VideoTransformerWeightPath("transformer_blocks.3.attn1.to_out.1.weight")).toBeNull();
    expect(ltx2VideoTransformerWeightPath("transformer_blocks.3.ff.net.1.weight")).toBeNull();
    expect(
      ltx2VideoTransformerWeightPath("transformer_blocks.3.prompt_scale_shift_table"),
    ).toBeNull();
    expect(
      ltx2VideoTransformerWeightPath("transformer_blocks.3.audio_prompt_scale_shift_table"),
    ).toBeNull();
    expect(ltx2VideoTransformerWeightPath("prompt_adaln.linear.weight")).toBeNull();
    expect(ltx2VideoTransformerWeightPath("audio_prompt_adaln.linear.weight")).toBeNull();
    expect(ltx2VideoTransformerWeightPath("norm_out.weight")).toBeNull();
    expect(ltx2VideoTransformerWeightPath("audio_norm_out.weight")).toBeNull();
    expect(ltx2VideoTransformerWeightPath("rope.freqs")).toBeNull();
  });

  test("loads a complete generated transformer safetensors shard", async () => {
    await withTempDirectory("mlxts-ltx2-transformer-weights-", async (directory) => {
      using target = new Ltx2VideoTransformer3DModel(tinyConfig());
      using source = new Ltx2VideoTransformer3DModel(tinyConfig());
      const checkpointPath = join(directory, "diffusion_pytorch_model.safetensors");
      await writeCheckpoint(
        checkpointPath,
        checkpointTensorsForLtx2Transformer(source.parameters()),
      );

      const result = await loadLtx2VideoTransformerWeights(
        target,
        transformerComponent([checkpointPath]),
      );

      expect(result.shardCount).toBe(1);
      expect(result.unexpectedWeights).toEqual([]);
      expect(result.assignedPaths).toContain("projIn.weight");
      expect(result.assignedPaths).toContain("audioProjIn.weight");
      expect(result.assignedPaths).toContain("transformerBlocks.0.audioToVideoAttn.toK.weight");
      expect(result.assignedPaths).toContain("transformerBlocks.0.audioFf.linear2.weight");
    });
  });

  test("loads transformer shards from a Diffusers safetensors index", async () => {
    await withTempDirectory("mlxts-ltx2-transformer-index-", async (directory) => {
      using target = new Ltx2VideoTransformer3DModel(tinyConfig());
      using source = new Ltx2VideoTransformer3DModel(tinyConfig());
      const shards = splitIndexedShards(checkpointTensorsForLtx2Transformer(source.parameters()));
      const firstPath = join(directory, "diffusion_pytorch_model-00001-of-00002.safetensors");
      const secondPath = join(directory, "diffusion_pytorch_model-00002-of-00002.safetensors");
      await writeCheckpoint(firstPath, shards.firstShard);
      await writeCheckpoint(secondPath, shards.secondShard);
      const indexPath = join(directory, "diffusion_pytorch_model.safetensors.index.json");
      writeJson(indexPath, { metadata: {}, weight_map: shards.weightMap });

      const result = await loadLtx2VideoTransformerWeights(
        target,
        transformerComponent([indexPath]),
      );

      expect(result.shardCount).toBe(2);
      expect(result.unexpectedWeights).toEqual([]);
    });
  });

  test("loads transformer directly from an inspected LTX-2 snapshot manifest", async () => {
    await withTempDirectory("mlxts-ltx2-transformer-snapshot-", async (directory) => {
      writeLtx2SnapshotMetadata(directory);
      using source = new Ltx2VideoTransformer3DModel(tinyConfig());
      await writeCheckpoint(
        join(directory, "transformer", "diffusion_pytorch_model.safetensors"),
        checkpointTensorsForLtx2Transformer(source.parameters()),
      );
      const manifest = await loadDiffusionSnapshotManifest(directory);

      using target = await loadLtx2VideoTransformerFromSnapshot(manifest);

      expect(target.projIn.weight.shape).toEqual([8, 4]);
      expect(target.audioProjIn.weight.shape).toEqual([8, 4]);
      expect(target.transformerBlocks[0]?.videoToAudioAttn.toV.weight.shape).toEqual([8, 8]);
    });
  });

  test("disposes partial snapshot transformer loads on failure", async () => {
    await withTempDirectory("mlxts-ltx2-transformer-snapshot-failure-", async (directory) => {
      writeLtx2SnapshotMetadata(directory);
      using source = new Ltx2VideoTransformer3DModel(tinyConfig());
      await writeCheckpoint(
        join(directory, "transformer", "diffusion_pytorch_model.safetensors"),
        checkpointTensorsForLtx2Transformer(source.parameters(), {
          omitPath: "transformerBlocks.0.audioAttn2.toOut.weight",
        }),
      );
      const manifest = await loadDiffusionSnapshotManifest(directory);

      await expect(loadLtx2VideoTransformerFromSnapshot(manifest)).rejects.toThrow(
        DiffusionMissingWeightsError,
      );
    });
  });

  test("rejects missing, mismatched, and strict unexpected transformer weights", async () => {
    await withTempDirectory("mlxts-ltx2-transformer-invalid-", async (directory) => {
      using source = new Ltx2VideoTransformer3DModel(tinyConfig());

      const missingPath = join(directory, "missing.safetensors");
      await writeCheckpoint(
        missingPath,
        checkpointTensorsForLtx2Transformer(source.parameters(), {
          omitPath: "transformerBlocks.0.audioToVideoAttn.toOut.weight",
        }),
      );
      using missingTarget = new Ltx2VideoTransformer3DModel(tinyConfig());
      await expect(
        loadLtx2VideoTransformerWeights(missingTarget, transformerComponent([missingPath])),
      ).rejects.toThrow(DiffusionMissingWeightsError);

      const mismatchPath = join(directory, "mismatch.safetensors");
      await writeCheckpoint(
        mismatchPath,
        checkpointTensorsForLtx2Transformer(source.parameters(), {
          mismatchPath: "transformerBlocks.0.attn1.toQ.weight",
        }),
      );
      using mismatchTarget = new Ltx2VideoTransformer3DModel(tinyConfig());
      await expect(
        loadLtx2VideoTransformerWeights(mismatchTarget, transformerComponent([mismatchPath])),
      ).rejects.toThrow(DiffusionWeightMismatchError);

      const unexpectedPath = join(directory, "unexpected.safetensors");
      await writeCheckpoint(
        unexpectedPath,
        checkpointTensorsForLtx2Transformer(source.parameters(), {
          extra: { "transformer_blocks.0.attn1.to_gate_logits.weight": zeros([2, 8]) },
        }),
      );
      using unexpectedTarget = new Ltx2VideoTransformer3DModel(tinyConfig());
      await expect(
        loadLtx2VideoTransformerWeights(unexpectedTarget, transformerComponent([unexpectedPath]), {
          strictUnexpectedWeights: true,
        }),
      ).rejects.toThrow("unexpected unmapped weights");
    });
  });
});
