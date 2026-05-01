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
import type { LtxVideoTransformerConfig } from "./config";
import { LtxVideoTransformer3DModel } from "./transformer";
import {
  loadLtxVideoTransformerFromSnapshot,
  loadLtxVideoTransformerWeights,
  ltxVideoTransformerWeightPath,
} from "./transformer-weights";

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

function tinyConfig(overrides: Partial<LtxVideoTransformerConfig> = {}): LtxVideoTransformerConfig {
  return {
    inChannels: 4,
    outChannels: 4,
    patchSize: 1,
    patchSizeT: 1,
    numAttentionHeads: 2,
    attentionHeadDim: 4,
    hiddenSize: 8,
    crossAttentionDim: 8,
    numLayers: 1,
    activationFn: "gelu-approximate",
    qkNorm: "rms_norm_across_heads",
    normElementwiseAffine: false,
    normEps: 1e-6,
    captionChannels: 6,
    attentionBias: true,
    attentionOutBias: true,
    rawConfig: {},
    ...overrides,
  };
}

function transformerComponent(weightPaths: readonly string[]): DiffusionSnapshotComponent {
  return {
    name: "transformer",
    role: "backbone",
    library: "diffusers",
    className: "LTXVideoTransformer3DModel",
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
  ["scaleShiftTable", "scale_shift_table"],
  ["timeEmbed.linear.weight", "time_embed.linear.weight"],
  ["timeEmbed.linear.bias", "time_embed.linear.bias"],
  [
    "timeEmbed.emb.timestepEmbedder.linear1.weight",
    "time_embed.emb.timestep_embedder.linear_1.weight",
  ],
  ["timeEmbed.emb.timestepEmbedder.linear1.bias", "time_embed.emb.timestep_embedder.linear_1.bias"],
  [
    "timeEmbed.emb.timestepEmbedder.linear2.weight",
    "time_embed.emb.timestep_embedder.linear_2.weight",
  ],
  ["timeEmbed.emb.timestepEmbedder.linear2.bias", "time_embed.emb.timestep_embedder.linear_2.bias"],
  ["captionProjection.linear1.weight", "caption_projection.linear_1.weight"],
  ["captionProjection.linear1.bias", "caption_projection.linear_1.bias"],
  ["captionProjection.linear2.weight", "caption_projection.linear_2.weight"],
  ["captionProjection.linear2.bias", "caption_projection.linear_2.bias"],
  ["projOut.weight", "proj_out.weight"],
  ["projOut.bias", "proj_out.bias"],
]);

const ATTENTION_CHECKPOINT_PATHS = new Map<string, string>([
  ["normQ", "norm_q"],
  ["normK", "norm_k"],
  ["toQ", "to_q"],
  ["toK", "to_k"],
  ["toV", "to_v"],
  ["toOut", "to_out.0"],
]);

function attentionCheckpointName(path: string): string | null {
  const attentionMatch = path.match(/^transformerBlocks\.(\d+)\.(attn[12])\.(.+)\.(.+)$/);
  if (attentionMatch !== null) {
    const [, index, attentionName, local, leaf] = attentionMatch;
    if (
      index === undefined ||
      attentionName === undefined ||
      local === undefined ||
      leaf === undefined
    ) {
      throw new Error("checkpointNameForTransformerPath: malformed attention path.");
    }
    const checkpointLocal = ATTENTION_CHECKPOINT_PATHS.get(local);
    if (checkpointLocal === undefined) {
      throw new Error(`checkpointNameForTransformerPath: unmapped path ${path}.`);
    }
    return `transformer_blocks.${index}.${attentionName}.${checkpointLocal}.${leaf}`;
  }
  return null;
}

function feedForwardCheckpointName(path: string): string | null {
  const firstFeedForward = path.match(/^transformerBlocks\.(\d+)\.ff\.linear1\.(.+)$/);
  if (firstFeedForward !== null) {
    const [, index, leaf] = firstFeedForward;
    if (index === undefined || leaf === undefined) {
      throw new Error("checkpointNameForTransformerPath: malformed linear1 path.");
    }
    return `transformer_blocks.${index}.ff.net.0.proj.${leaf}`;
  }

  const secondFeedForward = path.match(/^transformerBlocks\.(\d+)\.ff\.linear2\.(.+)$/);
  if (secondFeedForward !== null) {
    const [, index, leaf] = secondFeedForward;
    if (index === undefined || leaf === undefined) {
      throw new Error("checkpointNameForTransformerPath: malformed linear2 path.");
    }
    return `transformer_blocks.${index}.ff.net.2.${leaf}`;
  }
  return null;
}

function scaleShiftCheckpointName(path: string): string | null {
  const scaleShift = path.match(/^transformerBlocks\.(\d+)\.scaleShiftTable$/);
  if (scaleShift !== null) {
    const [, index] = scaleShift;
    if (index === undefined) {
      throw new Error("checkpointNameForTransformerPath: malformed scale-shift path.");
    }
    return `transformer_blocks.${index}.scale_shift_table`;
  }
  return null;
}

function checkpointNameForTransformerPath(path: string): string {
  const mapped =
    TOP_LEVEL_CHECKPOINT_PATHS.get(path) ??
    attentionCheckpointName(path) ??
    feedForwardCheckpointName(path) ??
    scaleShiftCheckpointName(path);
  if (mapped !== undefined && mapped !== null) {
    return mapped;
  }

  throw new Error(`checkpointNameForTransformerPath: unmapped path ${path}.`);
}

function checkpointTensorsForLtxTransformer(
  parameters: ParameterTree,
  options: { omitPath?: string; mismatchPath?: string; extra?: Record<string, MxArray> } = {},
): Record<string, MxArray> {
  const tensors: Record<string, MxArray> = {};
  for (const [path, parameter] of treeFlatten(parameters)) {
    const internalPath = path.join(".");
    if (internalPath === options.omitPath) {
      continue;
    }
    const checkpointName = checkpointNameForTransformerPath(internalPath);
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

function writeLtxSnapshotMetadata(directory: string): void {
  writeJson(join(directory, "model_index.json"), {
    _class_name: "LTXPipeline",
    scheduler: ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    text_encoder: ["transformers", "T5EncoderModel"],
    tokenizer: ["transformers", "T5Tokenizer"],
    transformer: ["diffusers", "LTXVideoTransformer3DModel"],
    vae: ["diffusers", "AutoencoderKLLTXVideo"],
  });

  const config = tinyConfig();
  const transformerDirectory = join(directory, "transformer");
  mkdirSync(transformerDirectory, { recursive: true });
  writeJson(join(transformerDirectory, "config.json"), {
    _class_name: "LTXVideoTransformer3DModel",
    activation_fn: config.activationFn,
    attention_bias: config.attentionBias,
    attention_head_dim: config.attentionHeadDim,
    attention_out_bias: config.attentionOutBias,
    caption_channels: config.captionChannels,
    cross_attention_dim: config.crossAttentionDim,
    in_channels: config.inChannels,
    norm_elementwise_affine: config.normElementwiseAffine,
    norm_eps: config.normEps,
    num_attention_heads: config.numAttentionHeads,
    num_layers: config.numLayers,
    out_channels: config.outChannels,
    patch_size: config.patchSize,
    patch_size_t: config.patchSizeT,
    qk_norm: config.qkNorm,
  });

  const vaeDirectory = join(directory, "vae");
  mkdirSync(vaeDirectory, { recursive: true });
  writeJson(join(vaeDirectory, "config.json"), {
    _class_name: "AutoencoderKLLTXVideo",
    block_out_channels: [4],
    decoder_block_out_channels: [4],
    decoder_inject_noise: [false, false],
    decoder_layers_per_block: [1, 1],
    decoder_spatio_temporal_scaling: [false],
    downsample_type: ["conv"],
    in_channels: 3,
    layers_per_block: [1, 1],
    latent_channels: config.inChannels,
    out_channels: 3,
    spatio_temporal_scaling: [false],
    upsample_factor: [1],
    upsample_residual: [false],
  });
  writeFileSync(join(vaeDirectory, "diffusion_pytorch_model.safetensors"), "");

  mkdirSync(join(directory, "scheduler"), { recursive: true });
  writeJson(join(directory, "scheduler", "scheduler_config.json"), {
    _class_name: "FlowMatchEulerDiscreteScheduler",
  });
  mkdirSync(join(directory, "text_encoder"), { recursive: true });
  writeJson(join(directory, "text_encoder", "config.json"), {});
  writeFileSync(join(directory, "text_encoder", "model.safetensors"), "");
  mkdirSync(join(directory, "tokenizer"), { recursive: true });
  writeFileSync(join(directory, "tokenizer", "tokenizer.json"), "{}");
}

describe("LTX-Video transformer weight mapping", () => {
  test("maps top-level Diffusers parameters", () => {
    expect(ltxVideoTransformerWeightPath("proj_in.weight")).toBe("projIn.weight");
    expect(ltxVideoTransformerWeightPath("scale_shift_table")).toBe("scaleShiftTable");
    expect(ltxVideoTransformerWeightPath("time_embed.linear.bias")).toBe("timeEmbed.linear.bias");
    expect(ltxVideoTransformerWeightPath("time_embed.emb.timestep_embedder.linear_1.weight")).toBe(
      "timeEmbed.emb.timestepEmbedder.linear1.weight",
    );
    expect(ltxVideoTransformerWeightPath("caption_projection.linear_2.bias")).toBe(
      "captionProjection.linear2.bias",
    );
    expect(ltxVideoTransformerWeightPath("proj_out.weight")).toBe("projOut.weight");
  });

  test("maps block attention, feed-forward, and scale-shift parameters", () => {
    expect(ltxVideoTransformerWeightPath("transformer_blocks.3.attn1.norm_q.weight")).toBe(
      "transformerBlocks.3.attn1.normQ.weight",
    );
    expect(ltxVideoTransformerWeightPath("transformer_blocks.3.attn2.to_k.bias")).toBe(
      "transformerBlocks.3.attn2.toK.bias",
    );
    expect(ltxVideoTransformerWeightPath("transformer_blocks.3.attn1.to_out.0.weight")).toBe(
      "transformerBlocks.3.attn1.toOut.weight",
    );
    expect(ltxVideoTransformerWeightPath("transformer_blocks.3.ff.net.0.proj.weight")).toBe(
      "transformerBlocks.3.ff.linear1.weight",
    );
    expect(ltxVideoTransformerWeightPath("transformer_blocks.3.ff.net.2.bias")).toBe(
      "transformerBlocks.3.ff.linear2.bias",
    );
    expect(ltxVideoTransformerWeightPath("transformer_blocks.3.scale_shift_table")).toBe(
      "transformerBlocks.3.scaleShiftTable",
    );
  });

  test("does not map no-parameter dropout and unknown checkpoint paths", () => {
    expect(ltxVideoTransformerWeightPath("transformer_blocks.3.attn1.to_out.1.weight")).toBeNull();
    expect(ltxVideoTransformerWeightPath("transformer_blocks.3.ff.net.1.weight")).toBeNull();
    expect(ltxVideoTransformerWeightPath("rope.freqs")).toBeNull();
  });

  test("loads a complete generated transformer safetensors shard", async () => {
    await withTempDirectory("mlxts-ltx-transformer-weights-", async (directory) => {
      using target = new LtxVideoTransformer3DModel(tinyConfig());
      using source = new LtxVideoTransformer3DModel(tinyConfig());
      const checkpointPath = join(directory, "diffusion_pytorch_model.safetensors");
      await writeCheckpoint(
        checkpointPath,
        checkpointTensorsForLtxTransformer(source.parameters()),
      );

      const result = await loadLtxVideoTransformerWeights(
        target,
        transformerComponent([checkpointPath]),
      );

      expect(result.shardCount).toBe(1);
      expect(result.unexpectedWeights).toEqual([]);
      expect(result.assignedPaths).toContain("projIn.weight");
      expect(result.assignedPaths).toContain("transformerBlocks.0.attn1.normQ.weight");
      expect(result.assignedPaths).toContain("transformerBlocks.0.ff.linear2.weight");
    });
  });

  test("loads transformer shards from a Diffusers safetensors index", async () => {
    await withTempDirectory("mlxts-ltx-transformer-index-", async (directory) => {
      using target = new LtxVideoTransformer3DModel(tinyConfig());
      using source = new LtxVideoTransformer3DModel(tinyConfig());
      const shards = splitIndexedShards(checkpointTensorsForLtxTransformer(source.parameters()));
      const firstPath = join(directory, "diffusion_pytorch_model-00001-of-00002.safetensors");
      const secondPath = join(directory, "diffusion_pytorch_model-00002-of-00002.safetensors");
      await writeCheckpoint(firstPath, shards.firstShard);
      await writeCheckpoint(secondPath, shards.secondShard);
      const indexPath = join(directory, "diffusion_pytorch_model.safetensors.index.json");
      writeJson(indexPath, { metadata: {}, weight_map: shards.weightMap });

      const result = await loadLtxVideoTransformerWeights(
        target,
        transformerComponent([indexPath]),
      );

      expect(result.shardCount).toBe(2);
      expect(result.unexpectedWeights).toEqual([]);
    });
  });

  test("rejects invalid LTX transformer shard manifests", async () => {
    await withTempDirectory("mlxts-ltx-transformer-shards-invalid-", async (directory) => {
      using target = new LtxVideoTransformer3DModel(tinyConfig());
      const firstIndex = join(directory, "first.safetensors.index.json");
      const secondIndex = join(directory, "second.safetensors.index.json");
      writeJson(firstIndex, { metadata: {}, weight_map: {} });
      writeJson(secondIndex, { metadata: {}, weight_map: {} });

      await expect(
        loadLtxVideoTransformerWeights(target, transformerComponent([])),
      ).rejects.toThrow("has no safetensors weight shards");
      await expect(
        loadLtxVideoTransformerWeights(target, transformerComponent([firstIndex, secondIndex])),
      ).rejects.toThrow("multiple safetensors index files");

      const invalidJsonPath = join(directory, "invalid.safetensors.index.json");
      writeFileSync(invalidJsonPath, "{");
      await expect(
        loadLtxVideoTransformerWeights(target, transformerComponent([invalidJsonPath])),
      ).rejects.toThrow("valid JSON");

      const nonObjectPath = join(directory, "non-object.safetensors.index.json");
      writeJson(nonObjectPath, []);
      await expect(
        loadLtxVideoTransformerWeights(target, transformerComponent([nonObjectPath])),
      ).rejects.toThrow("must be a JSON object");

      const badWeightMapPath = join(directory, "bad-weight-map.safetensors.index.json");
      writeJson(badWeightMapPath, { metadata: {}, weight_map: [] });
      await expect(
        loadLtxVideoTransformerWeights(target, transformerComponent([badWeightMapPath])),
      ).rejects.toThrow("weight_map must be a JSON object");

      const badShardPath = join(directory, "bad-shard.safetensors.index.json");
      writeJson(badShardPath, { metadata: {}, weight_map: { "proj_in.weight": "" } });
      await expect(
        loadLtxVideoTransformerWeights(target, transformerComponent([badShardPath])),
      ).rejects.toThrow("must be a shard filename");
    });
  });

  test("loads transformer directly from an inspected LTX-Video snapshot manifest", async () => {
    await withTempDirectory("mlxts-ltx-transformer-snapshot-", async (directory) => {
      writeLtxSnapshotMetadata(directory);
      using source = new LtxVideoTransformer3DModel(tinyConfig());
      await writeCheckpoint(
        join(directory, "transformer", "diffusion_pytorch_model.safetensors"),
        checkpointTensorsForLtxTransformer(source.parameters()),
      );
      const manifest = await loadDiffusionSnapshotManifest(directory);

      using target = await loadLtxVideoTransformerFromSnapshot(manifest);

      expect(target.projIn.weight.shape).toEqual([8, 4]);
      expect(target.transformerBlocks[0]?.attn1.toQ.weight.shape).toEqual([8, 8]);
    });
  });

  test("disposes partial snapshot transformer loads on failure", async () => {
    await withTempDirectory("mlxts-ltx-transformer-snapshot-failure-", async (directory) => {
      writeLtxSnapshotMetadata(directory);
      using source = new LtxVideoTransformer3DModel(tinyConfig());
      await writeCheckpoint(
        join(directory, "transformer", "diffusion_pytorch_model.safetensors"),
        checkpointTensorsForLtxTransformer(source.parameters(), {
          omitPath: "transformerBlocks.0.attn2.toOut.weight",
        }),
      );
      const manifest = await loadDiffusionSnapshotManifest(directory);

      await expect(loadLtxVideoTransformerFromSnapshot(manifest)).rejects.toThrow(
        DiffusionMissingWeightsError,
      );
    });
  });

  test("rejects missing, mismatched, and strict unexpected transformer weights", async () => {
    await withTempDirectory("mlxts-ltx-transformer-invalid-", async (directory) => {
      using source = new LtxVideoTransformer3DModel(tinyConfig());

      const missingPath = join(directory, "missing.safetensors");
      await writeCheckpoint(
        missingPath,
        checkpointTensorsForLtxTransformer(source.parameters(), {
          omitPath: "transformerBlocks.0.attn2.toOut.weight",
        }),
      );
      using missingTarget = new LtxVideoTransformer3DModel(tinyConfig());
      await expect(
        loadLtxVideoTransformerWeights(missingTarget, transformerComponent([missingPath])),
      ).rejects.toThrow(DiffusionMissingWeightsError);

      const mismatchPath = join(directory, "mismatch.safetensors");
      await writeCheckpoint(
        mismatchPath,
        checkpointTensorsForLtxTransformer(source.parameters(), {
          mismatchPath: "transformerBlocks.0.attn1.toQ.weight",
        }),
      );
      using mismatchTarget = new LtxVideoTransformer3DModel(tinyConfig());
      await expect(
        loadLtxVideoTransformerWeights(mismatchTarget, transformerComponent([mismatchPath])),
      ).rejects.toThrow(DiffusionWeightMismatchError);

      const unexpectedPath = join(directory, "unexpected.safetensors");
      await writeCheckpoint(
        unexpectedPath,
        checkpointTensorsForLtxTransformer(source.parameters(), {
          extra: { "transformer_blocks.0.attn1.to_out.1.weight": zeros([8]) },
        }),
      );
      using unexpectedTarget = new LtxVideoTransformer3DModel(tinyConfig());
      await expect(
        loadLtxVideoTransformerWeights(unexpectedTarget, transformerComponent([unexpectedPath]), {
          strictUnexpectedWeights: true,
        }),
      ).rejects.toThrow("unexpected unmapped weights");
    });
  });
});
