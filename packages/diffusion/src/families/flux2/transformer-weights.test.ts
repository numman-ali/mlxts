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
import type { Flux2KleinTransformerConfig } from "./config";
import { Flux2KleinTransformer2DModel } from "./transformer";
import {
  flux2KleinTransformerWeightPath,
  loadFlux2KleinTransformerFromSnapshot,
  loadFlux2KleinTransformerWeights,
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

function tinyFlux2TransformerConfig(
  overrides: Partial<Flux2KleinTransformerConfig> = {},
): Flux2KleinTransformerConfig {
  return {
    patchSize: 1,
    inChannels: 4,
    latentChannels: 1,
    outChannels: 4,
    numLayers: 1,
    numSingleLayers: 1,
    attentionHeadDim: 8,
    numAttentionHeads: 1,
    hiddenSize: 8,
    mlpRatio: 2,
    jointAttentionDim: 6,
    timestepGuidanceChannels: 8,
    axesDimsRope: [2, 2, 2, 2],
    ropeTheta: 2000,
    normEps: 1e-6,
    guidanceEmbeds: true,
    rawConfig: {},
    ...overrides,
  };
}

function transformerComponent(weightPaths: readonly string[]): DiffusionSnapshotComponent {
  return {
    name: "transformer",
    role: "backbone",
    library: "diffusers",
    className: "Flux2Transformer2DModel",
    enabled: true,
    optional: false,
    subfolder: "transformer",
    metadataPaths: [],
    weightPaths,
  };
}

const TOP_LEVEL_CHECKPOINT_PATHS = new Map<string, string>([
  [
    "timeGuidanceEmbed.timestepEmbedder.linear1.weight",
    "time_guidance_embed.timestep_embedder.linear_1.weight",
  ],
  [
    "timeGuidanceEmbed.timestepEmbedder.linear2.weight",
    "time_guidance_embed.timestep_embedder.linear_2.weight",
  ],
  [
    "timeGuidanceEmbed.guidanceEmbedder.linear1.weight",
    "time_guidance_embed.guidance_embedder.linear_1.weight",
  ],
  [
    "timeGuidanceEmbed.guidanceEmbedder.linear2.weight",
    "time_guidance_embed.guidance_embedder.linear_2.weight",
  ],
  ["doubleStreamModulationImg.linear.weight", "double_stream_modulation_img.linear.weight"],
  ["doubleStreamModulationTxt.linear.weight", "double_stream_modulation_txt.linear.weight"],
  ["singleStreamModulation.linear.weight", "single_stream_modulation.linear.weight"],
  ["xEmbedder.weight", "x_embedder.weight"],
  ["contextEmbedder.weight", "context_embedder.weight"],
  ["normOut.linear.weight", "norm_out.linear.weight"],
  ["projOut.weight", "proj_out.weight"],
]);

const DOUBLE_BLOCK_CHECKPOINT_PATHS = new Map<string, string>([
  ["attn.toQ", "attn.to_q"],
  ["attn.toK", "attn.to_k"],
  ["attn.toV", "attn.to_v"],
  ["attn.addQProj", "attn.add_q_proj"],
  ["attn.addKProj", "attn.add_k_proj"],
  ["attn.addVProj", "attn.add_v_proj"],
  ["attn.norm.queryNorm", "attn.norm_q"],
  ["attn.norm.keyNorm", "attn.norm_k"],
  ["attn.addedNorm.queryNorm", "attn.norm_added_q"],
  ["attn.addedNorm.keyNorm", "attn.norm_added_k"],
  ["attn.toOut", "attn.to_out.0"],
  ["attn.toAddOut", "attn.to_add_out"],
  ["ff.linearIn", "ff.linear_in"],
  ["ff.linearOut", "ff.linear_out"],
  ["ffContext.linearIn", "ff_context.linear_in"],
  ["ffContext.linearOut", "ff_context.linear_out"],
]);

const SINGLE_BLOCK_CHECKPOINT_PATHS = new Map<string, string>([
  ["attn.toQkvMlpProj", "attn.to_qkv_mlp_proj"],
  ["attn.norm.queryNorm", "attn.norm_q"],
  ["attn.norm.keyNorm", "attn.norm_k"],
  ["attn.toOut", "attn.to_out"],
]);

function checkpointNameForTransformerPath(path: string): string {
  const topLevel = TOP_LEVEL_CHECKPOINT_PATHS.get(path);
  if (topLevel !== undefined) {
    return topLevel;
  }

  const doubleMatch = path.match(/^transformerBlocks\.(\d+)\.(.+)\.weight$/);
  if (doubleMatch !== null) {
    const [, index, local] = doubleMatch;
    if (index === undefined || local === undefined) {
      throw new Error("checkpointNameForTransformerPath: malformed double block path.");
    }
    const checkpointLocal = DOUBLE_BLOCK_CHECKPOINT_PATHS.get(local);
    if (checkpointLocal === undefined) {
      throw new Error(`checkpointNameForTransformerPath: unmapped path ${path}.`);
    }
    return `transformer_blocks.${index}.${checkpointLocal}.weight`;
  }

  const singleMatch = path.match(/^singleTransformerBlocks\.(\d+)\.(.+)\.weight$/);
  if (singleMatch !== null) {
    const [, index, local] = singleMatch;
    if (index === undefined || local === undefined) {
      throw new Error("checkpointNameForTransformerPath: malformed single block path.");
    }
    const checkpointLocal = SINGLE_BLOCK_CHECKPOINT_PATHS.get(local);
    if (checkpointLocal === undefined) {
      throw new Error(`checkpointNameForTransformerPath: unmapped path ${path}.`);
    }
    return `single_transformer_blocks.${index}.${checkpointLocal}.weight`;
  }

  throw new Error(`checkpointNameForTransformerPath: unmapped path ${path}.`);
}

function checkpointTensorsForFlux2Transformer(
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

function writeFlux2SnapshotMetadata(
  directory: string,
  transformerOverrides: Partial<Flux2KleinTransformerConfig> = {},
): void {
  writeJson(join(directory, "model_index.json"), {
    _class_name: "Flux2KleinPipeline",
    is_distilled: true,
    scheduler: ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    text_encoder: ["transformers", "Qwen3ForCausalLM"],
    tokenizer: ["transformers", "Qwen2TokenizerFast"],
    transformer: ["diffusers", "Flux2Transformer2DModel"],
    vae: ["diffusers", "AutoencoderKLFlux2"],
  });
  const config = tinyFlux2TransformerConfig(transformerOverrides);
  const transformerDirectory = join(directory, "transformer");
  mkdirSync(transformerDirectory, { recursive: true });
  writeJson(join(transformerDirectory, "config.json"), {
    _class_name: "Flux2Transformer2DModel",
    patch_size: config.patchSize,
    in_channels: config.inChannels,
    out_channels: config.outChannels,
    num_layers: config.numLayers,
    num_single_layers: config.numSingleLayers,
    attention_head_dim: config.attentionHeadDim,
    num_attention_heads: config.numAttentionHeads,
    joint_attention_dim: config.jointAttentionDim,
    timestep_guidance_channels: config.timestepGuidanceChannels,
    mlp_ratio: config.mlpRatio,
    axes_dims_rope: config.axesDimsRope,
    rope_theta: config.ropeTheta,
    eps: config.normEps,
    guidance_embeds: config.guidanceEmbeds,
  });

  const vaeDirectory = join(directory, "vae");
  mkdirSync(vaeDirectory, { recursive: true });
  writeJson(join(vaeDirectory, "config.json"), {
    _class_name: "AutoencoderKLFlux2",
    block_out_channels: [4],
    decoder_block_out_channels: [4],
    down_block_types: ["DownEncoderBlock2D"],
    in_channels: 3,
    latent_channels: config.latentChannels,
    layers_per_block: 1,
    norm_num_groups: 1,
    out_channels: 3,
    patch_size: [2, 2],
    sample_size: 16,
    up_block_types: ["UpDecoderBlock2D"],
  });
  writeFileSync(join(vaeDirectory, "diffusion_pytorch_model.safetensors"), "");

  const schedulerDirectory = join(directory, "scheduler");
  mkdirSync(schedulerDirectory, { recursive: true });
  writeJson(join(schedulerDirectory, "scheduler_config.json"), {
    _class_name: "FlowMatchEulerDiscreteScheduler",
    num_train_timesteps: 1000,
    shift: 3,
  });

  mkdirSync(join(directory, "text_encoder"), { recursive: true });
  writeJson(join(directory, "text_encoder", "config.json"), {});
  writeFileSync(join(directory, "text_encoder", "model.safetensors"), "");
  mkdirSync(join(directory, "tokenizer"), { recursive: true });
  writeJson(join(directory, "tokenizer", "tokenizer.json"), {});
}

describe("FLUX.2 Klein transformer weight loading", () => {
  test("maps Diffusers transformer names onto the package parameter tree", () => {
    expect(
      flux2KleinTransformerWeightPath("time_guidance_embed.timestep_embedder.linear_1.weight"),
    ).toBe("timeGuidanceEmbed.timestepEmbedder.linear1.weight");
    expect(flux2KleinTransformerWeightPath("double_stream_modulation_img.linear.weight")).toBe(
      "doubleStreamModulationImg.linear.weight",
    );
    expect(flux2KleinTransformerWeightPath("transformer_blocks.0.attn.to_q.weight")).toBe(
      "transformerBlocks.0.attn.toQ.weight",
    );
    expect(
      flux2KleinTransformerWeightPath("transformer_blocks.0.ff_context.linear_out.weight"),
    ).toBe("transformerBlocks.0.ffContext.linearOut.weight");
    expect(
      flux2KleinTransformerWeightPath("single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight"),
    ).toBe("singleTransformerBlocks.0.attn.toQkvMlpProj.weight");
    expect(flux2KleinTransformerWeightPath("norm_out.linear.weight")).toBe("normOut.linear.weight");
    expect(flux2KleinTransformerWeightPath("transformer_blocks.0.attn.to_q.bias")).toBeNull();
  });

  test("loads a complete generated transformer safetensors shard", async () => {
    await withTempDirectory("mlxts-flux2-transformer-weights-", async (directory) => {
      using target = new Flux2KleinTransformer2DModel(tinyFlux2TransformerConfig());
      using source = new Flux2KleinTransformer2DModel(tinyFlux2TransformerConfig());
      const checkpointPath = join(directory, "diffusion_pytorch_model.safetensors");
      await writeCheckpoint(
        checkpointPath,
        checkpointTensorsForFlux2Transformer(source.parameters()),
      );

      const result = await loadFlux2KleinTransformerWeights(
        target,
        transformerComponent([checkpointPath]),
      );

      expect(result.shardCount).toBe(1);
      expect(result.unexpectedWeights).toEqual([]);
      expect(result.assignedPaths).toContain("xEmbedder.weight");
      expect(result.assignedPaths).toContain("transformerBlocks.0.ffContext.linearOut.weight");
      expect(result.assignedPaths).toContain("singleTransformerBlocks.0.attn.toQkvMlpProj.weight");
    });
  });

  test("loads transformer shards from a Diffusers safetensors index", async () => {
    await withTempDirectory("mlxts-flux2-transformer-index-", async (directory) => {
      using target = new Flux2KleinTransformer2DModel(tinyFlux2TransformerConfig());
      using source = new Flux2KleinTransformer2DModel(tinyFlux2TransformerConfig());
      const shards = splitIndexedShards(checkpointTensorsForFlux2Transformer(source.parameters()));
      const firstPath = join(directory, "diffusion_pytorch_model-00001-of-00002.safetensors");
      const secondPath = join(directory, "diffusion_pytorch_model-00002-of-00002.safetensors");
      await writeCheckpoint(firstPath, shards.firstShard);
      await writeCheckpoint(secondPath, shards.secondShard);
      const indexPath = join(directory, "diffusion_pytorch_model.safetensors.index.json");
      writeJson(indexPath, { metadata: {}, weight_map: shards.weightMap });

      const result = await loadFlux2KleinTransformerWeights(
        target,
        transformerComponent([indexPath]),
      );

      expect(result.shardCount).toBe(2);
      expect(result.unexpectedWeights).toEqual([]);
    });
  });

  test("loads transformer directly from an inspected FLUX.2 Klein snapshot manifest", async () => {
    await withTempDirectory("mlxts-flux2-transformer-snapshot-", async (directory) => {
      writeFlux2SnapshotMetadata(directory);
      using source = new Flux2KleinTransformer2DModel(tinyFlux2TransformerConfig());
      await writeCheckpoint(
        join(directory, "transformer", "diffusion_pytorch_model.safetensors"),
        checkpointTensorsForFlux2Transformer(source.parameters()),
      );
      const manifest = await loadDiffusionSnapshotManifest(directory);

      using target = await loadFlux2KleinTransformerFromSnapshot(manifest);

      expect(target.xEmbedder.weight.shape).toEqual([8, 4]);
      expect(target.transformerBlocks[0]?.attn.toQ.weight.shape).toEqual([8, 8]);
    });
  });

  test("rejects missing, mismatched, and strict unexpected transformer weights", async () => {
    await withTempDirectory("mlxts-flux2-transformer-invalid-", async (directory) => {
      using source = new Flux2KleinTransformer2DModel(tinyFlux2TransformerConfig());

      const missingPath = join(directory, "missing.safetensors");
      await writeCheckpoint(
        missingPath,
        checkpointTensorsForFlux2Transformer(source.parameters(), {
          omitPath: "singleTransformerBlocks.0.attn.toOut.weight",
        }),
      );
      using missingTarget = new Flux2KleinTransformer2DModel(tinyFlux2TransformerConfig());
      await expect(
        loadFlux2KleinTransformerWeights(missingTarget, transformerComponent([missingPath])),
      ).rejects.toThrow(DiffusionMissingWeightsError);

      const mismatchPath = join(directory, "mismatch.safetensors");
      await writeCheckpoint(
        mismatchPath,
        checkpointTensorsForFlux2Transformer(source.parameters(), {
          mismatchPath: "transformerBlocks.0.attn.toQ.weight",
        }),
      );
      using mismatchTarget = new Flux2KleinTransformer2DModel(tinyFlux2TransformerConfig());
      await expect(
        loadFlux2KleinTransformerWeights(mismatchTarget, transformerComponent([mismatchPath])),
      ).rejects.toThrow(DiffusionWeightMismatchError);

      const unexpectedPath = join(directory, "unexpected.safetensors");
      await writeCheckpoint(
        unexpectedPath,
        checkpointTensorsForFlux2Transformer(source.parameters(), {
          extra: { "transformer_blocks.0.attn.to_q.bias": zeros([8]) },
        }),
      );
      using unexpectedTarget = new Flux2KleinTransformer2DModel(tinyFlux2TransformerConfig());
      await expect(
        loadFlux2KleinTransformerWeights(unexpectedTarget, transformerComponent([unexpectedPath]), {
          strictUnexpectedWeights: true,
        }),
      ).rejects.toThrow("unexpected unmapped weights");
    });
  });

  test("loads non-guidance FLUX.2 transformer checkpoints without guidance embedder weights", async () => {
    await withTempDirectory("mlxts-flux2-transformer-no-guidance-", async (directory) => {
      const config = tinyFlux2TransformerConfig({ guidanceEmbeds: false });
      using target = new Flux2KleinTransformer2DModel(config);
      using source = new Flux2KleinTransformer2DModel(config);
      const checkpointPath = join(directory, "diffusion_pytorch_model.safetensors");
      await writeCheckpoint(
        checkpointPath,
        checkpointTensorsForFlux2Transformer(source.parameters()),
      );

      const result = await loadFlux2KleinTransformerWeights(
        target,
        transformerComponent([checkpointPath]),
      );

      expect(result.assignedPaths).not.toContain(
        "timeGuidanceEmbed.guidanceEmbedder.linear1.weight",
      );
      expect(result.unexpectedWeights).toEqual([]);
    });
  });
});
