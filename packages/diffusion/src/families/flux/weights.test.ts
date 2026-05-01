import { describe, expect, test } from "bun:test";
import {
  array,
  type MxArray,
  type ParameterTree,
  saveSafetensors,
  treeFlatten,
  zeros,
} from "@mlxts/core";
import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { DiffusionMissingWeightsError, DiffusionWeightMismatchError } from "../../errors";
import type { DiffusionSnapshotComponent } from "../../pretrained/snapshot-manifest";
import type { FluxTransformerConfig } from "./config";
import { FluxTransformer2DModel } from "./transformer";
import {
  fluxTransformerWeightPath,
  loadFluxTransformerWeights,
  transformFluxTransformerWeight,
} from "./weights";

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

function tinyFluxConfig(overrides: Partial<FluxTransformerConfig> = {}): FluxTransformerConfig {
  return {
    patchSize: 1,
    inChannels: 4,
    latentChannels: 1,
    outChannels: 4,
    numLayers: 1,
    numSingleLayers: 1,
    attentionHeadDim: 6,
    numAttentionHeads: 2,
    hiddenSize: 12,
    mlpRatio: 2,
    jointAttentionDim: 5,
    pooledProjectionDim: 7,
    guidanceEmbeds: true,
    axesDimsRope: [2, 2, 2],
    ropeTheta: 10000,
    qkvBias: true,
    rawConfig: {},
    ...overrides,
  };
}

function transformerComponent(weightPaths: readonly string[]): DiffusionSnapshotComponent {
  return {
    name: "transformer",
    role: "backbone",
    library: "diffusers",
    className: "FluxTransformer2DModel",
    enabled: true,
    optional: false,
    subfolder: "transformer",
    metadataPaths: [],
    weightPaths,
  };
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

function checkpointTensorsForFluxParameters(
  parameters: ParameterTree,
  config: FluxTransformerConfig,
): Record<string, MxArray> {
  const tensors: Record<string, MxArray> = {};
  for (const [path, parameter] of treeFlatten(parameters)) {
    const internalPath = path.join(".");
    addFluxCheckpointTensors(tensors, internalPath, parameter.shape, config);
  }
  return tensors;
}

function addFluxCheckpointTensors(
  tensors: Record<string, MxArray>,
  internalPath: string,
  shape: readonly number[],
  config: FluxTransformerConfig,
): void {
  const doubleQkv = internalPath.match(
    /^doubleBlocks\.(\d+)\.(imageAttention|textAttention)\.qkv\.(weight|bias)$/,
  );
  if (doubleQkv !== null) {
    const [, index, stream, leaf] = doubleQkv;
    if (index === undefined || stream === undefined || leaf === undefined) {
      throw new Error("addFluxCheckpointTensors: malformed double qkv path.");
    }
    const parts =
      stream === "imageAttention"
        ? ["attn.to_q", "attn.to_k", "attn.to_v"]
        : ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"];
    addFusedCheckpointTensors(tensors, `transformer_blocks.${index}`, parts, leaf, shape, [
      config.hiddenSize,
      config.hiddenSize,
      config.hiddenSize,
    ]);
    return;
  }

  const singleLinear1 = internalPath.match(/^singleBlocks\.(\d+)\.linear1\.(weight|bias)$/);
  if (singleLinear1 !== null) {
    const [, index, leaf] = singleLinear1;
    if (index === undefined || leaf === undefined) {
      throw new Error("addFluxCheckpointTensors: malformed single linear1 path.");
    }
    addFusedCheckpointTensors(
      tensors,
      `single_transformer_blocks.${index}`,
      ["attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp"],
      leaf,
      shape,
      [
        config.hiddenSize,
        config.hiddenSize,
        config.hiddenSize,
        Math.round(config.hiddenSize * config.mlpRatio),
      ],
    );
    return;
  }

  const checkpointName = fluxCheckpointNameForPath(internalPath);
  if (checkpointName !== null) {
    tensors[checkpointName] = zeros([...shape]);
  }
}

function addFusedCheckpointTensors(
  tensors: Record<string, MxArray>,
  prefix: string,
  parts: readonly string[],
  leaf: string,
  shape: readonly number[],
  partRows: readonly number[],
): void {
  const [, columns] = shape;
  if (partRows.length !== parts.length) {
    throw new Error("addFusedCheckpointTensors: row sizes must match fused parts.");
  }
  for (let index = 0; index < parts.length; index += 1) {
    const part = parts[index];
    const rows = partRows[index];
    if (part === undefined) {
      throw new Error("addFusedCheckpointTensors: missing fused part.");
    }
    if (rows === undefined) {
      throw new Error("addFusedCheckpointTensors: missing fused row size.");
    }
    const partShape = leaf === "weight" ? [rows, columns ?? 0] : [rows];
    tensors[`${prefix}.${part}.${leaf}`] = zeros(partShape);
  }
}

function fluxCheckpointNameForPath(path: string): string | null {
  const topLevel = new Map<string, string>([
    ["imageProjection.weight", "x_embedder.weight"],
    ["imageProjection.bias", "x_embedder.bias"],
    ["textProjection.weight", "context_embedder.weight"],
    ["textProjection.bias", "context_embedder.bias"],
    ["timeEmbedding.linear1.weight", "time_text_embed.timestep_embedder.linear_1.weight"],
    ["timeEmbedding.linear1.bias", "time_text_embed.timestep_embedder.linear_1.bias"],
    ["timeEmbedding.linear2.weight", "time_text_embed.timestep_embedder.linear_2.weight"],
    ["timeEmbedding.linear2.bias", "time_text_embed.timestep_embedder.linear_2.bias"],
    ["guidanceEmbedding.linear1.weight", "time_text_embed.guidance_embedder.linear_1.weight"],
    ["guidanceEmbedding.linear1.bias", "time_text_embed.guidance_embedder.linear_1.bias"],
    ["guidanceEmbedding.linear2.weight", "time_text_embed.guidance_embedder.linear_2.weight"],
    ["guidanceEmbedding.linear2.bias", "time_text_embed.guidance_embedder.linear_2.bias"],
    ["vectorEmbedding.linear1.weight", "time_text_embed.text_embedder.linear_1.weight"],
    ["vectorEmbedding.linear1.bias", "time_text_embed.text_embedder.linear_1.bias"],
    ["vectorEmbedding.linear2.weight", "time_text_embed.text_embedder.linear_2.weight"],
    ["vectorEmbedding.linear2.bias", "time_text_embed.text_embedder.linear_2.bias"],
    ["finalLayer.modulation.weight", "norm_out.linear.weight"],
    ["finalLayer.modulation.bias", "norm_out.linear.bias"],
    ["finalLayer.projection.weight", "proj_out.weight"],
    ["finalLayer.projection.bias", "proj_out.bias"],
  ]);
  const mapped = topLevel.get(path);
  if (mapped !== undefined) {
    return mapped;
  }

  return path
    .replaceAll("doubleBlocks", "transformer_blocks")
    .replaceAll("imageModulation.linear", "norm1.linear")
    .replaceAll("textModulation.linear", "norm1_context.linear")
    .replaceAll("imageFeedForward.linear1", "ff.net.0.proj")
    .replaceAll("imageFeedForward.linear2", "ff.net.2")
    .replaceAll("textFeedForward.linear1", "ff_context.net.0.proj")
    .replaceAll("textFeedForward.linear2", "ff_context.net.2")
    .replaceAll("imageAttention.norm.queryNorm", "attn.norm_q")
    .replaceAll("imageAttention.norm.keyNorm", "attn.norm_k")
    .replaceAll("textAttention.norm.queryNorm", "attn.norm_added_q")
    .replaceAll("textAttention.norm.keyNorm", "attn.norm_added_k")
    .replaceAll("imageAttention.projection", "attn.to_out.0")
    .replaceAll("textAttention.projection", "attn.to_add_out")
    .replaceAll("singleBlocks", "single_transformer_blocks")
    .replaceAll("modulation.linear", "norm.linear")
    .replaceAll("norm.queryNorm", "attn.norm_q")
    .replaceAll("norm.keyNorm", "attn.norm_k")
    .replaceAll("linear2", "proj_out");
}

describe("FLUX transformer weight loading", () => {
  test("maps Diffusers FLUX names onto the package parameter tree", () => {
    expect(fluxTransformerWeightPath("x_embedder.weight")).toBe("imageProjection.weight");
    expect(fluxTransformerWeightPath("time_text_embed.text_embedder.linear_2.bias")).toBe(
      "vectorEmbedding.linear2.bias",
    );
    expect(fluxTransformerWeightPath("transformer_blocks.0.norm1_context.linear.weight")).toBe(
      "doubleBlocks.0.textModulation.linear.weight",
    );
    expect(fluxTransformerWeightPath("transformer_blocks.0.attn.add_v_proj.bias")).toBe(
      "doubleBlocks.0.textAttention.qkv.bias",
    );
    expect(fluxTransformerWeightPath("single_transformer_blocks.0.attn.norm_k.weight")).toBe(
      "singleBlocks.0.norm.keyNorm.weight",
    );
    expect(fluxTransformerWeightPath("single_transformer_blocks.0.proj_mlp.weight")).toBe(
      "singleBlocks.0.linear1.weight",
    );
    expect(fluxTransformerWeightPath("proj_out.bias")).toBe("finalLayer.projection.bias");
  });

  test("loads a complete generated transformer safetensors shard", async () => {
    await withTempDirectory("mlxts-flux-transformer-weights-", async (directory) => {
      const config = tinyFluxConfig();
      using source = new FluxTransformer2DModel(config);
      const checkpointPath = join(directory, "diffusion_pytorch_model.safetensors");
      await writeCheckpoint(
        checkpointPath,
        checkpointTensorsForFluxParameters(source.parameters(), config),
      );
      using target = new FluxTransformer2DModel(config);

      const result = await loadFluxTransformerWeights(
        target,
        transformerComponent([checkpointPath]),
      );

      expect(result.shardCount).toBe(1);
      expect(result.unexpectedWeights).toEqual([]);
      expect(result.assignedPaths).toEqual(
        treeFlatten(target.parameters())
          .map(([path]) => path.join("."))
          .toSorted((left, right) => left.localeCompare(right)),
      );
      expect(target.doubleBlocks[0]?.imageAttention.qkv.weight.shape).toEqual([36, 12]);
      expect(target.singleBlocks[0]?.linear1.weight.shape).toEqual([60, 12]);
    });
  });

  test("loads transformer shards from a Diffusers safetensors index", async () => {
    await withTempDirectory("mlxts-flux-transformer-index-", async (directory) => {
      const config = tinyFluxConfig();
      using source = new FluxTransformer2DModel(config);
      const entries = checkpointTensorsForFluxParameters(source.parameters(), config);
      const { firstShard, secondShard, weightMap } = splitIndexedShards(entries);

      await saveSafetensors(
        firstShard,
        join(directory, "diffusion_pytorch_model-00001-of-00002.safetensors"),
      );
      await saveSafetensors(
        secondShard,
        join(directory, "diffusion_pytorch_model-00002-of-00002.safetensors"),
      );
      freeTensors(entries);
      const indexPath = join(directory, "diffusion_pytorch_model.safetensors.index.json");
      writeJson(indexPath, { weight_map: weightMap });

      using target = new FluxTransformer2DModel(config);
      const result = await loadFluxTransformerWeights(target, transformerComponent([indexPath]));

      expect(result.shardCount).toBe(2);
      expect(result.unexpectedWeights).toEqual([]);
    });
  });

  test("swaps Diffusers final modulation scale/shift halves", () => {
    using diffusersBias = array([1, 2, 3, 4], "float32");
    using transformedBias = transformFluxTransformerWeight(
      "finalLayer.modulation.bias",
      diffusersBias,
    );

    expect(transformedBias.toList()).toEqual([3, 4, 1, 2]);

    using diffusersWeight = array(
      [
        [1, 10],
        [2, 20],
        [3, 30],
        [4, 40],
      ],
      "float32",
    );
    using transformedWeight = transformFluxTransformerWeight(
      "finalLayer.modulation.weight",
      diffusersWeight,
    );

    expect(transformedWeight.toList()).toEqual([
      [3, 30],
      [4, 40],
      [1, 10],
      [2, 20],
    ]);
  });

  test("rejects missing, mismatched, and strict unexpected FLUX weights", async () => {
    await withTempDirectory("mlxts-flux-transformer-invalid-", async (directory) => {
      const config = tinyFluxConfig();
      using source = new FluxTransformer2DModel(config);
      const missingPath = join(directory, "missing.safetensors");
      const missingEntries = checkpointTensorsForFluxParameters(source.parameters(), config);
      delete missingEntries["single_transformer_blocks.0.attn.to_q.weight"];
      await writeCheckpoint(missingPath, missingEntries);

      using missingTarget = new FluxTransformer2DModel(config);
      await expect(
        loadFluxTransformerWeights(missingTarget, transformerComponent([missingPath])),
      ).rejects.toThrow(DiffusionMissingWeightsError);

      const mismatchPath = join(directory, "mismatch.safetensors");
      const mismatchEntries = checkpointTensorsForFluxParameters(source.parameters(), config);
      mismatchEntries["proj_out.bias"]?.free();
      mismatchEntries["proj_out.bias"] = zeros([1]);
      await writeCheckpoint(mismatchPath, mismatchEntries);

      using mismatchTarget = new FluxTransformer2DModel(config);
      await expect(
        loadFluxTransformerWeights(mismatchTarget, transformerComponent([mismatchPath])),
      ).rejects.toThrow(DiffusionWeightMismatchError);

      const unexpectedPath = join(directory, "unexpected.safetensors");
      const unexpectedEntries = checkpointTensorsForFluxParameters(source.parameters(), config);
      unexpectedEntries["unsupported.weight"] = zeros([1]);
      await writeCheckpoint(unexpectedPath, unexpectedEntries);

      using unexpectedTarget = new FluxTransformer2DModel(config);
      await expect(
        loadFluxTransformerWeights(unexpectedTarget, transformerComponent([unexpectedPath]), {
          strictUnexpectedWeights: true,
        }),
      ).rejects.toThrow("unexpected unmapped weights");
    });
  });
});
