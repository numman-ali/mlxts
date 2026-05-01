import { describe, expect, test } from "bun:test";
import { type MxArray, type ParameterTree, saveSafetensors, treeFlatten, zeros } from "@mlxts/core";
import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { DiffusionMissingWeightsError, DiffusionWeightMismatchError } from "../../errors";
import type { DiffusionSnapshotComponent } from "../../pretrained/snapshot-manifest";
import type { ZImageTransformerConfig } from "./config";
import { ZImageTransformer2DModel } from "./transformer";
import { loadZImageTransformerWeights, zImageTransformerWeightPath } from "./weights";

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

function tinyZImageConfig(): ZImageTransformerConfig {
  return {
    patchGeometries: [{ patchSize: 2, framePatchSize: 1, packedLatentChannels: 16 }],
    inChannels: 4,
    outChannels: 4,
    hiddenSize: 16,
    numLayers: 1,
    numRefinerLayers: 1,
    numAttentionHeads: 2,
    numKeyValueHeads: 2,
    attentionHeadDim: 8,
    normEps: 1e-5,
    qkNorm: true,
    captionFeatureDim: 8,
    siglipFeatureDim: null,
    ropeTheta: 256,
    timestepScale: 1000,
    sequenceMultiple: 32,
    latentPadDim: 64,
    axesDims: [4, 2, 2],
    axesLens: [128, 32, 32],
    rawConfig: {},
  };
}

function transformerComponent(weightPaths: readonly string[]): DiffusionSnapshotComponent {
  return {
    name: "transformer",
    role: "backbone",
    library: "diffusers",
    className: "ZImageTransformer2DModel",
    enabled: true,
    optional: false,
    subfolder: "transformer",
    metadataPaths: [],
    weightPaths,
  };
}

function checkpointNamesForConfig(config: ZImageTransformerConfig): string[] {
  const names = [
    "x_pad_token",
    "cap_pad_token",
    "t_embedder.mlp.0.weight",
    "t_embedder.mlp.0.bias",
    "t_embedder.mlp.2.weight",
    "t_embedder.mlp.2.bias",
    "cap_embedder.0.weight",
    "cap_embedder.1.weight",
    "cap_embedder.1.bias",
  ];

  for (const geometry of config.patchGeometries) {
    const key = `${geometry.patchSize}-${geometry.framePatchSize}`;
    names.push(
      `all_x_embedder.${key}.weight`,
      `all_x_embedder.${key}.bias`,
      `all_final_layer.${key}.linear.weight`,
      `all_final_layer.${key}.linear.bias`,
      `all_final_layer.${key}.adaLN_modulation.1.weight`,
      `all_final_layer.${key}.adaLN_modulation.1.bias`,
    );
  }

  const blockLocals = [
    "attention.to_q",
    "attention.to_k",
    "attention.to_v",
    "attention.to_out.0",
    "attention.norm_q",
    "attention.norm_k",
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
    "attention_norm1",
    "attention_norm2",
    "ffn_norm1",
    "ffn_norm2",
    "adaLN_modulation.0",
  ];
  const groups = [
    ["noise_refiner", config.numRefinerLayers],
    ["context_refiner", config.numRefinerLayers],
    ["layers", config.numLayers],
  ] as const;

  for (const [group, count] of groups) {
    for (let index = 0; index < count; index += 1) {
      for (const local of blockLocals) {
        names.push(`${group}.${index}.${local}.weight`, `${group}.${index}.${local}.bias`);
      }
    }
  }

  return names;
}

function checkpointNameForPath(config: ZImageTransformerConfig, path: string): string {
  for (const checkpointName of checkpointNamesForConfig(config)) {
    if (zImageTransformerWeightPath(config, checkpointName) === path) {
      return checkpointName;
    }
  }
  throw new Error(`checkpointNameForPath: no checkpoint name maps to ${path}.`);
}

function checkpointTensorsForZImageParameters(
  config: ZImageTransformerConfig,
  parameters: ParameterTree,
  options: { omitPath?: string; extra?: Record<string, MxArray>; mismatchPath?: string } = {},
): Record<string, MxArray> {
  const tensors: Record<string, MxArray> = {};
  for (const [path, parameter] of treeFlatten(parameters)) {
    const internalPath = path.join(".");
    if (internalPath === options.omitPath) {
      continue;
    }
    const shape = internalPath === options.mismatchPath ? [1] : [...parameter.shape];
    tensors[checkpointNameForPath(config, internalPath)] = zeros(shape);
  }
  return { ...tensors, ...(options.extra ?? {}) };
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

describe("Z-Image weight mapping", () => {
  test("maps top-level and geometry-specific transformer weights", () => {
    const config = tinyZImageConfig();

    expect(zImageTransformerWeightPath(config, "x_pad_token")).toBe("xPadToken");
    expect(zImageTransformerWeightPath(config, "all_x_embedder.2-1.weight")).toBe(
      "imageEmbedders.0.weight",
    );
    expect(zImageTransformerWeightPath(config, "all_final_layer.2-1.linear.bias")).toBe(
      "finalLayers.0.output.bias",
    );
    expect(
      zImageTransformerWeightPath(config, "all_final_layer.2-1.adaLN_modulation.1.weight"),
    ).toBe("finalLayers.0.modulation.weight");
    expect(zImageTransformerWeightPath(config, "cap_embedder.0.weight")).toBe(
      "captionEmbedder.norm.weight",
    );
  });

  test("maps refiner and main block weights", () => {
    const config = tinyZImageConfig();

    expect(zImageTransformerWeightPath(config, "noise_refiner.0.attention.to_q.weight")).toBe(
      "noiseRefiner.0.attention.query.weight",
    );
    expect(zImageTransformerWeightPath(config, "context_refiner.0.feed_forward.w2.weight")).toBe(
      "contextRefiner.0.feedForward.w2.weight",
    );
    expect(zImageTransformerWeightPath(config, "layers.0.adaLN_modulation.0.bias")).toBe(
      "layers.0.adaLnModulation.bias",
    );
    expect(zImageTransformerWeightPath(config, "layers.0.unknown.weight")).toBeNull();
  });

  test("loads complete Z-Image transformer safetensors weights", async () => {
    await withTempDirectory("mlxts-z-image-transformer-weights-", async (directory) => {
      const config = tinyZImageConfig();
      using source = new ZImageTransformer2DModel(config);
      const checkpointPath = join(directory, "diffusion_pytorch_model.safetensors");
      await writeCheckpoint(
        checkpointPath,
        checkpointTensorsForZImageParameters(config, source.parameters()),
      );
      using target = new ZImageTransformer2DModel(config);

      const result = await loadZImageTransformerWeights(
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
    });
  });

  test("loads Z-Image transformer shards from a Diffusers safetensors index", async () => {
    await withTempDirectory("mlxts-z-image-transformer-index-", async (directory) => {
      const config = tinyZImageConfig();
      using source = new ZImageTransformer2DModel(config);
      const entries = checkpointTensorsForZImageParameters(config, source.parameters());
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

      using target = new ZImageTransformer2DModel(config);
      const result = await loadZImageTransformerWeights(target, transformerComponent([indexPath]));

      expect(result.shardCount).toBe(2);
      expect(result.unexpectedWeights).toEqual([]);
    });
  });

  test("rejects missing, mismatched, and strict unexpected Z-Image transformer weights", async () => {
    await withTempDirectory("mlxts-z-image-transformer-invalid-", async (directory) => {
      const config = tinyZImageConfig();
      using source = new ZImageTransformer2DModel(config);

      const missingPath = join(directory, "missing.safetensors");
      await writeCheckpoint(
        missingPath,
        checkpointTensorsForZImageParameters(config, source.parameters(), {
          omitPath: "xPadToken",
        }),
      );
      using missingTarget = new ZImageTransformer2DModel(config);
      await expect(
        loadZImageTransformerWeights(missingTarget, transformerComponent([missingPath])),
      ).rejects.toThrow(DiffusionMissingWeightsError);

      const mismatchPath = join(directory, "mismatch.safetensors");
      await writeCheckpoint(
        mismatchPath,
        checkpointTensorsForZImageParameters(config, source.parameters(), {
          mismatchPath: "layers.0.attention.query.weight",
        }),
      );
      using mismatchTarget = new ZImageTransformer2DModel(config);
      await expect(
        loadZImageTransformerWeights(mismatchTarget, transformerComponent([mismatchPath])),
      ).rejects.toThrow(DiffusionWeightMismatchError);

      const unexpectedPath = join(directory, "unexpected.safetensors");
      await writeCheckpoint(
        unexpectedPath,
        checkpointTensorsForZImageParameters(config, source.parameters(), {
          extra: { "layers.0.extra.weight": zeros([1]) },
        }),
      );
      using unexpectedTarget = new ZImageTransformer2DModel(config);
      await expect(
        loadZImageTransformerWeights(unexpectedTarget, transformerComponent([unexpectedPath]), {
          strictUnexpectedWeights: true,
        }),
      ).rejects.toThrow("unexpected unmapped weights");
    });
  });

  test("rejects invalid Z-Image transformer shard manifests", async () => {
    await withTempDirectory("mlxts-z-image-transformer-manifest-invalid-", async (directory) => {
      using noWeightsTarget = new ZImageTransformer2DModel(tinyZImageConfig());
      await expect(
        loadZImageTransformerWeights(noWeightsTarget, transformerComponent([])),
      ).rejects.toThrow("no safetensors weight shards");

      const firstIndex = join(directory, "first.safetensors.index.json");
      const secondIndex = join(directory, "second.safetensors.index.json");
      writeJson(firstIndex, { weight_map: {} });
      writeJson(secondIndex, { weight_map: {} });
      using multipleIndexTarget = new ZImageTransformer2DModel(tinyZImageConfig());
      await expect(
        loadZImageTransformerWeights(
          multipleIndexTarget,
          transformerComponent([firstIndex, secondIndex]),
        ),
      ).rejects.toThrow("multiple safetensors index files");

      const malformedIndex = join(directory, "malformed.safetensors.index.json");
      writeFileSync(malformedIndex, "{");
      using malformedTarget = new ZImageTransformer2DModel(tinyZImageConfig());
      await expect(
        loadZImageTransformerWeights(malformedTarget, transformerComponent([malformedIndex])),
      ).rejects.toThrow("valid JSON");

      const badWeightMap = join(directory, "bad-map.safetensors.index.json");
      writeJson(badWeightMap, { weight_map: { x_pad_token: 1 } });
      using badMapTarget = new ZImageTransformer2DModel(tinyZImageConfig());
      await expect(
        loadZImageTransformerWeights(badMapTarget, transformerComponent([badWeightMap])),
      ).rejects.toThrow("must be a shard filename");
    });
  });
});
