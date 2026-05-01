import { describe, expect, test } from "bun:test";
import { MxArray, type ParameterTree, saveSafetensors, treeFlatten, zeros } from "@mlxts/core";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { DiffusionMissingWeightsError, DiffusionWeightMismatchError } from "../../errors";
import {
  type DiffusionSnapshotComponent,
  loadDiffusionSnapshotManifest,
} from "../../pretrained/snapshot-manifest";
import { Flux2KleinAutoencoderKL } from "./autoencoder";
import type { Flux2KleinAutoencoderConfig } from "./config";
import {
  flux2KleinAutoencoderWeightPath,
  loadFlux2KleinAutoencoderFromSnapshot,
  loadFlux2KleinAutoencoderWeights,
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

function tinyFlux2VaeConfig(
  overrides: Partial<Flux2KleinAutoencoderConfig> = {},
): Flux2KleinAutoencoderConfig {
  return {
    inChannels: 3,
    outChannels: 3,
    latentChannels: 2,
    latentChannelsOut: 4,
    packedLatentChannels: 8,
    useQuantConv: true,
    usePostQuantConv: true,
    blockOutChannels: [4, 8],
    decoderBlockOutChannels: [6, 10],
    layersPerBlock: 1,
    normNumGroups: 1,
    forceUpcast: false,
    midBlockAddAttention: true,
    batchNormEps: 0.0001,
    batchNormMomentum: 0.1,
    patchSize: [2, 2],
    sampleSize: 16,
    vaeScaleFactor: 2,
    downBlockTypes: ["DownEncoderBlock2D", "DownEncoderBlock2D"],
    upBlockTypes: ["UpDecoderBlock2D", "UpDecoderBlock2D"],
    rawConfig: {},
    ...overrides,
  };
}

function vaeComponent(weightPaths: readonly string[]): DiffusionSnapshotComponent {
  return {
    name: "vae",
    role: "vae",
    library: "diffusers",
    className: "AutoencoderKLFlux2",
    enabled: true,
    optional: false,
    subfolder: "vae",
    metadataPaths: [],
    weightPaths,
  };
}

function checkpointNameForPath(path: string): string {
  return path
    .replaceAll("downBlocks", "down_blocks")
    .replaceAll("upBlocks", "up_blocks")
    .replaceAll("downsample.conv", "downsamplers.0.conv")
    .replaceAll("upsample.conv", "upsamplers.0.conv")
    .replaceAll("midBlock.resnetIn", "mid_block.resnets.0")
    .replaceAll("midBlock.attention", "mid_block.attentions.0")
    .replaceAll("midBlock.resnetOut", "mid_block.resnets.1")
    .replaceAll("queryProjection", "to_q")
    .replaceAll("keyProjection", "to_k")
    .replaceAll("valueProjection", "to_v")
    .replaceAll("outputProjection", "to_out.0")
    .replaceAll("groupNorm", "group_norm")
    .replaceAll("convNormOut", "conv_norm_out")
    .replaceAll("convShortcut", "conv_shortcut")
    .replaceAll("convIn", "conv_in")
    .replaceAll("convOut", "conv_out")
    .replaceAll("postQuantConv", "post_quant_conv")
    .replaceAll("quantConv", "quant_conv");
}

function checkpointShapeFor(path: string, tensor: MxArray): number[] {
  const [outputChannels, kernelHeight, kernelWidth, inputChannels] = tensor.shape;
  if (
    path.endsWith(".weight") &&
    tensor.shape.length === 4 &&
    outputChannels !== undefined &&
    kernelHeight !== undefined &&
    kernelWidth !== undefined &&
    inputChannels !== undefined
  ) {
    return [outputChannels, inputChannels, kernelHeight, kernelWidth];
  }
  return [...tensor.shape];
}

function checkpointTensorsForFlux2Vae(
  parameters: ParameterTree,
  options: { omitPath?: string; mismatchPath?: string; extra?: Record<string, MxArray> } = {},
): Record<string, MxArray> {
  const tensors: Record<string, MxArray> = {};
  for (const [path, parameter] of treeFlatten(parameters)) {
    const internalPath = path.join(".");
    if (internalPath === options.omitPath) {
      continue;
    }
    const checkpointName = checkpointNameForPath(internalPath);
    const shape =
      internalPath === options.mismatchPath ? [1] : checkpointShapeFor(internalPath, parameter);
    tensors[checkpointName] = zeros(shape);
  }
  if (options.omitPath !== "bn.running_mean") {
    tensors["bn.running_mean"] = MxArray.fromData([0, 1, 2, 3, 4, 5, 6, 7], [8]);
  }
  if (options.omitPath !== "bn.running_var") {
    const shape = options.mismatchPath === "bn.running_var" ? [1] : [8];
    tensors["bn.running_var"] = zeros(shape);
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

function writeFlux2SnapshotMetadata(directory: string): void {
  writeJson(join(directory, "model_index.json"), {
    _class_name: "Flux2KleinPipeline",
    is_distilled: true,
    scheduler: ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    text_encoder: ["transformers", "Qwen3ForCausalLM"],
    tokenizer: ["transformers", "Qwen2TokenizerFast"],
    transformer: ["diffusers", "Flux2Transformer2DModel"],
    vae: ["diffusers", "AutoencoderKLFlux2"],
  });
  const transformerDirectory = join(directory, "transformer");
  mkdirSync(transformerDirectory, { recursive: true });
  writeJson(join(transformerDirectory, "config.json"), {
    _class_name: "Flux2Transformer2DModel",
    attention_head_dim: 8,
    axes_dims_rope: [2, 2, 2, 2],
    guidance_embeds: false,
    in_channels: 8,
    joint_attention_dim: 8,
    num_attention_heads: 1,
    out_channels: null,
  });
  writeFileSync(join(transformerDirectory, "diffusion_pytorch_model.safetensors"), "");

  const vaeDirectory = join(directory, "vae");
  mkdirSync(vaeDirectory, { recursive: true });
  writeJson(join(vaeDirectory, "config.json"), {
    _class_name: "AutoencoderKLFlux2",
    block_out_channels: [4, 8],
    decoder_block_out_channels: [6, 10],
    down_block_types: ["DownEncoderBlock2D", "DownEncoderBlock2D"],
    in_channels: 3,
    latent_channels: 2,
    layers_per_block: 1,
    norm_num_groups: 1,
    out_channels: 3,
    patch_size: [2, 2],
    sample_size: 16,
    up_block_types: ["UpDecoderBlock2D", "UpDecoderBlock2D"],
  });

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

describe("FLUX.2 Klein VAE weight loading", () => {
  test("maps FLUX.2 batch-norm buffers and shared VAE parameter names", () => {
    expect(flux2KleinAutoencoderWeightPath("bn.running_mean")).toBe("bn.running_mean");
    expect(flux2KleinAutoencoderWeightPath("bn.running_var")).toBe("bn.running_var");
    expect(flux2KleinAutoencoderWeightPath("bn.num_batches_tracked")).toBeNull();
    expect(flux2KleinAutoencoderWeightPath(" ")).toBeNull();
    expect(flux2KleinAutoencoderWeightPath("decoder.conv_in.weight")).toBe("decoder.convIn.weight");
  });

  test("loads complete Diffusers FLUX.2 VAE safetensors weights", async () => {
    await withTempDirectory("mlxts-flux2-vae-weights-", async (directory) => {
      using source = new Flux2KleinAutoencoderKL(tinyFlux2VaeConfig());
      const checkpointPath = join(directory, "diffusion_pytorch_model.safetensors");
      await writeCheckpoint(
        checkpointPath,
        checkpointTensorsForFlux2Vae(source.parameters(), {
          extra: { "bn.num_batches_tracked": MxArray.fromData([0], [1], "int32") },
        }),
      );
      using target = new Flux2KleinAutoencoderKL(tinyFlux2VaeConfig());

      const result = await loadFlux2KleinAutoencoderWeights(
        target,
        vaeComponent([checkpointPath]),
        { strictUnexpectedWeights: true },
      );

      expect(result.shardCount).toBe(1);
      expect(result.unexpectedWeights).toEqual([]);
      expect(result.assignedPaths).toEqual(
        [
          ...treeFlatten(target.parameters()).map(([path]) => path.join(".")),
          "bn.running_mean",
          "bn.running_var",
        ].toSorted((left, right) => left.localeCompare(right)),
      );
      expect(target.batchNormMean).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
      expect(target.batchNormVar).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);
      expect(target.decoder.convIn.weight.shape).toEqual([10, 3, 3, 2]);
    });
  });

  test("loads FLUX.2 VAE shards from a Diffusers safetensors index", async () => {
    await withTempDirectory("mlxts-flux2-vae-index-", async (directory) => {
      using source = new Flux2KleinAutoencoderKL(tinyFlux2VaeConfig());
      const entries = checkpointTensorsForFlux2Vae(source.parameters());
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

      using target = new Flux2KleinAutoencoderKL(tinyFlux2VaeConfig());
      const result = await loadFlux2KleinAutoencoderWeights(target, vaeComponent([indexPath]));

      expect(result.shardCount).toBe(2);
      expect(result.unexpectedWeights).toEqual([]);
      expect(target.batchNormMean).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    });
  });

  test("rejects missing, mismatched, and strict unexpected FLUX.2 VAE weights", async () => {
    await withTempDirectory("mlxts-flux2-vae-invalid-", async (directory) => {
      using source = new Flux2KleinAutoencoderKL(tinyFlux2VaeConfig());
      const missingPath = join(directory, "missing.safetensors");
      await writeCheckpoint(
        missingPath,
        checkpointTensorsForFlux2Vae(source.parameters(), {
          omitPath: "bn.running_var",
        }),
      );
      using missingTarget = new Flux2KleinAutoencoderKL(tinyFlux2VaeConfig());
      await expect(
        loadFlux2KleinAutoencoderWeights(missingTarget, vaeComponent([missingPath])),
      ).rejects.toThrow(DiffusionMissingWeightsError);

      const mismatchPath = join(directory, "mismatch.safetensors");
      await writeCheckpoint(
        mismatchPath,
        checkpointTensorsForFlux2Vae(source.parameters(), {
          mismatchPath: "bn.running_var",
        }),
      );
      using mismatchTarget = new Flux2KleinAutoencoderKL(tinyFlux2VaeConfig());
      await expect(
        loadFlux2KleinAutoencoderWeights(mismatchTarget, vaeComponent([mismatchPath])),
      ).rejects.toThrow(DiffusionWeightMismatchError);

      const parameterMismatchPath = join(directory, "parameter-mismatch.safetensors");
      await writeCheckpoint(
        parameterMismatchPath,
        checkpointTensorsForFlux2Vae(source.parameters(), {
          mismatchPath: "decoder.convOut.weight",
        }),
      );
      using parameterMismatchTarget = new Flux2KleinAutoencoderKL(tinyFlux2VaeConfig());
      await expect(
        loadFlux2KleinAutoencoderWeights(
          parameterMismatchTarget,
          vaeComponent([parameterMismatchPath]),
        ),
      ).rejects.toThrow(DiffusionWeightMismatchError);

      const unexpectedPath = join(directory, "unexpected.safetensors");
      await writeCheckpoint(
        unexpectedPath,
        checkpointTensorsForFlux2Vae(source.parameters(), {
          extra: { "decoder.extra.weight": zeros([1]) },
        }),
      );
      using unexpectedTarget = new Flux2KleinAutoencoderKL(tinyFlux2VaeConfig());
      await expect(
        loadFlux2KleinAutoencoderWeights(unexpectedTarget, vaeComponent([unexpectedPath]), {
          strictUnexpectedWeights: true,
        }),
      ).rejects.toThrow("unexpected unmapped weights");
    });
  });

  test("rejects invalid FLUX.2 VAE shard manifests", async () => {
    await withTempDirectory("mlxts-flux2-vae-shards-invalid-", async (directory) => {
      using noWeightsTarget = new Flux2KleinAutoencoderKL(tinyFlux2VaeConfig());
      await expect(
        loadFlux2KleinAutoencoderWeights(noWeightsTarget, vaeComponent([])),
      ).rejects.toThrow("no safetensors weight shards");

      const firstIndex = join(directory, "first.safetensors.index.json");
      const secondIndex = join(directory, "second.safetensors.index.json");
      writeJson(firstIndex, { weight_map: {} });
      writeJson(secondIndex, { weight_map: {} });
      using multipleIndexTarget = new Flux2KleinAutoencoderKL(tinyFlux2VaeConfig());
      await expect(
        loadFlux2KleinAutoencoderWeights(
          multipleIndexTarget,
          vaeComponent([firstIndex, secondIndex]),
        ),
      ).rejects.toThrow("multiple safetensors index files");

      const malformedIndex = join(directory, "malformed.safetensors.index.json");
      writeFileSync(malformedIndex, "{");
      using malformedTarget = new Flux2KleinAutoencoderKL(tinyFlux2VaeConfig());
      await expect(
        loadFlux2KleinAutoencoderWeights(malformedTarget, vaeComponent([malformedIndex])),
      ).rejects.toThrow("valid JSON");

      const arrayIndex = join(directory, "array.safetensors.index.json");
      writeJson(arrayIndex, []);
      using arrayTarget = new Flux2KleinAutoencoderKL(tinyFlux2VaeConfig());
      await expect(
        loadFlux2KleinAutoencoderWeights(arrayTarget, vaeComponent([arrayIndex])),
      ).rejects.toThrow("JSON object");

      const badMap = join(directory, "bad-map.safetensors.index.json");
      writeJson(badMap, { weight_map: { x: 1 } });
      using badMapTarget = new Flux2KleinAutoencoderKL(tinyFlux2VaeConfig());
      await expect(
        loadFlux2KleinAutoencoderWeights(badMapTarget, vaeComponent([badMap])),
      ).rejects.toThrow("shard filename");
    });
  });

  test("loads FLUX.2 VAE directly from an inspected snapshot manifest", async () => {
    await withTempDirectory("mlxts-flux2-vae-snapshot-", async (directory) => {
      writeFlux2SnapshotMetadata(directory);
      using source = new Flux2KleinAutoencoderKL(tinyFlux2VaeConfig());
      await writeCheckpoint(
        join(directory, "vae", "diffusion_pytorch_model.safetensors"),
        checkpointTensorsForFlux2Vae(source.parameters()),
      );
      const manifest = await loadDiffusionSnapshotManifest(directory);

      using target = await loadFlux2KleinAutoencoderFromSnapshot(manifest);

      expect(target.batchNormMean).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
      expect(target.decoder.convIn.weight.shape).toEqual([10, 3, 3, 2]);
    });
  });

  test("disposes partial FLUX.2 VAE snapshot loads on failure", async () => {
    await withTempDirectory("mlxts-flux2-vae-snapshot-fail-", async (directory) => {
      writeFlux2SnapshotMetadata(directory);
      using source = new Flux2KleinAutoencoderKL(tinyFlux2VaeConfig());
      await writeCheckpoint(
        join(directory, "vae", "diffusion_pytorch_model.safetensors"),
        checkpointTensorsForFlux2Vae(source.parameters(), {
          omitPath: "bn.running_mean",
        }),
      );
      const manifest = await loadDiffusionSnapshotManifest(directory);

      await expect(loadFlux2KleinAutoencoderFromSnapshot(manifest)).rejects.toThrow(
        DiffusionMissingWeightsError,
      );
    });
  });
});
