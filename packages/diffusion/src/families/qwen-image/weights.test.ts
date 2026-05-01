import { describe, expect, test } from "bun:test";
import {
  MxArray,
  mxEval,
  type ParameterTree,
  saveSafetensors,
  treeFlatten,
  zeros,
} from "@mlxts/core";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { DiffusionMissingWeightsError, DiffusionWeightMismatchError } from "../../errors";
import type { DiffusionSnapshotComponent } from "../../pretrained/snapshot-manifest";
import { loadDiffusionSnapshotManifest } from "../../pretrained/snapshot-manifest";
import { QwenImageAutoencoderKL } from "./autoencoder";
import type { QwenImageAutoencoderConfig } from "./config";
import {
  loadQwenImageAutoencoderFromSnapshot,
  loadQwenImageAutoencoderWeights,
  qwenImageAutoencoderWeightPath,
  transformQwenImageAutoencoderWeight,
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

function writeJson(path: string, payload: unknown): void {
  writeFileSync(path, JSON.stringify(payload));
}

function tinyAutoencoderConfig(): QwenImageAutoencoderConfig {
  return {
    baseDim: 2,
    latentChannels: 2,
    latentChannelsOut: 4,
    dimMultipliers: [1, 2],
    numResBlocks: 1,
    attentionScales: [],
    temporalDownsample: [false],
    temporalUpsample: [false],
    dropout: 0,
    inputChannels: 3,
    latentsMean: [0, 0],
    latentsStd: [1, 1],
    spatialCompressionRatio: 2,
    rawConfig: {},
  };
}

function diffusersTinyAutoencoderConfig(): Record<string, unknown> {
  return {
    _class_name: "AutoencoderKLQwenImage",
    base_dim: 2,
    z_dim: 2,
    dim_mult: [1, 2],
    num_res_blocks: 1,
    attn_scales: [],
    temperal_downsample: [false],
    dropout: 0,
    input_channels: 3,
    latents_mean: [0, 0],
    latents_std: [1, 1],
  };
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

function writeQwenImageSnapshot(snapshot: string): void {
  writeJson(join(snapshot, "model_index.json"), {
    _class_name: "QwenImagePipeline",
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
  writeJson(join(snapshot, "transformer", "config.json"), {
    _class_name: "QwenImageTransformer2DModel",
    attention_head_dim: 128,
    axes_dims_rope: [16, 56, 56],
    in_channels: 64,
    out_channels: 16,
    patch_size: 2,
  });
  writeComponentFiles(snapshot, "vae", ["config.json"], ["diffusion_pytorch_model.safetensors"]);
  writeJson(join(snapshot, "vae", "config.json"), diffusersTinyAutoencoderConfig());
  writeComponentFiles(snapshot, "scheduler", ["scheduler_config.json"]);
  writeJson(join(snapshot, "scheduler", "scheduler_config.json"), {
    _class_name: "FlowMatchEulerDiscreteScheduler",
  });
  writeComponentFiles(snapshot, "text_encoder", ["config.json"], ["model.safetensors"]);
  writeComponentFiles(snapshot, "tokenizer", ["tokenizer.json"]);
}

function vaeComponent(weightPaths: readonly string[]): DiffusionSnapshotComponent {
  return {
    name: "vae",
    role: "vae",
    library: "diffusers",
    className: "AutoencoderKLQwenImage",
    enabled: true,
    optional: false,
    subfolder: "vae",
    metadataPaths: [],
    weightPaths,
  };
}

function isQwenRmsNormWeightPath(path: string): boolean {
  return (
    path.endsWith(".weight") &&
    (path.includes(".norm") || path.includes("normOut") || path.endsWith("norm.weight"))
  );
}

function checkpointNameForPath(path: string): string {
  let checkpointName = path
    .replaceAll("postQuantConv", "post_quant_conv")
    .replaceAll("quantConv", "quant_conv")
    .replaceAll("downBlocks", "down_blocks")
    .replaceAll("upBlocks", "up_blocks")
    .replaceAll("midBlock", "mid_block")
    .replaceAll("convIn", "conv_in")
    .replaceAll("convOut", "conv_out")
    .replaceAll("convShortcut", "conv_shortcut")
    .replaceAll("normOut", "norm_out")
    .replaceAll("timeConv", "time_conv")
    .replaceAll("toQkv", "to_qkv")
    .replaceAll("upsampler", "upsamplers.0")
    .replaceAll("resample", "resample.1");
  if (isQwenRmsNormWeightPath(path)) {
    checkpointName = checkpointName.replaceAll(".weight", ".gamma");
  }
  return checkpointName;
}

function checkpointShapeFor(path: string, tensor: MxArray): number[] {
  if (isQwenRmsNormWeightPath(path)) {
    const channels = tensor.shape[0];
    if (channels === undefined) {
      throw new Error(`checkpointShapeFor: ${path} is missing channels.`);
    }
    return [channels, 1, 1, 1];
  }
  const [outputChannels, first, second, third, inputChannels] = tensor.shape;
  if (
    path.endsWith(".weight") &&
    tensor.shape.length === 5 &&
    outputChannels !== undefined &&
    first !== undefined &&
    second !== undefined &&
    third !== undefined &&
    inputChannels !== undefined
  ) {
    return [outputChannels, inputChannels, first, second, third];
  }
  const [convOutputChannels, kernelHeight, kernelWidth, convInputChannels] = tensor.shape;
  if (
    path.endsWith(".weight") &&
    tensor.shape.length === 4 &&
    convOutputChannels !== undefined &&
    kernelHeight !== undefined &&
    kernelWidth !== undefined &&
    convInputChannels !== undefined
  ) {
    return [convOutputChannels, convInputChannels, kernelHeight, kernelWidth];
  }
  return [...tensor.shape];
}

function checkpointTensorsForParameters(
  parameters: ParameterTree,
  options: { omitPath?: string; extra?: Record<string, MxArray>; mismatchPath?: string } = {},
): Record<string, MxArray> {
  const tensors: Record<string, MxArray> = {};
  for (const [path, parameter] of treeFlatten(parameters)) {
    const internalPath = path.join(".");
    if (internalPath === options.omitPath) {
      continue;
    }
    const shape =
      internalPath === options.mismatchPath ? [1] : checkpointShapeFor(internalPath, parameter);
    tensors[checkpointNameForPath(internalPath)] = zeros(shape);
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

function expectTensorValues(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBe(expected[index]);
  }
}

describe("Qwen-Image VAE weight mapping", () => {
  test("maps Diffusers VAE names onto the package parameter tree", () => {
    expect(qwenImageAutoencoderWeightPath("encoder.conv_in.weight")).toBe("encoder.convIn.weight");
    expect(qwenImageAutoencoderWeightPath("encoder.down_blocks.0.norm1.gamma")).toBe(
      "encoder.downBlocks.0.norm1.weight",
    );
    expect(
      qwenImageAutoencoderWeightPath("decoder.up_blocks.0.upsamplers.0.resample.1.weight"),
    ).toBe("decoder.upBlocks.0.upsampler.resample.weight");
    expect(
      qwenImageAutoencoderWeightPath("decoder.up_blocks.0.upsamplers.0.time_conv.weight"),
    ).toBe("decoder.upBlocks.0.upsampler.timeConv.weight");
    expect(qwenImageAutoencoderWeightPath("post_quant_conv.bias")).toBe("postQuantConv.bias");
    expect(qwenImageAutoencoderWeightPath("")).toBeNull();
    expect(qwenImageAutoencoderWeightPath("encoder.num_batches_tracked")).toBeNull();
  });

  test("transposes Diffusers Conv3d weights into MLX layout", () => {
    using checkpoint = MxArray.fromData([1, 2, 3, 4], [1, 2, 2, 1, 1]);
    using transformed = transformQwenImageAutoencoderWeight(
      "encoder.conv_in.weight",
      "encoder.convIn.weight",
      checkpoint,
    );

    mxEval(transformed);
    expect(transformed.shape).toEqual([1, 2, 1, 1, 2]);
    expectTensorValues(transformed.toTypedArray(), [1, 3, 2, 4]);
  });

  test("transposes Diffusers Conv2d weights into MLX layout", () => {
    using checkpoint = MxArray.fromData([1, 2, 3, 4], [1, 2, 1, 2]);
    using transformed = transformQwenImageAutoencoderWeight(
      "decoder.mid_block.attentions.0.to_qkv.weight",
      "decoder.midBlock.attentions.0.toQkv.weight",
      checkpoint,
    );

    mxEval(transformed);
    expect(transformed.shape).toEqual([1, 1, 2, 2]);
    expectTensorValues(transformed.toTypedArray(), [1, 3, 2, 4]);
  });

  test("squeezes RMS normalization gamma tensors", () => {
    using checkpoint = MxArray.fromData([5, 6], [2, 1, 1, 1]);
    using transformed = transformQwenImageAutoencoderWeight(
      "encoder.down_blocks.0.norm1.gamma",
      "encoder.downBlocks.0.norm1.weight",
      checkpoint,
    );

    mxEval(transformed);
    expect(transformed.shape).toEqual([2]);
    expectTensorValues(transformed.toTypedArray(), [5, 6]);
  });

  test("leaves bias tensors in their checkpoint layout", () => {
    using checkpoint = MxArray.fromData([1, 2], [2]);
    const transformed = transformQwenImageAutoencoderWeight(
      "encoder.conv_in.bias",
      "encoder.convIn.bias",
      checkpoint,
    );

    expect(transformed).toBe(checkpoint);
  });

  test("loads a complete generated VAE safetensors shard", async () => {
    await withTempDirectory("mlxts-qwen-image-vae-weights-", async (directory) => {
      using source = new QwenImageAutoencoderKL(tinyAutoencoderConfig());
      const checkpointPath = join(directory, "diffusion_pytorch_model.safetensors");
      await writeCheckpoint(checkpointPath, checkpointTensorsForParameters(source.parameters()));
      using target = new QwenImageAutoencoderKL(tinyAutoencoderConfig());

      const result = await loadQwenImageAutoencoderWeights(target, vaeComponent([checkpointPath]));

      expect(result.shardCount).toBe(1);
      expect(result.unexpectedWeights).toEqual([]);
      expect(result.assignedPaths).toEqual(
        treeFlatten(target.parameters())
          .map(([path]) => path.join("."))
          .toSorted((left, right) => left.localeCompare(right)),
      );
      expect(target.encoder.convIn.weight.shape).toEqual([2, 3, 3, 3, 3]);
      expect(target.decoder.upBlocks[0]?.upsampler?.resample?.weight.shape).toEqual([2, 3, 3, 4]);
    });
  });

  test("loads VAE shards from a Diffusers safetensors index", async () => {
    await withTempDirectory("mlxts-qwen-image-vae-index-", async (directory) => {
      using source = new QwenImageAutoencoderKL(tinyAutoencoderConfig());
      const entries = checkpointTensorsForParameters(source.parameters());
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

      using target = new QwenImageAutoencoderKL(tinyAutoencoderConfig());
      const result = await loadQwenImageAutoencoderWeights(target, vaeComponent([indexPath]));

      expect(result.shardCount).toBe(2);
      expect(result.unexpectedWeights).toEqual([]);
    });
  });

  test("loads a VAE directly from an inspected Qwen-Image snapshot manifest", async () => {
    await withTempDirectory("mlxts-qwen-image-vae-manifest-", async (directory) => {
      writeQwenImageSnapshot(directory);
      using source = new QwenImageAutoencoderKL(tinyAutoencoderConfig());
      await writeCheckpoint(
        join(directory, "vae", "diffusion_pytorch_model.safetensors"),
        checkpointTensorsForParameters(source.parameters()),
      );

      const manifest = await loadDiffusionSnapshotManifest(directory);
      using model = await loadQwenImageAutoencoderFromSnapshot(manifest);

      expect(model.latentChannels).toBe(2);
      expect(model.spatialCompressionRatio).toBe(2);
      expect(model.encoder.convIn.weight.shape).toEqual([2, 3, 3, 3, 3]);
    });
  });

  test("rejects missing, mismatched, and strict unexpected VAE weights", async () => {
    await withTempDirectory("mlxts-qwen-image-vae-invalid-", async (directory) => {
      using source = new QwenImageAutoencoderKL(tinyAutoencoderConfig());
      const missingPath = join(directory, "missing.safetensors");
      await writeCheckpoint(
        missingPath,
        checkpointTensorsForParameters(source.parameters(), {
          omitPath: "quantConv.bias",
        }),
      );
      using missingTarget = new QwenImageAutoencoderKL(tinyAutoencoderConfig());
      await expect(
        loadQwenImageAutoencoderWeights(missingTarget, vaeComponent([missingPath])),
      ).rejects.toThrow(DiffusionMissingWeightsError);

      const mismatchPath = join(directory, "mismatch.safetensors");
      await writeCheckpoint(
        mismatchPath,
        checkpointTensorsForParameters(source.parameters(), {
          mismatchPath: "encoder.convOut.weight",
        }),
      );
      using mismatchTarget = new QwenImageAutoencoderKL(tinyAutoencoderConfig());
      await expect(
        loadQwenImageAutoencoderWeights(mismatchTarget, vaeComponent([mismatchPath])),
      ).rejects.toThrow(DiffusionWeightMismatchError);

      const unexpectedPath = join(directory, "unexpected.safetensors");
      await writeCheckpoint(
        unexpectedPath,
        checkpointTensorsForParameters(source.parameters(), {
          extra: { "decoder.extra.weight": zeros([1]) },
        }),
      );
      using unexpectedTarget = new QwenImageAutoencoderKL(tinyAutoencoderConfig());
      await expect(
        loadQwenImageAutoencoderWeights(unexpectedTarget, vaeComponent([unexpectedPath]), {
          strictUnexpectedWeights: true,
        }),
      ).rejects.toThrow("unexpected unmapped weights");
    });
  });
});
