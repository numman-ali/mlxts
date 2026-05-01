import { describe, expect, test } from "bun:test";
import { type MxArray, type ParameterTree, saveSafetensors, treeFlatten, zeros } from "@mlxts/core";
import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { DiffusionMissingWeightsError, DiffusionWeightMismatchError } from "../../errors";
import type { DiffusionSnapshotComponent } from "../../pretrained/snapshot-manifest";
import { FluxAutoencoderKL, loadFluxAutoencoderWeights } from "./autoencoder";
import type { FluxAutoencoderConfig } from "./config";

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

function tinyFluxVaeConfig(): FluxAutoencoderConfig {
  return {
    inChannels: 3,
    outChannels: 3,
    latentChannels: 2,
    latentChannelsOut: 4,
    blockOutChannels: [4, 8],
    layersPerBlock: 1,
    normNumGroups: 1,
    scalingFactor: 0.3611,
    shiftFactor: 0.1159,
    vaeScaleFactor: 2,
    downBlockTypes: ["DownEncoderBlock2D", "DownEncoderBlock2D"],
    upBlockTypes: ["UpDecoderBlock2D", "UpDecoderBlock2D"],
    forceUpcast: false,
    rawConfig: {},
  };
}

function vaeComponent(weightPaths: readonly string[]): DiffusionSnapshotComponent {
  return {
    name: "vae",
    role: "vae",
    library: "diffusers",
    className: "AutoencoderKL",
    enabled: true,
    optional: false,
    subfolder: "vae",
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

function checkpointTensorsForFluxVaeParameters(
  parameters: ParameterTree,
  options: { omitPath?: string; extra?: Record<string, MxArray>; mismatchPath?: string } = {},
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

describe("FluxAutoencoderKL", () => {
  test("wraps the shared AutoencoderKL topology with FLUX latent metadata", () => {
    using vae = new FluxAutoencoderKL(tinyFluxVaeConfig());

    expect(vae.scalingFactor).toBe(0.3611);
    expect(vae.shiftFactor).toBe(0.1159);
    expect(vae.latentChannels).toBe(2);
    expect(vae.vaeScaleFactor).toBe(2);

    using latents = zeros([1, 2, 2, 2]);
    using decoded = vae.decode(latents);

    expect(decoded.shape).toEqual([1, 4, 4, 3]);
  });

  test("loads complete Diffusers FLUX VAE safetensors weights", async () => {
    await withTempDirectory("mlxts-flux-vae-weights-", async (directory) => {
      using source = new FluxAutoencoderKL(tinyFluxVaeConfig());
      const checkpointPath = join(directory, "diffusion_pytorch_model.safetensors");
      await writeCheckpoint(
        checkpointPath,
        checkpointTensorsForFluxVaeParameters(source.parameters()),
      );
      using target = new FluxAutoencoderKL(tinyFluxVaeConfig());

      const result = await loadFluxAutoencoderWeights(target, vaeComponent([checkpointPath]));

      expect(result.shardCount).toBe(1);
      expect(result.unexpectedWeights).toEqual([]);
      expect(result.assignedPaths).toEqual(
        treeFlatten(target.parameters())
          .map(([path]) => path.join("."))
          .toSorted((left, right) => left.localeCompare(right)),
      );
      expect(target.quantConv.weight.shape).toEqual([4, 1, 1, 4]);
      expect(target.postQuantConv.weight.shape).toEqual([2, 1, 1, 2]);
    });
  });

  test("loads FLUX VAE shards from a Diffusers safetensors index", async () => {
    await withTempDirectory("mlxts-flux-vae-index-", async (directory) => {
      using source = new FluxAutoencoderKL(tinyFluxVaeConfig());
      const entries = checkpointTensorsForFluxVaeParameters(source.parameters());
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

      using target = new FluxAutoencoderKL(tinyFluxVaeConfig());
      const result = await loadFluxAutoencoderWeights(target, vaeComponent([indexPath]));

      expect(result.shardCount).toBe(2);
      expect(result.unexpectedWeights).toEqual([]);
    });
  });

  test("rejects missing, mismatched, and strict unexpected FLUX VAE weights", async () => {
    await withTempDirectory("mlxts-flux-vae-invalid-", async (directory) => {
      using source = new FluxAutoencoderKL(tinyFluxVaeConfig());
      const missingPath = join(directory, "missing.safetensors");
      await writeCheckpoint(
        missingPath,
        checkpointTensorsForFluxVaeParameters(source.parameters(), {
          omitPath: "quantConv.bias",
        }),
      );
      using missingTarget = new FluxAutoencoderKL(tinyFluxVaeConfig());
      await expect(
        loadFluxAutoencoderWeights(missingTarget, vaeComponent([missingPath])),
      ).rejects.toThrow(DiffusionMissingWeightsError);

      const mismatchPath = join(directory, "mismatch.safetensors");
      await writeCheckpoint(
        mismatchPath,
        checkpointTensorsForFluxVaeParameters(source.parameters(), {
          mismatchPath: "decoder.convOut.weight",
        }),
      );
      using mismatchTarget = new FluxAutoencoderKL(tinyFluxVaeConfig());
      await expect(
        loadFluxAutoencoderWeights(mismatchTarget, vaeComponent([mismatchPath])),
      ).rejects.toThrow(DiffusionWeightMismatchError);

      const unexpectedPath = join(directory, "unexpected.safetensors");
      await writeCheckpoint(
        unexpectedPath,
        checkpointTensorsForFluxVaeParameters(source.parameters(), {
          extra: { "decoder.extra.weight": zeros([1]) },
        }),
      );
      using unexpectedTarget = new FluxAutoencoderKL(tinyFluxVaeConfig());
      await expect(
        loadFluxAutoencoderWeights(unexpectedTarget, vaeComponent([unexpectedPath]), {
          strictUnexpectedWeights: true,
        }),
      ).rejects.toThrow("unexpected unmapped weights");
    });
  });
});
