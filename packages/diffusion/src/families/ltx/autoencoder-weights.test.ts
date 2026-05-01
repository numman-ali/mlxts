import { describe, expect, test } from "bun:test";
import { MxArray, mxEval, saveSafetensors, treeFlatten, zeros } from "@mlxts/core";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import type { DiffusionSnapshotComponent } from "../../pretrained/snapshot-manifest";
import { LtxVideoAutoencoderKL } from "./autoencoder";
import {
  loadLtxVideoAutoencoderWeights,
  ltxVideoAutoencoderWeightPath,
  transformLtxVideoAutoencoderWeight,
} from "./autoencoder-weights";
import type { LtxVideoAutoencoderConfig } from "./config";

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

function tinyConfig(): LtxVideoAutoencoderConfig {
  return {
    inChannels: 3,
    outChannels: 1,
    latentChannels: 1,
    latentChannelsOut: 2,
    blockOutChannels: [1],
    downBlockTypes: ["LTXVideoDownBlock3D"],
    decoderBlockOutChannels: [1],
    layersPerBlock: [0, 0],
    decoderLayersPerBlock: [0, 0],
    spatioTemporalScaling: [false],
    decoderSpatioTemporalScaling: [false],
    decoderInjectNoise: [false, false],
    downsampleTypes: ["conv"],
    upsampleResidual: [false],
    upsampleFactors: [1],
    timestepConditioning: false,
    patchSize: 1,
    patchSizeT: 1,
    resnetNormEps: 1e-6,
    scalingFactor: 1,
    encoderCausal: true,
    decoderCausal: false,
    spatialCompressionRatio: null,
    temporalCompressionRatio: null,
    rawConfig: {},
  };
}

function vaeComponent(weightPaths: readonly string[]): DiffusionSnapshotComponent {
  return {
    name: "vae",
    role: "vae",
    library: "diffusers",
    className: "AutoencoderKLLTXVideo",
    enabled: true,
    optional: false,
    subfolder: "vae",
    metadataPaths: [],
    weightPaths,
  };
}

function freeTensors(tensors: Record<string, MxArray>): void {
  for (const tensor of Object.values(tensors)) {
    tensor.free();
  }
}

function minimalDecoderTensors(extra: Record<string, MxArray> = {}): Record<string, MxArray> {
  return {
    "decoder.conv_in.conv.weight": zeros([1, 1, 3, 3, 3]),
    "decoder.conv_in.conv.bias": zeros([1]),
    "decoder.conv_out.conv.weight": zeros([1, 1, 3, 3, 3]),
    "decoder.conv_out.conv.bias": zeros([1]),
    latents_mean: MxArray.fromData([2], [1]),
    latents_std: MxArray.fromData([3], [1]),
    ...extra,
  };
}

function expectCloseList(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBeCloseTo(expected[index] ?? Number.NaN, 6);
  }
}

describe("LTX video VAE weight loading", () => {
  test("maps decoder checkpoint paths and latent stats", () => {
    expect(ltxVideoAutoencoderWeightPath("latents_mean")).toBe("latents_mean");
    expect(ltxVideoAutoencoderWeightPath("latents_std")).toBe("latents_std");
    expect(ltxVideoAutoencoderWeightPath("decoder.conv_in.conv.weight")).toBe(
      "decoder.convIn.conv.weight",
    );
    expect(ltxVideoAutoencoderWeightPath("decoder.conv_out.conv.bias")).toBe(
      "decoder.convOut.conv.bias",
    );
    expect(ltxVideoAutoencoderWeightPath("decoder.up_blocks.1.upsamplers.0.conv.conv.weight")).toBe(
      "decoder.upBlocks.1.upsampler.conv.conv.weight",
    );
    expect(
      ltxVideoAutoencoderWeightPath("decoder.mid_block.resnets.0.conv_shortcut.conv.bias"),
    ).toBe("decoder.midBlock.resnets.0.convShortcut.conv.bias");
    expect(ltxVideoAutoencoderWeightPath("encoder.conv_in.conv.weight")).toBeNull();
    expect(ltxVideoAutoencoderWeightPath("")).toBeNull();
    expect(ltxVideoAutoencoderWeightPath("decoder.norm_out.num_batches_tracked")).toBeNull();
  });

  test("transposes Diffusers Conv3d kernels into MLX channel-last layout", () => {
    using checkpoint = MxArray.fromData([1, 2, 3, 4], [1, 2, 2, 1, 1]);
    using transformed = transformLtxVideoAutoencoderWeight(
      "decoder.convIn.conv.weight",
      checkpoint,
    );

    mxEval(transformed);
    expect(transformed.shape).toEqual([1, 2, 1, 1, 2]);
    expectCloseList(transformed.toTypedArray(), [1, 3, 2, 4]);
  });

  test("loads decoder-only weights and non-parameter latent stats", async () => {
    await withTempDirectory("ltx-video-vae-", async (directory) => {
      using model = new LtxVideoAutoencoderKL(tinyConfig());
      const tensors = minimalDecoderTensors();
      const checkpointPath = join(directory, "diffusion_pytorch_model.safetensors");
      try {
        await saveSafetensors(tensors, checkpointPath);
      } finally {
        freeTensors(tensors);
      }

      const result = await loadLtxVideoAutoencoderWeights(model, vaeComponent([checkpointPath]));
      const parameterPaths = treeFlatten(model.parameters()).map(([path]) => path.join("."));

      expect(result.shardCount).toBe(1);
      expect(result.unexpectedWeights).toEqual([]);
      expect(result.assignedPaths).toContain("latents_mean");
      expect(result.assignedPaths).toContain("latents_std");
      expect(parameterPaths).toContain("decoder.convIn.conv.weight");
      expect(parameterPaths).not.toContain("encoder.convIn.conv.weight");
      expect(model.latentsMean).toEqual([2]);
      expect(model.latentsStd).toEqual([3]);
    });
  });

  test("loads decoder weights through a safetensors index", async () => {
    await withTempDirectory("ltx-video-vae-index-", async (directory) => {
      using model = new LtxVideoAutoencoderKL(tinyConfig());
      const tensors = minimalDecoderTensors();
      const checkpointPath = join(directory, "decoder-00001-of-00001.safetensors");
      const weightMap = Object.fromEntries(
        Object.keys(tensors).map((name) => [name, "decoder-00001-of-00001.safetensors"]),
      );
      try {
        await saveSafetensors(tensors, checkpointPath);
      } finally {
        freeTensors(tensors);
      }
      const indexPath = join(directory, "diffusion_pytorch_model.safetensors.index.json");
      await Bun.write(indexPath, JSON.stringify({ metadata: {}, weight_map: weightMap }));

      const result = await loadLtxVideoAutoencoderWeights(model, vaeComponent([indexPath]));

      expect(result.shardCount).toBe(1);
      expect(result.assignedPaths).toContain("decoder.convIn.conv.weight");
      expect(model.latentsMean).toEqual([2]);
    });
  });

  test("rejects missing shards, missing stats, and strict unexpected tensors", async () => {
    await withTempDirectory("ltx-video-vae-errors-", async (directory) => {
      using emptyModel = new LtxVideoAutoencoderKL(tinyConfig());
      await expect(loadLtxVideoAutoencoderWeights(emptyModel, vaeComponent([]))).rejects.toThrow(
        "no safetensors",
      );

      using missingStatsModel = new LtxVideoAutoencoderKL(tinyConfig());
      const missingStatsTensors = minimalDecoderTensors();
      const latentMean = missingStatsTensors.latents_mean;
      const latentStd = missingStatsTensors.latents_std;
      if (latentMean === undefined || latentStd === undefined) {
        throw new Error("minimalDecoderTensors must include latent stats.");
      }
      latentMean.free();
      latentStd.free();
      delete missingStatsTensors.latents_mean;
      delete missingStatsTensors.latents_std;
      const missingStatsPath = join(directory, "missing-stats.safetensors");
      try {
        await saveSafetensors(missingStatsTensors, missingStatsPath);
      } finally {
        freeTensors(missingStatsTensors);
      }
      await expect(
        loadLtxVideoAutoencoderWeights(missingStatsModel, vaeComponent([missingStatsPath])),
      ).rejects.toThrow("missing required parameters");

      using strictModel = new LtxVideoAutoencoderKL(tinyConfig());
      const strictTensors = minimalDecoderTensors({
        "encoder.conv_in.conv.weight": zeros([1]),
      });
      const strictPath = join(directory, "unexpected.safetensors");
      try {
        await saveSafetensors(strictTensors, strictPath);
      } finally {
        freeTensors(strictTensors);
      }
      await expect(
        loadLtxVideoAutoencoderWeights(strictModel, vaeComponent([strictPath]), {
          strictUnexpectedWeights: true,
        }),
      ).rejects.toThrow("unexpected unmapped weights");
    });
  });
});
