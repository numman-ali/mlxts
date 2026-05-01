import { describe, expect, test } from "bun:test";
import { MxArray, mxEval, saveSafetensors, treeFlatten, zeros } from "@mlxts/core";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import type { DiffusionSnapshotComponent } from "../../pretrained/snapshot-manifest";
import { Ltx2VideoAutoencoderKL } from "./autoencoder-ltx2";
import {
  loadLtx2VideoAutoencoderWeights,
  ltx2VideoAutoencoderWeightPath,
  transformLtx2VideoAutoencoderWeight,
} from "./autoencoder-ltx2-weights";
import type { Ltx2VideoAutoencoderConfig } from "./config";

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

function tinyConfig(): Ltx2VideoAutoencoderConfig {
  return {
    inChannels: 3,
    outChannels: 1,
    latentChannels: 1,
    latentChannelsOut: 2,
    blockOutChannels: [8],
    downBlockTypes: ["LTX2VideoDownBlock3D"],
    decoderBlockOutChannels: [8],
    layersPerBlock: [0, 0],
    decoderLayersPerBlock: [0, 0],
    spatioTemporalScaling: [true],
    decoderSpatioTemporalScaling: [true],
    decoderInjectNoise: [false, false],
    downsampleTypes: ["spatiotemporal"],
    upsampleTypes: ["spatiotemporal"],
    upsampleResidual: [true],
    upsampleFactors: [2],
    timestepConditioning: false,
    patchSize: 1,
    patchSizeT: 1,
    resnetNormEps: 1e-6,
    scalingFactor: 1,
    encoderCausal: true,
    decoderCausal: true,
    encoderSpatialPaddingMode: "zeros",
    decoderSpatialPaddingMode: "reflect",
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
    className: "AutoencoderKLLTX2Video",
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
    "decoder.conv_in.conv.weight": zeros([8, 1, 3, 3, 3]),
    "decoder.conv_in.conv.bias": zeros([8]),
    "decoder.up_blocks.0.upsamplers.0.conv.conv.weight": zeros([32, 8, 3, 3, 3]),
    "decoder.up_blocks.0.upsamplers.0.conv.conv.bias": zeros([32]),
    "decoder.conv_out.conv.weight": zeros([1, 4, 3, 3, 3]),
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

describe("LTX-2 video VAE weight loading", () => {
  test("maps decoder checkpoint paths and latent stats", () => {
    expect(ltx2VideoAutoencoderWeightPath("latents_mean")).toBe("latents_mean");
    expect(ltx2VideoAutoencoderWeightPath("latents_std")).toBe("latents_std");
    expect(ltx2VideoAutoencoderWeightPath("decoder.conv_in.conv.weight")).toBe(
      "decoder.convIn.conv.weight",
    );
    expect(ltx2VideoAutoencoderWeightPath("decoder.conv_out.conv.bias")).toBe(
      "decoder.convOut.conv.bias",
    );
    expect(
      ltx2VideoAutoencoderWeightPath("decoder.up_blocks.0.upsamplers.0.conv.conv.weight"),
    ).toBe("decoder.upBlocks.0.upsamplers.0.conv.conv.weight");
    expect(ltx2VideoAutoencoderWeightPath("decoder.up_blocks.1.conv_in.conv_shortcut.weight")).toBe(
      "decoder.upBlocks.1.convIn.convShortcut.weight",
    );
    expect(ltx2VideoAutoencoderWeightPath("encoder.conv_in.conv.weight")).toBeNull();
    expect(ltx2VideoAutoencoderWeightPath("")).toBeNull();
    expect(ltx2VideoAutoencoderWeightPath("decoder.norm_out.num_batches_tracked")).toBeNull();
  });

  test("transposes Diffusers Conv3d kernels into MLX channel-last layout", () => {
    using checkpoint = MxArray.fromData([1, 2, 3, 4], [1, 2, 2, 1, 1]);
    using transformed = transformLtx2VideoAutoencoderWeight(
      "decoder.convIn.conv.weight",
      checkpoint,
    );

    mxEval(transformed);
    expect(transformed.shape).toEqual([1, 2, 1, 1, 2]);
    expectCloseList(transformed.toTypedArray(), [1, 3, 2, 4]);
  });

  test("loads decoder-only weights and non-parameter latent stats", async () => {
    await withTempDirectory("ltx2-video-vae-", async (directory) => {
      using model = new Ltx2VideoAutoencoderKL(tinyConfig());
      const tensors = minimalDecoderTensors();
      const checkpointPath = join(directory, "diffusion_pytorch_model.safetensors");
      try {
        await saveSafetensors(tensors, checkpointPath);
      } finally {
        freeTensors(tensors);
      }

      const result = await loadLtx2VideoAutoencoderWeights(model, vaeComponent([checkpointPath]));
      const parameterPaths = treeFlatten(model.parameters()).map(([path]) => path.join("."));

      expect(result.shardCount).toBe(1);
      expect(result.unexpectedWeights).toEqual([]);
      expect(result.assignedPaths).toContain("latents_mean");
      expect(result.assignedPaths).toContain("latents_std");
      expect(parameterPaths).toContain("decoder.convIn.conv.weight");
      expect(parameterPaths).toContain("decoder.upBlocks.0.upsamplers.0.conv.conv.weight");
      expect(parameterPaths).not.toContain("encoder.convIn.conv.weight");
      expect(model.latentsMean).toEqual([2]);
      expect(model.latentsStd).toEqual([3]);
    });
  });

  test("loads decoder weights through a safetensors index", async () => {
    await withTempDirectory("ltx2-video-vae-index-", async (directory) => {
      using model = new Ltx2VideoAutoencoderKL(tinyConfig());
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

      const result = await loadLtx2VideoAutoencoderWeights(model, vaeComponent([indexPath]));

      expect(result.shardCount).toBe(1);
      expect(result.assignedPaths).toContain("decoder.convIn.conv.weight");
      expect(model.latentsMean).toEqual([2]);
    });
  });

  test("rejects missing shards, missing stats, and strict unexpected tensors", async () => {
    await withTempDirectory("ltx2-video-vae-errors-", async (directory) => {
      using emptyModel = new Ltx2VideoAutoencoderKL(tinyConfig());
      await expect(loadLtx2VideoAutoencoderWeights(emptyModel, vaeComponent([]))).rejects.toThrow(
        "no safetensors",
      );

      using missingStatsModel = new Ltx2VideoAutoencoderKL(tinyConfig());
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
        loadLtx2VideoAutoencoderWeights(missingStatsModel, vaeComponent([missingStatsPath])),
      ).rejects.toThrow("missing required parameters");

      using strictModel = new Ltx2VideoAutoencoderKL(tinyConfig());
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
        loadLtx2VideoAutoencoderWeights(strictModel, vaeComponent([strictPath]), {
          strictUnexpectedWeights: true,
        }),
      ).rejects.toThrow("unexpected unmapped weights");
    });
  });
});
