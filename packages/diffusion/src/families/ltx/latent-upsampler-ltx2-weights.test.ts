import { describe, expect, test } from "bun:test";
import { MxArray, mxEval, ones, saveSafetensors, treeFlatten, zeros } from "@mlxts/core";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import type { DiffusionSnapshotComponent } from "../../pretrained/snapshot-manifest";
import { loadDiffusionSnapshotManifest } from "../../pretrained/snapshot-manifest";
import { Ltx2LatentUpsamplerModel } from "./latent-upsampler-ltx2";
import {
  loadLtx2LatentUpsamplerFromDirectory,
  loadLtx2LatentUpsamplerFromSnapshot,
  loadLtx2LatentUpsamplerWeights,
  ltx2LatentUpsamplerWeightPath,
  transformLtx2LatentUpsamplerWeight,
} from "./latent-upsampler-ltx2-weights";

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

function upsamplerComponent(weightPaths: readonly string[]): DiffusionSnapshotComponent {
  return {
    name: "latent_upsampler",
    role: "latent-upsampler",
    library: "ltx2",
    className: "LTX2LatentUpsamplerModel",
    enabled: true,
    optional: false,
    subfolder: "latent_upsampler",
    metadataPaths: [],
    weightPaths,
  };
}

function freeTensors(tensors: Record<string, MxArray>): void {
  for (const tensor of Object.values(tensors)) {
    tensor.free();
  }
}

function writeJson(path: string, payload: unknown): void {
  writeFileSync(path, JSON.stringify(payload));
}

function writeConfig(directory: string): void {
  writeJson(join(directory, "config.json"), {
    _class_name: "LTX2LatentUpsamplerModel",
    dims: 3,
    in_channels: 1,
    mid_channels: 32,
    num_blocks_per_stage: 0,
    rational_spatial_scale: 1.5,
    spatial_upsample: true,
    temporal_upsample: false,
    use_rational_resampler: true,
  });
}

function tinyModel(): Ltx2LatentUpsamplerModel {
  return new Ltx2LatentUpsamplerModel({
    inChannels: 1,
    midChannels: 32,
    numBlocksPerStage: 0,
    dims: 3,
    spatialUpsample: true,
    temporalUpsample: false,
    rationalSpatialScale: 1.5,
    useRationalResampler: true,
    rawConfig: {},
  });
}

function minimalTensors(extra: Record<string, MxArray> = {}): Record<string, MxArray> {
  return {
    "initial_conv.weight": zeros([32, 1, 3, 3, 3]),
    "initial_conv.bias": zeros([32]),
    "initial_norm.weight": ones([32]),
    "initial_norm.bias": zeros([32]),
    "upsampler.blur_down.kernel": zeros([1, 1, 5, 5]),
    "upsampler.conv.weight": zeros([288, 32, 3, 3]),
    "upsampler.conv.bias": zeros([288]),
    "final_conv.weight": zeros([1, 32, 3, 3, 3]),
    "final_conv.bias": zeros([1]),
    ...extra,
  };
}

function expectCloseList(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBeCloseTo(expected[index] ?? Number.NaN, 6);
  }
}

describe("LTX-2 latent upsampler weight loading", () => {
  test("maps Diffusers checkpoint paths onto the package tree", () => {
    expect(ltx2LatentUpsamplerWeightPath("initial_conv.weight")).toBe("initialConv.weight");
    expect(ltx2LatentUpsamplerWeightPath("initial_norm.bias")).toBe("initialNorm.bias");
    expect(ltx2LatentUpsamplerWeightPath("res_blocks.0.conv1.weight")).toBe(
      "resBlocks.0.conv1.weight",
    );
    expect(ltx2LatentUpsamplerWeightPath("upsampler.conv.weight")).toBe("upsampler.conv.weight");
    expect(ltx2LatentUpsamplerWeightPath("upsampler.0.weight")).toBe("upsampler.conv.weight");
    expect(ltx2LatentUpsamplerWeightPath("upsampler.blur_down.kernel")).toBeNull();
    expect(ltx2LatentUpsamplerWeightPath("post_upsample_res_blocks.0.norm2.bias")).toBe(
      "postUpsampleResBlocks.0.norm2.bias",
    );
    expect(ltx2LatentUpsamplerWeightPath("final_conv.bias")).toBe("finalConv.bias");
    expect(ltx2LatentUpsamplerWeightPath("unknown.weight")).toBeNull();
    expect(ltx2LatentUpsamplerWeightPath("")).toBeNull();
    expect(ltx2LatentUpsamplerWeightPath("initial_norm.num_batches_tracked")).toBeNull();
  });

  test("transposes Conv2d and Conv3d kernels into MLX channel-last layout", () => {
    using conv2d = MxArray.fromData([1, 2, 3, 4], [1, 2, 2, 1]);
    using conv3d = MxArray.fromData([1, 2, 3, 4], [1, 2, 2, 1, 1]);
    using transformed2d = transformLtx2LatentUpsamplerWeight("upsampler.conv.weight", conv2d);
    using transformed3d = transformLtx2LatentUpsamplerWeight("initialConv.weight", conv3d);

    mxEval(transformed2d, transformed3d);
    expect(transformed2d.shape).toEqual([1, 2, 1, 2]);
    expectCloseList(transformed2d.toTypedArray(), [1, 3, 2, 4]);
    expect(transformed3d.shape).toEqual([1, 2, 1, 1, 2]);
    expectCloseList(transformed3d.toTypedArray(), [1, 3, 2, 4]);
  });

  test("loads complete LTX-2 latent upsampler weights from a component directory", async () => {
    await withTempDirectory("ltx2-upsampler-", async (directory) => {
      writeConfig(directory);
      const tensors = minimalTensors();
      const checkpointPath = join(directory, "diffusion_pytorch_model.safetensors");
      try {
        await saveSafetensors(tensors, checkpointPath);
      } finally {
        freeTensors(tensors);
      }

      using model = await loadLtx2LatentUpsamplerFromDirectory(directory);
      const parameterPaths = treeFlatten(model.parameters()).map(([path]) => path.join("."));
      using latents = zeros([1, 1, 1, 4, 4]);
      using output = model.forward(latents);

      mxEval(output);
      expect(parameterPaths).toContain("upsampler.conv.weight");
      expect(output.shape).toEqual([1, 1, 1, 6, 6]);
    });
  });

  test("loads LTX-2 sidecar snapshot manifests and safetensors indices", async () => {
    await withTempDirectory("ltx2-upsampler-snapshot-", async (directory) => {
      writeJson(join(directory, "model_index.json"), {
        _class_name: "LTX2LatentUpsamplePipeline",
        latent_upsampler: ["ltx2", "LTX2LatentUpsamplerModel"],
        vae: ["diffusers", "AutoencoderKLLTX2Video"],
      });
      const componentDirectory = join(directory, "latent_upsampler");
      const vaeDirectory = join(directory, "vae");
      mkdirSync(componentDirectory, { recursive: true });
      mkdirSync(vaeDirectory, { recursive: true });
      writeConfig(componentDirectory);
      writeJson(join(vaeDirectory, "config.json"), {});
      writeFileSync(join(vaeDirectory, "diffusion_pytorch_model.safetensors"), "");
      const tensors = minimalTensors();
      const checkpointPath = join(componentDirectory, "upsampler-00001-of-00001.safetensors");
      const indexPath = join(componentDirectory, "diffusion_pytorch_model.safetensors.index.json");
      const weightMap = Object.fromEntries(
        Object.keys(tensors).map((name) => [name, "upsampler-00001-of-00001.safetensors"]),
      );
      try {
        await saveSafetensors(tensors, checkpointPath);
      } finally {
        freeTensors(tensors);
      }
      writeJson(indexPath, { metadata: {}, weight_map: weightMap });

      const manifest = await loadDiffusionSnapshotManifest(directory);
      using model = await loadLtx2LatentUpsamplerFromSnapshot(manifest);

      expect(model.config.inChannels).toBe(1);
      expect(model.config.midChannels).toBe(32);
      expect(model.config.rationalSpatialScale).toBe(1.5);
    });
  });

  test("rejects missing shards, strict unexpected tensors, and wrong pipeline manifests", async () => {
    await withTempDirectory("ltx2-upsampler-errors-", async (directory) => {
      using emptyModel = tinyModel();
      await expect(
        loadLtx2LatentUpsamplerWeights(emptyModel, upsamplerComponent([])),
      ).rejects.toThrow("no safetensors");

      using strictModel = tinyModel();
      const strictTensors = minimalTensors({ "unknown.weight": zeros([1]) });
      const strictPath = join(directory, "unexpected.safetensors");
      try {
        await saveSafetensors(strictTensors, strictPath);
      } finally {
        freeTensors(strictTensors);
      }
      await expect(
        loadLtx2LatentUpsamplerWeights(strictModel, upsamplerComponent([strictPath]), {
          strictUnexpectedWeights: true,
        }),
      ).rejects.toThrow("unexpected unmapped weights");

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
      mkdirSync(join(directory, "scheduler"), { recursive: true });
      writeJson(join(directory, "scheduler", "scheduler_config.json"), {
        _class_name: "FlowMatchEulerDiscreteScheduler",
      });
      for (const component of [
        "audio_vae",
        "connectors",
        "text_encoder",
        "transformer",
        "vae",
        "vocoder",
      ]) {
        const componentDirectory = join(directory, component);
        mkdirSync(componentDirectory, { recursive: true });
        writeJson(join(componentDirectory, "config.json"), {});
        writeFileSync(join(componentDirectory, "diffusion_pytorch_model.safetensors"), "");
      }
      mkdirSync(join(directory, "tokenizer"), { recursive: true });
      writeFileSync(join(directory, "tokenizer", "tokenizer.json"), "");
      const manifest = await loadDiffusionSnapshotManifest(directory);

      await expect(loadLtx2LatentUpsamplerFromSnapshot(manifest)).rejects.toThrow(
        "requires LTX2LatentUpsamplePipeline",
      );
    });
  });
});
