import { describe, expect, test } from "bun:test";
import { MxArray, mxEval, ones, saveSafetensors, treeFlatten, zeros } from "@mlxts/core";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import type { DiffusionSnapshotComponent } from "../../pretrained/snapshot-manifest";
import { loadDiffusionSnapshotManifest } from "../../pretrained/snapshot-manifest";
import { LtxVideoLatentUpsamplerModel } from "./latent-upsampler";
import {
  loadLtxVideoLatentUpsamplerFromDirectory,
  loadLtxVideoLatentUpsamplerFromSnapshot,
  loadLtxVideoLatentUpsamplerWeights,
  ltxVideoLatentUpsamplerWeightPath,
  transformLtxVideoLatentUpsamplerWeight,
} from "./latent-upsampler-weights";

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
    library: "ltx",
    className: "LTXLatentUpsamplerModel",
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
    _class_name: "LTXLatentUpsamplerModel",
    dims: 3,
    in_channels: 1,
    mid_channels: 32,
    num_blocks_per_stage: 0,
    spatial_upsample: true,
    temporal_upsample: false,
  });
}

function tinyModel(): LtxVideoLatentUpsamplerModel {
  return new LtxVideoLatentUpsamplerModel({
    inChannels: 1,
    midChannels: 32,
    numBlocksPerStage: 0,
    dims: 3,
    spatialUpsample: true,
    temporalUpsample: false,
    rawConfig: {},
  });
}

function minimalTensors(extra: Record<string, MxArray> = {}): Record<string, MxArray> {
  return {
    "initial_conv.weight": zeros([32, 1, 3, 3, 3]),
    "initial_conv.bias": zeros([32]),
    "initial_norm.weight": ones([32]),
    "initial_norm.bias": zeros([32]),
    "upsampler.0.weight": zeros([128, 32, 3, 3]),
    "upsampler.0.bias": zeros([128]),
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

describe("LTX video latent upsampler weight loading", () => {
  test("maps Diffusers checkpoint paths onto the package tree", () => {
    expect(ltxVideoLatentUpsamplerWeightPath("initial_conv.weight")).toBe("initialConv.weight");
    expect(ltxVideoLatentUpsamplerWeightPath("initial_norm.bias")).toBe("initialNorm.bias");
    expect(ltxVideoLatentUpsamplerWeightPath("res_blocks.0.conv1.weight")).toBe(
      "resBlocks.0.conv1.weight",
    );
    expect(ltxVideoLatentUpsamplerWeightPath("upsampler.0.weight")).toBe("upsampler.conv.weight");
    expect(ltxVideoLatentUpsamplerWeightPath("post_upsample_res_blocks.0.norm2.bias")).toBe(
      "postUpsampleResBlocks.0.norm2.bias",
    );
    expect(ltxVideoLatentUpsamplerWeightPath("final_conv.bias")).toBe("finalConv.bias");
    expect(ltxVideoLatentUpsamplerWeightPath("unknown.weight")).toBeNull();
    expect(ltxVideoLatentUpsamplerWeightPath("")).toBeNull();
    expect(ltxVideoLatentUpsamplerWeightPath("initial_norm.num_batches_tracked")).toBeNull();
  });

  test("transposes Conv2d and Conv3d kernels into MLX channel-last layout", () => {
    using conv2d = MxArray.fromData([1, 2, 3, 4], [1, 2, 2, 1]);
    using conv3d = MxArray.fromData([1, 2, 3, 4], [1, 2, 2, 1, 1]);
    using transformed2d = transformLtxVideoLatentUpsamplerWeight("upsampler.conv.weight", conv2d);
    using transformed3d = transformLtxVideoLatentUpsamplerWeight("initialConv.weight", conv3d);

    mxEval(transformed2d, transformed3d);
    expect(transformed2d.shape).toEqual([1, 2, 1, 2]);
    expectCloseList(transformed2d.toTypedArray(), [1, 3, 2, 4]);
    expect(transformed3d.shape).toEqual([1, 2, 1, 1, 2]);
    expectCloseList(transformed3d.toTypedArray(), [1, 3, 2, 4]);
  });

  test("loads complete latent upsampler weights from a component directory", async () => {
    await withTempDirectory("ltx-video-upsampler-", async (directory) => {
      writeConfig(directory);
      const tensors = minimalTensors();
      const checkpointPath = join(directory, "diffusion_pytorch_model.safetensors");
      try {
        await saveSafetensors(tensors, checkpointPath);
      } finally {
        freeTensors(tensors);
      }

      using model = await loadLtxVideoLatentUpsamplerFromDirectory(directory);
      const parameterPaths = treeFlatten(model.parameters()).map(([path]) => path.join("."));
      using latents = zeros([1, 1, 1, 1, 1]);
      using output = model.forward(latents);

      mxEval(output);
      expect(parameterPaths).toContain("upsampler.conv.weight");
      expect(output.shape).toEqual([1, 1, 1, 2, 2]);
    });
  });

  test("loads sidecar snapshot manifests and safetensors indices", async () => {
    await withTempDirectory("ltx-video-upsampler-snapshot-", async (directory) => {
      writeJson(join(directory, "model_index.json"), {
        _class_name: "LTXLatentUpsamplePipeline",
        latent_upsampler: ["ltx", "LTXLatentUpsamplerModel"],
        vae: ["diffusers", "AutoencoderKLLTXVideo"],
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
      using model = await loadLtxVideoLatentUpsamplerFromSnapshot(manifest);

      expect(model.config.inChannels).toBe(1);
      expect(model.config.midChannels).toBe(32);
    });
  });

  test("rejects missing shards, strict unexpected tensors, and wrong pipeline manifests", async () => {
    await withTempDirectory("ltx-video-upsampler-errors-", async (directory) => {
      using emptyModel = tinyModel();
      await expect(
        loadLtxVideoLatentUpsamplerWeights(emptyModel, upsamplerComponent([])),
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
        loadLtxVideoLatentUpsamplerWeights(strictModel, upsamplerComponent([strictPath]), {
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

      await expect(loadLtxVideoLatentUpsamplerFromSnapshot(manifest)).rejects.toThrow(
        "requires LTXLatentUpsamplePipeline",
      );
    });
  });
});
