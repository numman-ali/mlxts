import { describe, expect, test } from "bun:test";
import { MxArray, mxEval, saveSafetensors, treeFlatten, zeros } from "@mlxts/core";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import type { DiffusionSnapshotComponent } from "../../pretrained/snapshot-manifest";
import { loadDiffusionSnapshotManifest } from "../../pretrained/snapshot-manifest";
import type { Ltx2VocoderConfig } from "./config";
import { Ltx2Vocoder } from "./vocoder-ltx2";
import {
  loadLtx2VocoderFromSnapshot,
  loadLtx2VocoderWeights,
  ltx2VocoderWeightPath,
  transformLtx2VocoderWeight,
} from "./vocoder-ltx2-weights";

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

function tinyConfig(overrides: Partial<Ltx2VocoderConfig> = {}): Ltx2VocoderConfig {
  return {
    inChannels: 4,
    hiddenChannels: 4,
    outChannels: 1,
    upsampleKernelSizes: [4],
    upsampleFactors: [2],
    totalUpsampleFactor: 2,
    resnetKernelSizes: [3],
    resnetDilations: [[1]],
    actFn: "leaky_relu",
    leakyReluNegativeSlope: 0.1,
    antialias: false,
    antialiasRatio: 2,
    antialiasKernelSize: 12,
    finalActFn: "tanh",
    finalBias: true,
    outputSamplingRate: 20,
    rawConfig: {},
    ...overrides,
  };
}

function vocoderComponent(weightPaths: readonly string[]): DiffusionSnapshotComponent {
  return {
    name: "vocoder",
    role: "vocoder",
    library: "ltx2",
    className: "LTX2Vocoder",
    enabled: true,
    optional: false,
    subfolder: "vocoder",
    metadataPaths: [],
    weightPaths,
  };
}

function writeJson(path: string, payload: unknown): void {
  writeFileSync(path, JSON.stringify(payload));
}

function writeComponent(
  snapshot: string,
  name: string,
  config: Record<string, unknown>,
  needsWeights = true,
): void {
  const directory = join(snapshot, name);
  mkdirSync(directory, { recursive: true });
  writeJson(
    join(directory, name === "scheduler" ? "scheduler_config.json" : "config.json"),
    config,
  );
  if (needsWeights) {
    writeFileSync(join(directory, "diffusion_pytorch_model.safetensors"), "");
  }
}

function freeTensors(tensors: Record<string, MxArray>): void {
  for (const tensor of Object.values(tensors)) {
    tensor.free();
  }
}

function checkpointNameForVocoderPath(path: string): string {
  return path.replace(/^convIn\./, "conv_in.").replace(/^convOut\./, "conv_out.");
}

function checkpointTensorForPath(path: string, tensor: MxArray): MxArray {
  if (!path.endsWith(".weight") || tensor.shape.length !== 3) {
    return zeros([...tensor.shape]);
  }
  const [outputChannels, kernelSize, inputChannels] = tensor.shape;
  if (path.startsWith("upsamplers.")) {
    return zeros([inputChannels ?? 0, outputChannels ?? 0, kernelSize ?? 0]);
  }
  return zeros([outputChannels ?? 0, inputChannels ?? 0, kernelSize ?? 0]);
}

function checkpointTensorsForVocoder(model: Ltx2Vocoder): Record<string, MxArray> {
  const output: Record<string, MxArray> = {};
  for (const [path, tensor] of treeFlatten(model.parameters())) {
    const internalPath = path.join(".");
    output[checkpointNameForVocoderPath(internalPath)] = checkpointTensorForPath(
      internalPath,
      tensor,
    );
  }
  return output;
}

async function writeVocoderWeights(
  path: string,
  config: Ltx2VocoderConfig = tinyConfig(),
): Promise<void> {
  using model = new Ltx2Vocoder(config);
  const tensors = checkpointTensorsForVocoder(model);
  try {
    await saveSafetensors(tensors, path);
  } finally {
    freeTensors(tensors);
  }
}

function snapshotVocoderConfig(): Record<string, unknown> {
  return {
    _class_name: "LTX2Vocoder",
    hidden_channels: 4,
    in_channels: 4,
    out_channels: 1,
    output_sampling_rate: 20,
    resnet_dilations: [
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 1],
    ],
    resnet_kernel_sizes: [3, 5, 7],
    upsample_factors: [2],
    upsample_kernel_sizes: [4],
  };
}

function snapshotVocoderRuntimeConfig(): Ltx2VocoderConfig {
  return tinyConfig({
    resnetKernelSizes: [3, 5, 7],
    resnetDilations: [
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 1],
    ],
  });
}

async function writeLtx2VocoderSnapshot(snapshot: string): Promise<void> {
  writeJson(join(snapshot, "model_index.json"), {
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
  writeComponent(snapshot, "transformer", {
    _class_name: "LTX2VideoTransformer3DModel",
    audio_hop_length: 1,
    audio_in_channels: 1,
    audio_out_channels: 1,
    audio_sampling_rate: 10,
    in_channels: 1,
    out_channels: 1,
    vae_scale_factors: [8, 32, 32],
  });
  writeComponent(snapshot, "vae", {
    _class_name: "AutoencoderKLLTX2Video",
    latent_channels: 1,
  });
  writeComponent(snapshot, "audio_vae", {
    _class_name: "AutoencoderKLLTX2Audio",
    base_channels: 1,
    latent_channels: 1,
    mel_bins: 4,
    mel_hop_length: 1,
    output_channels: 1,
    sample_rate: 10,
  });
  writeComponent(snapshot, "connectors", { _class_name: "LTX2TextConnectors" });
  writeComponent(snapshot, "vocoder", snapshotVocoderConfig());
  writeComponent(snapshot, "scheduler", { _class_name: "FlowMatchEulerDiscreteScheduler" }, false);
  writeComponent(snapshot, "text_encoder", {});
  const tokenizer = join(snapshot, "tokenizer");
  mkdirSync(tokenizer, { recursive: true });
  writeFileSync(join(tokenizer, "tokenizer.json"), "{}");
  await writeVocoderWeights(
    join(snapshot, "vocoder", "diffusion_pytorch_model.safetensors"),
    snapshotVocoderRuntimeConfig(),
  );
}

function expectCloseList(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBeCloseTo(expected[index] ?? Number.NaN, 6);
  }
}

describe("LTX-2 vocoder weight loading", () => {
  test("maps Diffusers vocoder checkpoint paths", () => {
    expect(ltx2VocoderWeightPath("conv_in.weight")).toBe("convIn.weight");
    expect(ltx2VocoderWeightPath("conv_out.bias")).toBe("convOut.bias");
    expect(ltx2VocoderWeightPath("upsamplers.0.weight")).toBe("upsamplers.0.weight");
    expect(ltx2VocoderWeightPath("resnets.0.convs1.0.bias")).toBe("resnets.0.convs1.0.bias");
    expect(ltx2VocoderWeightPath("")).toBeNull();
    expect(ltx2VocoderWeightPath("resnets.0.convs1.0.num_batches_tracked")).toBeNull();
  });

  test("transposes Conv1d and ConvTranspose1d checkpoint kernels", () => {
    using conv = MxArray.fromData([1, 2, 3, 4], [1, 2, 2]);
    using transformedConv = transformLtx2VocoderWeight("convIn.weight", conv);
    mxEval(transformedConv);
    expect(transformedConv.shape).toEqual([1, 2, 2]);
    expectCloseList(transformedConv.toTypedArray(), [1, 3, 2, 4]);

    using deconv = MxArray.fromData([1, 2, 3, 4], [2, 1, 2]);
    using transformedDeconv = transformLtx2VocoderWeight("upsamplers.0.weight", deconv);
    mxEval(transformedDeconv);
    expect(transformedDeconv.shape).toEqual([1, 2, 2]);
    expectCloseList(transformedDeconv.toTypedArray(), [1, 3, 2, 4]);
  });

  test("loads vocoder weights and indexed shards", async () => {
    await withTempDirectory("ltx2-vocoder-", async (directory) => {
      using model = new Ltx2Vocoder(tinyConfig());
      const checkpointPath = join(directory, "diffusion_pytorch_model.safetensors");
      await writeVocoderWeights(checkpointPath);

      const result = await loadLtx2VocoderWeights(model, vocoderComponent([checkpointPath]));

      expect(result.shardCount).toBe(1);
      expect(result.unexpectedWeights).toEqual([]);
      expect(result.assignedPaths).toContain("convIn.weight");
      expect(result.assignedPaths).toContain("upsamplers.0.weight");
    });

    await withTempDirectory("ltx2-vocoder-index-", async (directory) => {
      using model = new Ltx2Vocoder(tinyConfig());
      const tensors = checkpointTensorsForVocoder(model);
      const shardName = "vocoder-00001-of-00001.safetensors";
      const checkpointPath = join(directory, shardName);
      const weightMap = Object.fromEntries(Object.keys(tensors).map((name) => [name, shardName]));
      try {
        await saveSafetensors(tensors, checkpointPath);
      } finally {
        freeTensors(tensors);
      }
      const indexPath = join(directory, "diffusion_pytorch_model.safetensors.index.json");
      writeJson(indexPath, { metadata: {}, weight_map: weightMap });

      const result = await loadLtx2VocoderWeights(model, vocoderComponent([indexPath]));

      expect(result.shardCount).toBe(1);
      expect(result.assignedPaths).toContain("convOut.weight");
    });
  });

  test("loads a vocoder directly from an inspected LTX-2 snapshot manifest", async () => {
    await withTempDirectory("ltx2-vocoder-snapshot-", async (directory) => {
      await writeLtx2VocoderSnapshot(directory);
      const manifest = await loadDiffusionSnapshotManifest(directory);

      using model = await loadLtx2VocoderFromSnapshot(manifest);

      expect(model.inChannels).toBe(4);
      expect(model.totalUpsampleFactor).toBe(2);
    });
  });

  test("rejects missing, mismatched, strict unexpected, and invalid shard inputs", async () => {
    await withTempDirectory("ltx2-vocoder-errors-", async (directory) => {
      using missingModel = new Ltx2Vocoder(tinyConfig());
      const missingTensors = checkpointTensorsForVocoder(missingModel);
      missingTensors["conv_in.weight"]?.free();
      delete missingTensors["conv_in.weight"];
      const missingPath = join(directory, "missing.safetensors");
      try {
        await saveSafetensors(missingTensors, missingPath);
      } finally {
        freeTensors(missingTensors);
      }
      await expect(
        loadLtx2VocoderWeights(missingModel, vocoderComponent([missingPath])),
      ).rejects.toThrow("convIn.weight");

      using mismatchModel = new Ltx2Vocoder(tinyConfig());
      const mismatchTensors = checkpointTensorsForVocoder(mismatchModel);
      mismatchTensors["conv_in.weight"]?.free();
      mismatchTensors["conv_in.weight"] = zeros([1, 1, 1]);
      const mismatchPath = join(directory, "mismatch.safetensors");
      try {
        await saveSafetensors(mismatchTensors, mismatchPath);
      } finally {
        freeTensors(mismatchTensors);
      }
      await expect(
        loadLtx2VocoderWeights(mismatchModel, vocoderComponent([mismatchPath])),
      ).rejects.toThrow("convIn.weight");

      using strictModel = new Ltx2Vocoder(tinyConfig());
      const strictTensors = checkpointTensorsForVocoder(strictModel);
      strictTensors["act_out.alpha"] = zeros([1]);
      const strictPath = join(directory, "strict.safetensors");
      try {
        await saveSafetensors(strictTensors, strictPath);
      } finally {
        freeTensors(strictTensors);
      }
      await expect(
        loadLtx2VocoderWeights(strictModel, vocoderComponent([strictPath]), {
          strictUnexpectedWeights: true,
        }),
      ).rejects.toThrow("unexpected unmapped weights");

      using emptyModel = new Ltx2Vocoder(tinyConfig());
      await expect(loadLtx2VocoderWeights(emptyModel, vocoderComponent([]))).rejects.toThrow(
        "no safetensors",
      );

      using invalidIndexModel = new Ltx2Vocoder(tinyConfig());
      const invalidIndex = join(directory, "invalid.safetensors.index.json");
      writeJson(invalidIndex, { metadata: {}, weight_map: { "conv_in.weight": "" } });
      await expect(
        loadLtx2VocoderWeights(invalidIndexModel, vocoderComponent([invalidIndex])),
      ).rejects.toThrow("shard filename");
    });
  });
});
