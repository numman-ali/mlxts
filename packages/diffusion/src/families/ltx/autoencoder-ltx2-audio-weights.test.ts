import { describe, expect, test } from "bun:test";
import { MxArray, mxEval, saveSafetensors, treeFlatten, zeros } from "@mlxts/core";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import {
  type DiffusionSnapshotComponent,
  loadDiffusionSnapshotManifest,
} from "../../pretrained/snapshot-manifest";
import { Ltx2AudioAutoencoderKL } from "./autoencoder-ltx2-audio";
import {
  loadLtx2AudioAutoencoderFromSnapshot,
  loadLtx2AudioAutoencoderWeights,
  ltx2AudioAutoencoderWeightPath,
  transformLtx2AudioAutoencoderWeight,
} from "./autoencoder-ltx2-audio-weights";
import type { Ltx2AudioAutoencoderConfig } from "./config";

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

function tinyConfig(
  overrides: Partial<Ltx2AudioAutoencoderConfig> = {},
): Ltx2AudioAutoencoderConfig {
  return {
    baseChannels: 1,
    outputChannels: 1,
    chMult: [1],
    numResBlocks: 0,
    attnResolutions: null,
    inChannels: 1,
    resolution: 4,
    latentChannels: 1,
    normType: "pixel",
    causalityAxis: "height",
    dropout: 0,
    midBlockAddAttention: false,
    sampleRate: 16000,
    melHopLength: 160,
    isCausal: true,
    melBins: 4,
    melCompressionRatio: 4,
    temporalCompressionRatio: 4,
    packedFeatureSize: 1,
    doubleZ: true,
    rawConfig: {},
    ...overrides,
  };
}

function audioVaeComponent(weightPaths: readonly string[]): DiffusionSnapshotComponent {
  return {
    name: "audio_vae",
    role: "audio-vae",
    library: "diffusers",
    className: "AutoencoderKLLTX2Audio",
    enabled: true,
    optional: false,
    subfolder: "audio_vae",
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
  writeJson(join(directory, "config.json"), config);
  if (needsWeights) {
    writeFileSync(join(directory, "diffusion_pytorch_model.safetensors"), "");
  }
}

function freeTensors(tensors: Record<string, MxArray>): void {
  for (const tensor of Object.values(tensors)) {
    tensor.free();
  }
}

function checkpointNameForAudioVaePath(path: string): string {
  return path
    .replaceAll("convIn", "conv_in")
    .replaceAll("convOut", "conv_out")
    .replaceAll("normOut", "norm_out")
    .replaceAll("block1", "block_1")
    .replaceAll("block2", "block_2")
    .replaceAll("attn1", "attn_1")
    .replaceAll("projOut", "proj_out")
    .replaceAll("ninShortcut", "nin_shortcut");
}

function checkpointTensorForPath(path: string, tensor: MxArray): MxArray {
  if (path.endsWith(".weight") && tensor.shape.length === 4) {
    const [outChannels, kernelHeight, kernelWidth, inputChannels] = tensor.shape;
    return zeros([outChannels ?? 0, inputChannels ?? 0, kernelHeight ?? 0, kernelWidth ?? 0]);
  }
  return zeros([...tensor.shape]);
}

function checkpointTensorsForAudioVae(model: Ltx2AudioAutoencoderKL): Record<string, MxArray> {
  const output: Record<string, MxArray> = {};
  for (const [path, tensor] of treeFlatten(model.parameters())) {
    const internalPath = path.join(".");
    output[checkpointNameForAudioVaePath(internalPath)] = checkpointTensorForPath(
      internalPath,
      tensor,
    );
  }
  output.latents_mean = MxArray.fromData([2], [1]);
  output.latents_std = MxArray.fromData([3], [1]);
  return output;
}

async function writeAudioVaeWeights(
  path: string,
  config: Ltx2AudioAutoencoderConfig = tinyConfig(),
): Promise<void> {
  using model = new Ltx2AudioAutoencoderKL(config);
  const tensors = checkpointTensorsForAudioVae(model);
  try {
    await saveSafetensors(tensors, path);
  } finally {
    freeTensors(tensors);
  }
}

async function writeLtx2AudioSnapshot(snapshot: string): Promise<void> {
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
    audio_in_channels: 1,
    audio_out_channels: 1,
  });
  writeComponent(snapshot, "vae", { _class_name: "AutoencoderKLLTX2Video" });
  writeComponent(snapshot, "audio_vae", {
    _class_name: "AutoencoderKLLTX2Audio",
    attn_resolutions: null,
    base_channels: 1,
    causality_axis: "height",
    ch_mult: [1],
    double_z: true,
    dropout: 0,
    in_channels: 1,
    is_causal: true,
    latent_channels: 1,
    mel_bins: 4,
    mel_hop_length: 160,
    mid_block_add_attention: false,
    norm_type: "pixel",
    num_res_blocks: 1,
    output_channels: 1,
    resolution: 4,
    sample_rate: 16000,
  });
  writeComponent(snapshot, "connectors", { _class_name: "LTX2TextConnectors" });
  writeComponent(snapshot, "vocoder", {
    _class_name: "LTX2Vocoder",
    in_channels: 4,
  });
  writeComponent(snapshot, "text_encoder", {});
  const scheduler = join(snapshot, "scheduler");
  mkdirSync(scheduler, { recursive: true });
  writeJson(join(scheduler, "scheduler_config.json"), {
    _class_name: "FlowMatchEulerDiscreteScheduler",
  });
  const tokenizer = join(snapshot, "tokenizer");
  mkdirSync(tokenizer, { recursive: true });
  writeFileSync(join(tokenizer, "tokenizer.json"), "{}");
  await writeAudioVaeWeights(
    join(snapshot, "audio_vae", "diffusion_pytorch_model.safetensors"),
    tinyConfig({ numResBlocks: 1 }),
  );
}

function expectCloseList(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    expect(actual[index]).toBeCloseTo(expected[index] ?? Number.NaN, 6);
  }
}

describe("LTX-2 audio VAE weight loading", () => {
  test("maps decoder checkpoint paths and packed latent stats", () => {
    expect(ltx2AudioAutoencoderWeightPath("latents_mean")).toBe("latents_mean");
    expect(ltx2AudioAutoencoderWeightPath("latents_std")).toBe("latents_std");
    expect(ltx2AudioAutoencoderWeightPath("decoder.conv_in.conv.weight")).toBe(
      "decoder.convIn.conv.weight",
    );
    expect(ltx2AudioAutoencoderWeightPath("decoder.mid.block_1.conv1.conv.bias")).toBe(
      "decoder.mid.block1.conv1.conv.bias",
    );
    expect(ltx2AudioAutoencoderWeightPath("decoder.mid.attn_1.proj_out.weight")).toBe(
      "decoder.mid.attn1.projOut.weight",
    );
    expect(ltx2AudioAutoencoderWeightPath("decoder.up.0.block.0.nin_shortcut.conv.bias")).toBe(
      "decoder.up.0.block.0.ninShortcut.conv.bias",
    );
    expect(ltx2AudioAutoencoderWeightPath("encoder.conv_in.conv.weight")).toBeNull();
    expect(ltx2AudioAutoencoderWeightPath("")).toBeNull();
    expect(ltx2AudioAutoencoderWeightPath("decoder.norm_out.num_batches_tracked")).toBeNull();
  });

  test("transposes Diffusers Conv2d kernels into MLX channel-last layout", () => {
    using checkpoint = MxArray.fromData([1, 2, 3, 4], [1, 2, 2, 1]);
    using transformed = transformLtx2AudioAutoencoderWeight(
      "decoder.convIn.conv.weight",
      checkpoint,
    );

    mxEval(transformed);
    expect(transformed.shape).toEqual([1, 2, 1, 2]);
    expectCloseList(transformed.toTypedArray(), [1, 3, 2, 4]);
  });

  test("loads decoder-only weights and packed latent stats", async () => {
    await withTempDirectory("ltx2-audio-vae-", async (directory) => {
      using model = new Ltx2AudioAutoencoderKL(tinyConfig());
      const tensors = checkpointTensorsForAudioVae(model);
      const checkpointPath = join(directory, "diffusion_pytorch_model.safetensors");
      try {
        await saveSafetensors(tensors, checkpointPath);
      } finally {
        freeTensors(tensors);
      }

      const result = await loadLtx2AudioAutoencoderWeights(
        model,
        audioVaeComponent([checkpointPath]),
      );

      expect(result.shardCount).toBe(1);
      expect(result.unexpectedWeights).toEqual([]);
      expect(result.assignedPaths).toContain("decoder.convIn.conv.weight");
      expect(result.assignedPaths).toContain("latents_mean");
      expect(result.assignedPaths).toContain("latents_std");
      expect(model.latentsMean).toEqual([2]);
      expect(model.latentsStd).toEqual([3]);
    });
  });

  test("loads decoder weights through a safetensors index", async () => {
    await withTempDirectory("ltx2-audio-vae-index-", async (directory) => {
      using model = new Ltx2AudioAutoencoderKL(tinyConfig());
      const tensors = checkpointTensorsForAudioVae(model);
      const shardName = "audio-00001-of-00001.safetensors";
      const checkpointPath = join(directory, shardName);
      const weightMap = Object.fromEntries(Object.keys(tensors).map((name) => [name, shardName]));
      try {
        await saveSafetensors(tensors, checkpointPath);
      } finally {
        freeTensors(tensors);
      }
      const indexPath = join(directory, "diffusion_pytorch_model.safetensors.index.json");
      writeJson(indexPath, { metadata: {}, weight_map: weightMap });

      const result = await loadLtx2AudioAutoencoderWeights(model, audioVaeComponent([indexPath]));

      expect(result.shardCount).toBe(1);
      expect(result.assignedPaths).toContain("decoder.convIn.conv.weight");
      expect(model.latentsMean).toEqual([2]);
    });
  });

  test("loads a decoder directly from an inspected LTX-2 snapshot manifest", async () => {
    await withTempDirectory("ltx2-audio-vae-snapshot-", async (directory) => {
      await writeLtx2AudioSnapshot(directory);
      const manifest = await loadDiffusionSnapshotManifest(directory);

      using model = await loadLtx2AudioAutoencoderFromSnapshot(manifest);

      expect(model.latentsMean).toEqual([2]);
      expect(model.latentsStd).toEqual([3]);
      expect(model.latentStatSize).toBe(1);
    });
  });

  test("rejects missing packed latent stats and strict unexpected weights", async () => {
    await withTempDirectory("ltx2-audio-vae-errors-", async (directory) => {
      using model = new Ltx2AudioAutoencoderKL(tinyConfig());
      const tensors = checkpointTensorsForAudioVae(model);
      const latentMean = tensors.latents_mean;
      if (latentMean === undefined) {
        throw new Error("expected latents_mean test tensor");
      }
      latentMean.free();
      delete tensors.latents_mean;
      const missingStatsPath = join(directory, "missing-stats.safetensors");
      try {
        await saveSafetensors(tensors, missingStatsPath);
      } finally {
        freeTensors(tensors);
      }
      await expect(
        loadLtx2AudioAutoencoderWeights(model, audioVaeComponent([missingStatsPath])),
      ).rejects.toThrow("latents_mean");

      using strictModel = new Ltx2AudioAutoencoderKL(tinyConfig());
      const strictTensors = checkpointTensorsForAudioVae(strictModel);
      strictTensors["encoder.conv_in.conv.weight"] = zeros([2, 1, 3, 3]);
      const strictPath = join(directory, "strict.safetensors");
      try {
        await saveSafetensors(strictTensors, strictPath);
      } finally {
        freeTensors(strictTensors);
      }
      await expect(
        loadLtx2AudioAutoencoderWeights(strictModel, audioVaeComponent([strictPath]), {
          strictUnexpectedWeights: true,
        }),
      ).rejects.toThrow("unexpected unmapped weights");

      using malformedStatsModel = new Ltx2AudioAutoencoderKL(tinyConfig());
      const malformedStatsTensors = checkpointTensorsForAudioVae(malformedStatsModel);
      malformedStatsTensors.latents_std?.free();
      malformedStatsTensors.latents_std = zeros([2]);
      const malformedStatsPath = join(directory, "malformed-stats.safetensors");
      try {
        await saveSafetensors(malformedStatsTensors, malformedStatsPath);
      } finally {
        freeTensors(malformedStatsTensors);
      }
      await expect(
        loadLtx2AudioAutoencoderWeights(
          malformedStatsModel,
          audioVaeComponent([malformedStatsPath]),
        ),
      ).rejects.toThrow("latents_std");
    });
  });

  test("rejects invalid shard metadata before loading audio tensors", async () => {
    await withTempDirectory("ltx2-audio-vae-shard-errors-", async (directory) => {
      using emptyModel = new Ltx2AudioAutoencoderKL(tinyConfig());
      await expect(
        loadLtx2AudioAutoencoderWeights(emptyModel, audioVaeComponent([])),
      ).rejects.toThrow("no safetensors");

      using duplicateIndexModel = new Ltx2AudioAutoencoderKL(tinyConfig());
      const firstIndex = join(directory, "first.safetensors.index.json");
      const secondIndex = join(directory, "second.safetensors.index.json");
      writeJson(firstIndex, { metadata: {}, weight_map: {} });
      writeJson(secondIndex, { metadata: {}, weight_map: {} });
      await expect(
        loadLtx2AudioAutoencoderWeights(
          duplicateIndexModel,
          audioVaeComponent([firstIndex, secondIndex]),
        ),
      ).rejects.toThrow("multiple safetensors index files");

      using invalidJsonModel = new Ltx2AudioAutoencoderKL(tinyConfig());
      const invalidJsonIndex = join(directory, "invalid.safetensors.index.json");
      writeFileSync(invalidJsonIndex, "{");
      await expect(
        loadLtx2AudioAutoencoderWeights(invalidJsonModel, audioVaeComponent([invalidJsonIndex])),
      ).rejects.toThrow("valid JSON");

      using invalidWeightMapModel = new Ltx2AudioAutoencoderKL(tinyConfig());
      const invalidWeightMapIndex = join(directory, "invalid-weight-map.safetensors.index.json");
      writeJson(invalidWeightMapIndex, {
        metadata: {},
        weight_map: { latents_mean: "" },
      });
      await expect(
        loadLtx2AudioAutoencoderWeights(
          invalidWeightMapModel,
          audioVaeComponent([invalidWeightMapIndex]),
        ),
      ).rejects.toThrow("shard filename");
    });
  });
});
