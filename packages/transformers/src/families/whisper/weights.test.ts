import { describe, expect, test } from "bun:test";
import { type MxArray, type ParameterTree, saveSafetensors, treeFlatten, zeros } from "@mlxts/core";
import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { MissingWeightsError, WeightMismatchError } from "../../types";
import { loadWhisperModel } from "./load";
import { WhisperForConditionalGeneration } from "./model";
import type { WhisperConfig } from "./types";
import { loadWhisperWeights, transformWhisperWeight, whisperWeightPath } from "./weights";

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

function whisperConfig(overrides: Partial<WhisperConfig> = {}): WhisperConfig {
  return {
    modelType: "whisper",
    rawConfig: {},
    vocabSize: 32,
    numMelBins: 4,
    encoderLayers: 1,
    encoderAttentionHeads: 2,
    decoderLayers: 1,
    decoderAttentionHeads: 2,
    encoderFfnDim: 16,
    decoderFfnDim: 16,
    dModel: 8,
    encoderHeadDim: 4,
    decoderHeadDim: 4,
    activationFunction: "gelu",
    maxSourcePositions: 4,
    maxTargetPositions: 6,
    padTokenId: 0,
    bosTokenId: 1,
    eosTokenId: 2,
    decoderStartTokenId: 1,
    scaleEmbedding: false,
    useCache: true,
    ...overrides,
  };
}

function writeConfig(directory: string, config: WhisperConfig): void {
  writeJson(join(directory, "config.json"), {
    model_type: "whisper",
    is_encoder_decoder: true,
    vocab_size: config.vocabSize,
    num_mel_bins: config.numMelBins,
    encoder_layers: config.encoderLayers,
    encoder_attention_heads: config.encoderAttentionHeads,
    decoder_layers: config.decoderLayers,
    decoder_attention_heads: config.decoderAttentionHeads,
    encoder_ffn_dim: config.encoderFfnDim,
    decoder_ffn_dim: config.decoderFfnDim,
    d_model: config.dModel,
    max_source_positions: config.maxSourcePositions,
    max_target_positions: config.maxTargetPositions,
    pad_token_id: config.padTokenId,
    bos_token_id: config.bosTokenId,
    eos_token_id: config.eosTokenId,
    decoder_start_token_id: config.decoderStartTokenId,
  });
}

function encoderCheckpointNameForPath(path: string): string | null {
  const encoderAttention = path.match(
    /^model\.encoder\.layers\.(\d+)\.selfAttention\.(q|k|v)Proj\.(weight|bias)$/,
  );
  if (encoderAttention !== null) {
    return `model.encoder.layers.${encoderAttention[1]}.self_attn.${encoderAttention[2]}_proj.${encoderAttention[3]}`;
  }

  const encoderOut = path.match(
    /^model\.encoder\.layers\.(\d+)\.selfAttention\.outProj\.(weight|bias)$/,
  );
  if (encoderOut !== null) {
    return `model.encoder.layers.${encoderOut[1]}.self_attn.out_proj.${encoderOut[2]}`;
  }

  const encoderNorm = path.match(
    /^model\.encoder\.layers\.(\d+)\.(selfAttentionLayerNorm|finalLayerNorm)\.(weight|bias)$/,
  );
  if (encoderNorm !== null) {
    const layerNorm =
      encoderNorm[2] === "selfAttentionLayerNorm" ? "self_attn_layer_norm" : "final_layer_norm";
    return `model.encoder.layers.${encoderNorm[1]}.${layerNorm}.${encoderNorm[3]}`;
  }

  const encoderMlp = path.match(/^model\.encoder\.layers\.(\d+)\.mlp\.(fc1|fc2)\.(weight|bias)$/);
  if (encoderMlp !== null) {
    return `model.encoder.layers.${encoderMlp[1]}.${encoderMlp[2]}.${encoderMlp[3]}`;
  }

  return null;
}

function decoderCheckpointNameForPath(path: string): string | null {
  const decoderAttention = path.match(
    /^model\.decoder\.layers\.(\d+)\.(selfAttention|crossAttention)\.(q|k|v)Proj\.(weight|bias)$/,
  );
  if (decoderAttention !== null) {
    const attentionName = decoderAttention[2] === "selfAttention" ? "self_attn" : "encoder_attn";
    return `model.decoder.layers.${decoderAttention[1]}.${attentionName}.${decoderAttention[3]}_proj.${decoderAttention[4]}`;
  }

  const decoderOut = path.match(
    /^model\.decoder\.layers\.(\d+)\.(selfAttention|crossAttention)\.outProj\.(weight|bias)$/,
  );
  if (decoderOut !== null) {
    const attentionName = decoderOut[2] === "selfAttention" ? "self_attn" : "encoder_attn";
    return `model.decoder.layers.${decoderOut[1]}.${attentionName}.out_proj.${decoderOut[3]}`;
  }

  const decoderNorm = path.match(
    /^model\.decoder\.layers\.(\d+)\.(selfAttentionLayerNorm|crossAttentionLayerNorm|finalLayerNorm)\.(weight|bias)$/,
  );
  if (decoderNorm !== null) {
    const layerNorm =
      decoderNorm[2] === "selfAttentionLayerNorm"
        ? "self_attn_layer_norm"
        : decoderNorm[2] === "crossAttentionLayerNorm"
          ? "encoder_attn_layer_norm"
          : "final_layer_norm";
    return `model.decoder.layers.${decoderNorm[1]}.${layerNorm}.${decoderNorm[3]}`;
  }

  const decoderMlp = path.match(/^model\.decoder\.layers\.(\d+)\.mlp\.(fc1|fc2)\.(weight|bias)$/);
  if (decoderMlp !== null) {
    return `model.decoder.layers.${decoderMlp[1]}.${decoderMlp[2]}.${decoderMlp[3]}`;
  }

  return null;
}

function checkpointNameForPath(path: string): string {
  const rootMapping: Record<string, string> = {
    "model.encoder.conv1.weight": "model.encoder.conv1.weight",
    "model.encoder.conv1.bias": "model.encoder.conv1.bias",
    "model.encoder.conv2.weight": "model.encoder.conv2.weight",
    "model.encoder.conv2.bias": "model.encoder.conv2.bias",
    "model.encoder.positionEmbedding.weight": "model.encoder.embed_positions.weight",
    "model.encoder.layerNorm.weight": "model.encoder.layer_norm.weight",
    "model.encoder.layerNorm.bias": "model.encoder.layer_norm.bias",
    "model.decoder.tokenEmbedding.weight": "model.decoder.embed_tokens.weight",
    "model.decoder.positionEmbedding.weight": "model.decoder.embed_positions.weight",
    "model.decoder.layerNorm.weight": "model.decoder.layer_norm.weight",
    "model.decoder.layerNorm.bias": "model.decoder.layer_norm.bias",
  };
  const checkpointName =
    rootMapping[path] ?? encoderCheckpointNameForPath(path) ?? decoderCheckpointNameForPath(path);
  if (checkpointName !== null) {
    return checkpointName;
  }
  throw new Error(`No checkpoint mapping for ${path}`);
}

function checkpointShapeForPath(path: string, parameter: MxArray): number[] {
  if (path === "model.encoder.conv1.weight" || path === "model.encoder.conv2.weight") {
    const [outputChannels, kernelSize, inputChannels] = parameter.shape;
    if (outputChannels === undefined || inputChannels === undefined || kernelSize === undefined) {
      throw new Error(`Malformed Conv1d parameter shape for ${path}.`);
    }
    return [outputChannels, inputChannels, kernelSize];
  }
  return [...parameter.shape];
}

function checkpointTensorsForParameters(
  parameters: ParameterTree,
  options: {
    omitPath?: string;
    extra?: Record<string, MxArray>;
    mismatchPath?: string;
  } = {},
): Record<string, MxArray> {
  const tensors: Record<string, MxArray> = {};
  for (const [path, parameter] of treeFlatten(parameters)) {
    const internalPath = path.join(".");
    if (internalPath === options.omitPath) {
      continue;
    }
    const shape =
      internalPath === options.mismatchPath ? [1] : checkpointShapeForPath(internalPath, parameter);
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

describe("Whisper weight mapping", () => {
  test("maps Hugging Face Whisper names into the package parameter tree", () => {
    expect(whisperWeightPath("model.encoder.conv1.weight")).toBe("model.encoder.conv1.weight");
    expect(whisperWeightPath("model.encoder.embed_positions.weight")).toBe(
      "model.encoder.positionEmbedding.weight",
    );
    expect(whisperWeightPath("model.encoder.layers.0.self_attn.q_proj.weight")).toBe(
      "model.encoder.layers.0.selfAttention.qProj.weight",
    );
    expect(whisperWeightPath("model.decoder.layers.0.encoder_attn.k_proj.weight")).toBe(
      "model.decoder.layers.0.crossAttention.kProj.weight",
    );
    expect(whisperWeightPath("model.decoder.layers.0.encoder_attn_layer_norm.bias")).toBe(
      "model.decoder.layers.0.crossAttentionLayerNorm.bias",
    );
    expect(whisperWeightPath("proj_out.weight")).toBeNull();
  });

  test("transposes raw Hugging Face Conv1d kernels into MLX channel-last layout", () => {
    using rawWeight = zeros([8, 4, 3], "float32");
    using transformed = transformWhisperWeight("model.encoder.conv1.weight", rawWeight);
    expect(transformed.shape).toEqual([8, 3, 4]);
  });
});

describe("loadWhisperWeights", () => {
  test("loads complete Whisper weights and ignores tied projection weights", async () => {
    await withTempDirectory("mlxts-whisper-weights-", async (directory) => {
      using source = new WhisperForConditionalGeneration(whisperConfig());
      const checkpointPath = join(directory, "model.safetensors");
      await writeCheckpoint(
        checkpointPath,
        checkpointTensorsForParameters(source.parameters(), {
          extra: {
            "proj_out.weight": zeros([32, 8]),
          },
        }),
      );
      using target = new WhisperForConditionalGeneration(whisperConfig());
      const snapshot = {
        source: "local" as const,
        directory,
        files: [{ relativePath: "model.safetensors", localPath: checkpointPath, size: 1 }],
        totalBytes: 1,
      };

      const result = await loadWhisperWeights(target, snapshot, {
        strictUnexpectedWeights: true,
      });

      expect(target.isTraining).toBe(false);
      expect(result.shardCount).toBe(1);
      expect(result.unexpectedWeights).toEqual([]);
      expect(result.assignedPaths).toEqual(
        treeFlatten(target.parameters())
          .map(([path]) => path.join("."))
          .toSorted((left, right) => left.localeCompare(right)),
      );
    });
  });

  test("rejects missing, mismatched, and strict unexpected weights", async () => {
    await withTempDirectory("mlxts-whisper-invalid-weights-", async (directory) => {
      using source = new WhisperForConditionalGeneration(whisperConfig());
      const missingPath = join(directory, "missing.safetensors");
      await writeCheckpoint(
        missingPath,
        checkpointTensorsForParameters(source.parameters(), {
          omitPath: "model.decoder.layerNorm.weight",
        }),
      );
      using missingTarget = new WhisperForConditionalGeneration(whisperConfig());
      await expect(
        loadWhisperWeights(missingTarget, {
          source: "local",
          directory,
          files: [{ relativePath: "missing.safetensors", localPath: missingPath, size: 1 }],
          totalBytes: 1,
        }),
      ).rejects.toThrow(MissingWeightsError);

      const mismatchPath = join(directory, "mismatch.safetensors");
      await writeCheckpoint(
        mismatchPath,
        checkpointTensorsForParameters(source.parameters(), {
          mismatchPath: "model.encoder.layers.0.selfAttention.qProj.weight",
        }),
      );
      using mismatchTarget = new WhisperForConditionalGeneration(whisperConfig());
      await expect(
        loadWhisperWeights(mismatchTarget, {
          source: "local",
          directory,
          files: [{ relativePath: "mismatch.safetensors", localPath: mismatchPath, size: 1 }],
          totalBytes: 1,
        }),
      ).rejects.toThrow(WeightMismatchError);

      const unexpectedPath = join(directory, "unexpected.safetensors");
      await writeCheckpoint(
        unexpectedPath,
        checkpointTensorsForParameters(source.parameters(), {
          extra: { "vision_model.embeddings.class_embedding": zeros([1]) },
        }),
      );
      using unexpectedTarget = new WhisperForConditionalGeneration(whisperConfig());
      await expect(
        loadWhisperWeights(
          unexpectedTarget,
          {
            source: "local",
            directory,
            files: [{ relativePath: "unexpected.safetensors", localPath: unexpectedPath, size: 1 }],
            totalBytes: 1,
          },
          { strictUnexpectedWeights: true },
        ),
      ).rejects.toThrow("unexpected unmapped weights");
    });
  });
});

describe("Whisper model loading", () => {
  test("loads a Whisper model from a local snapshot", async () => {
    await withTempDirectory("mlxts-whisper-load-", async (directory) => {
      const config = whisperConfig();
      writeConfig(directory, config);
      using source = new WhisperForConditionalGeneration(config);
      await writeCheckpoint(
        join(directory, "model.safetensors"),
        checkpointTensorsForParameters(source.parameters()),
      );

      using loaded = await loadWhisperModel(directory, { strictUnexpectedWeights: true });
      using inputFeatures = zeros([1, 8, 4], "float32");
      using decoderInputIds = zeros([1, 2], "int32");
      const logits = loaded.forward(inputFeatures, decoderInputIds);
      try {
        expect(logits.shape).toEqual([1, 2, config.vocabSize]);
      } finally {
        logits.free();
      }
    });
  });
});
