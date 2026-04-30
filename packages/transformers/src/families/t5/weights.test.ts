import { describe, expect, test } from "bun:test";
import { type MxArray, type ParameterTree, saveSafetensors, treeFlatten, zeros } from "@mlxts/core";
import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { MissingWeightsError, WeightMismatchError } from "../../types";
import { loadT5EncoderModel } from "./load";
import { disposeT5EncoderModelOutput, T5EncoderModel } from "./model";
import type { T5EncoderConfig } from "./types";
import { loadT5EncoderWeights, t5EncoderWeightPath } from "./weights";

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

function t5EncoderConfig(overrides: Partial<T5EncoderConfig> = {}): T5EncoderConfig {
  return {
    modelType: "t5_encoder_model",
    rawConfig: {},
    vocabSize: 32,
    dModel: 8,
    dKv: 4,
    dFf: 16,
    numLayers: 2,
    numHeads: 2,
    innerDim: 8,
    relativeAttentionNumBuckets: 8,
    relativeAttentionMaxDistance: 16,
    layerNormEps: 1e-6,
    dropoutRate: 0,
    feedForwardProjection: "gated-gelu",
    denseActivation: "gelu_new",
    isGatedActivation: true,
    padTokenId: 0,
    eosTokenId: 1,
    ...overrides,
  };
}

function writeConfig(directory: string, config: T5EncoderConfig): void {
  writeJson(join(directory, "config.json"), {
    model_type: "t5",
    vocab_size: config.vocabSize,
    d_model: config.dModel,
    d_kv: config.dKv,
    d_ff: config.dFf,
    num_layers: config.numLayers,
    num_heads: config.numHeads,
    relative_attention_num_buckets: config.relativeAttentionNumBuckets,
    relative_attention_max_distance: config.relativeAttentionMaxDistance,
    layer_norm_epsilon: config.layerNormEps,
    dropout_rate: config.dropoutRate,
    feed_forward_proj: config.feedForwardProjection,
    is_encoder_decoder: true,
    is_decoder: false,
    pad_token_id: config.padTokenId,
    eos_token_id: config.eosTokenId,
  });
}

function checkpointNameForPath(path: string): string {
  if (path === "tokenEmbedding.weight") {
    return "shared.weight";
  }

  const selfAttention = path.match(/^layers\.(\d+)\.selfAttention\.attention\.(q|k|v|o)\.weight$/);
  if (selfAttention !== null) {
    return `encoder.block.${selfAttention[1]}.layer.0.SelfAttention.${selfAttention[2]}.weight`;
  }

  const relativeBias = path.match(
    /^layers\.(\d+)\.selfAttention\.attention\.relativeAttentionBias\.weight$/,
  );
  if (relativeBias !== null) {
    return `encoder.block.${relativeBias[1]}.layer.0.SelfAttention.relative_attention_bias.weight`;
  }

  const selfNorm = path.match(/^layers\.(\d+)\.selfAttention\.layerNorm\.weight$/);
  if (selfNorm !== null) {
    return `encoder.block.${selfNorm[1]}.layer.0.layer_norm.weight`;
  }

  const dense = path.match(/^layers\.(\d+)\.feedForward\.dense\.(wi|wi0|wi1|wo)\.weight$/);
  if (dense !== null) {
    const projection = dense[2] === "wi0" ? "wi_0" : dense[2] === "wi1" ? "wi_1" : dense[2];
    return `encoder.block.${dense[1]}.layer.1.DenseReluDense.${projection}.weight`;
  }

  const ffNorm = path.match(/^layers\.(\d+)\.feedForward\.layerNorm\.weight$/);
  if (ffNorm !== null) {
    return `encoder.block.${ffNorm[1]}.layer.1.layer_norm.weight`;
  }

  if (path === "finalLayerNorm.weight") {
    return "encoder.final_layer_norm.weight";
  }

  throw new Error(`No checkpoint mapping for ${path}`);
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
    const shape = internalPath === options.mismatchPath ? [1] : [...parameter.shape];
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

describe("T5 encoder weight mapping", () => {
  test("maps Hugging Face encoder names into the package parameter tree", () => {
    expect(t5EncoderWeightPath("shared.weight")).toBe("tokenEmbedding.weight");
    expect(t5EncoderWeightPath("encoder.embed_tokens.weight")).toBe("tokenEmbedding.weight");
    expect(t5EncoderWeightPath("encoder.block.1.layer.0.SelfAttention.q.weight")).toBe(
      "layers.1.selfAttention.attention.q.weight",
    );
    expect(
      t5EncoderWeightPath("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"),
    ).toBe("layers.0.selfAttention.attention.relativeAttentionBias.weight");
    expect(t5EncoderWeightPath("encoder.block.1.layer.1.DenseReluDense.wi_0.weight")).toBe(
      "layers.1.feedForward.dense.wi0.weight",
    );
    expect(t5EncoderWeightPath("encoder.final_layer_norm.weight")).toBe("finalLayerNorm.weight");
  });
});

describe("loadT5EncoderWeights", () => {
  test("loads complete gated T5 encoder weights and ignores decoder-only tensors", async () => {
    await withTempDirectory("mlxts-t5-weights-", async (directory) => {
      using source = new T5EncoderModel(t5EncoderConfig());
      const checkpointPath = join(directory, "model.safetensors");
      await writeCheckpoint(
        checkpointPath,
        checkpointTensorsForParameters(source.parameters(), {
          extra: {
            "decoder.block.0.layer.0.SelfAttention.q.weight": zeros([8, 8]),
            "lm_head.weight": zeros([32, 8]),
          },
        }),
      );
      using target = new T5EncoderModel(t5EncoderConfig());
      const snapshot = {
        source: "local" as const,
        directory,
        files: [{ relativePath: "model.safetensors", localPath: checkpointPath, size: 1 }],
        totalBytes: 1,
      };

      const result = await loadT5EncoderWeights(target, snapshot, {
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

  test("loads complete non-gated T5 encoder weights", async () => {
    await withTempDirectory("mlxts-t5-nongated-weights-", async (directory) => {
      const config = t5EncoderConfig({
        feedForwardProjection: "relu",
        denseActivation: "relu",
        isGatedActivation: false,
      });
      using source = new T5EncoderModel(config);
      const checkpointPath = join(directory, "model.safetensors");
      await writeCheckpoint(checkpointPath, checkpointTensorsForParameters(source.parameters()));
      using target = new T5EncoderModel(config);
      const snapshot = {
        source: "local" as const,
        directory,
        files: [{ relativePath: "model.safetensors", localPath: checkpointPath, size: 1 }],
        totalBytes: 1,
      };

      const result = await loadT5EncoderWeights(target, snapshot);

      expect(result.assignedPaths).toContain("layers.0.feedForward.dense.wi.weight");
    });
  });

  test("rejects missing, mismatched, and strict unexpected weights", async () => {
    await withTempDirectory("mlxts-t5-invalid-weights-", async (directory) => {
      using source = new T5EncoderModel(t5EncoderConfig());
      const missingPath = join(directory, "missing.safetensors");
      await writeCheckpoint(
        missingPath,
        checkpointTensorsForParameters(source.parameters(), {
          omitPath: "finalLayerNorm.weight",
        }),
      );
      using missingTarget = new T5EncoderModel(t5EncoderConfig());
      await expect(
        loadT5EncoderWeights(missingTarget, {
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
          mismatchPath: "layers.0.selfAttention.attention.q.weight",
        }),
      );
      using mismatchTarget = new T5EncoderModel(t5EncoderConfig());
      await expect(
        loadT5EncoderWeights(mismatchTarget, {
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
      using unexpectedTarget = new T5EncoderModel(t5EncoderConfig());
      await expect(
        loadT5EncoderWeights(
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

describe("T5 encoder model loading", () => {
  test("loads a T5EncoderModel from a local snapshot", async () => {
    await withTempDirectory("mlxts-t5-load-", async (directory) => {
      const config = t5EncoderConfig();
      writeConfig(directory, config);
      using source = new T5EncoderModel(config);
      await writeCheckpoint(
        join(directory, "model.safetensors"),
        checkpointTensorsForParameters(source.parameters()),
      );

      using loaded = await loadT5EncoderModel(directory, { strictUnexpectedWeights: true });
      using inputIds = zeros([1, 3], "int32");
      const output = loaded.run(inputIds);
      try {
        expect(output.lastHiddenState.shape).toEqual([1, 3, config.dModel]);
      } finally {
        disposeT5EncoderModelOutput(output);
      }
    });
  });
});
