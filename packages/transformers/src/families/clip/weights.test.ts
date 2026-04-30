import { describe, expect, test } from "bun:test";
import { type MxArray, type ParameterTree, saveSafetensors, treeFlatten, zeros } from "@mlxts/core";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { MissingWeightsError, WeightMismatchError } from "../../types";
import { loadCLIPTextModel, loadCLIPTextModelWithProjection } from "./load";
import { CLIPTextModel, CLIPTextModelWithProjection, disposeCLIPTextModelOutput } from "./model";
import type { CLIPTextConfig } from "./types";
import { clipTextWeightPath, loadCLIPTextWeights } from "./weights";

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

function clipTextConfig(overrides: Partial<CLIPTextConfig> = {}): CLIPTextConfig {
  return {
    modelType: "clip_text_model",
    rawConfig: {},
    vocabSize: 16,
    hiddenSize: 8,
    intermediateSize: 16,
    projectionDim: 6,
    numHiddenLayers: 2,
    numAttentionHeads: 2,
    headDim: 4,
    maxPositionEmbeddings: 8,
    hiddenAct: "quick_gelu",
    layerNormEps: 1e-5,
    attentionDropout: 0,
    padTokenId: 0,
    bosTokenId: 1,
    eosTokenId: 15,
    ...overrides,
  };
}

function writeConfig(directory: string, config: CLIPTextConfig): void {
  writeJson(join(directory, "config.json"), {
    model_type: "clip_text_model",
    vocab_size: config.vocabSize,
    hidden_size: config.hiddenSize,
    intermediate_size: config.intermediateSize,
    projection_dim: config.projectionDim,
    num_hidden_layers: config.numHiddenLayers,
    num_attention_heads: config.numAttentionHeads,
    max_position_embeddings: config.maxPositionEmbeddings,
    hidden_act: config.hiddenAct,
    layer_norm_eps: config.layerNormEps,
    attention_dropout: config.attentionDropout,
    pad_token_id: config.padTokenId,
    bos_token_id: config.bosTokenId,
    eos_token_id: config.eosTokenId,
  });
}

function checkpointNameForPath(path: string): string {
  if (path === "textProjection.weight") {
    return "text_projection.weight";
  }

  const textPath = path.startsWith("textModel.") ? path.slice("textModel.".length) : path;
  return `text_model.${textPath}`
    .replace("text_model.layers.", "text_model.encoder.layers.")
    .replaceAll("tokenEmbedding", "token_embedding")
    .replaceAll("positionEmbedding", "position_embedding")
    .replaceAll("selfAttention", "self_attn")
    .replaceAll("qProj", "q_proj")
    .replaceAll("kProj", "k_proj")
    .replaceAll("vProj", "v_proj")
    .replaceAll("outProj", "out_proj")
    .replaceAll("layerNorm1", "layer_norm1")
    .replaceAll("layerNorm2", "layer_norm2")
    .replaceAll("finalLayerNorm", "final_layer_norm");
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

describe("CLIP text weight mapping", () => {
  test("maps Hugging Face text_model names into the package parameter tree", () => {
    expect(clipTextWeightPath("text_model.embeddings.token_embedding.weight", "text")).toBe(
      "embeddings.tokenEmbedding.weight",
    );
    expect(clipTextWeightPath("text_model.encoder.layers.1.self_attn.q_proj.bias", "text")).toBe(
      "layers.1.selfAttention.qProj.bias",
    );
    expect(clipTextWeightPath("text_model.encoder.layers.0.mlp.fc2.weight", "text")).toBe(
      "layers.0.mlp.fc2.weight",
    );
    expect(clipTextWeightPath("text_model.final_layer_norm.bias", "text")).toBe(
      "finalLayerNorm.bias",
    );
    expect(clipTextWeightPath("text_model.embeddings.position_embedding.weight", "projected")).toBe(
      "textModel.embeddings.positionEmbedding.weight",
    );
    expect(clipTextWeightPath("text_projection.weight", "projected")).toBe("textProjection.weight");
    expect(clipTextWeightPath("text_projection.weight", "text")).toBeNull();
  });

  test("maps unprefixed CLIPTextModel names", () => {
    expect(clipTextWeightPath("embeddings.token_embedding.weight", "text")).toBe(
      "embeddings.tokenEmbedding.weight",
    );
    expect(clipTextWeightPath("encoder.layers.0.layer_norm2.weight", "text")).toBe(
      "layers.0.layerNorm2.weight",
    );
  });
});

describe("loadCLIPTextWeights", () => {
  test("loads complete plain CLIP text weights", async () => {
    await withTempDirectory("mlxts-clip-text-weights-", async (directory) => {
      using source = new CLIPTextModel(clipTextConfig());
      const checkpointPath = join(directory, "model.safetensors");
      await writeCheckpoint(checkpointPath, checkpointTensorsForParameters(source.parameters()));
      using target = new CLIPTextModel(clipTextConfig());
      const snapshot = {
        source: "local" as const,
        directory,
        files: [{ relativePath: "model.safetensors", localPath: checkpointPath, size: 1 }],
        totalBytes: 1,
      };

      const result = await loadCLIPTextWeights(target, snapshot, "text");

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

  test("loads projected CLIP text weights and ignores position_ids", async () => {
    await withTempDirectory("mlxts-clip-projected-weights-", async (directory) => {
      using source = new CLIPTextModelWithProjection(clipTextConfig({ projectionDim: 5 }));
      const checkpointPath = join(directory, "model.safetensors");
      await writeCheckpoint(
        checkpointPath,
        checkpointTensorsForParameters(source.parameters(), {
          extra: { "text_model.embeddings.position_ids": zeros([1, 8], "int32") },
        }),
      );
      using target = new CLIPTextModelWithProjection(clipTextConfig({ projectionDim: 5 }));
      const snapshot = {
        source: "local" as const,
        directory,
        files: [{ relativePath: "model.safetensors", localPath: checkpointPath, size: 1 }],
        totalBytes: 1,
      };

      const result = await loadCLIPTextWeights(target, snapshot, "projected", {
        strictUnexpectedWeights: true,
      });

      expect(result.unexpectedWeights).toEqual([]);
      expect(result.assignedPaths).toContain("textProjection.weight");
    });
  });

  test("rejects missing, mismatched, and strict unexpected weights", async () => {
    await withTempDirectory("mlxts-clip-invalid-weights-", async (directory) => {
      using source = new CLIPTextModel(clipTextConfig());
      const missingPath = join(directory, "missing.safetensors");
      await writeCheckpoint(
        missingPath,
        checkpointTensorsForParameters(source.parameters(), {
          omitPath: "finalLayerNorm.bias",
        }),
      );
      using missingTarget = new CLIPTextModel(clipTextConfig());
      await expect(
        loadCLIPTextWeights(
          missingTarget,
          {
            source: "local",
            directory,
            files: [{ relativePath: "missing.safetensors", localPath: missingPath, size: 1 }],
            totalBytes: 1,
          },
          "text",
        ),
      ).rejects.toThrow(MissingWeightsError);

      const mismatchPath = join(directory, "mismatch.safetensors");
      await writeCheckpoint(
        mismatchPath,
        checkpointTensorsForParameters(source.parameters(), {
          mismatchPath: "layers.0.selfAttention.qProj.weight",
        }),
      );
      using mismatchTarget = new CLIPTextModel(clipTextConfig());
      await expect(
        loadCLIPTextWeights(
          mismatchTarget,
          {
            source: "local",
            directory,
            files: [{ relativePath: "mismatch.safetensors", localPath: mismatchPath, size: 1 }],
            totalBytes: 1,
          },
          "text",
        ),
      ).rejects.toThrow(WeightMismatchError);

      const unexpectedPath = join(directory, "unexpected.safetensors");
      await writeCheckpoint(
        unexpectedPath,
        checkpointTensorsForParameters(source.parameters(), {
          extra: { "vision_model.embeddings.class_embedding": zeros([1]) },
        }),
      );
      using unexpectedTarget = new CLIPTextModel(clipTextConfig());
      await expect(
        loadCLIPTextWeights(
          unexpectedTarget,
          {
            source: "local",
            directory,
            files: [{ relativePath: "unexpected.safetensors", localPath: unexpectedPath, size: 1 }],
            totalBytes: 1,
          },
          "text",
          { strictUnexpectedWeights: true },
        ),
      ).rejects.toThrow("unexpected unmapped weights");
    });
  });
});

describe("CLIP text model loading", () => {
  test("loads a plain CLIP text model from a local snapshot", async () => {
    await withTempDirectory("mlxts-clip-text-load-", async (directory) => {
      const config = clipTextConfig();
      writeConfig(directory, config);
      using source = new CLIPTextModel(config);
      await writeCheckpoint(
        join(directory, "model.safetensors"),
        checkpointTensorsForParameters(source.parameters()),
      );

      using loaded = await loadCLIPTextModel(directory, { strictUnexpectedWeights: true });
      using inputIds = zeros([1, 3], "int32");
      const output = loaded.run(inputIds);
      try {
        expect(output.lastHiddenState.shape).toEqual([1, 3, config.hiddenSize]);
        expect(output.pooledOutput.shape).toEqual([1, config.hiddenSize]);
      } finally {
        disposeCLIPTextModelOutput(output);
      }
    });
  });

  test("loads a projected CLIP text model from a local snapshot", async () => {
    await withTempDirectory("mlxts-clip-projected-load-", async (directory) => {
      const config = clipTextConfig({ projectionDim: 5 });
      writeConfig(directory, config);
      using source = new CLIPTextModelWithProjection(config);
      await writeCheckpoint(
        join(directory, "model.safetensors"),
        checkpointTensorsForParameters(source.parameters()),
      );

      using loaded = await loadCLIPTextModelWithProjection(directory, {
        strictUnexpectedWeights: true,
      });
      using inputIds = zeros([1, 3], "int32");
      using output = loaded.forward(inputIds);
      expect(output.shape).toEqual([1, 5]);
    });
  });

  test("loads Diffusers text_encoder and text_encoder_2 component directories", async () => {
    await withTempDirectory("mlxts-clip-component-load-", async (directory) => {
      const textEncoderDirectory = join(directory, "text_encoder");
      const projectedDirectory = join(directory, "text_encoder_2");
      const plainConfig = clipTextConfig();
      const projectedConfig = clipTextConfig({ projectionDim: 5 });
      writeJson(join(directory, "model_index.json"), {
        _class_name: "StableDiffusionXLPipeline",
        text_encoder: ["transformers", "CLIPTextModel"],
        text_encoder_2: ["transformers", "CLIPTextModelWithProjection"],
      });
      mkdirSync(textEncoderDirectory, { recursive: true });
      mkdirSync(projectedDirectory, { recursive: true });
      writeConfig(textEncoderDirectory, plainConfig);
      writeConfig(projectedDirectory, projectedConfig);

      using plainSource = new CLIPTextModel(plainConfig);
      await writeCheckpoint(
        join(textEncoderDirectory, "model.safetensors"),
        checkpointTensorsForParameters(plainSource.parameters()),
      );
      using projectedSource = new CLIPTextModelWithProjection(projectedConfig);
      await writeCheckpoint(
        join(projectedDirectory, "model.safetensors"),
        checkpointTensorsForParameters(projectedSource.parameters()),
      );

      using plain = await loadCLIPTextModel(textEncoderDirectory, {
        strictUnexpectedWeights: true,
      });
      using projected = await loadCLIPTextModelWithProjection(projectedDirectory, {
        strictUnexpectedWeights: true,
      });

      using plainInputIds = zeros([1, 3], "int32");
      const plainOutput = plain.run(plainInputIds);
      using projectedInputIds = zeros([1, 3], "int32");
      using projectedOutput = projected.forward(projectedInputIds);
      try {
        expect(plainOutput.lastHiddenState.shape).toEqual([1, 3, plainConfig.hiddenSize]);
        expect(projectedOutput.shape).toEqual([1, 5]);
      } finally {
        disposeCLIPTextModelOutput(plainOutput);
      }
    });
  });
});
