import { describe, expect, test } from "bun:test";

import { MxArray, saveSafetensors } from "@mlxts/core";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import type { Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig } from "./types";
import {
  detectQwen3_5CheckpointStyle,
  sanitizeQwen3_5TextWeight,
  sanitizeQwen3_5Weight,
  transformQwen3_5CheckpointTensor,
} from "./weights";

function qwen3_5TextConfig(overrides: Partial<Qwen3_5TextConfig> = {}): Qwen3_5TextConfig {
  return {
    family: "qwen",
    modelType: "qwen3_5_text",
    rawConfig: {},
    vocabSize: 32,
    hiddenSize: 8,
    intermediateSize: 16,
    numHiddenLayers: 2,
    numAttentionHeads: 2,
    numKeyValueHeads: 1,
    headDim: 4,
    hiddenAct: "silu",
    maxPositionEmbeddings: 128,
    initializerRange: 0.02,
    rmsNormEps: 1e-6,
    useCache: true,
    tieWordEmbeddings: false,
    attentionBias: false,
    attentionDropout: 0,
    attnOutputGate: true,
    outputGateType: null,
    linearConvKernelDim: 2,
    linearKeyHeadDim: 2,
    linearValueHeadDim: 2,
    linearNumKeyHeads: 1,
    linearNumValueHeads: 2,
    layerTypes: ["linear_attention", "full_attention"],
    fullAttentionInterval: 2,
    ropeParameters: {
      ropeType: "default",
      ropeTheta: 10000,
      partialRotaryFactor: 1,
      mropeSection: [1, 1, 0],
      mropeInterleaved: true,
    },
    partialRotaryFactor: 1,
    mtpNumHiddenLayers: 0,
    mtpUseDedicatedEmbeddings: false,
    mambaSsmDtype: null,
    bosTokenId: null,
    eosTokenId: null,
    padTokenId: null,
    ...overrides,
  };
}

function qwen3_5VisionConfig(overrides: Partial<Qwen3_5VisionConfig> = {}): Qwen3_5VisionConfig {
  return {
    family: "qwen",
    modelType: "qwen3_5",
    rawConfig: {},
    depth: 2,
    hiddenSize: 8,
    hiddenAct: "gelu_pytorch_tanh",
    intermediateSize: 16,
    numHeads: 2,
    inChannels: 3,
    patchSize: 2,
    spatialMergeSize: 1,
    temporalPatchSize: 1,
    outHiddenSize: 8,
    numPositionEmbeddings: 16,
    deepstackVisualIndexes: [],
    initializerRange: 0.02,
    ...overrides,
  };
}

function qwen3_5Config(overrides: Partial<Qwen3_5Config> = {}): Qwen3_5Config {
  const textConfig = qwen3_5TextConfig();
  const visionConfig = qwen3_5VisionConfig();
  return {
    family: "qwen",
    modelType: "qwen3_5",
    rawConfig: {},
    vocabSize: textConfig.vocabSize,
    hiddenSize: textConfig.hiddenSize,
    numHiddenLayers: textConfig.numHiddenLayers,
    textConfig,
    visionConfig,
    imageTokenId: 248056,
    videoTokenId: 248057,
    visionStartTokenId: 248053,
    visionEndTokenId: 248054,
    tieWordEmbeddings: false,
    languageModelOnly: false,
    ...overrides,
  };
}

describe("Qwen 3.5 weight mapping", () => {
  test("maps both full-attention and linear-attention checkpoint names", () => {
    const config = qwen3_5TextConfig();

    expect(sanitizeQwen3_5TextWeight(config, "model.embed_tokens.weight")).toBe(
      "model.embedTokens.weight",
    );
    expect(sanitizeQwen3_5TextWeight(config, "model.layers.0.linear_attn.in_proj_qkv.weight")).toBe(
      "model.layers.0.linearAttention.inProjectionQkv.weight",
    );
    expect(sanitizeQwen3_5TextWeight(config, "model.layers.0.linear_attn.conv1d.weight")).toBe(
      "model.layers.0.linearAttention.conv1d.weight",
    );
    expect(sanitizeQwen3_5TextWeight(config, "model.layers.1.self_attn.q_proj.weight")).toBe(
      "model.layers.1.selfAttention.qProjection.weight",
    );
    expect(sanitizeQwen3_5TextWeight(config, "model.layers.1.mlp.down_proj.weight")).toBe(
      "model.layers.1.mlp.downProjection.weight",
    );
    expect(sanitizeQwen3_5TextWeight(config, "lm_head.weight")).toBe("lmHead.weight");
  });

  test("drops tied lm head and unknown weights", () => {
    const tiedConfig = qwen3_5TextConfig({ tieWordEmbeddings: true });
    expect(sanitizeQwen3_5TextWeight(tiedConfig, "lm_head.weight")).toBeNull();
    expect(sanitizeQwen3_5TextWeight(tiedConfig, "unknown.weight")).toBeNull();
  });

  test("maps top-level multimodal wrapper weights for both text and vision namespaces", () => {
    const config = qwen3_5Config();

    expect(sanitizeQwen3_5Weight(config, "model.language_model.embed_tokens.weight")).toBe(
      "model.languageModel.embedTokens.weight",
    );
    expect(sanitizeQwen3_5Weight(config, "language_model.model.embed_tokens.weight")).toBe(
      "model.languageModel.embedTokens.weight",
    );
    expect(
      sanitizeQwen3_5Weight(config, "model.language_model.layers.0.linear_attn.conv1d.weight"),
    ).toBe("model.languageModel.layers.0.linearAttention.conv1d.weight");
    expect(
      sanitizeQwen3_5Weight(config, "language_model.model.layers.0.linear_attn.conv1d.weight"),
    ).toBe("model.languageModel.layers.0.linearAttention.conv1d.weight");
    expect(sanitizeQwen3_5Weight(config, "model.visual.patch_embed.proj.weight")).toBe(
      "model.visual.patchEmbed.weight",
    );
    expect(sanitizeQwen3_5Weight(config, "vision_tower.patch_embed.proj.weight")).toBe(
      "model.visual.patchEmbed.weight",
    );
    expect(sanitizeQwen3_5Weight(config, "model.visual.blocks.1.attn.qkv.bias")).toBe(
      "model.visual.blocks.1.attention.qkv.bias",
    );
    expect(sanitizeQwen3_5Weight(config, "vision_tower.blocks.1.attn.qkv.bias")).toBe(
      "model.visual.blocks.1.attention.qkv.bias",
    );
    expect(sanitizeQwen3_5Weight(config, "model.visual.merger.linear_fc2.weight")).toBe(
      "model.visual.merger.linearFc2.weight",
    );
    expect(sanitizeQwen3_5Weight(config, "vision_tower.merger.linear_fc2.weight")).toBe(
      "model.visual.merger.linearFc2.weight",
    );
    expect(sanitizeQwen3_5Weight(config, "lm_head.weight")).toBe("lmHead.weight");
    expect(sanitizeQwen3_5Weight(config, "language_model.lm_head.weight")).toBe("lmHead.weight");
  });

  test("detects converted checkpoint layout without modifying already-direct norm weights", async () => {
    const directory = mkdtempSync(join(tmpdir(), "mlxts-qwen3_5-mlx-"));
    const snapshotPath = join(directory, "model.safetensors");
    try {
      using normTensor = MxArray.fromData([1.25, 0.75], [2]);
      using convTensor = MxArray.fromData(
        Array.from({ length: 8 }, (_, index) => index + 1),
        [2, 4, 1],
      );
      await saveSafetensors(
        {
          "language_model.model.layers.0.input_layernorm.weight": normTensor,
          "language_model.model.layers.0.linear_attn.conv1d.weight": convTensor,
        },
        snapshotPath,
      );

      const style = await detectQwen3_5CheckpointStyle({
        source: "local",
        directory,
        files: [
          {
            relativePath: "model.safetensors",
            localPath: snapshotPath,
            size: Bun.file(snapshotPath).size,
          },
        ],
        totalBytes: Bun.file(snapshotPath).size,
      });

      expect(style).toBe("mlx-converted");

      using norm = MxArray.fromData([1.25, 0.75], [2]);
      using transformed = transformQwen3_5CheckpointTensor(
        style,
        "model.layers.0.inputLayerNorm.weight",
        norm,
      );
      expect(transformed.toList()).toEqual([1.25, 0.75]);
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  test("reorients raw HF conv1d weights and shifts zero-centered norm weights", () => {
    using weight = MxArray.fromData([1, 2, 3, 4, 5, 6], [2, 1, 3]);
    using transformed = transformQwen3_5CheckpointTensor(
      "raw-hf",
      "model.layers.0.linearAttention.conv1d.weight",
      weight,
    );

    expect(transformed.shape).toEqual([2, 3, 1]);
    expect(transformed.toList()).toEqual([
      [[1], [2], [3]],
      [[4], [5], [6]],
    ]);

    using norm = MxArray.fromData([0.25, -0.25], [2]);
    using shiftedNorm = transformQwen3_5CheckpointTensor(
      "raw-hf",
      "model.layers.3.selfAttention.qNorm.weight",
      norm,
    );
    expect(shiftedNorm.toList()).toEqual([1.25, 0.75]);
  });
});
