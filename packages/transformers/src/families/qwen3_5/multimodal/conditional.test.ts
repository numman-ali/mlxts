import { describe, expect, test } from "bun:test";

import { array, retainArray, zeros } from "@mlxts/core";
import { Qwen3_5TextBatchCache } from "../cache/batch-cache";
import { Qwen3_5TextCausalLM } from "../model";
import type { Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig } from "../types";
import {
  createQwen3_5MmTokenTypeIds,
  expandQwen3_5ImageTokens,
  prepareQwen3_5ImagePrompt,
  Qwen3_5ForConditionalGeneration,
} from "./conditional";

function qwen3_5TextConfig(overrides: Partial<Qwen3_5TextConfig> = {}): Qwen3_5TextConfig {
  return {
    family: "qwen",
    modelType: "qwen3_5_text",
    rawConfig: {},
    vocabSize: 32,
    hiddenSize: 8,
    intermediateSize: 16,
    feedForwardKind: "dense",
    moeIntermediateSize: null,
    sharedExpertIntermediateSize: null,
    numExperts: null,
    numExpertsPerToken: null,
    routerAuxLossCoef: null,
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
    imageTokenId: 28,
    videoTokenId: 29,
    visionStartTokenId: 26,
    visionEndTokenId: 27,
    tieWordEmbeddings: false,
    languageModelOnly: false,
    ...overrides,
  };
}

function samplePixelValues(): number[][] {
  return [
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
  ];
}

describe("Qwen3_5ForConditionalGeneration", () => {
  test("expands image placeholders and derives modality ids for image-only prompts", () => {
    using imageGridThw = array([[1, 2, 2]], "int32");
    const tokenIds = [7, 28, 9];

    expect(expandQwen3_5ImageTokens(tokenIds, imageGridThw, 28, 1)).toEqual([7, 28, 28, 28, 28, 9]);
    expect(createQwen3_5MmTokenTypeIds([7, 28, 28, 28, 28, 9], 28, 29)).toEqual([0, 1, 1, 1, 1, 0]);
  });

  test("prepareImagePrompt auto-expands one image token per image and returns aligned tensors", () => {
    using model = new Qwen3_5ForConditionalGeneration(qwen3_5Config());
    using imageGridThw = array([[1, 2, 2]], "int32");
    using pixelValues = array(samplePixelValues(), "float32");

    const prompt = model.prepareImagePrompt([7, 28, 9], pixelValues, imageGridThw);

    try {
      expect(prompt.tokenIds).toEqual([7, 28, 28, 28, 28, 9]);
      expect(prompt.inputEmbeddings?.shape).toEqual([1, 6, 8]);
      expect(prompt.positionIds?.shape).toEqual([3, 1, 6]);
    } finally {
      prompt.inputEmbeddings?.free();
      prompt.positionIds?.free();
    }
  });

  test("prepareQwen3_5ImagePrompt rejects non-conditional models", () => {
    using textOnlyModel = new Qwen3_5TextCausalLM(qwen3_5TextConfig());
    using imageGridThw = array([[1, 2, 2]], "int32");
    using pixelValues = array(samplePixelValues(), "float32");

    expect(() =>
      prepareQwen3_5ImagePrompt(textOnlyModel, [7, 28, 9], pixelValues, imageGridThw),
    ).toThrow("expected a Qwen 3.5 conditional checkpoint");
  });

  test("prepareImagePrompt validates multimodal token-type ids", () => {
    using model = new Qwen3_5ForConditionalGeneration(qwen3_5Config());
    using imageGridThw = array([[1, 2, 2]], "int32");
    using pixelValues = array(samplePixelValues(), "float32");

    expect(() =>
      model.prepareImagePrompt([7, 28, 9], pixelValues, imageGridThw, [0, 1, 0]),
    ).toThrow("mmTokenTypeIds length 3 must match prepared token count 6");
    expect(() =>
      model.prepareImagePrompt([7, 28, 9], pixelValues, imageGridThw, [0, 2, 2, 2, 2, 0]),
    ).toThrow("video token types are not implemented yet");
  });

  test("accepts text batch cache for text-only wrapper forward", () => {
    using model = new Qwen3_5ForConditionalGeneration(qwen3_5Config());
    const originalRun = model.model.run.bind(model.model);
    let sawBatchCache = false;
    let sawPositionIds = false;

    model.model.run = ((inputIds, cache, _inputEmbeddings, positionIds) => {
      sawBatchCache = cache instanceof Qwen3_5TextBatchCache;
      sawPositionIds = positionIds !== undefined;
      const [batchSize, sequenceLength] = inputIds.shape;
      return zeros([batchSize ?? 0, sequenceLength ?? 0, model.config.hiddenSize], "float32");
    }) as typeof model.model.run;

    try {
      using cache = model.createBatchCache([0, 1]);
      using inputIds = array(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        "int32",
      );
      using logits = model.forward(inputIds, { cache });

      expect(cache).toBeInstanceOf(Qwen3_5TextBatchCache);
      expect(cache.layerKinds).toEqual(["linear-recurrent", "full"]);
      expect(logits.shape).toEqual([2, 3, 32]);
      expect(sawBatchCache).toBe(true);
      expect(sawPositionIds).toBe(false);
      expect(cache.offsets).toEqual([3, 2]);
    } finally {
      model.model.run = originalRun;
    }
  });

  test("reuses per-batch rope deltas during cached decode", () => {
    using model = new Qwen3_5ForConditionalGeneration(qwen3_5Config());
    const originalRun = model.model.run.bind(model.model);
    const capturedPositionIds: number[][][][] = [];

    model.model.run = ((inputIds, _cache, _inputEmbeddings, positionIds) => {
      if (positionIds !== undefined) {
        using retained = retainArray(positionIds);
        capturedPositionIds.push(retained.toList() as number[][][]);
      }
      const [batchSize, sequenceLength] = inputIds.shape;
      return zeros([batchSize ?? 0, sequenceLength ?? 0, model.config.hiddenSize], "float32");
    }) as typeof model.model.run;

    try {
      using cache = model.createCache();
      using firstInputIds = array(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        "int32",
      );
      using firstPositionIds = array(
        [
          [
            [0, 1, 2],
            [5, 6, 7],
          ],
          [
            [0, 1, 2],
            [5, 6, 7],
          ],
          [
            [0, 1, 2],
            [5, 6, 7],
          ],
        ],
        "int32",
      );
      using secondInputIds = array([[8], [9]], "int32");

      using firstLogits = model.forward(firstInputIds, { cache, positionIds: firstPositionIds });
      using secondLogits = model.forward(secondInputIds, { cache });
      void firstLogits;
      void secondLogits;

      expect(capturedPositionIds).toHaveLength(2);
      expect(capturedPositionIds[1]).toEqual([
        [[3], [8]],
        [[3], [8]],
        [[3], [8]],
      ]);

      capturedPositionIds.length = 0;
      using cacheForSnapshot = model.createCache();
      using snapshotInputIds = array([[1, 2, 3]], "int32");
      using snapshotPositionIds = array([[[0, 1, 2]], [[0, 1, 2]], [[0, 1, 2]]], "int32");
      using snapshotLogits = model.forward(snapshotInputIds, {
        cache: cacheForSnapshot,
        positionIds: snapshotPositionIds,
      });
      void snapshotLogits;
      using snapshot = cacheForSnapshot.snapshot();
      using forkedCache = snapshot.fork();
      using staleCache = model.createCache();
      using stalePositionIds = array([[[10, 11, 12]], [[10, 11, 12]], [[10, 11, 12]]], "int32");
      using staleLogits = model.forward(snapshotInputIds, {
        cache: staleCache,
        positionIds: stalePositionIds,
      });
      using nextInputIds = array([[8]], "int32");
      using forkedLogits = model.forward(nextInputIds, { cache: forkedCache });
      void staleLogits;
      void forkedLogits;

      expect(capturedPositionIds[2]).toEqual([[[3]], [[3]], [[3]]]);
    } finally {
      model.model.run = originalRun;
    }
  });
});
