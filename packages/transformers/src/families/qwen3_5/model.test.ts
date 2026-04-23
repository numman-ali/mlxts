import { describe, expect, test } from "bun:test";
import { array, retainArray, zeros } from "@mlxts/core";
import { Qwen3_5TextCache } from "./cache";
import { Qwen3_5TextCausalLM } from "./model";
import type { Qwen3_5TextConfig } from "./types";

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
    tieWordEmbeddings: true,
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

describe("Qwen3_5TextCausalLM", () => {
  test("runs the hybrid decoder and advances cache offsets", () => {
    const config = qwen3_5TextConfig();
    using model = new Qwen3_5TextCausalLM(config);
    using cache = model.createCache();
    if (!(cache instanceof Qwen3_5TextCache)) {
      throw new Error("expected a Qwen3_5TextCache");
    }

    using prompt = array([[1, 2, 3]], "int32");
    using promptLogits = model.forward(prompt, { cache });
    expect(promptLogits.shape).toEqual([1, 3, config.vocabSize]);
    expect(cache.offset).toBe(3);
    const linearState = cache.linearState(0);
    expect(linearState.convState?.shape).toEqual([1, 1, 8]);
    expect(linearState.recurrentState?.shape).toEqual([1, 2, 2, 2]);

    using nextToken = array([[4]], "int32");
    using nextLogits = model.forward(nextToken, { cache });
    expect(nextLogits.shape).toEqual([1, 1, config.vocabSize]);
    expect(cache.offset).toBe(4);
  });

  test("supports external prompt embeddings for prefill", () => {
    const config = qwen3_5TextConfig({ layerTypes: ["full_attention"], numHiddenLayers: 1 });
    using model = new Qwen3_5TextCausalLM(config);
    using inputIds = array([[1, 2]], "int32");
    using inputEmbeddings = array(
      [
        [
          [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
          [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        ],
      ],
      "float32",
    );
    using logits = model.forward(inputIds, { inputEmbeddings });
    expect(logits.shape).toEqual([1, 2, config.vocabSize]);
  });

  test("forwards explicit position ids into the text model", () => {
    using model = new Qwen3_5TextCausalLM(qwen3_5TextConfig());
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
      using inputIds = array([[1, 2]], "int32");
      using positionIds = array([[[0, 1]], [[0, 1]], [[0, 1]]], "int32");
      using logits = model.forward(inputIds, { positionIds });
      void logits;

      expect(capturedPositionIds).toEqual([[[[0, 1]], [[0, 1]], [[0, 1]]]]);
    } finally {
      model.model.run = originalRun;
    }
  });

  test("supports full-attention projections wider than the hidden size", () => {
    const config = qwen3_5TextConfig({
      hiddenSize: 10,
      intermediateSize: 20,
      numHiddenLayers: 1,
      numAttentionHeads: 3,
      numKeyValueHeads: 1,
      headDim: 4,
      layerTypes: ["full_attention"],
      fullAttentionInterval: 1,
    });
    using model = new Qwen3_5TextCausalLM(config);
    using inputIds = array([[1, 2]], "int32");
    using logits = model.forward(inputIds);
    expect(logits.shape).toEqual([1, 2, config.vocabSize]);
  });
});
