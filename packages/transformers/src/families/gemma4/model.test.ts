import { describe, expect, test } from "bun:test";
import { array, full, type MxArray, retainArray, zeros } from "@mlxts/core";
import { Module } from "@mlxts/nn";

import {
  createContinuousBatchTokenScheduler,
  generateBatchTokens,
  generateTokens,
} from "../../generation";
import type { TransformerCache } from "../../types";
import { Gemma4TextCausalLM, Gemma4TextModel } from "./model";
import type { Gemma4SharedKeyValues, Gemma4TextConfig } from "./types";

function gemma4Config(overrides: Partial<Gemma4TextConfig> = {}): Gemma4TextConfig {
  return {
    family: "gemma",
    modelType: "gemma4_text",
    rawConfig: {},
    vocabSize: 8,
    vocabSizePerLayerInput: 8,
    hiddenSize: 4,
    intermediateSize: 8,
    numHiddenLayers: 1,
    numAttentionHeads: 1,
    numKeyValueHeads: 1,
    numGlobalKeyValueHeads: null,
    headDim: 4,
    globalHeadDim: 4,
    maxPositionEmbeddings: 32,
    slidingWindow: 16,
    layerTypes: ["sliding_attention"],
    rmsNormEps: 1e-5,
    attentionBias: false,
    tieWordEmbeddings: true,
    hiddenSizePerLayerInput: 4,
    useDoubleWideMLP: false,
    attentionKEqV: false,
    numKvSharedLayers: 0,
    slidingRopeTheta: 10_000,
    fullRopeTheta: 1_000_000,
    fullRotaryDimensions: 4,
    finalLogitSoftcapping: 30,
    embeddingScale: 2,
    ...overrides,
  };
}

function flattenNumbers(value: unknown): number[] {
  if (typeof value === "number") {
    return [value];
  }
  if (!Array.isArray(value)) {
    throw new Error("Expected nested numeric arrays");
  }
  return value.flatMap((entry) => flattenNumbers(entry));
}

class CapturePerLayerInputBlock extends Module {
  selfAttention = { layerType: "sliding_attention" as const };
  captured: MxArray | null = null;

  forward(x: MxArray): MxArray {
    return retainArray(x);
  }

  run(
    x: MxArray,
    _cache?: TransformerCache,
    _sharedKeyValues?: Gemma4SharedKeyValues,
    perLayerInput?: MxArray,
  ): { hidden: MxArray; keyValues: Gemma4SharedKeyValues | null } {
    this.captured?.free();
    this.captured = perLayerInput === undefined ? null : retainArray(perLayerInput);
    return {
      hidden: retainArray(x),
      keyValues: null,
    };
  }

  override [Symbol.dispose](): void {
    this.captured?.free();
    this.captured = null;
    super[Symbol.dispose]();
  }
}

describe("Gemma4TextModel", () => {
  test("static greedy batch generation matches separate single-prompt generation with shared KV", () => {
    using model = new Gemma4TextCausalLM(
      gemma4Config({
        numHiddenLayers: 3,
        layerTypes: ["sliding_attention", "full_attention", "sliding_attention"],
        hiddenSizePerLayerInput: 0,
        numKvSharedLayers: 1,
        slidingWindow: 2,
      }),
    );
    const prompts = [
      [1, 2, 3, 4],
      [5, 6],
    ];
    const options = { maxTokens: 2, temperature: 0, eosTokenIds: [] };

    const separate = prompts.map((prompt) => generateTokens(model, prompt, options));
    const batched = generateBatchTokens(model, prompts, options);

    expect(batched).toEqual(separate);
  });

  test("continuous greedy batch generation admits staggered shared-KV rows", async () => {
    using model = new Gemma4TextCausalLM(
      gemma4Config({
        numHiddenLayers: 3,
        layerTypes: ["sliding_attention", "full_attention", "sliding_attention"],
        hiddenSizePerLayerInput: 0,
        numKvSharedLayers: 1,
        slidingWindow: 2,
      }),
    );
    const firstPrompt = [1, 2, 3, 4];
    const secondPrompt = [5, 6];
    const firstSeparate = generateTokens(model, firstPrompt, {
      maxTokens: 3,
      temperature: 0,
      eosTokenIds: [],
    });
    const secondSeparate = generateTokens(model, secondPrompt, {
      maxTokens: 2,
      temperature: 0,
      eosTokenIds: [],
    });
    const scheduler = createContinuousBatchTokenScheduler(model, {
      maxBatchSize: 2,
      temperature: 0,
      eosTokenIds: [],
    });
    let second: Promise<unknown> | undefined;

    const first = scheduler.enqueue({
      id: "first",
      promptTokenIds: firstPrompt,
      maxTokens: 3,
      onToken(_tokenId, tokenIds) {
        if (tokenIds.length === 1 && second === undefined) {
          second = scheduler.enqueue({
            id: "second",
            promptTokenIds: secondPrompt,
            maxTokens: 2,
          });
        }
      },
    });

    await expect(first).resolves.toEqual(firstSeparate);
    await expect(second).resolves.toEqual(secondSeparate);
  });

  test("scales per-layer token embeddings by sqrt(hiddenSizePerLayerInput)", () => {
    const config = gemma4Config();
    using model = new Gemma4TextModel(config);

    for (const layer of model.layers) {
      layer[Symbol.dispose]();
    }

    const captureBlock = new CapturePerLayerInputBlock();
    model.layers = [captureBlock as unknown as (typeof model.layers)[number]];

    if (
      model.embedTokensPerLayer === null ||
      model.perLayerModelProjection === null ||
      model.perLayerProjectionNorm === null
    ) {
      throw new Error("Expected per-layer input modules to be enabled for Gemma 4.");
    }

    model.embedTokens.weight.free();
    model.embedTokens.weight = zeros([config.vocabSize, config.hiddenSize]);
    model.embedTokensPerLayer.weight.free();
    model.embedTokensPerLayer.weight = full(
      [config.vocabSizePerLayerInput, config.numHiddenLayers * config.hiddenSizePerLayerInput],
      1,
      "float32",
    );
    model.perLayerModelProjection.weight.free();
    model.perLayerModelProjection.weight = zeros([
      config.numHiddenLayers * config.hiddenSizePerLayerInput,
      config.hiddenSize,
    ]);

    using inputIds = array([[0]], "int32");
    using _hidden = model.run(inputIds);
    const captured = captureBlock.captured;
    if (captured === null) {
      throw new Error("Expected the capture block to receive per-layer inputs.");
    }

    const values = flattenNumbers(captured.toList());
    expect(values).toHaveLength(config.hiddenSizePerLayerInput);
    const expected = Math.sqrt(2);
    for (const value of values) {
      expect(value).toBeCloseTo(expected, 6);
    }
  });
});
