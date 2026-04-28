import { describe, expect, test } from "bun:test";
import { array, mxEval } from "@mlxts/core";

import { Gemma4TextRouter } from "./moe";
import type { Gemma4TextConfig } from "./types";

function gemma4Config(overrides: Partial<Gemma4TextConfig> = {}): Gemma4TextConfig {
  return {
    family: "gemma",
    modelType: "gemma4_text",
    rawConfig: {},
    vocabSize: 8,
    vocabSizePerLayerInput: 8,
    hiddenSize: 1,
    intermediateSize: 1,
    enableMoeBlock: true,
    moeIntermediateSize: 1,
    numExperts: 3,
    topKExperts: 2,
    numHiddenLayers: 1,
    numAttentionHeads: 1,
    numKeyValueHeads: 1,
    numGlobalKeyValueHeads: null,
    headDim: 1,
    globalHeadDim: 1,
    maxPositionEmbeddings: 32,
    slidingWindow: 16,
    layerTypes: ["sliding_attention"],
    rmsNormEps: 0,
    attentionBias: false,
    tieWordEmbeddings: true,
    hiddenSizePerLayerInput: 0,
    useDoubleWideMLP: false,
    attentionKEqV: false,
    numKvSharedLayers: 0,
    slidingRopeTheta: 10_000,
    fullRopeTheta: 1_000_000,
    fullRotaryDimensions: 1,
    finalLogitSoftcapping: 30,
    embeddingScale: 1,
    ...overrides,
  };
}

function firstRow(value: unknown): number[] {
  if (!Array.isArray(value) || !Array.isArray(value[0])) {
    throw new Error("Expected a two-dimensional numeric array.");
  }
  return value[0].map((entry) => {
    if (typeof entry !== "number") {
      throw new Error("Expected numeric route entries.");
    }
    return entry;
  });
}

describe("Gemma4TextRouter", () => {
  test("applies RMS scaling, top-k renormalization, and per-expert scale", () => {
    using router = new Gemma4TextRouter(gemma4Config());
    router.proj.weight.free();
    router.scale.free();
    router.perExpertScale.free();
    router.proj.weight = array([[2], [1], [-1]], "float32");
    router.scale = array([1], "float32");
    router.perExpertScale = array([1, 10, 1], "float32");
    using hidden = array([[2]], "float32");
    const routing = router.route(hidden);
    try {
      mxEval(routing.indices, routing.weights);
      const indices = firstRow(routing.indices.toList());
      const weights = firstRow(routing.weights.toList());
      const expectedByExpert = new Map([
        [0, Math.exp(2) / (Math.exp(2) + Math.exp(1))],
        [1, (10 * Math.exp(1)) / (Math.exp(2) + Math.exp(1))],
      ]);

      expect(new Set(indices)).toEqual(new Set([0, 1]));
      for (let index = 0; index < indices.length; index += 1) {
        const expert = indices[index];
        const actual = weights[index];
        if (expert === undefined || actual === undefined) {
          throw new Error("Expected routed expert and weight entries.");
        }
        expect(actual).toBeCloseTo(expectedByExpert.get(expert) ?? 0, 5);
      }
    } finally {
      routing.indices.free();
      routing.weights.free();
    }
  });
});
