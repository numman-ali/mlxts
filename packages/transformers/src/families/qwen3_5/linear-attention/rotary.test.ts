import { describe, expect, test } from "bun:test";
import { array } from "@mlxts/core";
import { RoPE } from "@mlxts/nn";
import type { Qwen3_5TextConfig } from "../types";
import { createSequentialQwen3_5PositionIds, Qwen3_5TextRotaryEmbedding } from "./rotary";

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

function expectNestedCloseTo(actual: unknown, expected: unknown, precision = 5): void {
  if (Array.isArray(actual) || Array.isArray(expected)) {
    if (!Array.isArray(actual) || !Array.isArray(expected) || actual.length !== expected.length) {
      throw new Error("expectNestedCloseTo: expected matching nested array shapes.");
    }
    actual.forEach((value, index) => {
      expectNestedCloseTo(value, expected[index], precision);
    });
    return;
  }
  if (typeof actual !== "number" || typeof expected !== "number") {
    throw new Error(
      `expectNestedCloseTo: expected numeric leaf values, got ${JSON.stringify(actual)} and ${JSON.stringify(expected)}.`,
    );
  }
  expect(actual).toEqual(expect.closeTo(expected, precision));
}

describe("Qwen3_5TextRotaryEmbedding", () => {
  test("matches standard RoPE when all 3 position axes are identical", () => {
    const config = qwen3_5TextConfig({
      ropeParameters: {
        ropeType: "default",
        ropeTheta: 10000,
        partialRotaryFactor: 1,
        mropeSection: [1, 1, 0],
        mropeInterleaved: true,
      },
    });

    using rotary = new Qwen3_5TextRotaryEmbedding(config);
    using baseline = new RoPE(config.headDim, false, config.ropeParameters.ropeTheta);
    using queries = array(
      [
        [
          [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
          ],
          [
            [2, 3, 4, 5],
            [6, 7, 8, 9],
          ],
        ],
      ],
      "float32",
    );
    using keys = array(
      [
        [
          [
            [4, 3, 2, 1],
            [8, 7, 6, 5],
          ],
        ],
      ],
      "float32",
    );
    using positionIds = createSequentialQwen3_5PositionIds(1, 2, 0);
    const rotated = rotary.apply(queries, keys, positionIds);

    using expectedQueries = baseline.forward(queries, 0);
    using expectedKeys = baseline.forward(keys, 0);
    expectNestedCloseTo(rotated.queries.toList(), expectedQueries.toList());
    expectNestedCloseTo(rotated.keys.toList(), expectedKeys.toList());

    rotated.queries.free();
    rotated.keys.free();
  });
});
