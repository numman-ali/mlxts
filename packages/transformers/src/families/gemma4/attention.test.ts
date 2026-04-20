import { describe, expect, test } from "bun:test";
import {
  array,
  type MxArray,
  mxEval,
  reshape,
  scaledDotProductAttention,
  transpose,
} from "@mlxts/core";

import { createCausalMask } from "../../infrastructure/masks";
import { Gemma4TextAttention } from "./attention";
import type { Gemma4TextConfig } from "./types";

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

function freeKeyValues(keyValues: { keys: MxArray; values: MxArray } | null): void {
  keyValues?.keys.free();
  keyValues?.values.free();
}

describe("Gemma4TextAttention", () => {
  test("matches the eager attention path for standard key/value layers", () => {
    using attention = new Gemma4TextAttention(gemma4Config(), 0);
    using input = array(
      [
        [
          [0.1, -0.2, 0.3, -0.4],
          [0.5, -0.6, 0.7, -0.8],
        ],
        [
          [-0.9, 1.0, -1.1, 1.2],
          [1.3, -1.4, 1.5, -1.6],
        ],
      ],
      "float32",
    );

    const { output, keyValues } = attention.run(input);
    try {
      if (attention.vProjection === null) {
        throw new Error("Expected a value projection for the standard Gemma 4 attention path.");
      }

      const batch = input.shape[0] ?? 0;
      const sequenceLength = input.shape[1] ?? 0;
      using projectedQueries = attention.qProjection.forward(input);
      using queryInputs = reshape(projectedQueries, [batch, sequenceLength, 1, 4]);
      using normalizedQueries = attention.qNorm.forward(queryInputs);
      using queryHeads = transpose(normalizedQueries, [0, 2, 1, 3]);
      using rotatedQueries = attention.rope.forward(queryHeads, 0);

      using projectedKeys = attention.kProjection.forward(input);
      using keyInputs = reshape(projectedKeys, [batch, sequenceLength, 1, 4]);
      using normalizedKeys = attention.kNorm.forward(keyInputs);
      using keyHeads = transpose(normalizedKeys, [0, 2, 1, 3]);
      using rotatedKeys = attention.rope.forward(keyHeads, 0);

      using projectedValues = attention.vProjection.forward(input);
      using valueInputs = reshape(projectedValues, [batch, sequenceLength, 1, 4]);
      using normalizedValues = attention.vNorm.forward(valueInputs);
      using valueHeads = transpose(normalizedValues, [0, 2, 1, 3]);
      using mask = createCausalMask(sequenceLength, sequenceLength, 0, rotatedQueries.dtype, 16);
      using attentionOutput =
        mask === null
          ? scaledDotProductAttention(rotatedQueries, rotatedKeys, valueHeads, { scale: 1.0 })
          : scaledDotProductAttention(rotatedQueries, rotatedKeys, valueHeads, {
              scale: 1.0,
              maskArray: mask,
            });
      using transposedOutput = transpose(attentionOutput, [0, 2, 1, 3]);
      using mergedOutput = reshape(transposedOutput, [batch, sequenceLength, 4]);
      using expected = attention.outputProjection.forward(mergedOutput);

      mxEval(output, expected);
      expect(output.toList()).toEqual(expected.toList());
    } finally {
      output.free();
      freeKeyValues(keyValues);
    }
  });

  test("matches the eager attention path for attentionKEqV full-attention layers", () => {
    using attention = new Gemma4TextAttention(
      gemma4Config({
        layerTypes: ["full_attention"],
        attentionKEqV: true,
        slidingWindow: 16,
      }),
      0,
    );
    using input = array(
      [
        [
          [0.1, -0.2, 0.3, -0.4],
          [0.5, -0.6, 0.7, -0.8],
        ],
      ],
      "float32",
    );

    const { output, keyValues } = attention.run(input);
    try {
      if (attention.vProjection !== null) {
        throw new Error("Expected KEqV full attention to omit the value projection.");
      }

      const batch = input.shape[0] ?? 0;
      const sequenceLength = input.shape[1] ?? 0;
      using projectedQueries = attention.qProjection.forward(input);
      using queryInputs = reshape(projectedQueries, [batch, sequenceLength, 1, 4]);
      using normalizedQueries = attention.qNorm.forward(queryInputs);
      using queryHeads = transpose(normalizedQueries, [0, 2, 1, 3]);
      using rotatedQueries = attention.rope.forward(queryHeads, 0);

      using projectedKeys = attention.kProjection.forward(input);
      using keyInputs = reshape(projectedKeys, [batch, sequenceLength, 1, 4]);
      using normalizedKeys = attention.kNorm.forward(keyInputs);
      using normalizedValues = attention.vNorm.forward(keyInputs);
      using keyHeads = transpose(normalizedKeys, [0, 2, 1, 3]);
      using valueHeads = transpose(normalizedValues, [0, 2, 1, 3]);
      using rotatedKeys = attention.rope.forward(keyHeads, 0);
      using mask = createCausalMask(sequenceLength, sequenceLength, 0, rotatedQueries.dtype);
      using attentionOutput =
        mask === null
          ? scaledDotProductAttention(rotatedQueries, rotatedKeys, valueHeads, { scale: 1.0 })
          : scaledDotProductAttention(rotatedQueries, rotatedKeys, valueHeads, {
              scale: 1.0,
              maskArray: mask,
            });
      using transposedOutput = transpose(attentionOutput, [0, 2, 1, 3]);
      using mergedOutput = reshape(transposedOutput, [batch, sequenceLength, 4]);
      using expected = attention.outputProjection.forward(mergedOutput);

      mxEval(output, expected);
      expect(output.toList()).toEqual(expected.toList());
    } finally {
      output.free();
      freeKeyValues(keyValues);
    }
  });
});
