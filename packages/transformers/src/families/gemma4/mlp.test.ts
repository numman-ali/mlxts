import { describe, expect, test } from "bun:test";
import { array, matmul, mxEval, transpose } from "@mlxts/core";

import { gegluApprox } from "../../infrastructure/gated-activations";
import { Gemma4TextMLP } from "./mlp";
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
    numHiddenLayers: 2,
    numAttentionHeads: 1,
    numKeyValueHeads: 1,
    numGlobalKeyValueHeads: null,
    headDim: 4,
    globalHeadDim: 4,
    maxPositionEmbeddings: 32,
    slidingWindow: 16,
    layerTypes: ["sliding_attention", "sliding_attention"],
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

describe("Gemma4TextMLP", () => {
  test("matches the eager gated MLP math", () => {
    using mlp = new Gemma4TextMLP(gemma4Config(), 0);
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
    using output = mlp.forward(input);
    using gateWeightTranspose = transpose(mlp.gateProjection.weight);
    using upWeightTranspose = transpose(mlp.upProjection.weight);
    using downWeightTranspose = transpose(mlp.downProjection.weight);
    using gate = matmul(input, gateWeightTranspose);
    using value = matmul(input, upWeightTranspose);
    using activated = gegluApprox(gate, value);
    using expected = matmul(activated, downWeightTranspose);

    mxEval(output, expected);

    expect(output.toList()).toEqual(expected.toList());
  });

  test("reuses the compiled MLP path across leading dimensions", () => {
    using mlp = new Gemma4TextMLP(gemma4Config(), 0);
    using firstInput = array([[[0.25, -0.5, 0.75, -1.0]]], "float32");
    using secondInput = array(
      [
        [
          [1.0, 0.5, 0.25, 0.125],
          [-0.125, -0.25, -0.5, -1.0],
        ],
        [
          [0.0, 0.25, 0.5, 0.75],
          [1.25, 1.5, 1.75, 2.0],
        ],
      ],
      "float32",
    );
    using firstOutput = mlp.forward(firstInput);
    using secondOutput = mlp.forward(secondInput);

    mxEval(firstOutput, secondOutput);

    expect(firstOutput.shape).toEqual([1, 1, 4]);
    expect(secondOutput.shape).toEqual([2, 2, 4]);
  });
});
