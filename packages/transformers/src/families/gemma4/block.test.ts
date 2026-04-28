import { describe, expect, test } from "bun:test";
import { add, array, type MxArray, multiply, mxEval, reshape, retainArray } from "@mlxts/core";

import { gegluApprox } from "../../infrastructure/gated-activations";
import type { AttentionMask } from "../../infrastructure/masks";
import type { TransformerCache } from "../../types";
import { Gemma4TextDecoderBlock } from "./block";
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
    enableMoeBlock: false,
    moeIntermediateSize: null,
    numExperts: null,
    topKExperts: null,
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

class FixedAttentionStub {
  readonly layerType = "sliding_attention" as const;
  #output: MxArray;

  constructor(output: MxArray) {
    this.#output = retainArray(output);
  }

  run(
    _x: MxArray,
    _cache?: TransformerCache,
    _sharedKeyValues?: Gemma4SharedKeyValues,
    _attentionMask?: AttentionMask,
  ): { output: MxArray; keyValues: Gemma4SharedKeyValues | null } {
    return { output: retainArray(this.#output), keyValues: null };
  }

  [Symbol.dispose](): void {
    this.#output.free();
  }
}

describe("Gemma4TextDecoderBlock", () => {
  test("matches the eager feedforward tail without per-layer input", () => {
    using block = new Gemma4TextDecoderBlock(gemma4Config(), 0);
    using attentionOutput = array(
      [
        [
          [0.2, -0.1, 0.3, -0.4],
          [0.5, -0.6, 0.7, -0.8],
        ],
        [
          [-0.9, 1.0, -1.1, 1.2],
          [1.3, -1.4, 1.5, -1.6],
        ],
      ],
      "float32",
    );
    using input = array(
      [
        [
          [0.1, 0.2, 0.3, 0.4],
          [-0.5, -0.6, -0.7, -0.8],
        ],
        [
          [0.9, 1.0, 1.1, 1.2],
          [-1.3, -1.4, -1.5, -1.6],
        ],
      ],
      "float32",
    );
    block.selfAttention[Symbol.dispose]();
    using stub = new FixedAttentionStub(attentionOutput);
    block.selfAttention = stub as unknown as typeof block.selfAttention;

    const { hidden } = block.run(input);
    using normalizedAttentionOutput = block.postAttentionLayerNorm.forward(attentionOutput);
    using residualAfterAttention = add(input, normalizedAttentionOutput);
    using normalizedForMlp = block.preFeedforwardLayerNorm.forward(residualAfterAttention);
    using mlpOutput = block.mlp.forward(normalizedForMlp);
    using normalizedMlpOutput = block.postFeedforwardLayerNorm.forward(mlpOutput);
    using eagerHidden = add(residualAfterAttention, normalizedMlpOutput);
    using expected = multiply(eagerHidden, block.layerScalar);

    mxEval(hidden, expected);

    expect(hidden.toList()).toEqual(expected.toList());
    hidden.free();
  });

  test("matches the eager dense-plus-MoE feedforward tail", () => {
    using block = new Gemma4TextDecoderBlock(
      gemma4Config({
        enableMoeBlock: true,
        hiddenSizePerLayerInput: 0,
        moeIntermediateSize: 4,
        numExperts: 3,
        topKExperts: 2,
      }),
      0,
    );
    using attentionOutput = array([[[0.2, -0.1, 0.3, -0.4]]], "float32");
    using input = array([[[0.1, 0.2, 0.3, 0.4]]], "float32");
    block.selfAttention[Symbol.dispose]();
    using stub = new FixedAttentionStub(attentionOutput);
    block.selfAttention = stub as unknown as typeof block.selfAttention;

    const { hidden } = block.run(input);
    if (
      block.router === null ||
      block.experts === null ||
      block.preFeedforwardLayerNorm2 === null ||
      block.postFeedforwardLayerNorm1 === null ||
      block.postFeedforwardLayerNorm2 === null
    ) {
      throw new Error("Expected Gemma 4 MoE modules to be enabled.");
    }

    using normalizedAttentionOutput = block.postAttentionLayerNorm.forward(attentionOutput);
    using residualAfterAttention = add(input, normalizedAttentionOutput);
    using normalizedForMlp = block.preFeedforwardLayerNorm.forward(residualAfterAttention);
    using mlpOutput = block.mlp.forward(normalizedForMlp);
    using denseOutput = block.postFeedforwardLayerNorm1.forward(mlpOutput);
    using flatResidual = reshape(residualAfterAttention, [1, 4]);
    const routing = block.router.route(flatResidual);
    try {
      using expertInput = block.preFeedforwardLayerNorm2.forward(flatResidual);
      using expertOutputFlat = block.experts.forward(expertInput, routing.indices, routing.weights);
      using expertOutput = reshape(expertOutputFlat, [1, 1, 4]);
      using normalizedExpertOutput = block.postFeedforwardLayerNorm2.forward(expertOutput);
      using combinedOutput = add(denseOutput, normalizedExpertOutput);
      using normalizedCombinedOutput = block.postFeedforwardLayerNorm.forward(combinedOutput);
      using hiddenAfterMoe = add(residualAfterAttention, normalizedCombinedOutput);
      using expected = multiply(hiddenAfterMoe, block.layerScalar);

      mxEval(hidden, expected);

      expect(hidden.toList()).toEqual(expected.toList());
    } finally {
      routing.indices.free();
      routing.weights.free();
      hidden.free();
    }
  });

  test("matches the eager feedforward tail with per-layer input", () => {
    using block = new Gemma4TextDecoderBlock(gemma4Config(), 0);
    using attentionOutput = array([[[0.2, -0.1, 0.3, -0.4]]], "float32");
    using input = array([[[0.1, 0.2, 0.3, 0.4]]], "float32");
    using perLayerInput = array([[[1.0, -1.5, 2.0, -2.5]]], "float32");
    block.selfAttention[Symbol.dispose]();
    using stub = new FixedAttentionStub(attentionOutput);
    block.selfAttention = stub as unknown as typeof block.selfAttention;

    const { hidden } = block.run(input, undefined, undefined, perLayerInput);
    if (
      block.perLayerInputGate === null ||
      block.perLayerProjection === null ||
      block.postPerLayerInputNorm === null
    ) {
      throw new Error("Expected per-layer input modules to be enabled.");
    }

    using normalizedAttentionOutput = block.postAttentionLayerNorm.forward(attentionOutput);
    using residualAfterAttention = add(input, normalizedAttentionOutput);
    using normalizedForMlp = block.preFeedforwardLayerNorm.forward(residualAfterAttention);
    using mlpOutput = block.mlp.forward(normalizedForMlp);
    using normalizedMlpOutput = block.postFeedforwardLayerNorm.forward(mlpOutput);
    using eagerHidden = add(residualAfterAttention, normalizedMlpOutput);
    using gatedInput = block.perLayerInputGate.forward(eagerHidden);
    using gatedPerLayerInput = gegluApprox(gatedInput, perLayerInput);
    using projectedPerLayerInput = block.perLayerProjection.forward(gatedPerLayerInput);
    using normalizedPerLayerInput = block.postPerLayerInputNorm.forward(projectedPerLayerInput);
    using eagerHiddenWithPerLayerInput = add(eagerHidden, normalizedPerLayerInput);
    using expected = multiply(eagerHiddenWithPerLayerInput, block.layerScalar);

    mxEval(hidden, expected);

    expect(hidden.toList()).toEqual(expected.toList());
    hidden.free();
  });
});
