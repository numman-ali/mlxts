import { describe, expect, test } from "bun:test";
import { array, mxEval } from "@mlxts/core";

import { PackedSwitchGLUExperts } from "../../infrastructure/moe";
import { createQwen3_5TextFeedForward, Qwen3_5TextMLP, Qwen3_5TextMoE } from "./mlp";
import type { Qwen3_5TextConfig } from "./types";

function qwen3_5Config(overrides: Partial<Qwen3_5TextConfig> = {}): Qwen3_5TextConfig {
  return {
    family: "qwen",
    modelType: "qwen3_5_text",
    rawConfig: {},
    vocabSize: 16,
    hiddenSize: 4,
    intermediateSize: 8,
    feedForwardKind: "dense",
    moeIntermediateSize: null,
    sharedExpertIntermediateSize: null,
    numExperts: null,
    numExpertsPerToken: null,
    routerAuxLossCoef: null,
    numHiddenLayers: 1,
    numAttentionHeads: 1,
    numKeyValueHeads: 1,
    headDim: 4,
    hiddenAct: "silu",
    maxPositionEmbeddings: 32,
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
    linearNumValueHeads: 1,
    layerTypes: ["full_attention"],
    fullAttentionInterval: 1,
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

function flattenNumbers(value: unknown): number[] {
  if (typeof value === "number") {
    return [value];
  }
  if (!Array.isArray(value)) {
    throw new Error("Expected nested numeric arrays.");
  }
  return value.flatMap((entry) => flattenNumbers(entry));
}

describe("Qwen3_5TextFeedForward", () => {
  test("creates dense and MoE modules from the same semantic config seam", () => {
    using dense = createQwen3_5TextFeedForward(qwen3_5Config());
    using moe = createQwen3_5TextFeedForward(
      qwen3_5Config({
        modelType: "qwen3_5_moe_text",
        feedForwardKind: "moe",
        moeIntermediateSize: 2,
        sharedExpertIntermediateSize: 2,
        numExperts: 3,
        numExpertsPerToken: 2,
        routerAuxLossCoef: 0.001,
      }),
    );

    expect(dense).toBeInstanceOf(Qwen3_5TextMLP);
    expect(moe).toBeInstanceOf(Qwen3_5TextMoE);
  });

  test("runs the Qwen MoE forward path with packed routed and shared experts", () => {
    using moe = new Qwen3_5TextMoE(
      qwen3_5Config({
        modelType: "qwen3_5_moe_text",
        feedForwardKind: "moe",
        moeIntermediateSize: 2,
        sharedExpertIntermediateSize: 2,
        numExperts: 3,
        numExpertsPerToken: 2,
        routerAuxLossCoef: 0.001,
      }),
    );
    using hidden = array(
      [
        [
          [0.1, -0.2, 0.3, -0.4],
          [0.5, -0.6, 0.7, -0.8],
        ],
      ],
      "float32",
    );

    using output = moe.forward(hidden);
    mxEval(output);

    expect(output.shape).toEqual([1, 2, 4]);
  });

  test("mixes routed and shared Qwen MoE experts with fixed weights", () => {
    using moe = new Qwen3_5TextMoE(
      qwen3_5Config({
        modelType: "qwen3_5_moe_text",
        hiddenSize: 1,
        intermediateSize: 1,
        feedForwardKind: "moe",
        moeIntermediateSize: 1,
        sharedExpertIntermediateSize: 1,
        numExperts: 2,
        numExpertsPerToken: 1,
        routerAuxLossCoef: 0.001,
      }),
    );
    moe.gate.weight.free();
    if (!(moe.experts instanceof PackedSwitchGLUExperts)) {
      throw new Error("Expected Qwen MoE test fixture to start with packed experts.");
    }
    moe.experts.gateUpProjection.free();
    moe.experts.downProjection.free();
    moe.sharedExpert.gateProjection.weight.free();
    moe.sharedExpert.upProjection.weight.free();
    moe.sharedExpert.downProjection.weight.free();
    moe.sharedExpertGate.weight.free();
    moe.gate.weight = array([[10], [0]], "float32");
    moe.experts.gateUpProjection = array(
      [
        [[1], [2]],
        [[0], [0]],
      ],
      "float32",
    );
    moe.experts.downProjection = array([[[3]], [[0]]], "float32");
    moe.sharedExpert.gateProjection.weight = array([[0]], "float32");
    moe.sharedExpert.upProjection.weight = array([[0]], "float32");
    moe.sharedExpert.downProjection.weight = array([[0]], "float32");
    moe.sharedExpertGate.weight = array([[0]], "float32");
    using hidden = array([[[1]]], "float32");
    using output = moe.forward(hidden);
    mxEval(output);

    const [actual] = flattenNumbers(output.toList());
    const expected = 6 * (1 / (1 + Math.exp(-1)));
    expect(actual ?? 0).toBeCloseTo(expected, 5);
  });
});
