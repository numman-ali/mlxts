import { describe, expect, test } from "bun:test";
import { array, type MxArray, mxEval } from "@mlxts/core";

import { T5Attention, t5RelativePositionBucket } from "./attention";
import { disposeT5EncoderModelOutput, T5EncoderModel } from "./model";
import type { T5EncoderConfig } from "./types";

function t5EncoderConfig(overrides: Partial<T5EncoderConfig> = {}): T5EncoderConfig {
  return {
    modelType: "t5_encoder_model",
    rawConfig: {},
    vocabSize: 32,
    dModel: 8,
    dKv: 4,
    dFf: 16,
    numLayers: 2,
    numHeads: 2,
    innerDim: 8,
    relativeAttentionNumBuckets: 8,
    relativeAttentionMaxDistance: 16,
    layerNormEps: 1e-6,
    dropoutRate: 0,
    feedForwardProjection: "gated-gelu",
    denseActivation: "gelu_new",
    isGatedActivation: true,
    padTokenId: 0,
    eosTokenId: 1,
    ...overrides,
  };
}

function firstTokenValues(hidden: MxArray): number[] {
  const rows = hidden.toList() as number[][][];
  return rows[0]?.[0] ?? [];
}

function expectDifferentVector(left: readonly number[], right: readonly number[]): void {
  expect(left).toHaveLength(right.length);
  const maxDelta = left.reduce((max, value, index) => {
    const other = right[index] ?? Number.NaN;
    return Math.max(max, Math.abs(value - other));
  }, 0);
  expect(maxDelta).toBeGreaterThan(1e-6);
}

function setIdentityAttentionWeights(attention: T5Attention): void {
  attention.q.weight.free();
  attention.q.weight = array(
    [
      [1, 0],
      [0, 1],
    ],
    "float32",
  );
  attention.k.weight.free();
  attention.k.weight = array(
    [
      [1, 0],
      [0, 1],
    ],
    "float32",
  );
  attention.v.weight.free();
  attention.v.weight = array(
    [
      [1, 0],
      [0, 1],
    ],
    "float32",
  );
  attention.o.weight.free();
  attention.o.weight = array(
    [
      [1, 0],
      [0, 1],
    ],
    "float32",
  );
}

describe("T5 relative position bucketing", () => {
  test("matches bidirectional bucket boundaries", () => {
    expect(t5RelativePositionBucket(0, true, 32, 128)).toBe(0);
    expect(t5RelativePositionBucket(-1, true, 32, 128)).toBe(1);
    expect(t5RelativePositionBucket(1, true, 32, 128)).toBe(17);
    expect(t5RelativePositionBucket(-1024, true, 32, 128)).toBe(15);
    expect(t5RelativePositionBucket(1024, true, 32, 128)).toBe(31);
  });
});

describe("T5Attention", () => {
  test("uses additive relative-position bias as the attention mask", () => {
    using attention = new T5Attention(
      t5EncoderConfig({
        dModel: 2,
        dKv: 2,
        dFf: 4,
        numHeads: 1,
        innerDim: 2,
        relativeAttentionNumBuckets: 4,
      }),
      false,
    );
    setIdentityAttentionWeights(attention);
    using hiddenStates = array(
      [
        [
          [1, 0],
          [0, 1],
        ],
      ],
      "float32",
    );
    using neutralBias = array(
      [
        [
          [
            [0, 0],
            [0, 0],
          ],
        ],
      ],
      "float32",
    );
    using biasedTowardSecondToken = array(
      [
        [
          [
            [0, 10],
            [0, 10],
          ],
        ],
      ],
      "float32",
    );

    using neutral = attention.run(hiddenStates, neutralBias);
    using biased = attention.run(hiddenStates, biasedTowardSecondToken);

    mxEval(neutral, biased);
    expectDifferentVector(firstTokenValues(neutral), firstTokenValues(biased));
  });
});

describe("T5EncoderModel", () => {
  test("returns last hidden state and retained hidden states", () => {
    using model = new T5EncoderModel(t5EncoderConfig());
    using inputIds = array(
      [
        [0, 4, 1],
        [0, 5, 1],
      ],
      "int32",
    );

    const output = model.run(inputIds, { outputHiddenStates: true });
    try {
      mxEval(output.lastHiddenState, ...(output.hiddenStates ?? []));
      expect(output.lastHiddenState.shape).toEqual([2, 3, 8]);
      expect(output.hiddenStates?.map((hidden) => hidden.shape)).toEqual([
        [2, 3, 8],
        [2, 3, 8],
        [2, 3, 8],
      ]);
    } finally {
      disposeT5EncoderModelOutput(output);
    }
  });

  test("uses bidirectional encoder attention", () => {
    using model = new T5EncoderModel(t5EncoderConfig({ numLayers: 1 }));
    using firstInput = array([[0, 4, 1]], "int32");
    using secondInput = array([[0, 9, 1]], "int32");

    const firstOutput = model.run(firstInput);
    const secondOutput = model.run(secondInput);
    try {
      mxEval(firstOutput.lastHiddenState, secondOutput.lastHiddenState);
      expectDifferentVector(
        firstTokenValues(firstOutput.lastHiddenState),
        firstTokenValues(secondOutput.lastHiddenState),
      );
    } finally {
      disposeT5EncoderModelOutput(firstOutput);
      disposeT5EncoderModelOutput(secondOutput);
    }
  });

  test("supports non-gated feed-forward layers", () => {
    using model = new T5EncoderModel(
      t5EncoderConfig({
        feedForwardProjection: "relu",
        denseActivation: "relu",
        isGatedActivation: false,
      }),
    );
    using inputIds = array([[0, 4, 1]], "int32");

    using output = model.forward(inputIds);
    expect(output.shape).toEqual([1, 3, 8]);
  });

  test("handles zero-layer encoder configs", () => {
    using model = new T5EncoderModel(t5EncoderConfig({ numLayers: 0 }));
    using inputIds = array([[0, 4, 1]], "int32");

    const output = model.run(inputIds, { outputHiddenStates: true });
    try {
      mxEval(output.lastHiddenState, ...(output.hiddenStates ?? []));
      expect(output.lastHiddenState.shape).toEqual([1, 3, 8]);
      expect(output.hiddenStates?.map((hidden) => hidden.shape)).toEqual([[1, 3, 8]]);
    } finally {
      disposeT5EncoderModelOutput(output);
    }
  });

  test("rejects invalid input ids", () => {
    using model = new T5EncoderModel(t5EncoderConfig());
    using rankThree = array([[[0, 1]]], "int32");
    using floats = array([[0, 1]], "float32");

    expect(() => model.run(rankThree)).toThrow("rank-2");
    expect(() => model.run(floats)).toThrow("integer token ids");
  });
});
