import { describe, expect, test } from "bun:test";
import { array, type MxArray, mxEval } from "@mlxts/core";

import {
  CLIPTextModel,
  CLIPTextModelWithProjection,
  disposeCLIPTextModelOutput,
  disposeCLIPTextProjectionOutput,
} from "./model";
import type { CLIPTextConfig } from "./types";

function clipTextConfig(overrides: Partial<CLIPTextConfig> = {}): CLIPTextConfig {
  return {
    modelType: "clip_text_model",
    rawConfig: {},
    vocabSize: 16,
    hiddenSize: 8,
    intermediateSize: 16,
    projectionDim: 6,
    numHiddenLayers: 2,
    numAttentionHeads: 2,
    headDim: 4,
    maxPositionEmbeddings: 8,
    hiddenAct: "quick_gelu",
    layerNormEps: 1e-5,
    attentionDropout: 0,
    padTokenId: 0,
    bosTokenId: 1,
    eosTokenId: 15,
    ...overrides,
  };
}

function firstTokenValues(hidden: MxArray): number[] {
  const rows = hidden.toList() as number[][][];
  return rows[0]?.[0] ?? [];
}

function expectCloseVector(actual: readonly number[], expected: readonly number[]): void {
  expect(actual).toHaveLength(expected.length);
  for (let index = 0; index < actual.length; index += 1) {
    expect(actual[index]).toBeCloseTo(expected[index] ?? Number.NaN, 5);
  }
}

describe("CLIPTextModel", () => {
  test("returns last hidden state, pooled output, and layer hidden states", () => {
    using model = new CLIPTextModel(clipTextConfig());
    using inputIds = array(
      [
        [1, 4, 15],
        [1, 5, 15],
      ],
      "int32",
    );

    const output = model.run(inputIds, { outputHiddenStates: true });
    try {
      mxEval(output.lastHiddenState, output.pooledOutput, ...(output.hiddenStates ?? []));
      expect(output.lastHiddenState.shape).toEqual([2, 3, 8]);
      expect(output.pooledOutput.shape).toEqual([2, 8]);
      expect(output.hiddenStates?.map((hidden) => hidden.shape)).toEqual([
        [2, 3, 8],
        [2, 3, 8],
        [2, 3, 8],
      ]);
    } finally {
      disposeCLIPTextModelOutput(output);
    }
  });

  test("selects pooled output from the first explicit EOS token", () => {
    using model = new CLIPTextModel(clipTextConfig({ numHiddenLayers: 0 }));
    using inputIds = array([[1, 15, 3, 15]], "int32");

    const output = model.run(inputIds);
    try {
      mxEval(output.lastHiddenState, output.pooledOutput);
      const hiddenRows = output.lastHiddenState.toList() as number[][][];
      const pooled = output.pooledOutput.toList() as number[][];
      expectCloseVector(pooled[0] ?? [], hiddenRows[0]?.[1] ?? []);
    } finally {
      disposeCLIPTextModelOutput(output);
    }
  });

  test("uses causal self-attention so future tokens do not affect earlier positions", () => {
    using model = new CLIPTextModel(clipTextConfig({ numHiddenLayers: 1 }));
    using firstInput = array([[1, 4, 15]], "int32");
    using secondInput = array([[1, 9, 15]], "int32");

    const firstOutput = model.run(firstInput);
    const secondOutput = model.run(secondInput);
    try {
      mxEval(firstOutput.lastHiddenState, secondOutput.lastHiddenState);
      expectCloseVector(
        firstTokenValues(firstOutput.lastHiddenState),
        firstTokenValues(secondOutput.lastHiddenState),
      );
    } finally {
      disposeCLIPTextModelOutput(firstOutput);
      disposeCLIPTextModelOutput(secondOutput);
    }
  });

  test("rejects sequences beyond max_position_embeddings", () => {
    using model = new CLIPTextModel(clipTextConfig({ maxPositionEmbeddings: 2 }));
    using inputIds = array([[1, 4, 15]], "int32");

    expect(() => model.run(inputIds)).toThrow("exceeds max_position_embeddings");
  });
});

describe("CLIPTextModelWithProjection", () => {
  test("projects pooled CLIP text features", () => {
    using model = new CLIPTextModelWithProjection(clipTextConfig({ projectionDim: 5 }));
    using inputIds = array([[1, 4, 15]], "int32");

    const output = model.run(inputIds, { outputHiddenStates: true });
    try {
      mxEval(output.lastHiddenState, output.pooledOutput, output.textEmbeds);
      expect(output.lastHiddenState.shape).toEqual([1, 3, 8]);
      expect(output.pooledOutput.shape).toEqual([1, 8]);
      expect(output.textEmbeds.shape).toEqual([1, 5]);
      expect(output.hiddenStates?.length).toBe(3);
    } finally {
      disposeCLIPTextProjectionOutput(output);
    }
  });

  test("requires projection_dim for projected CLIP text models", () => {
    expect(() => new CLIPTextModelWithProjection(clipTextConfig({ projectionDim: null }))).toThrow(
      "projection_dim is required",
    );
  });
});
