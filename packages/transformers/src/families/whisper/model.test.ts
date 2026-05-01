import { describe, expect, test } from "bun:test";
import { array, zeros } from "@mlxts/core";

import {
  disposeWhisperConditionalGenerationOutput,
  disposeWhisperModelOutput,
  WhisperForConditionalGeneration,
  WhisperTextDecoder,
} from "./model";
import type { WhisperConfig } from "./types";

function whisperConfig(overrides: Partial<WhisperConfig> = {}): WhisperConfig {
  return {
    modelType: "whisper",
    rawConfig: {},
    vocabSize: 32,
    numMelBins: 4,
    encoderLayers: 1,
    encoderAttentionHeads: 2,
    decoderLayers: 1,
    decoderAttentionHeads: 2,
    encoderFfnDim: 16,
    decoderFfnDim: 16,
    dModel: 8,
    encoderHeadDim: 4,
    decoderHeadDim: 4,
    activationFunction: "gelu",
    maxSourcePositions: 4,
    maxTargetPositions: 6,
    padTokenId: 0,
    bosTokenId: 1,
    eosTokenId: 2,
    decoderStartTokenId: 1,
    scaleEmbedding: false,
    useCache: true,
    ...overrides,
  };
}

describe("WhisperForConditionalGeneration", () => {
  test("runs an encoder-decoder logits pass over channel-last audio features", () => {
    using model = new WhisperForConditionalGeneration(whisperConfig());
    using inputFeatures = zeros([1, 8, 4], "float32");
    using decoderInputIds = array([[1, 2, 3]], "int32");
    const output = model.run(inputFeatures, decoderInputIds, { outputHiddenStates: true });

    try {
      expect(output.encoderLastHiddenState.shape).toEqual([1, 4, 8]);
      expect(output.lastHiddenState.shape).toEqual([1, 3, 8]);
      expect(output.logits.shape).toEqual([1, 3, 32]);
      expect(output.encoderHiddenStates).toHaveLength(3);
      expect(output.decoderHiddenStates).toHaveLength(3);
    } finally {
      disposeWhisperConditionalGenerationOutput(output);
    }
  });

  test("forward returns caller-owned logits and frees intermediate outputs", () => {
    using model = new WhisperForConditionalGeneration(whisperConfig());
    using inputFeatures = zeros([1, 8, 4], "float32");
    using decoderInputIds = array([[1, 2]], "int32");
    const logits = model.forward(inputFeatures, decoderInputIds);

    try {
      expect(logits.shape).toEqual([1, 2, 32]);
    } finally {
      logits.free();
    }
  });

  test("rejects malformed audio and decoder inputs before execution", () => {
    using model = new WhisperForConditionalGeneration(whisperConfig());
    using shortFeatures = zeros([1, 7, 4], "float32");
    using goodFeatures = zeros([1, 8, 4], "float32");
    using decoderInputIds = array([[1, 2]], "int32");
    using tooLongDecoderInputIds = array([[1, 2, 3, 4, 5, 6, 7]], "int32");
    using floatDecoderInputIds = zeros([1, 2], "float32");

    expect(() => model.forward(shortFeatures, decoderInputIds)).toThrow("[batch, 8, 4]");
    expect(() => model.forward(goodFeatures, tooLongDecoderInputIds)).toThrow(
      "max_target_positions",
    );
    expect(() => model.forward(goodFeatures, floatDecoderInputIds)).toThrow("integer token ids");
  });
});

describe("WhisperTextDecoder", () => {
  test("rejects encoder hidden states with the wrong hidden size", () => {
    const config = whisperConfig();
    using decoder = new WhisperTextDecoder(config);
    using inputIds = array([[1, 2]], "int32");
    using encoderHiddenStates = zeros([1, 4, 7], "float32");

    expect(() => decoder.forward(inputIds, encoderHiddenStates)).toThrow("[batch, seq, 8]");
  });
});

describe("WhisperModel", () => {
  test("returns decoder hidden states without projecting logits", () => {
    using wrapper = new WhisperForConditionalGeneration(whisperConfig());
    const model = wrapper.model;
    using inputFeatures = zeros([1, 8, 4], "float32");
    using decoderInputIds = array([[1, 2]], "int32");
    const output = model.run(inputFeatures, decoderInputIds);

    try {
      expect(output.lastHiddenState.shape).toEqual([1, 2, 8]);
      expect(output.encoderLastHiddenState.shape).toEqual([1, 4, 8]);
    } finally {
      disposeWhisperModelOutput(output);
    }
  });
});
