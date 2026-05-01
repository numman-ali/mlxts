import { describe, expect, test } from "bun:test";
import { array, max, min, mxEval, zeros } from "@mlxts/core";

import { parseWhisperFeatureExtractorConfig } from "./config";
import { createWhisperMelFilterBank, prepareWhisperAudioFeatures } from "./preprocessing";

const tinyFeatureConfig = parseWhisperFeatureExtractorConfig({
  feature_size: 4,
  sampling_rate: 16,
  hop_length: 4,
  chunk_length: 1,
  n_fft: 8,
  padding_value: 0,
});

describe("createWhisperMelFilterBank", () => {
  test("creates Slaney-normalized mel filters with Whisper geometry", () => {
    using filters = createWhisperMelFilterBank(tinyFeatureConfig);
    expect(filters.shape).toEqual([4, 5]);
    using peak = max(filters);
    peak.eval();
    expect(peak.item()).toBeGreaterThan(0);
  });
});

describe("prepareWhisperAudioFeatures", () => {
  test("pads silence into channel-last log-mel features", () => {
    using audio = zeros([8]);
    const prepared = prepareWhisperAudioFeatures(audio, tinyFeatureConfig);
    try {
      expect(prepared.inputFeatures.shape).toEqual([1, 4, 4]);
      using minValue = min(prepared.inputFeatures);
      using maxValue = max(prepared.inputFeatures);
      mxEval(minValue, maxValue);
      expect(minValue.item()).toBeCloseTo(-1.5, 5);
      expect(maxValue.item()).toBeCloseTo(-1.5, 5);
    } finally {
      prepared.inputFeatures.free();
    }
  });

  test("keeps non-silent audio distinct from the silence floor", () => {
    using audio = array([1, 0, 0, 0, 0, 0, 0, 0], "float32");
    const prepared = prepareWhisperAudioFeatures(audio, tinyFeatureConfig);
    try {
      using minValue = min(prepared.inputFeatures);
      using maxValue = max(prepared.inputFeatures);
      mxEval(minValue, maxValue);
      expect(maxValue.item()).toBeGreaterThan(minValue.item());
    } finally {
      prepared.inputFeatures.free();
    }
  });

  test("rejects non-mono tensors", () => {
    using audio = zeros([1, 16]);
    expect(() => prepareWhisperAudioFeatures(audio, tinyFeatureConfig)).toThrow(
      "expected rank-1 mono audio",
    );
  });
});
