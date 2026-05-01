import { describe, expect, test } from "bun:test";

import { ConfigParseError } from "../../types";
import { parseWhisperConfig, parseWhisperFeatureExtractorConfig } from "./config";

describe("parseWhisperConfig", () => {
  test("parses a Hugging Face Whisper config with defaults", () => {
    const config = parseWhisperConfig({
      model_type: "whisper",
      vocab_size: 100,
      d_model: 16,
      encoder_attention_heads: 4,
      decoder_attention_heads: 2,
    });

    expect(config.modelType).toBe("whisper");
    expect(config.vocabSize).toBe(100);
    expect(config.encoderHeadDim).toBe(4);
    expect(config.decoderHeadDim).toBe(8);
    expect(config.numMelBins).toBe(80);
    expect(config.maxSourcePositions).toBe(1500);
    expect(config.decoderStartTokenId).toBe(50257);
  });

  test("rejects non-Whisper and decoder-only configs", () => {
    expect(() => parseWhisperConfig({ model_type: "t5" })).toThrow(ConfigParseError);
    expect(() => parseWhisperConfig({ is_encoder_decoder: false })).toThrow(
      "is_encoder_decoder must be true",
    );
  });

  test("rejects invalid head geometry and unsupported activations", () => {
    expect(() =>
      parseWhisperConfig({
        d_model: 10,
        encoder_attention_heads: 3,
      }),
    ).toThrow("encoder_attention_heads must divide d_model");
    expect(() => parseWhisperConfig({ activation_function: "silu" })).toThrow(
      'activation_function must be "gelu"',
    );
  });

  test("preserves list-shaped eos token ids and rejects list-shaped bos ids", () => {
    const config = parseWhisperConfig({
      eos_token_id: [50256, 50257],
    });

    expect(config.eosTokenId).toEqual([50256, 50257]);
    expect(() => parseWhisperConfig({ bos_token_id: [50256] })).toThrow(
      "bos_token_id list values are not supported here",
    );
  });
});

describe("parseWhisperFeatureExtractorConfig", () => {
  test("parses feature extractor sidecar defaults and derived sizes", () => {
    const config = parseWhisperFeatureExtractorConfig({});

    expect(config.featureSize).toBe(80);
    expect(config.samplingRate).toBe(16000);
    expect(config.hopLength).toBe(160);
    expect(config.nSamples).toBe(480000);
    expect(config.nFrames).toBe(3000);
  });

  test("parses explicit feature extractor geometry", () => {
    const config = parseWhisperFeatureExtractorConfig({
      feature_size: 4,
      sampling_rate: 16,
      hop_length: 4,
      chunk_length: 1,
      n_fft: 8,
      padding_value: 0,
    });

    expect(config.featureSize).toBe(4);
    expect(config.nSamples).toBe(16);
    expect(config.nFrames).toBe(4);
  });
});
