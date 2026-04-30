import { describe, expect, test } from "bun:test";

import { ConfigParseError } from "../../types";
import { parseT5EncoderConfig } from "./config";

function t5Config(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    model_type: "t5",
    vocab_size: 32128,
    d_model: 4096,
    d_kv: 64,
    d_ff: 10240,
    num_layers: 24,
    num_heads: 64,
    relative_attention_num_buckets: 32,
    relative_attention_max_distance: 128,
    layer_norm_epsilon: 1e-6,
    dropout_rate: 0.1,
    feed_forward_proj: "gated-gelu",
    is_encoder_decoder: true,
    is_decoder: false,
    pad_token_id: 0,
    eos_token_id: 1,
    ...overrides,
  };
}

describe("parseT5EncoderConfig", () => {
  test("parses T5 v1.1 encoder config for FLUX-style text conditioning", () => {
    const parsed = parseT5EncoderConfig(t5Config());

    expect(parsed.modelType).toBe("t5_encoder_model");
    expect(parsed.vocabSize).toBe(32128);
    expect(parsed.dModel).toBe(4096);
    expect(parsed.dKv).toBe(64);
    expect(parsed.dFf).toBe(10240);
    expect(parsed.numLayers).toBe(24);
    expect(parsed.numHeads).toBe(64);
    expect(parsed.innerDim).toBe(4096);
    expect(parsed.feedForwardProjection).toBe("gated-gelu");
    expect(parsed.denseActivation).toBe("gelu_new");
    expect(parsed.isGatedActivation).toBe(true);
    expect(parsed.padTokenId).toBe(0);
    expect(parsed.eosTokenId).toBe(1);
  });

  test("parses defaults for small T5 encoder configs", () => {
    const parsed = parseT5EncoderConfig({});

    expect(parsed.vocabSize).toBe(32128);
    expect(parsed.dModel).toBe(512);
    expect(parsed.dKv).toBe(64);
    expect(parsed.dFf).toBe(2048);
    expect(parsed.numLayers).toBe(6);
    expect(parsed.numHeads).toBe(8);
    expect(parsed.innerDim).toBe(512);
    expect(parsed.feedForwardProjection).toBe("relu");
    expect(parsed.denseActivation).toBe("relu");
    expect(parsed.isGatedActivation).toBe(false);
  });

  test("parses supported feed-forward projection variants", () => {
    expect(parseT5EncoderConfig(t5Config({ feed_forward_proj: "relu" })).denseActivation).toBe(
      "relu",
    );
    expect(
      parseT5EncoderConfig(t5Config({ feed_forward_proj: "gated-silu" })).denseActivation,
    ).toBe("silu");
  });

  test("rejects unsupported T5 encoder config shapes", () => {
    expect(() => parseT5EncoderConfig([])).toThrow("JSON object");
    expect(() => parseT5EncoderConfig(t5Config({ model_type: "mt5" }))).toThrow(ConfigParseError);
    expect(() => parseT5EncoderConfig(t5Config({ is_decoder: true }))).toThrow(
      "is_decoder must be false",
    );
    expect(() => parseT5EncoderConfig(t5Config({ d_model: 0 }))).toThrow("must be positive");
    expect(() => parseT5EncoderConfig(t5Config({ num_layers: -1 }))).toThrow(
      "must be non-negative",
    );
    expect(() => parseT5EncoderConfig(t5Config({ feed_forward_proj: "gated-relu" }))).toThrow(
      "feed_forward_proj",
    );
  });
});
