import { describe, expect, test } from "bun:test";

import { ConfigParseError } from "../../types";
import { parseCLIPTextConfig } from "./config";

describe("parseCLIPTextConfig", () => {
  test("parses a plain CLIP text config with defaults", () => {
    const config = parseCLIPTextConfig({
      model_type: "clip_text_model",
      vocab_size: 100,
      hidden_size: 8,
      intermediate_size: 16,
      num_hidden_layers: 2,
      num_attention_heads: 2,
      max_position_embeddings: 77,
    });

    expect(config.modelType).toBe("clip_text_model");
    expect(config.hiddenAct).toBe("quick_gelu");
    expect(config.headDim).toBe(4);
    expect(config.projectionDim).toBe(512);
    expect(config.eosTokenId).toBe(49407);
  });

  test("parses nested CLIP configs from composite config.json", () => {
    const config = parseCLIPTextConfig({
      model_type: "clip",
      text_config: {
        vocab_size: 100,
        hidden_size: 12,
        intermediate_size: 24,
        projection_dim: null,
        num_hidden_layers: 1,
        num_attention_heads: 3,
        max_position_embeddings: 32,
        hidden_act: "gelu",
        pad_token_id: null,
        bos_token_id: null,
        eos_token_id: null,
      },
    });

    expect(config.hiddenAct).toBe("gelu");
    expect(config.projectionDim).toBeNull();
    expect(config.padTokenId).toBeNull();
    expect(config.bosTokenId).toBeNull();
    expect(config.eosTokenId).toBeNull();
  });

  test("rejects unsupported activations and invalid attention geometry", () => {
    expect(() =>
      parseCLIPTextConfig({
        hidden_size: 10,
        num_attention_heads: 3,
      }),
    ).toThrow(ConfigParseError);
    expect(() =>
      parseCLIPTextConfig({
        hidden_act: "silu",
      }),
    ).toThrow('hidden_act must be "quick_gelu" or "gelu"');
  });

  test("rejects list-shaped eos token ids", () => {
    expect(() =>
      parseCLIPTextConfig({
        eos_token_id: [49407],
      }),
    ).toThrow("eos_token_id list values are not supported");
  });

  test("rejects nonzero attention dropout", () => {
    expect(() =>
      parseCLIPTextConfig({
        attention_dropout: 0.1,
      }),
    ).toThrow("attention_dropout must be 0 for CLIP text inference");
  });
});
