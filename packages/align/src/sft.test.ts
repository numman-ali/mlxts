import { describe, expect, test } from "bun:test";
import { collateTokenSupervisionBatch } from "@mlxts/data";
import { Adam } from "@mlxts/optimizers";
import { resolveFamily } from "@mlxts/transformers";

import { sftTrain } from "./sft";

describe("sftTrain", () => {
  test("runs one supervised fine-tuning step", () => {
    const registration = resolveFamily("llama");
    using model = registration.createModel(
      registration.parseConfig({
        model_type: "llama",
        vocab_size: 8,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 1,
        num_attention_heads: 4,
        num_key_value_heads: 4,
        max_position_embeddings: 64,
        rope_theta: 10000,
        rms_norm_eps: 1e-6,
        attention_bias: false,
        tie_word_embeddings: false,
      }),
    );
    using optimizer = new Adam({ learningRate: 1e-3 });

    const batch = collateTokenSupervisionBatch(
      [
        {
          inputIds: [1, 2, 3],
          targetIds: [2, 3, 4],
          lossMask: [0, 1, 1],
        },
      ],
      0,
    );
    const result = sftTrain(model, {
      optimizer,
      batches: [batch],
      learningRate: 1e-3,
    });

    expect(result.averageLoss).toBeGreaterThan(0);
  });

  test("rejects empty SFT batch lists", () => {
    const registration = resolveFamily("llama");
    using model = registration.createModel(
      registration.parseConfig({
        model_type: "llama",
        vocab_size: 8,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 1,
        num_attention_heads: 4,
        num_key_value_heads: 4,
        max_position_embeddings: 64,
        rope_theta: 10000,
        rms_norm_eps: 1e-6,
        attention_bias: false,
        tie_word_embeddings: false,
      }),
    );
    using optimizer = new Adam({ learningRate: 1e-3 });

    expect(() =>
      sftTrain(model, {
        optimizer,
        batches: [],
      }),
    ).toThrow("expected at least one batch");
  });
});
