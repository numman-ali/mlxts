import { describe, expect, test } from "bun:test";
import { collatePreferenceBatch } from "@mlxts/data";
import { Adam } from "@mlxts/optimizers";
import { resolveFamily } from "@mlxts/transformers";

import { dpoTrain } from "./dpo";

describe("dpoTrain", () => {
  test("runs one DPO step against a reference model", () => {
    const registration = resolveFamily("llama");
    const policyModel = registration.createModel(
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
    const referenceModel = registration.createModel(
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

    const batch = collatePreferenceBatch(
      [
        {
          promptIds: [1, 2],
          chosenIds: [3, 4],
          rejectedIds: [5, 6],
        },
      ],
      0,
    );
    try {
      const result = dpoTrain(policyModel, {
        referenceModel,
        optimizer,
        batches: [batch],
        learningRate: 1e-3,
        beta: 0.1,
      });

      expect(result.averageLoss).toBeGreaterThan(0);
    } finally {
      policyModel[Symbol.dispose]();
      referenceModel[Symbol.dispose]();
    }
  });
});
