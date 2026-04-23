import { describe, expect, test } from "bun:test";
import { Adam } from "@mlxts/optimizers";
import { resolveFamily } from "@mlxts/transformers";

import {
  evaluatePreferenceDatasetLoss,
  evaluatePreferenceMetrics,
  evaluateSupervisionDatasetLoss,
  runPreferenceTrainingSteps,
  runSupervisionTrainingSteps,
} from "./recipes";

function createTinyLlama() {
  const registration = resolveFamily("llama");
  return registration.createModel(
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
}

describe("align recipes", () => {
  test("evaluates and trains supervision examples", () => {
    using model = createTinyLlama();
    using optimizer = new Adam({ learningRate: 1e-3 });
    const examples = [
      {
        inputIds: [1, 2, 3],
        targetIds: [2, 3, 4],
        lossMask: [0, 1, 1],
      },
      {
        inputIds: [2, 3, 4],
        targetIds: [3, 4, 5],
        lossMask: [0, 1, 1],
      },
    ] as const;

    const evalLoss = evaluateSupervisionDatasetLoss(model, {
      examples,
      padTokenId: 0,
      batchSize: 2,
    });
    const result = runSupervisionTrainingSteps(model, {
      optimizer,
      examples,
      padTokenId: 0,
      batchSize: 1,
      steps: 2,
      seed: 7,
      learningRate: 1e-3,
      maxGradNorm: 1,
    });

    expect(evalLoss).toBeGreaterThan(0);
    expect(result.averageLoss).toBeGreaterThan(0);
  });

  test("evaluates preference metrics with a reference-aware zero point", () => {
    using policyModel = createTinyLlama();
    const examples = [
      {
        promptIds: [1, 2],
        chosenIds: [3, 4],
        rejectedIds: [5, 6],
      },
    ] as const;

    const metrics = evaluatePreferenceMetrics(policyModel, {
      referenceModel: policyModel,
      examples,
      padTokenId: 0,
      batchSize: 1,
      beta: 0.1,
    });

    expect(metrics.rewardAccuracy).toBe(0);
    expect(metrics.rewardMargin).toBe(0);
    expect(metrics.chosenReward).toBe(0);
    expect(metrics.rejectedReward).toBe(0);
    expect(metrics.rawPreferenceAccuracy).toBeGreaterThanOrEqual(0);
  });

  test("evaluates and trains preference examples", () => {
    using policyModel = createTinyLlama();
    using referenceModel = createTinyLlama();
    using optimizer = new Adam({ learningRate: 1e-3 });
    const examples = [
      {
        promptIds: [1, 2],
        chosenIds: [3, 4],
        rejectedIds: [5, 6],
      },
      {
        promptIds: [2, 3],
        chosenIds: [4, 5],
        rejectedIds: [6, 7],
      },
    ] as const;

    const evalLoss = evaluatePreferenceDatasetLoss(policyModel, {
      referenceModel,
      examples,
      padTokenId: 0,
      batchSize: 2,
    });
    const result = runPreferenceTrainingSteps(policyModel, {
      referenceModel,
      optimizer,
      examples,
      padTokenId: 0,
      batchSize: 1,
      steps: 2,
      seed: 11,
      beta: 0.1,
      learningRate: 1e-3,
      maxGradNorm: 1,
    });

    expect(evalLoss).toBeGreaterThan(0);
    expect(result.averageLoss).toBeGreaterThan(0);
  });

  test("rejects empty supervision datasets", () => {
    using model = createTinyLlama();

    expect(() =>
      evaluateSupervisionDatasetLoss(model, {
        examples: [],
        padTokenId: 0,
        batchSize: 1,
      }),
    ).toThrow("expected at least one example");
  });

  test("rejects invalid batch sizing and step counts", () => {
    using model = createTinyLlama();
    using optimizer = new Adam({ learningRate: 1e-3 });
    const preferenceExamples = [
      {
        promptIds: [1, 2],
        chosenIds: [3, 4],
        rejectedIds: [5, 6],
      },
    ] as const;

    expect(() =>
      evaluateSupervisionDatasetLoss(model, {
        examples: [
          {
            inputIds: [1, 2, 3],
            targetIds: [2, 3, 4],
            lossMask: [0, 1, 1],
          },
        ],
        padTokenId: 0,
        batchSize: 0,
      }),
    ).toThrow("batchSize must be a positive integer");

    expect(() =>
      evaluatePreferenceDatasetLoss(model, {
        referenceModel: model,
        examples: [],
        padTokenId: 0,
        batchSize: 1,
      }),
    ).toThrow("expected at least one example");

    expect(() =>
      runPreferenceTrainingSteps(model, {
        referenceModel: model,
        optimizer,
        examples: preferenceExamples,
        padTokenId: 0,
        batchSize: 1,
        steps: 0,
        seed: 13,
      }),
    ).toThrow("steps must be a positive integer");
  });
});
