import { mxEval } from "@mlxts/core";
import type { TokenBatch } from "@mlxts/data";
import { Module, valueAndGrad } from "@mlxts/nn";
import { applyGradientStep, materializeTrainingState } from "@mlxts/train";
import type { CausalLM } from "@mlxts/transformers";

import { maskedTokenLoss } from "./loss-utils";
import type { SFTTrainOptions, TrainableCausalLM } from "./sft-types";

function assertTrainableModel(model: CausalLM): asserts model is TrainableCausalLM {
  if (!(model instanceof Module)) {
    throw new Error("align.sftTrain: model must also be an nn.Module.");
  }
}

function freeBatch(batch: TokenBatch): void {
  batch.inputIds.free();
  batch.targetIds.free();
  batch.lossMask.free();
}

/** SFT loss for one padded token batch. */
export function sftLoss(model: CausalLM, batch: TokenBatch) {
  using logits = model.forward(batch.inputIds);
  return maskedTokenLoss(logits, batch);
}

/** Run one supervised fine-tuning optimizer step over one or more micro-batches. */
export function sftTrain(model: CausalLM, options: SFTTrainOptions): { averageLoss: number } {
  assertTrainableModel(model);
  if (options.batches.length === 0) {
    throw new Error("align.sftTrain: expected at least one batch.");
  }

  if (options.learningRate !== undefined) {
    options.optimizer.setLearningRate?.(options.learningRate);
  }

  let batchIndex = 0;
  using lossAndGrad = valueAndGrad(model, (inputIds, targetIds, lossMask) =>
    sftLoss(model, {
      inputIds,
      targetIds,
      lossMask,
    }),
  );

  return applyGradientStep({
    gradAccumSteps: options.batches.length,
    maxGradNorm: options.maxGradNorm ?? null,
    takeMicroStep: () => {
      const batch = options.batches[batchIndex];
      batchIndex += 1;
      if (batch === undefined) {
        throw new Error("align.sftTrain: missing micro-batch.");
      }

      try {
        const [loss, gradients] = lossAndGrad(batch.inputIds, batch.targetIds, batch.lossMask);
        try {
          mxEval(loss);
          return {
            lossValue: loss.item(),
            gradients,
          };
        } finally {
          loss.free();
        }
      } finally {
        freeBatch(batch);
      }
    },
    applyGradients: (gradients) => {
      options.optimizer.update(model, gradients);
    },
    materialize: () => {
      materializeTrainingState(model, options.optimizer);
    },
  });
}
