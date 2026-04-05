import { mxEval } from "@mlxts/core";
import type { PreferenceBatch } from "@mlxts/data";
import { Module, valueAndGrad } from "@mlxts/nn";
import { applyGradientStep, materializeTrainingState } from "@mlxts/train";
import type { CausalLM } from "@mlxts/transformers";
import type { DPOTrainOptions } from "./dpo-types";
import { directPreferenceLoss } from "./loss-utils";
import type { TrainableCausalLM } from "./sft-types";

function assertTrainableModel(model: CausalLM): asserts model is TrainableCausalLM {
  if (!(model instanceof Module)) {
    throw new Error("align.dpoTrain: policy model must also be an nn.Module.");
  }
}

function freePreferenceBatch(batch: PreferenceBatch): void {
  batch.chosen.inputIds.free();
  batch.chosen.targetIds.free();
  batch.chosen.lossMask.free();
  batch.rejected.inputIds.free();
  batch.rejected.targetIds.free();
  batch.rejected.lossMask.free();
}

/** DPO loss for one preference batch. */
export function dpoLoss(
  policyModel: CausalLM,
  referenceModel: CausalLM,
  batch: PreferenceBatch,
  beta = 0.1,
) {
  return directPreferenceLoss(policyModel, referenceModel, batch, beta);
}

/** Run one DPO optimizer step over one or more micro-batches. */
export function dpoTrain(policyModel: CausalLM, options: DPOTrainOptions): { averageLoss: number } {
  assertTrainableModel(policyModel);
  if (options.batches.length === 0) {
    throw new Error("align.dpoTrain: expected at least one batch.");
  }

  if (options.learningRate !== undefined) {
    options.optimizer.setLearningRate?.(options.learningRate);
  }

  let batchIndex = 0;
  using lossAndGrad = valueAndGrad(
    policyModel,
    (
      chosenInputIds,
      chosenTargetIds,
      chosenLossMask,
      rejectedInputIds,
      rejectedTargetIds,
      rejectedLossMask,
    ) =>
      dpoLoss(
        policyModel,
        options.referenceModel,
        {
          chosen: {
            inputIds: chosenInputIds,
            targetIds: chosenTargetIds,
            lossMask: chosenLossMask,
          },
          rejected: {
            inputIds: rejectedInputIds,
            targetIds: rejectedTargetIds,
            lossMask: rejectedLossMask,
          },
        },
        options.beta ?? 0.1,
      ),
  );

  return applyGradientStep({
    gradAccumSteps: options.batches.length,
    maxGradNorm: options.maxGradNorm ?? null,
    takeMicroStep: () => {
      const batch = options.batches[batchIndex];
      batchIndex += 1;
      if (batch === undefined) {
        throw new Error("align.dpoTrain: missing micro-batch.");
      }

      try {
        const [loss, gradients] = lossAndGrad(
          batch.chosen.inputIds,
          batch.chosen.targetIds,
          batch.chosen.lossMask,
          batch.rejected.inputIds,
          batch.rejected.targetIds,
          batch.rejected.lossMask,
        );
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
        freePreferenceBatch(batch);
      }
    },
    applyGradients: (gradients) => {
      options.optimizer.update(policyModel, gradients);
    },
    materialize: () => {
      materializeTrainingState(policyModel, options.optimizer);
    },
  });
}
