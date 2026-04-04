/**
 * Reusable step-level training orchestration.
 *
 * @module
 */

import type { MxArray, ParameterTree } from "@mlxts/core";
import { mxEval, synchronize, treeLeaves } from "@mlxts/core";

import type { ParameterizedModel } from "./checkpoint-types";
import {
  accumulateGradients,
  clipGradientTree,
  freeGradientTree,
  scaleGradientTree,
} from "./gradients";

/** Result of one caller-owned micro-step. */
export interface GradientMicroStepResult {
  lossValue: number;
  gradients: ParameterTree;
}

/** Minimal optimizer surface needed for post-step materialization. */
export interface OptimizerStateOwner {
  stateArrays(): MxArray[];
}

/** Generic gradient-step orchestration inputs. */
export interface ApplyGradientStepOptions {
  gradAccumSteps: number;
  maxGradNorm: number | null;
  takeMicroStep: () => GradientMicroStepResult;
  applyGradients: (gradients: ParameterTree) => void;
  materialize?: () => void;
}

/**
 * Materialize updated model parameters and optimizer state.
 *
 * This keeps lazy MLX work explicit at the package boundary after each step.
 */
export function materializeTrainingState(
  model: ParameterizedModel,
  optimizer: OptimizerStateOwner,
): void {
  const arrays = [...treeLeaves(model.parameters()), ...optimizer.stateArrays()];
  if (arrays.length > 0) {
    mxEval(...arrays);
  }
  synchronize();
}

function mergeAccumulatedGradients(
  accumulated: ParameterTree | null,
  gradients: ParameterTree,
): ParameterTree {
  if (accumulated === null) {
    return gradients;
  }

  try {
    const combined = accumulateGradients(accumulated, gradients);
    freeGradientTree(accumulated);
    freeGradientTree(gradients);
    return combined;
  } catch (error) {
    freeGradientTree(accumulated);
    freeGradientTree(gradients);
    throw error;
  }
}

function normalizePreparedGradients(
  gradients: ParameterTree,
  gradAccumSteps: number,
  maxGradNorm: number | null,
): ParameterTree {
  let prepared = gradients;

  if (gradAccumSteps > 1) {
    const scaled = scaleGradientTree(prepared, 1 / gradAccumSteps);
    freeGradientTree(prepared);
    prepared = scaled;
  }

  const clipped = clipGradientTree(prepared, maxGradNorm);
  if (clipped !== prepared) {
    freeGradientTree(prepared);
    prepared = clipped;
  }

  return prepared;
}

/**
 * Run one optimizer step with gradient accumulation and optional clipping.
 *
 * The caller owns micro-step creation and finite-value checks; this helper owns
 * tree combination, scaling, clipping, update application, and cleanup.
 */
export function applyGradientStep(options: ApplyGradientStepOptions): { averageLoss: number } {
  const { gradAccumSteps, maxGradNorm, takeMicroStep, applyGradients, materialize } = options;
  if (!Number.isInteger(gradAccumSteps) || gradAccumSteps <= 0) {
    throw new Error("train.applyGradientStep: gradAccumSteps must be a positive integer");
  }

  let accumulated: ParameterTree | null = null;
  let prepared: ParameterTree | null = null;
  let totalLoss = 0;

  try {
    for (let microStep = 0; microStep < gradAccumSteps; microStep++) {
      const { lossValue, gradients } = takeMicroStep();
      totalLoss += lossValue;
      accumulated = mergeAccumulatedGradients(accumulated, gradients);
    }

    if (accumulated === null) {
      throw new Error("train.applyGradientStep: no gradients were produced");
    }

    prepared = accumulated;
    accumulated = null;
    prepared = normalizePreparedGradients(prepared, gradAccumSteps, maxGradNorm);

    try {
      applyGradients(prepared);
    } finally {
      freeGradientTree(prepared);
      prepared = null;
    }

    materialize?.();
    return { averageLoss: totalLoss / gradAccumSteps };
  } catch (error) {
    if (accumulated !== null) {
      freeGradientTree(accumulated);
    }
    if (prepared !== null) {
      freeGradientTree(prepared);
    }
    throw error;
  }
}
