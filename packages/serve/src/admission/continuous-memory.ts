/**
 * Memory reservation inputs for continuous transformer scheduler admission.
 * @module
 */

import type { CausalLM, ContinuousBatchAdmissionRequest } from "@mlxts/transformers";
import { readGenerationMemoryUsage } from "../runtime/memory";
import { estimateGenerationMemory } from "../runtime/model-context";
import type { ServeRuntimeStrategy } from "../runtime/strategy";

export type ContinuousSchedulerMemoryBudgetOptions = {
  maxScheduledMemoryBytes: number;
  estimateMemoryBytes(request: ContinuousBatchAdmissionRequest): number | undefined;
};

export function continuousSchedulerMemoryBudgetOptions(
  model: CausalLM,
  strategy: ServeRuntimeStrategy,
): ContinuousSchedulerMemoryBudgetOptions | undefined {
  if (strategy.memory.policy === "none") {
    return undefined;
  }
  const memory = readGenerationMemoryUsage();
  if (memory === undefined) {
    return undefined;
  }
  const maxScheduledMemoryBytes = Math.max(
    0,
    Math.floor(memory.limitBytes * strategy.memory.gpuMemoryUtilization) - memory.activeBytes,
  );
  return {
    maxScheduledMemoryBytes,
    estimateMemoryBytes(request) {
      return estimateGenerationMemory(model, {
        promptTokens: request.promptTokens,
        totalTokens: request.totalTokens,
        prefillStepSize: strategy.scheduler.prefillStepSize,
      })?.totalBytes;
    },
  };
}
