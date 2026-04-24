/**
 * Admission metadata derived from loaded transformer model configs.
 * @module
 */

import type { CausalLM } from "@mlxts/transformers";

export type ModelAdmissionMetadata = {
  contextWindow?: number;
  maxPromptTokens?: number;
  maxTotalTokens?: number;
  effectiveTotalTokens?: number;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function positiveIntegerField(record: Record<string, unknown>, key: string): number | undefined {
  const value = record[key];
  return Number.isInteger(value) && typeof value === "number" && value > 0 ? value : undefined;
}

/** Read the checkpoint-declared text context window when it is available. */
export function modelContextWindow(model: CausalLM): number | undefined {
  const rawConfig = model.config.rawConfig;
  return (
    positiveIntegerField(rawConfig, "max_position_embeddings") ??
    (isRecord(rawConfig.text_config)
      ? positiveIntegerField(rawConfig.text_config, "max_position_embeddings")
      : undefined)
  );
}

/** Return the total-token admission limit after server and checkpoint caps are combined. */
export function effectiveTotalTokenLimit(options: {
  maxTotalTokens?: number | undefined;
  contextWindow?: number | undefined;
}): number | undefined {
  const limits = [options.maxTotalTokens, options.contextWindow].filter(
    (limit): limit is number => limit !== undefined,
  );
  return limits.length === 0 ? undefined : Math.min(...limits);
}

/** Format model-level admission metadata for operator-facing introspection. */
export function modelAdmissionMetadata(
  model: CausalLM,
  limits: { maxPromptTokens?: number; maxTotalTokens?: number },
): ModelAdmissionMetadata {
  const contextWindow = modelContextWindow(model);
  const effectiveTotalTokens = effectiveTotalTokenLimit({
    maxTotalTokens: limits.maxTotalTokens,
    contextWindow,
  });
  return {
    ...(contextWindow === undefined ? {} : { contextWindow }),
    ...(limits.maxPromptTokens === undefined ? {} : { maxPromptTokens: limits.maxPromptTokens }),
    ...(limits.maxTotalTokens === undefined ? {} : { maxTotalTokens: limits.maxTotalTokens }),
    ...(effectiveTotalTokens === undefined ? {} : { effectiveTotalTokens }),
  };
}
