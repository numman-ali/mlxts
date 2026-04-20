import type { Tokenizer } from "@mlxts/tokenizers";

import type { CausalLM, GenerationDefaults, GenerationOptions } from "../../types";

function readNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function readEosTokenIds(value: unknown): number[] | undefined {
  if (typeof value === "number" && Number.isInteger(value)) {
    return [value];
  }

  if (!Array.isArray(value)) {
    return undefined;
  }

  const tokenIds: number[] = [];
  for (const entry of value) {
    if (typeof entry === "number" && Number.isInteger(entry)) {
      tokenIds.push(entry);
    }
  }
  return tokenIds.length === 0 ? undefined : tokenIds;
}

function mergedEosTokenIds(
  explicit: readonly number[] | undefined,
  modelDefaults: readonly number[] | undefined,
  tokenizer: Tokenizer | undefined,
): number[] | undefined {
  if (explicit !== undefined) {
    return [...explicit];
  }

  const merged = [...new Set([...(modelDefaults ?? []), ...(tokenizer?.eosTokenIds ?? [])])];
  return merged.length === 0 ? undefined : merged;
}

/** Parse generation defaults from a checkpoint `generation_config.json` payload. */
export function parseGenerationDefaults(
  config: Record<string, unknown>,
): GenerationDefaults | undefined {
  const defaults: GenerationDefaults = {};
  const doSample = config.do_sample;

  if (doSample === false) {
    defaults.temperature = 0;
  } else {
    const temperature = readNumber(config.temperature);
    if (temperature !== undefined) {
      defaults.temperature = temperature;
    }
  }

  const topK = readNumber(config.top_k);
  if (topK !== undefined) {
    defaults.topK = topK;
  }

  const topP = readNumber(config.top_p);
  if (topP !== undefined) {
    defaults.topP = topP;
  }

  const minP = readNumber(config.min_p);
  if (minP !== undefined) {
    defaults.minP = minP;
  }

  const repetitionPenalty = readNumber(config.repetition_penalty);
  if (repetitionPenalty !== undefined) {
    defaults.repetitionPenalty = repetitionPenalty;
  }

  const eosTokenIds = readEosTokenIds(config.eos_token_id);
  if (eosTokenIds !== undefined) {
    defaults.eosTokenIds = eosTokenIds;
  }

  return Object.keys(defaults).length === 0 ? undefined : defaults;
}

/** Merge explicit generation options with checkpoint defaults and tokenizer EOS ids. */
export function resolveGenerationOptions(
  model: CausalLM,
  tokenizer: Tokenizer | undefined,
  options: GenerationOptions,
): GenerationOptions {
  const modelDefaults = model.config.generationDefaults;
  const eosTokenIds = mergedEosTokenIds(options.eosTokenIds, modelDefaults?.eosTokenIds, tokenizer);
  return {
    ...modelDefaults,
    ...options,
    ...(eosTokenIds === undefined ? {} : { eosTokenIds }),
  };
}
