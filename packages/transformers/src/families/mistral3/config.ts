/**
 * Mistral 3 text-decoder wrapper config parsing.
 * @module
 */

import { expectConfigRecord, expectString } from "../../infrastructure/config-parsing";
import type { FamilyRegistration } from "../../types";
import { ConfigParseError } from "../../types";
import { LlamaLikeCausalLM } from "../llama-like/model";
import type { LlamaLikeConfig } from "../llama-like/types";
import { parseMistralConfig } from "../mistral/config";
import { isIgnoredMistral3Weight, sanitizeMistral3Weight } from "./weights";

export function parseMistral3Config(rawConfig: Record<string, unknown>): LlamaLikeConfig {
  const config = expectConfigRecord(rawConfig, "Mistral 3 config");
  const modelType = expectString(config, "model_type", "Mistral 3 config");
  if (modelType !== "mistral3") {
    throw new Error(`Mistral 3 config.model_type must be "mistral3", got "${modelType}".`);
  }

  const textConfig = expectConfigRecord(config.text_config, "Mistral 3 config.text_config");
  const textModelType = expectString(textConfig, "model_type", "Mistral 3 config.text_config");
  if (textModelType !== "mistral") {
    throw new ConfigParseError(
      `Mistral 3 text_config.model_type must be "mistral" for the Phase 7 text-only path, got "${textModelType}".`,
    );
  }

  const parsed = parseMistralConfig(textConfig);
  return {
    ...parsed,
    family: "mistral",
    modelType: "mistral3",
    rawConfig: config,
  };
}

export const mistral3Family: FamilyRegistration<LlamaLikeConfig> = {
  family: "mistral",
  modelTypes: ["mistral3"],
  parseConfig: parseMistral3Config,
  createModel: (config) => new LlamaLikeCausalLM(config),
  sanitizeWeight: sanitizeMistral3Weight,
  isIgnoredWeight: isIgnoredMistral3Weight,
};
