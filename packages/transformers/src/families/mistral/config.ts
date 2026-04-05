/**
 * Mistral config parsing.
 * @module
 */

import {
  expectConfigRecord,
  expectInteger,
  expectString,
  optionalBoolean,
  optionalInteger,
  optionalNumber,
} from "../../infrastructure/config-parsing";
import type { FamilyRegistration } from "../../types";
import { LlamaLikeCausalLM } from "../llama-like/model";
import type { LlamaLikeConfig } from "../llama-like/types";
import { isIgnoredMistralWeight, sanitizeMistralWeight } from "./weights";

export function parseMistralConfig(rawConfig: Record<string, unknown>): LlamaLikeConfig {
  const config = expectConfigRecord(rawConfig, "Mistral config");
  const modelType = expectString(config, "model_type", "Mistral config");
  if (modelType !== "mistral") {
    throw new Error(`Mistral config.model_type must be "mistral", got "${modelType}".`);
  }

  const hiddenSize = expectInteger(config, "hidden_size", "Mistral config");
  const numAttentionHeads = expectInteger(config, "num_attention_heads", "Mistral config");
  const numKeyValueHeads =
    optionalInteger(config, "num_key_value_heads", "Mistral config") ?? numAttentionHeads;

  const parsed: LlamaLikeConfig = {
    family: "mistral",
    modelType,
    rawConfig: config,
    vocabSize: expectInteger(config, "vocab_size", "Mistral config"),
    hiddenSize,
    intermediateSize: expectInteger(config, "intermediate_size", "Mistral config"),
    numHiddenLayers: expectInteger(config, "num_hidden_layers", "Mistral config"),
    numAttentionHeads,
    numKeyValueHeads,
    headDim: Math.floor(hiddenSize / numAttentionHeads),
    maxPositionEmbeddings: expectInteger(config, "max_position_embeddings", "Mistral config"),
    ropeTheta: optionalNumber(config, "rope_theta", "Mistral config") ?? 10000,
    rmsNormEps: optionalNumber(config, "rms_norm_eps", "Mistral config") ?? 1e-6,
    tieWordEmbeddings: optionalBoolean(config, "tie_word_embeddings", "Mistral config") ?? false,
    attentionBias: optionalBoolean(config, "attention_bias", "Mistral config") ?? false,
    mlpActivation: "swiglu",
  };

  const slidingWindow = optionalInteger(config, "sliding_window", "Mistral config");
  if (slidingWindow !== undefined) {
    parsed.slidingWindow = slidingWindow;
  }
  return parsed;
}

export const mistralFamily: FamilyRegistration<LlamaLikeConfig> = {
  family: "mistral",
  modelTypes: ["mistral"],
  parseConfig: parseMistralConfig,
  createModel: (config) => new LlamaLikeCausalLM(config),
  sanitizeWeight: sanitizeMistralWeight,
  isIgnoredWeight: isIgnoredMistralWeight,
};
