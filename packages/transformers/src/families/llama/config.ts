/**
 * LLaMA family config parsing.
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
import { isIgnoredLlamaWeight, sanitizeLlamaWeight } from "./weights";

export function parseLlamaConfig(rawConfig: Record<string, unknown>): LlamaLikeConfig {
  const config = expectConfigRecord(rawConfig, "LLaMA config");
  const modelType = expectString(config, "model_type", "LLaMA config");
  if (modelType !== "llama") {
    throw new Error(`LLaMA config.model_type must be "llama", got "${modelType}".`);
  }

  const hiddenSize = expectInteger(config, "hidden_size", "LLaMA config");
  const numAttentionHeads = expectInteger(config, "num_attention_heads", "LLaMA config");
  const numKeyValueHeads =
    optionalInteger(config, "num_key_value_heads", "LLaMA config") ?? numAttentionHeads;

  return {
    family: "llama",
    modelType,
    rawConfig: config,
    vocabSize: expectInteger(config, "vocab_size", "LLaMA config"),
    hiddenSize,
    intermediateSize: expectInteger(config, "intermediate_size", "LLaMA config"),
    numHiddenLayers: expectInteger(config, "num_hidden_layers", "LLaMA config"),
    numAttentionHeads,
    numKeyValueHeads,
    headDim: Math.floor(hiddenSize / numAttentionHeads),
    maxPositionEmbeddings: expectInteger(config, "max_position_embeddings", "LLaMA config"),
    ropeTheta: optionalNumber(config, "rope_theta", "LLaMA config") ?? 10000,
    rmsNormEps: optionalNumber(config, "rms_norm_eps", "LLaMA config") ?? 1e-6,
    tieWordEmbeddings: optionalBoolean(config, "tie_word_embeddings", "LLaMA config") ?? false,
    attentionBias: optionalBoolean(config, "attention_bias", "LLaMA config") ?? false,
    mlpActivation: "swiglu",
  };
}

export const llamaFamily: FamilyRegistration<LlamaLikeConfig> = {
  family: "llama",
  modelTypes: ["llama"],
  parseConfig: parseLlamaConfig,
  createModel: (config) => new LlamaLikeCausalLM(config),
  sanitizeWeight: sanitizeLlamaWeight,
  isIgnoredWeight: isIgnoredLlamaWeight,
};
