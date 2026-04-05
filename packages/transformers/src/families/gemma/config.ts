/**
 * Gemma config parsing.
 * @module
 */

import {
  expectConfigRecord,
  expectInteger,
  expectString,
  optionalBoolean,
  optionalInteger,
  optionalNumber,
  optionalString,
} from "../../infrastructure/config-parsing";
import type { FamilyRegistration } from "../../types";
import { LlamaLikeCausalLM } from "../llama-like/model";
import type { LlamaLikeConfig } from "../llama-like/types";
import { isIgnoredGemmaWeight, sanitizeGemmaWeight } from "./weights";

export function parseGemmaConfig(rawConfig: Record<string, unknown>): LlamaLikeConfig {
  const config = expectConfigRecord(rawConfig, "Gemma config");
  const modelType = expectString(config, "model_type", "Gemma config");
  if (modelType !== "gemma") {
    throw new Error(`Gemma config.model_type must be "gemma", got "${modelType}".`);
  }

  const hiddenSize = expectInteger(config, "hidden_size", "Gemma config");
  const numAttentionHeads = expectInteger(config, "num_attention_heads", "Gemma config");
  const numKeyValueHeads =
    optionalInteger(config, "num_key_value_heads", "Gemma config") ?? numAttentionHeads;
  const hiddenAct = optionalString(config, "hidden_act", "Gemma config") ?? "gelu_pytorch_tanh";

  return {
    family: "gemma",
    modelType,
    rawConfig: config,
    vocabSize: expectInteger(config, "vocab_size", "Gemma config"),
    hiddenSize,
    intermediateSize: expectInteger(config, "intermediate_size", "Gemma config"),
    numHiddenLayers: expectInteger(config, "num_hidden_layers", "Gemma config"),
    numAttentionHeads,
    numKeyValueHeads,
    headDim:
      optionalInteger(config, "head_dim", "Gemma config") ??
      Math.floor(hiddenSize / numAttentionHeads),
    maxPositionEmbeddings: expectInteger(config, "max_position_embeddings", "Gemma config"),
    ropeTheta: optionalNumber(config, "rope_theta", "Gemma config") ?? 10000,
    rmsNormEps: optionalNumber(config, "rms_norm_eps", "Gemma config") ?? 1e-6,
    tieWordEmbeddings: optionalBoolean(config, "tie_word_embeddings", "Gemma config") ?? true,
    attentionBias: optionalBoolean(config, "attention_bias", "Gemma config") ?? false,
    embeddingScale: Math.sqrt(hiddenSize),
    normWeightOffset: true,
    mlpActivation: hiddenAct === "gelu_pytorch_tanh" ? "gelu_pytorch_tanh" : "swiglu",
  };
}

export const gemmaFamily: FamilyRegistration<LlamaLikeConfig> = {
  family: "gemma",
  modelTypes: ["gemma"],
  parseConfig: parseGemmaConfig,
  createModel: (config) => new LlamaLikeCausalLM(config),
  sanitizeWeight: sanitizeGemmaWeight,
  isIgnoredWeight: isIgnoredGemmaWeight,
};
