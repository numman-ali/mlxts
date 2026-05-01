/**
 * Qwen3 dense text config parsing.
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
import { ConfigParseError } from "../../types";
import { LlamaLikeCausalLM } from "../llama-like/model";
import type { LlamaLikeConfig } from "../llama-like/types";
import { isIgnoredQwen3Weight, sanitizeQwen3Weight } from "./weights";

/** Parse a dense Qwen3 text config into the shared LLaMA-like runtime shape. */
export function parseQwen3Config(rawConfig: Record<string, unknown>): LlamaLikeConfig {
  const config = expectConfigRecord(rawConfig, "Qwen3 config");
  const modelType = expectString(config, "model_type", "Qwen3 config");
  if (modelType !== "qwen3") {
    throw new ConfigParseError(`Qwen3 config.model_type must be "qwen3", got "${modelType}".`);
  }

  const hiddenAct = optionalString(config, "hidden_act", "Qwen3 config") ?? "silu";
  if (hiddenAct !== "silu") {
    throw new ConfigParseError(`Qwen3 config.hidden_act must be "silu", got "${hiddenAct}".`);
  }

  const hiddenSize = expectInteger(config, "hidden_size", "Qwen3 config");
  const numAttentionHeads = expectInteger(config, "num_attention_heads", "Qwen3 config");
  const numKeyValueHeads =
    optionalInteger(config, "num_key_value_heads", "Qwen3 config") ?? numAttentionHeads;
  const headDim =
    optionalInteger(config, "head_dim", "Qwen3 config") ??
    Math.floor(hiddenSize / numAttentionHeads);

  const parsed: LlamaLikeConfig = {
    family: "qwen",
    modelType,
    rawConfig: config,
    vocabSize: expectInteger(config, "vocab_size", "Qwen3 config"),
    hiddenSize,
    intermediateSize: expectInteger(config, "intermediate_size", "Qwen3 config"),
    numHiddenLayers: expectInteger(config, "num_hidden_layers", "Qwen3 config"),
    numAttentionHeads,
    numKeyValueHeads,
    headDim,
    maxPositionEmbeddings: expectInteger(config, "max_position_embeddings", "Qwen3 config"),
    ropeTheta: optionalNumber(config, "rope_theta", "Qwen3 config") ?? 1_000_000,
    rmsNormEps: optionalNumber(config, "rms_norm_eps", "Qwen3 config") ?? 1e-6,
    tieWordEmbeddings: optionalBoolean(config, "tie_word_embeddings", "Qwen3 config") ?? true,
    attentionBias: optionalBoolean(config, "attention_bias", "Qwen3 config") ?? false,
    queryKeyNorm: true,
    mlpActivation: "swiglu",
  };

  const slidingWindow =
    config.sliding_window === null
      ? undefined
      : optionalInteger(config, "sliding_window", "Qwen3 config");
  if (slidingWindow !== undefined) {
    parsed.slidingWindow = slidingWindow;
  }
  return parsed;
}

/** Family registration for dense Qwen3 text checkpoints. */
export const qwen3Family: FamilyRegistration<LlamaLikeConfig> = {
  family: "qwen",
  modelTypes: ["qwen3"],
  parseConfig: parseQwen3Config,
  createModel: (config) => new LlamaLikeCausalLM(config),
  sanitizeWeight: sanitizeQwen3Weight,
  isIgnoredWeight: isIgnoredQwen3Weight,
};
