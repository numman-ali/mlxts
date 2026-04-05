/**
 * Phi-3 config parsing.
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
import { isIgnoredPhiWeight, sanitizePhiWeight } from "./weights";

export function parsePhiConfig(rawConfig: Record<string, unknown>): LlamaLikeConfig {
  const config = expectConfigRecord(rawConfig, "Phi config");
  const modelType = expectString(config, "model_type", "Phi config");
  if (modelType !== "phi3") {
    throw new Error(`Phi config.model_type must be "phi3", got "${modelType}".`);
  }

  const hiddenSize = expectInteger(config, "hidden_size", "Phi config");
  const numAttentionHeads = expectInteger(config, "num_attention_heads", "Phi config");
  const numKeyValueHeads =
    optionalInteger(config, "num_key_value_heads", "Phi config") ?? numAttentionHeads;
  const headDim =
    optionalInteger(config, "head_dim", "Phi config") ?? Math.floor(hiddenSize / numAttentionHeads);
  const partialRotaryFactor = optionalNumber(config, "partial_rotary_factor", "Phi config") ?? 1.0;
  if (
    !Number.isFinite(partialRotaryFactor) ||
    partialRotaryFactor <= 0 ||
    partialRotaryFactor > 1
  ) {
    throw new ConfigParseError(
      `Phi config.partial_rotary_factor must be a number in the range (0, 1], got ${partialRotaryFactor}.`,
    );
  }
  const rotaryDimensions = Math.floor(headDim * partialRotaryFactor);
  if (rotaryDimensions <= 0 || rotaryDimensions % 2 !== 0) {
    throw new ConfigParseError(
      `Phi config.partial_rotary_factor yields ${rotaryDimensions} rotary dimensions, expected a positive even integer.`,
    );
  }

  const hiddenAct = optionalString(config, "hidden_act", "Phi config") ?? "silu";
  if (hiddenAct !== "silu") {
    throw new ConfigParseError(
      `Phi config.hidden_act must be "silu" for the supported dense Phi-3 path, got "${hiddenAct}".`,
    );
  }

  const parsed: LlamaLikeConfig = {
    family: "phi",
    modelType,
    rawConfig: config,
    vocabSize: expectInteger(config, "vocab_size", "Phi config"),
    hiddenSize,
    intermediateSize: expectInteger(config, "intermediate_size", "Phi config"),
    numHiddenLayers: expectInteger(config, "num_hidden_layers", "Phi config"),
    numAttentionHeads,
    numKeyValueHeads,
    headDim,
    maxPositionEmbeddings: expectInteger(config, "max_position_embeddings", "Phi config"),
    ropeTheta: optionalNumber(config, "rope_theta", "Phi config") ?? 10000,
    rmsNormEps: optionalNumber(config, "rms_norm_eps", "Phi config") ?? 1e-5,
    tieWordEmbeddings: optionalBoolean(config, "tie_word_embeddings", "Phi config") ?? false,
    attentionBias: optionalBoolean(config, "attention_bias", "Phi config") ?? false,
    rotaryDimensions,
    attentionProjectionLayout: "packed_qkv",
    mlpProjectionLayout: "packed_gate_up",
    mlpActivation: "swiglu",
  };
  const slidingWindow = optionalInteger(config, "sliding_window", "Phi config");
  if (slidingWindow !== undefined) {
    parsed.slidingWindow = slidingWindow;
  }
  return parsed;
}

export const phiFamily: FamilyRegistration<LlamaLikeConfig> = {
  family: "phi",
  modelTypes: ["phi3"],
  parseConfig: parsePhiConfig,
  createModel: (config) => new LlamaLikeCausalLM(config),
  sanitizeWeight: sanitizePhiWeight,
  isIgnoredWeight: isIgnoredPhiWeight,
};
