/**
 * CLIP text encoder config parsing.
 * @module
 */

import {
  expectConfigRecord,
  expectInteger,
  optionalInteger,
  optionalNumber,
  optionalString,
} from "../../infrastructure/config-parsing";
import { ConfigParseError } from "../../types";
import type { CLIPHiddenActivation, CLIPTextConfig } from "./types";

function positiveInteger(
  record: Record<string, unknown>,
  key: string,
  context: string,
  fallback?: number,
): number {
  const value =
    fallback === undefined
      ? expectInteger(record, key, context)
      : (optionalInteger(record, key, context) ?? fallback);
  if (value <= 0) {
    throw new ConfigParseError(`${context}.${key} must be positive, got ${value}.`);
  }
  return value;
}

function nonNegativeInteger(
  record: Record<string, unknown>,
  key: string,
  context: string,
  fallback: number,
): number {
  const value = optionalInteger(record, key, context) ?? fallback;
  if (value < 0) {
    throw new ConfigParseError(`${context}.${key} must be non-negative, got ${value}.`);
  }
  return value;
}

function parseNullableInteger(
  record: Record<string, unknown>,
  key: string,
  context: string,
  fallback: number | null,
): number | null {
  const value = record[key];
  if (value === undefined) {
    return fallback;
  }
  if (value === null) {
    return null;
  }
  if (typeof value !== "number" || !Number.isInteger(value)) {
    throw new ConfigParseError(`${context}.${key} must be an integer or null when present.`);
  }
  return value;
}

function parseEosTokenId(record: Record<string, unknown>, context: string): number | null {
  const value = record.eos_token_id;
  if (value === undefined) {
    return 49407;
  }
  if (value === null) {
    return null;
  }
  if (Array.isArray(value)) {
    throw new ConfigParseError(
      `${context}.eos_token_id list values are not supported for CLIP text.`,
    );
  }
  if (typeof value !== "number" || !Number.isInteger(value)) {
    throw new ConfigParseError(`${context}.eos_token_id must be an integer or null when present.`);
  }
  return value;
}

function parseHiddenActivation(
  record: Record<string, unknown>,
  context: string,
): CLIPHiddenActivation {
  const hiddenAct = optionalString(record, "hidden_act", context) ?? "quick_gelu";
  if (hiddenAct !== "quick_gelu" && hiddenAct !== "gelu") {
    throw new ConfigParseError(
      `${context}.hidden_act must be "quick_gelu" or "gelu", got "${hiddenAct}".`,
    );
  }
  return hiddenAct;
}

function parseSourceConfig(
  rawConfig: Record<string, unknown>,
  context: string,
): Record<string, unknown> {
  const modelType = optionalString(rawConfig, "model_type", context);
  if (modelType === "clip") {
    return expectConfigRecord(rawConfig.text_config, `${context}.text_config`);
  }
  if (modelType !== undefined && modelType !== "clip_text_model") {
    throw new ConfigParseError(
      `${context}.model_type must be "clip_text_model" or "clip", got "${modelType}".`,
    );
  }
  return rawConfig;
}

/** Parse a Hugging Face CLIP text encoder config. */
export function parseCLIPTextConfig(
  rawConfig: Record<string, unknown>,
  context = "CLIP text config",
): CLIPTextConfig {
  const sourceConfig = parseSourceConfig(rawConfig, context);
  const hiddenSize = positiveInteger(sourceConfig, "hidden_size", context, 512);
  const numAttentionHeads = positiveInteger(sourceConfig, "num_attention_heads", context, 8);
  if (hiddenSize % numAttentionHeads !== 0) {
    throw new ConfigParseError(
      `${context}.hidden_size (${hiddenSize}) must be divisible by num_attention_heads (${numAttentionHeads}).`,
    );
  }

  const attentionDropout = optionalNumber(sourceConfig, "attention_dropout", context) ?? 0;
  if (attentionDropout !== 0) {
    throw new ConfigParseError(
      `${context}.attention_dropout must be 0 for CLIP text inference, got ${attentionDropout}.`,
    );
  }

  return {
    modelType: "clip_text_model",
    rawConfig,
    vocabSize: positiveInteger(sourceConfig, "vocab_size", context, 49408),
    hiddenSize,
    intermediateSize: positiveInteger(sourceConfig, "intermediate_size", context, 2048),
    projectionDim: parseNullableInteger(sourceConfig, "projection_dim", context, 512),
    numHiddenLayers: nonNegativeInteger(sourceConfig, "num_hidden_layers", context, 12),
    numAttentionHeads,
    headDim: hiddenSize / numAttentionHeads,
    maxPositionEmbeddings: positiveInteger(sourceConfig, "max_position_embeddings", context, 77),
    hiddenAct: parseHiddenActivation(sourceConfig, context),
    layerNormEps: optionalNumber(sourceConfig, "layer_norm_eps", context) ?? 1e-5,
    attentionDropout,
    padTokenId: parseNullableInteger(sourceConfig, "pad_token_id", context, 1),
    bosTokenId: parseNullableInteger(sourceConfig, "bos_token_id", context, 49406),
    eosTokenId: parseEosTokenId(sourceConfig, context),
  };
}
