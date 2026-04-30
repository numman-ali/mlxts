/**
 * T5 encoder config parsing.
 * @module
 */

import {
  expectConfigRecord,
  expectInteger,
  optionalBoolean,
  optionalInteger,
  optionalNumber,
  optionalString,
} from "../../infrastructure/config-parsing";
import { ConfigParseError } from "../../types";
import type { T5DenseActivation, T5EncoderConfig, T5FeedForwardProjection } from "./types";

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
    return 1;
  }
  if (value === null) {
    return null;
  }
  if (Array.isArray(value)) {
    const first = value[0];
    if (typeof first === "number" && Number.isInteger(first)) {
      return first;
    }
    throw new ConfigParseError(`${context}.eos_token_id list must start with an integer.`);
  }
  if (typeof value !== "number" || !Number.isInteger(value)) {
    throw new ConfigParseError(`${context}.eos_token_id must be an integer or null when present.`);
  }
  return value;
}

function parseFeedForwardProjection(
  record: Record<string, unknown>,
  context: string,
): {
  projection: T5FeedForwardProjection;
  activation: T5DenseActivation;
  gated: boolean;
} {
  const projection = optionalString(record, "feed_forward_proj", context) ?? "relu";
  if (projection === "relu") {
    return { projection, activation: "relu", gated: false };
  }
  if (projection === "gated-gelu") {
    return { projection, activation: "gelu_new", gated: true };
  }
  if (projection === "gated-silu") {
    return { projection, activation: "silu", gated: true };
  }

  throw new ConfigParseError(
    `${context}.feed_forward_proj must be "relu", "gated-gelu", or "gated-silu", got "${projection}".`,
  );
}

function assertEncoderShape(record: Record<string, unknown>, context: string): void {
  const modelType = optionalString(record, "model_type", context);
  if (modelType !== undefined && modelType !== "t5") {
    throw new ConfigParseError(`${context}.model_type must be "t5", got "${modelType}".`);
  }

  const isDecoder = optionalBoolean(record, "is_decoder", context) ?? false;
  if (isDecoder) {
    throw new ConfigParseError(`${context}.is_decoder must be false for T5EncoderModel.`);
  }
}

/** Parse a Hugging Face T5 encoder config. */
export function parseT5EncoderConfig(
  rawConfig: unknown,
  context = "T5 encoder config",
): T5EncoderConfig {
  const sourceConfig = expectConfigRecord(rawConfig, context);
  assertEncoderShape(sourceConfig, context);

  const dModel = positiveInteger(sourceConfig, "d_model", context, 512);
  const dKv = positiveInteger(sourceConfig, "d_kv", context, 64);
  const numHeads = positiveInteger(sourceConfig, "num_heads", context, 8);
  const projection = parseFeedForwardProjection(sourceConfig, context);
  const dropoutRate = optionalNumber(sourceConfig, "dropout_rate", context) ?? 0;

  return {
    modelType: "t5_encoder_model",
    rawConfig: sourceConfig,
    vocabSize: positiveInteger(sourceConfig, "vocab_size", context, 32128),
    dModel,
    dKv,
    dFf: positiveInteger(sourceConfig, "d_ff", context, 4 * dModel),
    numLayers: nonNegativeInteger(sourceConfig, "num_layers", context, 6),
    numHeads,
    innerDim: numHeads * dKv,
    relativeAttentionNumBuckets: positiveInteger(
      sourceConfig,
      "relative_attention_num_buckets",
      context,
      32,
    ),
    relativeAttentionMaxDistance: positiveInteger(
      sourceConfig,
      "relative_attention_max_distance",
      context,
      128,
    ),
    layerNormEps: optionalNumber(sourceConfig, "layer_norm_epsilon", context) ?? 1e-6,
    dropoutRate,
    feedForwardProjection: projection.projection,
    denseActivation: projection.activation,
    isGatedActivation: projection.gated,
    padTokenId: parseNullableInteger(sourceConfig, "pad_token_id", context, 0),
    eosTokenId: parseEosTokenId(sourceConfig, context),
  };
}
