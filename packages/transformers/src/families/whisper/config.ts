/**
 * Whisper config and feature-extractor config parsing.
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
import type { WhisperActivation, WhisperConfig, WhisperFeatureExtractorConfig } from "./types";

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
  if (Array.isArray(value)) {
    throw new ConfigParseError(`${context}.${key} list values are not supported here.`);
  }
  if (typeof value !== "number" || !Number.isInteger(value)) {
    throw new ConfigParseError(`${context}.${key} must be an integer or null when present.`);
  }
  return value;
}

function parseEosTokenId(
  record: Record<string, unknown>,
  context: string,
): number | readonly number[] | null {
  const value = record.eos_token_id;
  if (value === undefined) {
    return 50256;
  }
  if (value === null) {
    return null;
  }
  if (Array.isArray(value)) {
    if (value.length === 0) {
      throw new ConfigParseError(`${context}.eos_token_id list must contain at least one token.`);
    }
    return value.map((tokenId, index) => {
      if (typeof tokenId !== "number" || !Number.isInteger(tokenId)) {
        throw new ConfigParseError(`${context}.eos_token_id[${index}] must be an integer.`);
      }
      return tokenId;
    });
  }
  if (typeof value !== "number" || !Number.isInteger(value)) {
    throw new ConfigParseError(
      `${context}.eos_token_id must be an integer, integer list, or null when present.`,
    );
  }
  return value;
}

function parseActivation(record: Record<string, unknown>, context: string): WhisperActivation {
  const activation = optionalString(record, "activation_function", context) ?? "gelu";
  if (activation !== "gelu") {
    throw new ConfigParseError(
      `${context}.activation_function must be "gelu", got "${activation}".`,
    );
  }
  return activation;
}

function assertWhisperConfigShape(record: Record<string, unknown>, context: string): void {
  const modelType = optionalString(record, "model_type", context);
  if (modelType !== undefined && modelType !== "whisper") {
    throw new ConfigParseError(`${context}.model_type must be "whisper", got "${modelType}".`);
  }

  const isEncoderDecoder = optionalBoolean(record, "is_encoder_decoder", context) ?? true;
  if (!isEncoderDecoder) {
    throw new ConfigParseError(`${context}.is_encoder_decoder must be true for Whisper.`);
  }
}

function headDim(dModel: number, heads: number, context: string, key: string): number {
  if (dModel % heads !== 0) {
    throw new ConfigParseError(`${context}.${key} must divide d_model ${dModel}.`);
  }
  return dModel / heads;
}

/** Parse a Hugging Face Whisper config. */
export function parseWhisperConfig(rawConfig: unknown, context = "Whisper config"): WhisperConfig {
  const sourceConfig = expectConfigRecord(rawConfig, context);
  assertWhisperConfigShape(sourceConfig, context);

  const dModel = positiveInteger(sourceConfig, "d_model", context, 384);
  const encoderAttentionHeads = positiveInteger(
    sourceConfig,
    "encoder_attention_heads",
    context,
    6,
  );
  const decoderAttentionHeads = positiveInteger(
    sourceConfig,
    "decoder_attention_heads",
    context,
    6,
  );

  return {
    modelType: "whisper",
    rawConfig: sourceConfig,
    vocabSize: positiveInteger(sourceConfig, "vocab_size", context, 51865),
    numMelBins: positiveInteger(sourceConfig, "num_mel_bins", context, 80),
    encoderLayers: positiveInteger(sourceConfig, "encoder_layers", context, 4),
    encoderAttentionHeads,
    decoderLayers: positiveInteger(sourceConfig, "decoder_layers", context, 4),
    decoderAttentionHeads,
    encoderFfnDim: positiveInteger(sourceConfig, "encoder_ffn_dim", context, 1536),
    decoderFfnDim: positiveInteger(sourceConfig, "decoder_ffn_dim", context, 1536),
    dModel,
    encoderHeadDim: headDim(dModel, encoderAttentionHeads, context, "encoder_attention_heads"),
    decoderHeadDim: headDim(dModel, decoderAttentionHeads, context, "decoder_attention_heads"),
    activationFunction: parseActivation(sourceConfig, context),
    maxSourcePositions: positiveInteger(sourceConfig, "max_source_positions", context, 1500),
    maxTargetPositions: positiveInteger(sourceConfig, "max_target_positions", context, 448),
    padTokenId: parseNullableInteger(sourceConfig, "pad_token_id", context, 50256),
    bosTokenId: parseNullableInteger(sourceConfig, "bos_token_id", context, 50256),
    eosTokenId: parseEosTokenId(sourceConfig, context),
    decoderStartTokenId: positiveInteger(sourceConfig, "decoder_start_token_id", context, 50257),
    scaleEmbedding: optionalBoolean(sourceConfig, "scale_embedding", context) ?? false,
    useCache: optionalBoolean(sourceConfig, "use_cache", context) ?? true,
  };
}

/** Parse a Hugging Face Whisper feature-extractor config. */
export function parseWhisperFeatureExtractorConfig(
  rawConfig: unknown,
  context = "Whisper feature extractor config",
): WhisperFeatureExtractorConfig {
  const sourceConfig = expectConfigRecord(rawConfig, context);
  const featureSize = positiveInteger(sourceConfig, "feature_size", context, 80);
  const samplingRate = positiveInteger(sourceConfig, "sampling_rate", context, 16000);
  const hopLength = positiveInteger(sourceConfig, "hop_length", context, 160);
  const chunkLength = positiveInteger(sourceConfig, "chunk_length", context, 30);
  const nFft = positiveInteger(sourceConfig, "n_fft", context, 400);
  const paddingValue = optionalNumber(sourceConfig, "padding_value", context) ?? 0;
  const nSamples = chunkLength * samplingRate;

  return {
    featureSize,
    samplingRate,
    hopLength,
    chunkLength,
    nFft,
    paddingValue,
    nSamples,
    nFrames: Math.floor(nSamples / hopLength),
  };
}
