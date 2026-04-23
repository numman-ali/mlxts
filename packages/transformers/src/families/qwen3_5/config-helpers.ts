import {
  expectConfigRecord,
  optionalBoolean,
  optionalInteger,
  optionalNumber,
  optionalString,
} from "../../infrastructure/config-parsing";
import { ConfigParseError } from "../../types";
import type {
  Qwen3_5EosTokenId,
  Qwen3_5LayerType,
  Qwen3_5PatchShape,
  Qwen3_5TextRopeParameters,
} from "./types";

const DEFAULT_FULL_ATTENTION_INTERVAL = 4;
const DEFAULT_QWEN3_5_ROPE_SECTION = [11, 11, 10] as const;

export function expectPositiveInteger(value: number, field: string): number {
  if (value <= 0) {
    throw new ConfigParseError(`${field} must be positive, got ${value}.`);
  }
  return value;
}

function parseIntegerArray(value: unknown, context: string, allowEmpty = false): number[] {
  if (!Array.isArray(value) || (!allowEmpty && value.length === 0)) {
    const description = allowEmpty ? "an array of integers" : "a non-empty array of integers";
    throw new ConfigParseError(`${context} must be ${description}.`);
  }

  return value.map((entry, index) => {
    if (typeof entry !== "number" || !Number.isInteger(entry)) {
      throw new ConfigParseError(`${context}[${index}] must be an integer.`);
    }
    return entry;
  });
}

export function parseIntegerArrayField(
  config: Record<string, unknown>,
  key: string,
  context: string,
  allowEmpty = false,
): number[] | undefined {
  const value = config[key];
  if (value === undefined) {
    return undefined;
  }
  return parseIntegerArray(value, `${context}.${key}`, allowEmpty);
}

export function parseIntegerOrIntegerArray(
  config: Record<string, unknown>,
  key: string,
  context: string,
): Qwen3_5PatchShape {
  const value = config[key];
  if (typeof value === "number" && Number.isInteger(value)) {
    return expectPositiveInteger(value, `${context}.${key}`);
  }
  return parseIntegerArray(value, `${context}.${key}`);
}

export function optionalIntegerOrNull(
  config: Record<string, unknown>,
  key: string,
  context: string,
): number | null {
  const value = config[key];
  if (value === undefined || value === null) {
    return null;
  }
  if (typeof value !== "number" || !Number.isInteger(value)) {
    throw new ConfigParseError(`${context}.${key} must be an integer when present.`);
  }
  return value;
}

function optionalIntegerArrayOrNull(
  config: Record<string, unknown>,
  key: string,
  context: string,
): number[] | null {
  const value = config[key];
  if (value === undefined || value === null) {
    return null;
  }
  return parseIntegerArray(value, `${context}.${key}`);
}

export function parseEosTokenId(
  config: Record<string, unknown>,
  context: string,
): Qwen3_5EosTokenId {
  const value = config.eos_token_id;
  if (value === undefined || value === null) {
    return null;
  }
  if (typeof value === "number" && Number.isInteger(value)) {
    return value;
  }
  if (Array.isArray(value)) {
    return optionalIntegerArrayOrNull(config, "eos_token_id", context);
  }

  throw new ConfigParseError(`${context}.eos_token_id must be an integer or integer array.`);
}

function buildLayerTypes(
  numHiddenLayers: number,
  fullAttentionInterval: number,
): Qwen3_5LayerType[] {
  return Array.from({ length: numHiddenLayers }, (_, layerIndex) =>
    (layerIndex + 1) % fullAttentionInterval === 0 ? "full_attention" : "linear_attention",
  );
}

function inferFullAttentionInterval(layerTypes: readonly Qwen3_5LayerType[]): number | null {
  const firstFullAttentionIndex = layerTypes.indexOf("full_attention");
  if (firstFullAttentionIndex < 0) {
    return null;
  }

  const interval = firstFullAttentionIndex + 1;
  for (let layerIndex = 0; layerIndex < layerTypes.length; layerIndex += 1) {
    const expected = (layerIndex + 1) % interval === 0 ? "full_attention" : "linear_attention";
    if (layerTypes[layerIndex] !== expected) {
      return null;
    }
  }
  return interval;
}

export function parseLayerTypes(
  config: Record<string, unknown>,
  numHiddenLayers: number,
  context: string,
): { layerTypes: Qwen3_5LayerType[]; fullAttentionInterval: number | null } {
  const rawFullAttentionInterval = optionalInteger(config, "full_attention_interval", context);
  if (rawFullAttentionInterval !== undefined) {
    expectPositiveInteger(rawFullAttentionInterval, `${context}.full_attention_interval`);
  }

  const rawLayerTypes = config.layer_types;
  if (rawLayerTypes === undefined) {
    const fullAttentionInterval = rawFullAttentionInterval ?? DEFAULT_FULL_ATTENTION_INTERVAL;
    return {
      layerTypes: buildLayerTypes(numHiddenLayers, fullAttentionInterval),
      fullAttentionInterval,
    };
  }

  if (!Array.isArray(rawLayerTypes) || rawLayerTypes.length !== numHiddenLayers) {
    throw new ConfigParseError(
      `${context}.layer_types must be an array with ${numHiddenLayers} entries when present.`,
    );
  }

  const layerTypes = rawLayerTypes.map((entry, layerIndex) => {
    if (entry !== "linear_attention" && entry !== "full_attention") {
      throw new ConfigParseError(
        `${context}.layer_types[${layerIndex}] must be "linear_attention" or "full_attention".`,
      );
    }
    return entry;
  });

  if (rawFullAttentionInterval !== undefined) {
    const expectedLayerTypes = buildLayerTypes(numHiddenLayers, rawFullAttentionInterval);
    const matchesInterval = expectedLayerTypes.every(
      (expectedLayerType, layerIndex) => layerTypes[layerIndex] === expectedLayerType,
    );
    if (!matchesInterval) {
      throw new ConfigParseError(
        `${context}.layer_types must match full_attention_interval=${rawFullAttentionInterval} when both fields are present.`,
      );
    }
  }

  return {
    layerTypes,
    fullAttentionInterval: rawFullAttentionInterval ?? inferFullAttentionInterval(layerTypes),
  };
}

export function parseTextRopeParameters(
  config: Record<string, unknown>,
  context: string,
): Qwen3_5TextRopeParameters {
  const rawRopeParameters =
    config.rope_parameters === undefined
      ? {}
      : expectConfigRecord(config.rope_parameters, `${context}.rope_parameters`);

  const ropeType =
    optionalString(rawRopeParameters, "rope_type", `${context}.rope_parameters`) ?? "default";
  const ropeTheta =
    optionalNumber(rawRopeParameters, "rope_theta", `${context}.rope_parameters`) ??
    optionalNumber(config, "rope_theta", context) ??
    10_000_000;
  const nestedPartialRotaryFactor =
    optionalNumber(rawRopeParameters, "partial_rotary_factor", `${context}.rope_parameters`) ??
    optionalNumber(config, "partial_rotary_factor", context) ??
    0.25;
  const mropeSection = parseIntegerArrayField(
    rawRopeParameters,
    "mrope_section",
    `${context}.rope_parameters`,
  ) ?? [...DEFAULT_QWEN3_5_ROPE_SECTION];
  const mropeInterleaved =
    optionalBoolean(rawRopeParameters, "mrope_interleaved", `${context}.rope_parameters`) ?? true;

  return {
    ropeType,
    ropeTheta,
    partialRotaryFactor: nestedPartialRotaryFactor,
    mropeSection,
    mropeInterleaved,
  };
}
