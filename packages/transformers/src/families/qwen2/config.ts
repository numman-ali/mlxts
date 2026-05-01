/**
 * Qwen2 and Qwen2.5-VL text config parsing.
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
import { isIgnoredQwen2Weight, sanitizeQwen2Weight } from "./weights";

const SUPPORTED_MODEL_TYPES = new Set(["qwen2", "qwen2_5_vl"]);

function parseLayerTypes(config: Record<string, unknown>, context: string): readonly string[] {
  const value = config.layer_types;
  if (value === undefined || value === null) {
    return [];
  }
  if (!Array.isArray(value)) {
    throw new ConfigParseError(`${context}.layer_types must be an array when present.`);
  }
  return value.map((entry, index) => {
    if (typeof entry !== "string") {
      throw new ConfigParseError(`${context}.layer_types[${index}] must be a string.`);
    }
    return entry;
  });
}

function assertSupportedLayerTypes(config: Record<string, unknown>, context: string): void {
  const useSlidingWindow = optionalBoolean(config, "use_sliding_window", context) ?? false;
  const layerTypes = parseLayerTypes(config, context);
  if (useSlidingWindow || layerTypes.some((layerType) => layerType !== "full_attention")) {
    throw new ConfigParseError(
      `${context} with sliding-window Qwen2 attention is not supported by the lean text runtime yet.`,
    );
  }
}

function optionalRopeField(
  config: Record<string, unknown>,
  key: "rope_scaling" | "rope_parameters",
  context: string,
): Record<string, unknown> | undefined {
  const value = config[key];
  if (value === undefined || value === null) {
    return undefined;
  }
  return expectConfigRecord(value, `${context}.${key}`);
}

function assertDefaultRopeRecord(rope: Record<string, unknown>, context: string): void {
  const ropeType = rope.rope_type ?? rope.type;
  if (ropeType !== undefined && ropeType !== "default") {
    throw new ConfigParseError(`${context}.rope_type must be "default".`);
  }
}

function assertDefaultRope(config: Record<string, unknown>, context: string): void {
  const ropeScaling = optionalRopeField(config, "rope_scaling", context);
  const ropeParameters = optionalRopeField(config, "rope_parameters", context);
  if (ropeScaling !== undefined) {
    assertDefaultRopeRecord(ropeScaling, `${context}.rope_scaling`);
  }
  if (ropeParameters !== undefined) {
    assertDefaultRopeRecord(ropeParameters, `${context}.rope_parameters`);
  }
}

function ropeTheta(config: Record<string, unknown>, context: string): number | undefined {
  const ropeParameters = optionalRopeField(config, "rope_parameters", context);
  if (ropeParameters !== undefined) {
    const theta = optionalNumber(ropeParameters, "rope_theta", `${context}.rope_parameters`);
    if (theta !== undefined) {
      return theta;
    }
  }
  return optionalNumber(config, "rope_theta", context);
}

function textConfigForQwen2Family(rawConfig: Record<string, unknown>): {
  modelType: string;
  config: Record<string, unknown>;
  rootConfig: Record<string, unknown>;
} {
  const rootConfig = expectConfigRecord(rawConfig, "Qwen2 config");
  const modelType = expectString(rootConfig, "model_type", "Qwen2 config");
  if (!SUPPORTED_MODEL_TYPES.has(modelType)) {
    throw new ConfigParseError(
      `Qwen2 config.model_type must be "qwen2" or "qwen2_5_vl", got "${modelType}".`,
    );
  }

  if (modelType === "qwen2_5_vl") {
    const textConfig = expectConfigRecord(rootConfig.text_config, "Qwen2 config.text_config");
    const textModelType = expectString(textConfig, "model_type", "Qwen2 config.text_config");
    if (textModelType !== "qwen2_5_vl_text") {
      throw new ConfigParseError(
        `Qwen2 config.text_config.model_type must be "qwen2_5_vl_text", got "${textModelType}".`,
      );
    }
    return { modelType, config: textConfig, rootConfig };
  }

  return { modelType, config: rootConfig, rootConfig };
}

/** Parse Qwen2-family text configs into the shared LLaMA-like runtime shape. */
export function parseQwen2Config(rawConfig: Record<string, unknown>): LlamaLikeConfig {
  const { modelType, config, rootConfig } = textConfigForQwen2Family(rawConfig);
  const context = modelType === "qwen2_5_vl" ? "Qwen2 config.text_config" : "Qwen2 config";

  const hiddenAct = optionalString(config, "hidden_act", context) ?? "silu";
  if (hiddenAct !== "silu") {
    throw new ConfigParseError(`${context}.hidden_act must be "silu", got "${hiddenAct}".`);
  }
  assertSupportedLayerTypes(config, context);
  assertDefaultRope(config, context);

  const hiddenSize = expectInteger(config, "hidden_size", context);
  const numAttentionHeads = expectInteger(config, "num_attention_heads", context);
  const numKeyValueHeads =
    optionalInteger(config, "num_key_value_heads", context) ?? numAttentionHeads;

  return {
    family: "qwen",
    modelType,
    rawConfig: rootConfig,
    vocabSize: expectInteger(config, "vocab_size", context),
    hiddenSize,
    intermediateSize: expectInteger(config, "intermediate_size", context),
    numHiddenLayers: expectInteger(config, "num_hidden_layers", context),
    numAttentionHeads,
    numKeyValueHeads,
    headDim:
      optionalInteger(config, "head_dim", context) ?? Math.floor(hiddenSize / numAttentionHeads),
    maxPositionEmbeddings: expectInteger(config, "max_position_embeddings", context),
    ropeTheta: ropeTheta(config, context) ?? ropeTheta(rootConfig, "Qwen2 config") ?? 1_000_000,
    rmsNormEps: optionalNumber(config, "rms_norm_eps", context) ?? 1e-6,
    tieWordEmbeddings:
      optionalBoolean(rootConfig, "tie_word_embeddings", "Qwen2 config") ??
      optionalBoolean(config, "tie_word_embeddings", context) ??
      false,
    attentionBias: true,
    attentionOutputBias: false,
    mlpActivation: "swiglu",
  };
}

/** Family registration for Qwen2-family text checkpoints. */
export const qwen2Family: FamilyRegistration<LlamaLikeConfig> = {
  family: "qwen",
  modelTypes: ["qwen2", "qwen2_5_vl"],
  parseConfig: parseQwen2Config,
  createModel: (config) => new LlamaLikeCausalLM(config),
  sanitizeWeight: sanitizeQwen2Weight,
  isIgnoredWeight: isIgnoredQwen2Weight,
};
