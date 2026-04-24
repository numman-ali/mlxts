/**
 * Admission metadata derived from loaded transformer model configs.
 * @module
 */

import type { CausalLM } from "@mlxts/transformers";

export type ModelAdmissionMetadata = {
  contextWindow?: number;
  maxPromptTokens?: number;
  maxTotalTokens?: number;
  effectiveTotalTokens?: number;
};

export type GenerationMemoryEstimate = {
  kvCacheBytes: number;
  fixedStateBytes: number;
  prefillTemporaryBytes: number;
  totalBytes: number;
  bytesPerToken: number;
  kvCacheLayers: number;
  keyValueHeads: number;
  attentionHeads: number;
  headDim: number;
  dtypeSizeBytes: number;
  batchSize: number;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function positiveIntegerField(record: Record<string, unknown>, key: string): number | undefined {
  const value = record[key];
  return Number.isInteger(value) && typeof value === "number" && value > 0 ? value : undefined;
}

function textConfigRecord(rawConfig: Record<string, unknown>): Record<string, unknown> {
  return isRecord(rawConfig.text_config) ? rawConfig.text_config : rawConfig;
}

function stringField(record: Record<string, unknown>, key: string): string | undefined {
  const value = record[key];
  return typeof value === "string" ? value : undefined;
}

function booleanField(record: Record<string, unknown>, key: string): boolean | undefined {
  const value = record[key];
  return typeof value === "boolean" ? value : undefined;
}

function stringArrayField(record: Record<string, unknown>, key: string): string[] | undefined {
  const value = record[key];
  if (!Array.isArray(value)) {
    return undefined;
  }
  const strings: string[] = [];
  for (const entry of value) {
    if (typeof entry !== "string") {
      return undefined;
    }
    strings.push(entry);
  }
  return strings;
}

/** Read the checkpoint-declared text context window when it is available. */
export function modelContextWindow(model: CausalLM): number | undefined {
  const rawConfig = textConfigRecord(model.config.rawConfig);
  return (
    positiveIntegerField(rawConfig, "max_position_embeddings") ??
    positiveIntegerField(model.config.rawConfig, "max_position_embeddings")
  );
}

/** Return the total-token admission limit after server and checkpoint caps are combined. */
export function effectiveTotalTokenLimit(options: {
  maxTotalTokens?: number | undefined;
  contextWindow?: number | undefined;
}): number | undefined {
  const limits = [options.maxTotalTokens, options.contextWindow].filter(
    (limit): limit is number => limit !== undefined,
  );
  return limits.length === 0 ? undefined : Math.min(...limits);
}

/** Format model-level admission metadata for operator-facing introspection. */
export function modelAdmissionMetadata(
  model: CausalLM,
  limits: { maxPromptTokens?: number; maxTotalTokens?: number },
): ModelAdmissionMetadata {
  const contextWindow = modelContextWindow(model);
  const effectiveTotalTokens = effectiveTotalTokenLimit({
    maxTotalTokens: limits.maxTotalTokens,
    contextWindow,
  });
  return {
    ...(contextWindow === undefined ? {} : { contextWindow }),
    ...(limits.maxPromptTokens === undefined ? {} : { maxPromptTokens: limits.maxPromptTokens }),
    ...(limits.maxTotalTokens === undefined ? {} : { maxTotalTokens: limits.maxTotalTokens }),
    ...(effectiveTotalTokens === undefined ? {} : { effectiveTotalTokens }),
  };
}

function dtypeSizeBytes(config: Record<string, unknown>): number {
  const dtype = stringField(config, "torch_dtype") ?? stringField(config, "dtype");
  switch (dtype) {
    case "float32":
    case "fp32":
      return 4;
    case "float8":
    case "float8_e4m3fn":
    case "fp8":
      return 1;
    default:
      return 2;
  }
}

function buildQwenLayerTypes(config: Record<string, unknown>, layerCount: number): string[] {
  const explicitLayerTypes = stringArrayField(config, "layer_types");
  if (explicitLayerTypes !== undefined) {
    return explicitLayerTypes;
  }

  const fullAttentionInterval = positiveIntegerField(config, "full_attention_interval") ?? 4;
  return Array.from({ length: layerCount }, (_, layerIndex) =>
    (layerIndex + 1) % fullAttentionInterval === 0 ? "full_attention" : "linear_attention",
  );
}

function buildGemma3LayerTypes(config: Record<string, unknown>, layerCount: number): string[] {
  const explicitLayerTypes = stringArrayField(config, "layer_types");
  if (explicitLayerTypes !== undefined) {
    return explicitLayerTypes;
  }

  const slidingWindowPattern = positiveIntegerField(config, "sliding_window_pattern") ?? 6;
  return Array.from({ length: layerCount }, (_, layerIndex) =>
    (layerIndex + 1) % slidingWindowPattern === 0 ? "full_attention" : "sliding_attention",
  );
}

function buildLayerTypes(config: Record<string, unknown>, layerCount: number): string[] {
  const modelType = stringField(config, "model_type");
  if (modelType === "qwen3_5_text") {
    return buildQwenLayerTypes(config, layerCount);
  }
  if (modelType === "gemma3_text") {
    return buildGemma3LayerTypes(config, layerCount);
  }

  return stringArrayField(config, "layer_types") ?? Array(layerCount).fill("full_attention");
}

function defaultHeadDim(
  config: Record<string, unknown>,
  hiddenSize: number | undefined,
  attentionHeads: number | undefined,
): number | undefined {
  return (
    positiveIntegerField(config, "head_dim") ??
    (hiddenSize !== undefined && attentionHeads !== undefined
      ? Math.floor(hiddenSize / attentionHeads)
      : undefined)
  );
}

type CacheShape = {
  kvCacheBytes: number;
  fixedStateBytes: number;
  bytesPerToken: number;
  kvCacheLayers: number;
  keyValueHeads: number;
  attentionHeads: number;
  headDim: number;
  dtypeSizeBytes: number;
};

function kvLayerBytes(options: {
  retainedTokens: number;
  batchSize: number;
  keyValueHeads: number;
  headDim: number;
  dtypeSizeBytes: number;
}): number {
  return (
    options.batchSize *
    options.retainedTokens *
    options.keyValueHeads *
    options.headDim *
    options.dtypeSizeBytes *
    2
  );
}

function qwenFixedStateBytes(
  config: Record<string, unknown>,
  layerTypes: readonly string[],
  batchSize: number,
  dtypeBytes: number,
): number {
  const linearLayers = layerTypes.filter((layerType) => layerType === "linear_attention").length;
  if (linearLayers === 0) {
    return 0;
  }

  const keyHeads = positiveIntegerField(config, "linear_num_key_heads") ?? 16;
  const valueHeads = positiveIntegerField(config, "linear_num_value_heads") ?? 32;
  const keyHeadDim = positiveIntegerField(config, "linear_key_head_dim") ?? 128;
  const valueHeadDim = positiveIntegerField(config, "linear_value_head_dim") ?? 128;
  const convKernelDim = positiveIntegerField(config, "linear_conv_kernel_dim") ?? 4;
  const convStateLength = Math.max(0, convKernelDim - 1);
  const convDim = keyHeads * keyHeadDim * 2 + valueHeads * valueHeadDim;
  const convStateBytes = batchSize * convStateLength * convDim * dtypeBytes;
  const recurrentStateBytes = batchSize * valueHeads * valueHeadDim * keyHeadDim * 4;
  return linearLayers * (convStateBytes + recurrentStateBytes);
}

function gemma4LayerShape(
  config: Record<string, unknown>,
  layerType: string,
): { keyValueHeads: number; headDim: number } | undefined {
  const defaultKeyValueHeads =
    positiveIntegerField(config, "num_key_value_heads") ??
    positiveIntegerField(config, "num_attention_heads");
  if (defaultKeyValueHeads === undefined) {
    return undefined;
  }

  if (layerType !== "full_attention") {
    const headDim = positiveIntegerField(config, "head_dim");
    return headDim === undefined ? undefined : { keyValueHeads: defaultKeyValueHeads, headDim };
  }

  const usesAlternativeAttention = booleanField(config, "attention_k_eq_v") ?? false;
  const globalKeyValueHeads = positiveIntegerField(config, "num_global_key_value_heads");
  const keyValueHeads =
    usesAlternativeAttention && globalKeyValueHeads !== undefined
      ? globalKeyValueHeads
      : defaultKeyValueHeads;
  const headDim = positiveIntegerField(config, "global_head_dim");
  return keyValueHeads === undefined || headDim === undefined
    ? undefined
    : { keyValueHeads, headDim };
}

function genericCacheShape(
  config: Record<string, unknown>,
  options: {
    totalTokens: number;
    batchSize: number;
  },
): CacheShape | undefined {
  const hiddenSize = positiveIntegerField(config, "hidden_size");
  const attentionHeads = positiveIntegerField(config, "num_attention_heads");
  const keyValueHeads =
    positiveIntegerField(config, "num_key_value_heads") ??
    positiveIntegerField(config, "n_kv_heads") ??
    attentionHeads;
  const layerCount = positiveIntegerField(config, "num_hidden_layers");
  const headDim = defaultHeadDim(config, hiddenSize, attentionHeads);
  if (
    attentionHeads === undefined ||
    keyValueHeads === undefined ||
    layerCount === undefined ||
    headDim === undefined
  ) {
    return undefined;
  }

  const layerTypes = buildLayerTypes(config, layerCount);
  if (
    layerTypes.length !== layerCount ||
    layerTypes.some(
      (layerType) =>
        layerType !== "full_attention" &&
        layerType !== "sliding_attention" &&
        layerType !== "linear_attention",
    )
  ) {
    return undefined;
  }
  const dtypeBytes = dtypeSizeBytes(config);
  const isGemma4 = stringField(config, "model_type") === "gemma4_text";
  let kvCacheBytes = 0;
  let bytesPerToken = 0;
  let kvCacheLayers = 0;
  let largestHeadDim = headDim;

  for (const layerType of layerTypes) {
    if (layerType === "linear_attention") {
      continue;
    }
    const retainedTokens =
      layerType === "sliding_attention"
        ? Math.min(
            options.totalTokens,
            positiveIntegerField(config, "sliding_window") ?? options.totalTokens,
          )
        : options.totalTokens;
    const layerShape = isGemma4 ? gemma4LayerShape(config, layerType) : { keyValueHeads, headDim };
    if (layerShape === undefined) {
      return undefined;
    }
    largestHeadDim = Math.max(largestHeadDim, layerShape.headDim);
    kvCacheBytes += kvLayerBytes({
      retainedTokens,
      batchSize: options.batchSize,
      keyValueHeads: layerShape.keyValueHeads,
      headDim: layerShape.headDim,
      dtypeSizeBytes: dtypeBytes,
    });
    bytesPerToken +=
      options.batchSize * layerShape.keyValueHeads * layerShape.headDim * dtypeBytes * 2;
    kvCacheLayers += 1;
  }

  return {
    kvCacheBytes,
    fixedStateBytes: qwenFixedStateBytes(config, layerTypes, options.batchSize, dtypeBytes),
    bytesPerToken,
    kvCacheLayers,
    keyValueHeads,
    attentionHeads,
    headDim: largestHeadDim,
    dtypeSizeBytes: dtypeBytes,
  };
}

/** Estimate request-local KV cache and prefill temporary memory from model config. */
export function estimateGenerationMemory(
  model: CausalLM,
  options: {
    promptTokens: number;
    totalTokens: number;
    prefillStepSize: number;
    batchSize?: number;
  },
): GenerationMemoryEstimate | undefined {
  const batchSize = options.batchSize ?? 1;
  const config = textConfigRecord(model.config.rawConfig);
  const shape = genericCacheShape(config, {
    totalTokens: Math.max(0, options.totalTokens),
    batchSize,
  });
  if (shape === undefined) {
    return undefined;
  }

  const prefillChunkTokens = Math.min(options.prefillStepSize, Math.max(0, options.promptTokens));
  const prefillTemporaryBytes =
    shape.headDim > 128
      ? batchSize *
        prefillChunkTokens *
        Math.max(0, options.promptTokens) *
        shape.attentionHeads *
        4
      : 0;

  return {
    ...shape,
    prefillTemporaryBytes,
    totalBytes: shape.kvCacheBytes + shape.fixedStateBytes + prefillTemporaryBytes,
    batchSize,
  };
}
