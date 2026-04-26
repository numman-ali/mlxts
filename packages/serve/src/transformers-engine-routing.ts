/**
 * Route-decision helpers for transformer-backed serving.
 * @module
 */

import type { CausalLM } from "@mlxts/transformers";
import type { TransformersGenerationEngineOptions } from "./transformers-engine";
import type {
  GenerationRoute,
  GenerationRouteDecisionReason,
  NormalizedGenerationRequest,
} from "./types";

const CONTINUOUS_BATCH_MODEL_TYPES = new Set(["gemma", "llama", "mistral", "mistral3", "phi3"]);
const GEMMA_LAYER_PATTERN_BATCH_MODEL_TYPES = new Set(["gemma3_text", "gemma4_text", "gemma4"]);
const STATIC_BATCH_MODEL_TYPES = new Set([
  ...CONTINUOUS_BATCH_MODEL_TYPES,
  ...GEMMA_LAYER_PATTERN_BATCH_MODEL_TYPES,
  "qwen3_5_text",
]);

function configHasSlidingWindow(model: CausalLM): boolean {
  const config = model.config;
  if (!("slidingWindow" in config)) {
    return false;
  }
  const value = config.slidingWindow;
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

function effectiveTemperature(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
): number {
  return (
    request.sampling.temperature ?? options.model.config.generationDefaults?.temperature ?? 1.0
  );
}

function effectiveRepetitionPenalty(options: TransformersGenerationEngineOptions): number {
  return options.model.config.generationDefaults?.repetitionPenalty ?? 1.0;
}

function hasGemmaLayerPatternBatchCache(model: CausalLM): boolean {
  if (!GEMMA_LAYER_PATTERN_BATCH_MODEL_TYPES.has(model.config.modelType)) {
    return false;
  }

  const layerTypes: unknown = Reflect.get(model.config, "layerTypes");
  const slidingWindow: unknown = Reflect.get(model.config, "slidingWindow");
  return (
    Array.isArray(layerTypes) &&
    layerTypes.length === model.layerCount &&
    layerTypes.every(
      (layerType) => layerType === "full_attention" || layerType === "sliding_attention",
    ) &&
    typeof slidingWindow === "number" &&
    Number.isInteger(slidingWindow) &&
    slidingWindow > 0
  );
}

export function staticBatchIneligibilityReason(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
): GenerationRouteDecisionReason {
  if (request.stream) {
    return "streaming";
  }
  if (configHasSlidingWindow(options.model) && !hasGemmaLayerPatternBatchCache(options.model)) {
    return "sliding_window_cache";
  }
  if (!STATIC_BATCH_MODEL_TYPES.has(options.model.config.modelType)) {
    return "unsupported_model_type";
  }
  if (effectiveTemperature(request, options) !== 0) {
    return "sampled_generation";
  }
  if (effectiveRepetitionPenalty(options) !== 1.0) {
    return "repetition_penalty";
  }
  return "eligible";
}

export function continuousBatchIneligibilityReason(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
): GenerationRouteDecisionReason {
  if (configHasSlidingWindow(options.model)) {
    return "sliding_window_cache";
  }
  if (!CONTINUOUS_BATCH_MODEL_TYPES.has(options.model.config.modelType)) {
    return "unsupported_model_type";
  }
  if (effectiveTemperature(request, options) !== 0) {
    return "sampled_generation";
  }
  if (effectiveRepetitionPenalty(options) !== 1.0) {
    return "repetition_penalty";
  }
  return "eligible";
}

export function canUseStaticBatchGeneration(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
): boolean {
  return staticBatchIneligibilityReason(request, options) === "eligible";
}

export function routeDecisionForRequest(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
): {
  route: GenerationRoute;
  eligible: boolean;
  reason: GenerationRouteDecisionReason;
} {
  const reason = continuousBatchIneligibilityReason(request, options);
  if (reason !== "eligible") {
    return { route: "single", eligible: false, reason };
  }
  if ((options.maxBatchSize ?? 1) <= 1) {
    return { route: "single", eligible: false, reason: "max_batch_size" };
  }
  return { route: "continuous", eligible: true, reason };
}

export function emitGenerationRouteDecision(
  options: TransformersGenerationEngineOptions,
  request: NormalizedGenerationRequest,
  route: GenerationRoute,
  eligible: boolean,
  reason: GenerationRouteDecisionReason,
): void {
  options.onEvent?.({
    type: "generation_route_decision",
    id: request.id,
    protocol: request.protocol,
    model: request.model,
    route,
    eligible,
    reason,
    modelType: options.model.config.modelType,
    maxBatchSize: options.maxBatchSize ?? 1,
    stream: request.stream,
  });
}
