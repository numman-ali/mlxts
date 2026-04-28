/**
 * Route-decision helpers for transformer-backed serving.
 * @module
 */

import type { CacheLayerKind, CausalLM } from "@mlxts/transformers";
import { transformersRuntimeStrategy } from "../runtime/strategy";
import type {
  GenerationRoute,
  GenerationRouteDecisionReason,
  NormalizedGenerationRequest,
} from "../types";
import type { TransformersGenerationEngineOptions } from "./index";

const CONTINUOUS_BATCH_MODEL_TYPES = new Set(["gemma", "llama", "mistral", "mistral3", "phi3"]);
const GEMMA_LAYER_PATTERN_BATCH_MODEL_TYPES = new Set(["gemma3_text", "gemma4_text", "gemma4"]);
const STATIC_BATCH_MODEL_TYPES = new Set([
  ...CONTINUOUS_BATCH_MODEL_TYPES,
  ...GEMMA_LAYER_PATTERN_BATCH_MODEL_TYPES,
]);

const MODEL_CACHE_LAYER_KINDS = new WeakMap<CausalLM, readonly CacheLayerKind[]>();

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

function cacheLayerKinds(model: CausalLM): readonly CacheLayerKind[] {
  const cached = MODEL_CACHE_LAYER_KINDS.get(model);
  if (cached !== undefined) {
    return cached;
  }

  using cache = model.createCache();
  const layerKinds = [...cache.layerKinds];
  MODEL_CACHE_LAYER_KINDS.set(model, layerKinds);
  return layerKinds;
}

function hasCacheLayerKind(layerKinds: readonly CacheLayerKind[], kind: CacheLayerKind): boolean {
  return layerKinds.includes(kind);
}

function isAttentionCacheLayerKind(kind: CacheLayerKind): boolean {
  return kind === "full" || kind === "sliding";
}

function hasGemmaLayerPatternBatchCache(model: CausalLM): boolean {
  if (!GEMMA_LAYER_PATTERN_BATCH_MODEL_TYPES.has(model.config.modelType)) {
    return false;
  }

  const layerKinds = cacheLayerKinds(model);
  return (
    layerKinds.length === model.layerCount &&
    layerKinds.every(isAttentionCacheLayerKind) &&
    hasCacheLayerKind(layerKinds, "sliding")
  );
}

function hasModelOwnedBatchCache(model: CausalLM): boolean {
  return typeof Reflect.get(model, "createBatchCache") === "function";
}

function hasLinearRecurrentBatchCache(model: CausalLM): boolean {
  return (
    hasCacheLayerKind(cacheLayerKinds(model), "linear-recurrent") && hasModelOwnedBatchCache(model)
  );
}

export function staticBatchIneligibilityReason(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
): GenerationRouteDecisionReason {
  if (request.input.kind === "content") {
    return "media_input";
  }
  if (request.stream) {
    return "streaming";
  }
  const layerKinds = cacheLayerKinds(options.model);
  const hasLayerPatternBatchCache = hasGemmaLayerPatternBatchCache(options.model);
  const hasLinearRecurrentCache = hasLinearRecurrentBatchCache(options.model);
  if (hasCacheLayerKind(layerKinds, "sliding") && !hasLayerPatternBatchCache) {
    return "sliding_window_cache";
  }
  if (!STATIC_BATCH_MODEL_TYPES.has(options.model.config.modelType) && !hasLinearRecurrentCache) {
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
  if (request.input.kind === "content") {
    return "media_input";
  }
  const layerKinds = cacheLayerKinds(options.model);
  const hasLayerPatternBatchCache = hasGemmaLayerPatternBatchCache(options.model);
  const hasLinearRecurrentCache = hasLinearRecurrentBatchCache(options.model);
  if (hasCacheLayerKind(layerKinds, "sliding") && !hasLayerPatternBatchCache) {
    return "sliding_window_cache";
  }
  if (
    !CONTINUOUS_BATCH_MODEL_TYPES.has(options.model.config.modelType) &&
    !hasLayerPatternBatchCache &&
    !hasLinearRecurrentCache
  ) {
    return "unsupported_model_type";
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
  const strategy = transformersRuntimeStrategy(options);
  if (reason !== "eligible") {
    return { route: "single", eligible: false, reason };
  }
  if (strategy.scheduler.maxBatchSize <= 1) {
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
  const strategy = transformersRuntimeStrategy(options);
  options.onEvent?.({
    type: "generation_route_decision",
    id: request.id,
    protocol: request.protocol,
    model: request.model,
    route,
    eligible,
    reason,
    modelType: options.model.config.modelType,
    maxBatchSize: strategy.scheduler.maxBatchSize,
    schedulerMode: strategy.scheduler.mode,
    cacheBackend: strategy.cache.backend,
    attentionBackend: strategy.attention.backend,
    decodingBackend: strategy.decoding.backend,
    stream: request.stream,
  });
}
