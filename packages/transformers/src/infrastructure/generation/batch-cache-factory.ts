/**
 * Batch-cache selection shared by static and continuous generation.
 * @module
 */

import type { CausalLM, TransformerBatchCache } from "../../types";
import { BatchKVCache, cacheLayerKindFromAttentionType, LayerPatternBatchKVCache } from "../cache";

type BatchCacheFactory = (leftPadding: readonly number[]) => unknown;

function isBatchCacheFactory(value: unknown): value is BatchCacheFactory {
  return typeof value === "function";
}

function isTransformerBatchCache(value: unknown): value is TransformerBatchCache {
  const layerKinds =
    typeof value === "object" && value !== null ? Reflect.get(value, "layerKinds") : undefined;
  return (
    typeof value === "object" &&
    value !== null &&
    Array.isArray(layerKinds) &&
    typeof Reflect.get(value, "updateAndFetch") === "function" &&
    typeof Reflect.get(value, "advance") === "function" &&
    typeof Reflect.get(value, "filter") === "function" &&
    typeof Reflect.get(value, "extend") === "function" &&
    typeof Reflect.get(value, "extract") === "function" &&
    typeof Reflect.get(value, "offsetTensor") === "function" &&
    typeof Reflect.get(value, "leftPaddingTensor") === "function" &&
    typeof Reflect.get(value, "arrays") === "function" &&
    typeof Reflect.get(value, Symbol.dispose) === "function"
  );
}

function createModelOwnedBatchCache(
  model: CausalLM,
  leftPadding: readonly number[],
  context: string,
): TransformerBatchCache | null {
  const factory = Reflect.get(model, "createBatchCache");
  if (!isBatchCacheFactory(factory)) {
    return null;
  }
  const cache = Reflect.apply(factory, model, [leftPadding]);
  if (!isTransformerBatchCache(cache)) {
    throw new Error(`${context}: model-owned batch cache is not a TransformerBatchCache.`);
  }
  return cache;
}

function gemmaLayerWindowSizes(model: CausalLM, context: string): (number | undefined)[] | null {
  if (model.config.family !== "gemma") {
    return null;
  }
  if (
    model.config.modelType !== "gemma3_text" &&
    model.config.modelType !== "gemma4_text" &&
    model.config.modelType !== "gemma4"
  ) {
    return null;
  }

  const layerTypes = Reflect.get(model.config, "layerTypes");
  const slidingWindow = Reflect.get(model.config, "slidingWindow");
  if (!Array.isArray(layerTypes) || layerTypes.length !== model.layerCount) {
    throw new Error(`${context}: Gemma batch cache requires one layer type per layer.`);
  }
  if (typeof slidingWindow !== "number" || !Number.isInteger(slidingWindow) || slidingWindow <= 0) {
    throw new Error(`${context}: Gemma batch cache requires a positive sliding window.`);
  }

  const layerWindowSizes: (number | undefined)[] = [];
  for (const layerType of layerTypes) {
    const layerKind =
      typeof layerType === "string" ? cacheLayerKindFromAttentionType(layerType) : null;
    if (layerKind === "sliding") {
      layerWindowSizes.push(slidingWindow);
      continue;
    }
    if (layerKind === "full") {
      layerWindowSizes.push(undefined);
      continue;
    }
    throw new Error(`${context}: unsupported Gemma layer type ${String(layerType)}.`);
  }
  return layerWindowSizes;
}

/** Create the most specific batch cache a model can consume. */
export function createBatchCacheForModel(
  model: CausalLM,
  leftPadding: readonly number[],
  context: string,
): TransformerBatchCache {
  const modelOwnedCache = createModelOwnedBatchCache(model, leftPadding, context);
  if (modelOwnedCache !== null) {
    return modelOwnedCache;
  }
  const layerWindowSizes = gemmaLayerWindowSizes(model, context);
  return layerWindowSizes === null
    ? new BatchKVCache(model.layerCount, leftPadding)
    : new LayerPatternBatchKVCache(model.layerCount, leftPadding, layerWindowSizes);
}
