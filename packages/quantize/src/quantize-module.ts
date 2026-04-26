import { Embedding, Linear, type Module, QuantizedEmbedding, QuantizedLinear } from "@mlxts/nn";

import { resolveQuantizationParameters } from "./parameters";
import { visitEmbeddingChildren, visitLinearChildren } from "./traversal";
import type { QuantizedModuleResult, QuantizeModuleOptions } from "./types";

function canQuantizeLinear(linear: Linear, groupSize: number): boolean {
  const inputDims = linear.weight.shape[1];
  return inputDims !== undefined && inputDims % groupSize === 0;
}

function canQuantizeEmbedding(embedding: Embedding, groupSize: number): boolean {
  const embeddingDims = embedding.weight.shape[1];
  return embeddingDims !== undefined && embeddingDims % groupSize === 0;
}

/** Quantize eligible dense linear and embedding children in place. */
export function quantizeModule(
  module: Module,
  options: QuantizeModuleOptions = {},
): QuantizedModuleResult {
  const result: QuantizedModuleResult = {
    targets: [],
    skipped: [],
  };

  visitLinearChildren(module, (slot) => {
    if (!(slot.child instanceof Linear) || slot.child instanceof QuantizedLinear) {
      return;
    }

    const decision = options.select?.(slot.path, slot.child) ?? true;
    if (decision === false) {
      result.skipped.push({
        path: slot.path,
        reason: "selection predicate returned false",
      });
      return;
    }

    const overrides = typeof decision === "boolean" ? {} : decision;
    const params = resolveQuantizationParameters(overrides, resolveQuantizationParameters(options));
    if (!canQuantizeLinear(slot.child, params.groupSize)) {
      result.skipped.push({
        path: slot.path,
        reason: `input feature dimension is not divisible by groupSize ${params.groupSize}`,
      });
      return;
    }

    const quantized = QuantizedLinear.fromLinear(slot.child, params);
    const previous = slot.parent.replaceChild(slot.key, quantized);
    previous[Symbol.dispose]();
    result.targets.push({
      path: slot.path,
      params,
    });
  });

  visitEmbeddingChildren(module, (slot) => {
    if (!(slot.child instanceof Embedding) || slot.child instanceof QuantizedEmbedding) {
      return;
    }

    const decision = options.select?.(slot.path, slot.child) ?? true;
    if (decision === false) {
      result.skipped.push({
        path: slot.path,
        reason: "selection predicate returned false",
      });
      return;
    }

    const overrides = typeof decision === "boolean" ? {} : decision;
    const params = resolveQuantizationParameters(overrides, resolveQuantizationParameters(options));
    if (!canQuantizeEmbedding(slot.child, params.groupSize)) {
      result.skipped.push({
        path: slot.path,
        reason: `embedding dimension is not divisible by groupSize ${params.groupSize}`,
      });
      return;
    }

    const quantized = QuantizedEmbedding.fromEmbedding(slot.child, params);
    const previous = slot.parent.replaceChild(slot.key, quantized);
    previous[Symbol.dispose]();
    result.targets.push({
      path: slot.path,
      params,
    });
  });

  return result;
}

/** Alias for quantizing a model tree in place. */
export const quantizeModel = quantizeModule;
