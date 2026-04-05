import { Linear, type Module, QuantizedLinear } from "@mlxts/nn";

import { resolveQuantizationParameters } from "./parameters";
import { visitLinearChildren } from "./traversal";
import type { QuantizedModuleResult, QuantizeModuleOptions } from "./types";

function canQuantizeLinear(linear: Linear, groupSize: number): boolean {
  const inputDims = linear.weight.shape[1];
  return inputDims !== undefined && inputDims % groupSize === 0;
}

/** Quantize eligible dense linear children in place. */
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

  return result;
}

/** Alias for quantizing a model tree in place. */
export const quantizeModel = quantizeModule;
