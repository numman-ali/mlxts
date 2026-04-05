import { Linear, type Module, QuantizedLinear } from "@mlxts/nn";

import { resolveQuantizationParameters } from "./parameters";
import { visitLinearChildren } from "./traversal";
import type { QuantizedCheckpointPlan, QuantizedModuleResult } from "./types";

function canQuantizeLinear(linear: Linear, groupSize: number): boolean {
  const inputDims = linear.weight.shape[1];
  return inputDims !== undefined && inputDims % groupSize === 0;
}

function planRuleMap(
  plan: QuantizedCheckpointPlan,
): Map<string, QuantizedCheckpointPlan["rules"][number]> {
  const rules = new Map<string, QuantizedCheckpointPlan["rules"][number]>();
  for (const rule of plan.rules) {
    rules.set(rule.path, rule);
  }
  return rules;
}

/** Prepare a dense module tree to receive MLX-native quantized checkpoint weights. */
export function setupQuantizedModule(
  module: Module,
  plan: QuantizedCheckpointPlan,
): QuantizedModuleResult {
  if (plan.provider !== "mlx") {
    throw new Error(
      `quantize: provider "${plan.provider}" requires a checkpoint transform that is not implemented in setupQuantizedModule.`,
    );
  }

  const ruleMap = planRuleMap(plan);
  const result: QuantizedModuleResult = {
    targets: [],
    skipped: [],
  };

  visitLinearChildren(module, (slot) => {
    if (!(slot.child instanceof Linear)) {
      return;
    }

    const rule = ruleMap.get(slot.path);
    if (rule?.enabled === false) {
      result.skipped.push({
        path: slot.path,
        reason: "explicit checkpoint rule disabled quantization",
      });
      return;
    }

    if (rule === undefined && plan.explicitOnly) {
      result.skipped.push({
        path: slot.path,
        reason: "checkpoint plan only quantizes explicit module paths",
      });
      return;
    }

    const params = resolveQuantizationParameters(rule?.params, plan.defaults);
    if (!canQuantizeLinear(slot.child, params.groupSize)) {
      if (rule?.enabled) {
        throw new Error(
          `quantize: explicit checkpoint rule "${slot.path}" is incompatible with groupSize ${params.groupSize}.`,
        );
      }
      result.skipped.push({
        path: slot.path,
        reason: `input feature dimension is not divisible by groupSize ${params.groupSize}`,
      });
      return;
    }

    const quantized = new QuantizedLinear(
      slot.child.weight.shape[1] ?? 0,
      slot.child.weight.shape[0] ?? 0,
      {
        bias: slot.child.bias !== null,
        ...params,
      },
    );
    const previous = slot.parent.replaceChild(slot.key, quantized);
    previous[Symbol.dispose]();
    result.targets.push({
      path: slot.path,
      params,
    });
  });

  return result;
}
