import { dequantize, inspectSafetensors, type MxArray } from "@mlxts/core";
import { Module } from "@mlxts/nn";
import {
  type QuantizedCheckpointPlan,
  resolveCheckpointQuantizationPlan,
  resolveQuantizationParameters,
  setupQuantizedModule,
  translateCheckpointQuantizationPlanPaths,
} from "@mlxts/quantize";

import { PackedSwitchGLUExperts, SwitchLinear } from "./infrastructure/moe";
import { assignWeightPath } from "./infrastructure/weight-assignment";
import type { ResolvedSnapshot, SnapshotInspection } from "./pretrained/types";
import { listSafetensorShardPaths, parseSafetensorIndex } from "./pretrained/weights";
import type { BaseModelConfig, CausalLM } from "./types";

export type StagedDenseQuantizedLeaf = {
  weight?: MxArray;
  scales?: MxArray;
  biases?: MxArray;
};

type ModuleChildSlot = {
  path: string;
  parent: Module;
  key: string;
  child: Module;
};

async function quantizedCheckpointPaths(
  snapshot: ResolvedSnapshot,
  inspection: SnapshotInspection,
): Promise<string[]> {
  const parsedIndex = parseSafetensorIndex(inspection.safetensorsIndex);
  if (parsedIndex !== null) {
    return Object.keys(parsedIndex.weight_map)
      .filter((name) => name.endsWith(".scales"))
      .map((name) => name.slice(0, -".scales".length));
  }

  const quantizedPaths = new Set<string>();
  for (const shardPath of listSafetensorShardPaths(snapshot)) {
    const shardInspection = await inspectSafetensors(shardPath);
    for (const tensor of shardInspection.tensors) {
      if (tensor.name.endsWith(".scales")) {
        const quantizedPath = tensor.name.slice(0, -".scales".length);
        quantizedPaths.add(quantizedPath);
      }
    }
  }
  return [...quantizedPaths];
}

function realizedCheckpointQuantizationPlan(
  plan: QuantizedCheckpointPlan,
  quantizedCheckpointPaths: readonly string[],
  translateCheckpointPath: (checkpointPath: string) => string | null,
): QuantizedCheckpointPlan {
  if (quantizedCheckpointPaths.length === 0) {
    return plan;
  }

  const translatedRules = quantizedCheckpointPaths.flatMap((checkpointPath) => {
    const translatedPath = translateCheckpointPath(checkpointPath);
    return translatedPath === null ? [] : [{ path: translatedPath, enabled: true as const }];
  });

  return {
    ...plan,
    explicitOnly: true,
    rules: translatedRules,
  };
}

function childPath(prefix: string, key: string): string {
  return prefix === "" ? key : `${prefix}.${key}`;
}

function moduleArray(value: unknown, path: string): Module[] | null {
  if (!Array.isArray(value) || value.length === 0) {
    return null;
  }
  const hasModule = value.some((entry) => entry instanceof Module);
  if (!hasModule) {
    return null;
  }
  if (!value.every((entry) => entry instanceof Module)) {
    throw new Error(`loadCausalLM: "${path}" mixes Module and non-Module values.`);
  }
  return value;
}

function visitChildModules(
  module: Module,
  visitor: (slot: ModuleChildSlot) => void,
  prefix = "",
): void {
  for (const key of Object.keys(module)) {
    const value = Reflect.get(module, key);
    const path = childPath(prefix, key);
    if (value instanceof Module) {
      visitor({ path, parent: module, key, child: value });
      visitChildModules(value, visitor, path);
      continue;
    }

    const children = moduleArray(value, path);
    if (children === null) {
      continue;
    }

    for (let index = 0; index < children.length; index += 1) {
      const child = children[index];
      if (child === undefined) {
        continue;
      }
      const indexedPath = childPath(path, String(index));
      visitor({ path: indexedPath, parent: module, key, child });
      visitChildModules(child, visitor, indexedPath);
    }
  }
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

function hasSplitSwitchRule(plan: QuantizedCheckpointPlan, path: string): boolean {
  const prefixes = [
    `${path}.gateProjection`,
    `${path}.upProjection`,
    `${path}.downProjection`,
  ] as const;
  return plan.rules.some((rule) => rule.enabled && prefixes.some((prefix) => rule.path === prefix));
}

function replacePackedExpertsForSplitCheckpoint(
  module: Module,
  plan: QuantizedCheckpointPlan,
): void {
  visitChildModules(module, (slot) => {
    if (!(slot.child instanceof PackedSwitchGLUExperts) || !hasSplitSwitchRule(plan, slot.path)) {
      return;
    }

    const splitExperts = slot.child.toSwitchGLUExperts();
    const previous = slot.parent.replaceChild(slot.key, splitExperts);
    previous[Symbol.dispose]();
  });
}

function prepareQuantizedSwitchLinears(module: Module, plan: QuantizedCheckpointPlan): void {
  const ruleMap = planRuleMap(plan);
  visitChildModules(module, (slot) => {
    if (!(slot.child instanceof SwitchLinear)) {
      return;
    }

    const rule = ruleMap.get(slot.path);
    if (rule?.enabled === false || (rule === undefined && plan.explicitOnly)) {
      return;
    }

    const params = resolveQuantizationParameters(rule?.params, plan.defaults);
    slot.child.prepareQuantized(params);
  });
}

export async function prepareQuantizedCheckpointModel(options: {
  snapshot: ResolvedSnapshot;
  inspection: SnapshotInspection;
  model: Module;
  configRecord: Record<string, unknown>;
  translateCheckpointPath: (checkpointPath: string) => string | null;
}): Promise<void> {
  const checkpointQuantizationPlan = resolveCheckpointQuantizationPlan(options.configRecord);
  if (checkpointQuantizationPlan === null) {
    return;
  }

  const quantizedPaths = await quantizedCheckpointPaths(options.snapshot, options.inspection);
  const translatedPlan = realizedCheckpointQuantizationPlan(
    translateCheckpointQuantizationPlanPaths(checkpointQuantizationPlan, (path) =>
      options.translateCheckpointPath(path),
    ),
    quantizedPaths,
    options.translateCheckpointPath,
  );
  setupQuantizedModule(options.model, translatedPlan);
  replacePackedExpertsForSplitCheckpoint(options.model, translatedPlan);
  prepareQuantizedSwitchLinears(options.model, translatedPlan);
}

function withQuantizedLeafSuffix(path: string, suffix: ".scales" | ".biases"): string | null {
  if (!path.endsWith(".weight")) {
    return null;
  }
  return `${path.slice(0, -".weight".length)}${suffix}`;
}

function quantizedBaseWeightPath(path: string): string | null {
  if (path.endsWith(".weight")) {
    return path;
  }
  if (path.endsWith(".scales")) {
    return `${path.slice(0, -".scales".length)}.weight`;
  }
  if (path.endsWith(".biases")) {
    return `${path.slice(0, -".biases".length)}.weight`;
  }
  return null;
}

function supportsQuantizedLeaf(expectedPaths: ReadonlySet<string>, weightPath: string): boolean {
  const scalesPath = withQuantizedLeafSuffix(weightPath, ".scales");
  return scalesPath !== null && expectedPaths.has(scalesPath);
}

export function stagedDenseQuantizedWeightPath(
  path: string,
  tensor: MxArray,
  expectedPaths: ReadonlySet<string>,
): string | null {
  const weightPath = quantizedBaseWeightPath(path);
  if (weightPath === null || !expectedPaths.has(weightPath)) {
    return null;
  }
  if (supportsQuantizedLeaf(expectedPaths, weightPath)) {
    return null;
  }
  if (path.endsWith(".weight")) {
    return tensor.dtype === "uint32" ? weightPath : null;
  }
  return path.endsWith(".scales") || path.endsWith(".biases") ? weightPath : null;
}

export function stageDenseQuantizedLeaf(
  stagedLeaves: Map<string, StagedDenseQuantizedLeaf>,
  weightPath: string,
  path: string,
  tensor: MxArray,
): void {
  const staged = stagedLeaves.get(weightPath) ?? {};
  if (path.endsWith(".weight")) {
    staged.weight = tensor;
  } else if (path.endsWith(".scales")) {
    staged.scales = tensor;
  } else if (path.endsWith(".biases")) {
    staged.biases = tensor;
  }
  stagedLeaves.set(weightPath, staged);
}

export function assignStagedDenseQuantizedLeaves(
  model: CausalLM,
  config: BaseModelConfig,
  stagedLeaves: ReadonlyMap<string, StagedDenseQuantizedLeaf>,
  assignedPaths: Set<string>,
): void {
  if (stagedLeaves.size === 0) {
    return;
  }

  const checkpointQuantizationPlan = resolveCheckpointQuantizationPlan(config.rawConfig);
  if (checkpointQuantizationPlan === null) {
    throw new Error(
      "loadCausalLM: encountered quantized dense checkpoint weights without quantization metadata.",
    );
  }

  for (const [weightPath, staged] of stagedLeaves) {
    try {
      if (staged.weight === undefined || staged.scales === undefined) {
        throw new Error(
          `loadCausalLM: incomplete quantized dense checkpoint tensors for "${weightPath}".`,
        );
      }

      const denseWeight = dequantize(staged.weight, staged.scales, {
        groupSize: checkpointQuantizationPlan.defaults.groupSize,
        bits: checkpointQuantizationPlan.defaults.bits,
        mode: checkpointQuantizationPlan.defaults.mode,
        ...(staged.biases === undefined ? {} : { biases: staged.biases }),
      });

      try {
        assignWeightPath(model, weightPath, denseWeight);
        assignedPaths.add(weightPath);
      } catch (error) {
        denseWeight.free();
        throw error;
      }
    } finally {
      staged.weight?.free();
      staged.scales?.free();
      staged.biases?.free();
    }
  }
}
