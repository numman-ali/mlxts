import { LoRALinear, type Module } from "@mlxts/nn";

import { collectLoRATargetSlots, collectLoRAWrapperSlots } from "./traversal";
import type { ApplyLoRAOptions, LoRAConfig, LoRAModuleResult } from "./types";

const KEY_ALIASES: Record<string, readonly string[]> = {
  qProjection: ["q_proj"],
  kProjection: ["k_proj"],
  vProjection: ["v_proj"],
  outputProjection: ["o_proj"],
  gateProjection: ["gate_proj"],
  upProjection: ["up_proj"],
  downProjection: ["down_proj"],
  qkvProjection: ["qkv_proj"],
  gateUpProjection: ["gate_up_proj"],
};

function layerIndex(path: string): number | null {
  const match = path.match(/^model\.layers\.(\d+)\./);
  if (match === null) {
    return null;
  }
  const rawIndex = match[1];
  if (rawIndex === undefined) {
    return null;
  }
  return Number.parseInt(rawIndex, 10);
}

function aliasSet(path: string): Set<string> {
  const leaf = path.split(".").at(-1) ?? path;
  return new Set([leaf, ...(KEY_ALIASES[leaf] ?? [])]);
}

function selectedLayerSet(
  paths: readonly string[],
  lastLayers: number | undefined,
): Set<number> | null {
  const indices = [
    ...new Set(paths.map((path) => layerIndex(path)).filter((value) => value !== null)),
  ];
  if (indices.length === 0) {
    return null;
  }
  indices.sort((left, right) => left - right);
  const selected = lastLayers === undefined ? indices : indices.slice(-Math.max(lastLayers, 0));
  return new Set(selected);
}

function matchesKeys(path: string, keys: readonly string[] | undefined): boolean {
  if (keys === undefined || keys.length === 0) {
    return true;
  }
  const aliases = aliasSet(path);
  return keys.some((key) => aliases.has(key));
}

function normalizeConfig(options: ApplyLoRAOptions): LoRAConfig {
  const config: LoRAConfig = {};
  if (options.rank !== undefined) {
    config.rank = options.rank;
  }
  if (options.alpha !== undefined) {
    config.alpha = options.alpha;
  }
  if (options.dropout !== undefined) {
    config.dropout = options.dropout;
  }
  return config;
}

function selectedByDefaultPath(
  path: string,
  layerSet: Set<number> | null,
  options: ApplyLoRAOptions,
): boolean {
  if (options.paths !== undefined) {
    return options.paths.includes(path);
  }

  const index = layerIndex(path);
  if (index === null || layerSet === null || !layerSet.has(index)) {
    return false;
  }
  return matchesKeys(path, options.keys);
}

/** Apply LoRA wrappers to the selected dense or quantized linear child modules. */
export function applyLoRAToModule(
  module: Module,
  options: ApplyLoRAOptions = {},
): LoRAModuleResult {
  const result: LoRAModuleResult = {
    targets: [],
    skipped: [],
  };
  const targets = collectLoRATargetSlots(module);
  const wrappers = collectLoRAWrapperSlots(module);
  const existing = new Set(wrappers.map((slot) => slot.path));
  const layerSet = selectedLayerSet(
    [...targets.map((slot) => slot.path), ...wrappers.map((slot) => slot.path)],
    options.lastLayers,
  );
  const config = normalizeConfig(options);

  for (const slot of wrappers) {
    if (!selectedByDefaultPath(slot.path, layerSet, options)) {
      continue;
    }
    result.skipped.push({
      path: slot.path,
      reason: "already wrapped with LoRA",
    });
  }

  for (const slot of targets) {
    if (existing.has(slot.path)) {
      result.skipped.push({
        path: slot.path,
        reason: "already wrapped with LoRA",
      });
      continue;
    }

    const defaultSelected = selectedByDefaultPath(slot.path, layerSet, options);
    if (!defaultSelected) {
      result.skipped.push({
        path: slot.path,
        reason: "did not match the selected layer slice",
      });
      continue;
    }

    if (options.select !== undefined && !options.select(slot.path, slot.child)) {
      result.skipped.push({
        path: slot.path,
        reason: "selection predicate returned false",
      });
      continue;
    }

    const wrapped = LoRALinear.fromBase(slot.child, config);
    slot.parent.replaceChild(slot.key, wrapped);
    result.targets.push(slot.path);
  }

  return result;
}

/** Merge LoRA wrappers into their base layers and replace the wrappers in place. */
export function mergeLoRAInModule(
  module: Module,
  options: { dequantize?: boolean } = {},
): LoRAModuleResult {
  const result: LoRAModuleResult = {
    targets: [],
    skipped: [],
  };

  for (const slot of collectLoRAWrapperSlots(module)) {
    const merged = slot.child.merge(options);
    slot.parent.replaceChild(slot.key, merged);
    slot.child[Symbol.dispose]();
    result.targets.push(slot.path);
  }

  return result;
}

/** Remove LoRA wrappers while keeping their original base layers. */
export function removeLoRAFromModule(module: Module): LoRAModuleResult {
  const result: LoRAModuleResult = {
    targets: [],
    skipped: [],
  };

  for (const slot of collectLoRAWrapperSlots(module)) {
    const base = slot.child.takeBase();
    slot.parent.replaceChild(slot.key, base);
    slot.child[Symbol.dispose]();
    result.targets.push(slot.path);
  }

  return result;
}

/** Alias for applying LoRA to a model tree. */
export const applyLoRA = applyLoRAToModule;

/** Alias for merging LoRA wrappers in a model tree. */
export const mergeLoRA = mergeLoRAInModule;

/** Alias for removing LoRA wrappers from a model tree. */
export const removeLoRA = removeLoRAFromModule;
