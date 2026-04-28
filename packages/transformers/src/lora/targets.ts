import type { CausalLM } from "../types";
import { collectLinearModulePaths } from "./module-traversal";

/** Named LoRA target bundles for causal decoder fine-tuning. */
export type LoRATargetPreset = "attention" | "attention+mlp" | "all-linear";

/** Options for resolving a readable target preset into explicit module paths. */
export type ResolveLoRATargetsOptions = {
  preset: LoRATargetPreset;
  lastLayers?: number;
  includeLmHead?: boolean;
};

/** Explicit LoRA targets derived from a preset for one loaded decoder family. */
export type ResolvedLoRATargets = {
  preset: LoRATargetPreset;
  modelType: string;
  paths: string[];
};

const ATTENTION_LEAVES = new Set([
  "qProjection",
  "kProjection",
  "vProjection",
  "outputProjection",
  "qkvProjection",
]);

const MLP_LEAVES = new Set([
  "gateProjection",
  "upProjection",
  "downProjection",
  "gateUpProjection",
]);

function leafName(path: string): string {
  return path.split(".").at(-1) ?? path;
}

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

function selectedLayerSet(
  paths: readonly string[],
  lastLayers: number | undefined,
): Set<number> | null {
  if (lastLayers === undefined) {
    return null;
  }

  const indices = [
    ...new Set(paths.map((path) => layerIndex(path)).filter((value) => value !== null)),
  ];
  if (indices.length === 0) {
    return null;
  }

  indices.sort((left, right) => left - right);
  return new Set(indices.slice(-Math.max(lastLayers, 0)));
}

function matchesPreset(path: string, preset: LoRATargetPreset): boolean {
  const leaf = leafName(path);
  if (preset === "all-linear") {
    return true;
  }

  if (ATTENTION_LEAVES.has(leaf)) {
    return true;
  }

  return preset === "attention+mlp" && MLP_LEAVES.has(leaf);
}

function filterForPreset(
  paths: readonly string[],
  preset: LoRATargetPreset,
  lastLayers: number | undefined,
  includeLmHead: boolean,
): string[] {
  const presetPaths = paths.filter((path) => matchesPreset(path, preset));
  const selectedLayers = selectedLayerSet(presetPaths, lastLayers);

  return presetPaths.filter((path) => {
    if (!includeLmHead && path === "lmHead") {
      return false;
    }

    const index = layerIndex(path);
    if (index === null || selectedLayers === null) {
      return true;
    }

    return selectedLayers.has(index);
  });
}

/** Resolve a readable LoRA target preset into explicit module paths for a loaded decoder. */
export function resolveLoRATargets(
  model: CausalLM,
  options: ResolveLoRATargetsOptions,
): ResolvedLoRATargets {
  const paths = filterForPreset(
    collectLinearModulePaths(model),
    options.preset,
    options.lastLayers,
    options.includeLmHead ?? false,
  );

  if (paths.length === 0) {
    throw new Error(
      `transformers.resolveLoRATargets: preset "${options.preset}" found no compatible linear targets for model_type "${model.config.modelType}".`,
    );
  }

  return {
    preset: options.preset,
    modelType: model.config.modelType,
    paths,
  };
}
