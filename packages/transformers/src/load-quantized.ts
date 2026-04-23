import { dequantize, inspectSafetensors, type MxArray } from "@mlxts/core";
import type { Module } from "@mlxts/nn";
import {
  type QuantizedCheckpointPlan,
  resolveCheckpointQuantizationPlan,
  setupQuantizedModule,
  translateCheckpointQuantizationPlanPaths,
} from "@mlxts/quantize";

import { assignWeightPath } from "./infrastructure/weight-assignment";
import type { ResolvedSnapshot, SnapshotInspection } from "./pretrained/types";
import { listSafetensorShardPaths, parseSafetensorIndex } from "./pretrained/weights";
import type { BaseModelConfig, CausalLM } from "./types";

export type StagedDenseQuantizedLeaf = {
  weight?: MxArray;
  scales?: MxArray;
  biases?: MxArray;
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
