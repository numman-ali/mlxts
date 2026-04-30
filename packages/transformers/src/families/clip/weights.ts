/**
 * CLIP text encoder checkpoint weight mapping and loading.
 * @module
 */

import { mxEval, treeFlatten } from "@mlxts/core";

import { assignWeightPath, listParameterPaths } from "../../infrastructure/weight-assignment";
import type { ResolvedSnapshot } from "../../pretrained/types";
import { iterateSafetensorWeights, listSafetensorShardPaths } from "../../pretrained/weights";
import { MissingWeightsError } from "../../types";
import type { CLIPTextModel, CLIPTextModelWithProjection } from "./model";

export type CLIPTextWeightTarget = "text" | "projected";

export type CLIPTextWeightLoadOptions = {
  /** Throw when the checkpoint contains unsupported tensor names. */
  strictUnexpectedWeights?: boolean;
};

/** Assignment summary returned after loading CLIP text weights. */
export type CLIPTextWeightLoadResult = {
  assignedPaths: readonly string[];
  unexpectedWeights: readonly string[];
  shardCount: number;
};

const TEXT_MODEL_PREFIX = "text_model.";
const TEXT_WEIGHT_PATTERNS = [
  {
    pattern: /^embeddings\.token_embedding\.(weight)$/,
    target: "embeddings.tokenEmbedding.$1",
  },
  {
    pattern: /^embeddings\.position_embedding\.(weight)$/,
    target: "embeddings.positionEmbedding.$1",
  },
  {
    pattern: /^encoder\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$/,
    target: "layers.$1.selfAttention.qProj.$2",
  },
  {
    pattern: /^encoder\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$/,
    target: "layers.$1.selfAttention.kProj.$2",
  },
  {
    pattern: /^encoder\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$/,
    target: "layers.$1.selfAttention.vProj.$2",
  },
  {
    pattern: /^encoder\.layers\.(\d+)\.self_attn\.out_proj\.(weight|bias)$/,
    target: "layers.$1.selfAttention.outProj.$2",
  },
  {
    pattern: /^encoder\.layers\.(\d+)\.layer_norm1\.(weight|bias)$/,
    target: "layers.$1.layerNorm1.$2",
  },
  {
    pattern: /^encoder\.layers\.(\d+)\.layer_norm2\.(weight|bias)$/,
    target: "layers.$1.layerNorm2.$2",
  },
  {
    pattern: /^encoder\.layers\.(\d+)\.mlp\.fc1\.(weight|bias)$/,
    target: "layers.$1.mlp.fc1.$2",
  },
  {
    pattern: /^encoder\.layers\.(\d+)\.mlp\.fc2\.(weight|bias)$/,
    target: "layers.$1.mlp.fc2.$2",
  },
  {
    pattern: /^final_layer_norm\.(weight|bias)$/,
    target: "finalLayerNorm.$1",
  },
] as const;

function replacePattern(name: string, pattern: RegExp, target: string): string | null {
  if (!pattern.test(name)) {
    return null;
  }
  return name.replace(pattern, target);
}

function textModelWeightPath(checkpointName: string, target: CLIPTextWeightTarget): string | null {
  const textName = checkpointName.startsWith(TEXT_MODEL_PREFIX)
    ? checkpointName.slice(TEXT_MODEL_PREFIX.length)
    : checkpointName;
  for (const rule of TEXT_WEIGHT_PATTERNS) {
    const path = replacePattern(textName, rule.pattern, rule.target);
    if (path !== null) {
      return target === "projected" ? `textModel.${path}` : path;
    }
  }
  return null;
}

function isIgnoredCLIPTextWeight(checkpointName: string): boolean {
  return (
    checkpointName === "text_model.embeddings.position_ids" ||
    checkpointName === "embeddings.position_ids"
  );
}

/** Map a CLIP checkpoint tensor name onto the package-owned parameter tree. */
export function clipTextWeightPath(
  checkpointName: string,
  target: CLIPTextWeightTarget,
): string | null {
  if (target === "projected" && checkpointName === "text_projection.weight") {
    return "textProjection.weight";
  }
  return textModelWeightPath(checkpointName, target);
}

function sorted(values: Iterable<string>): string[] {
  return [...values].toSorted((left, right) => left.localeCompare(right));
}

function throwIfMissingWeights(
  expectedPaths: ReadonlySet<string>,
  assignedPaths: ReadonlySet<string>,
): void {
  const missingPaths = [...expectedPaths].filter((path) => !assignedPaths.has(path));
  if (missingPaths.length > 0) {
    throw new MissingWeightsError(missingPaths);
  }
}

function throwIfUnexpectedWeights(
  unexpectedWeights: readonly string[],
  options: CLIPTextWeightLoadOptions,
): void {
  if (unexpectedWeights.length === 0 || options.strictUnexpectedWeights !== true) {
    return;
  }
  throw new Error(
    `loadCLIPTextWeights: checkpoint contained unexpected unmapped weights: ${unexpectedWeights.join(", ")}.`,
  );
}

function evalAssignedParameters(model: CLIPTextModel | CLIPTextModelWithProjection): void {
  const parameters = treeFlatten(model.parameters()).map(([, tensor]) => tensor);
  mxEval(...parameters);
}

/** Load CLIP text encoder weights into an existing model. */
export async function loadCLIPTextWeights(
  model: CLIPTextModel | CLIPTextModelWithProjection,
  snapshot: ResolvedSnapshot,
  target: CLIPTextWeightTarget,
  options: CLIPTextWeightLoadOptions = {},
): Promise<CLIPTextWeightLoadResult> {
  const expectedPaths = new Set(listParameterPaths(model.parameters()));
  const assignedPaths = new Set<string>();
  const unexpectedWeights: string[] = [];

  for await (const { name, tensor } of iterateSafetensorWeights(snapshot)) {
    const path = clipTextWeightPath(name, target);
    if (path === null || !expectedPaths.has(path)) {
      tensor.free();
      if (!isIgnoredCLIPTextWeight(name)) {
        unexpectedWeights.push(name);
      }
      continue;
    }

    try {
      assignWeightPath(model, path, tensor);
      assignedPaths.add(path);
    } catch (error) {
      tensor.free();
      throw error;
    }
  }

  const finalUnexpectedWeights = sorted(unexpectedWeights);
  throwIfMissingWeights(expectedPaths, assignedPaths);
  throwIfUnexpectedWeights(finalUnexpectedWeights, options);
  model.eval();
  evalAssignedParameters(model);

  return {
    assignedPaths: sorted(assignedPaths),
    unexpectedWeights: finalUnexpectedWeights,
    shardCount: listSafetensorShardPaths(snapshot).length,
  };
}
