/**
 * Whisper checkpoint weight mapping and loading.
 * @module
 */

import { contiguous, type MxArray, mxEval, transpose, treeFlatten } from "@mlxts/core";

import { assignWeightPath, listParameterPaths } from "../../infrastructure/weight-assignment";
import type { ResolvedSnapshot } from "../../pretrained/types";
import { iterateSafetensorWeights, listSafetensorShardPaths } from "../../pretrained/weights";
import { MissingWeightsError } from "../../types";
import type { WhisperForConditionalGeneration } from "./model";

export type WhisperWeightLoadOptions = {
  /** Throw when the checkpoint contains unsupported tensor names. */
  strictUnexpectedWeights?: boolean;
};

/** Assignment summary returned after loading Whisper weights. */
export type WhisperWeightLoadResult = {
  assignedPaths: readonly string[];
  unexpectedWeights: readonly string[];
  shardCount: number;
};

const WEIGHT_PATTERNS = [
  {
    pattern: /^encoder\.layers\.(\d+)\.self_attn\.(q|k|v)_proj\.(weight|bias)$/,
    target: "model.encoder.layers.$1.selfAttention.$2Proj.$3",
  },
  {
    pattern: /^encoder\.layers\.(\d+)\.self_attn\.out_proj\.(weight|bias)$/,
    target: "model.encoder.layers.$1.selfAttention.outProj.$2",
  },
  {
    pattern: /^encoder\.layers\.(\d+)\.self_attn_layer_norm\.(weight|bias)$/,
    target: "model.encoder.layers.$1.selfAttentionLayerNorm.$2",
  },
  {
    pattern: /^encoder\.layers\.(\d+)\.fc1\.(weight|bias)$/,
    target: "model.encoder.layers.$1.mlp.fc1.$2",
  },
  {
    pattern: /^encoder\.layers\.(\d+)\.fc2\.(weight|bias)$/,
    target: "model.encoder.layers.$1.mlp.fc2.$2",
  },
  {
    pattern: /^encoder\.layers\.(\d+)\.final_layer_norm\.(weight|bias)$/,
    target: "model.encoder.layers.$1.finalLayerNorm.$2",
  },
  {
    pattern: /^decoder\.layers\.(\d+)\.self_attn\.(q|k|v)_proj\.(weight|bias)$/,
    target: "model.decoder.layers.$1.selfAttention.$2Proj.$3",
  },
  {
    pattern: /^decoder\.layers\.(\d+)\.self_attn\.out_proj\.(weight|bias)$/,
    target: "model.decoder.layers.$1.selfAttention.outProj.$2",
  },
  {
    pattern: /^decoder\.layers\.(\d+)\.self_attn_layer_norm\.(weight|bias)$/,
    target: "model.decoder.layers.$1.selfAttentionLayerNorm.$2",
  },
  {
    pattern: /^decoder\.layers\.(\d+)\.encoder_attn\.(q|k|v)_proj\.(weight|bias)$/,
    target: "model.decoder.layers.$1.crossAttention.$2Proj.$3",
  },
  {
    pattern: /^decoder\.layers\.(\d+)\.encoder_attn\.out_proj\.(weight|bias)$/,
    target: "model.decoder.layers.$1.crossAttention.outProj.$2",
  },
  {
    pattern: /^decoder\.layers\.(\d+)\.encoder_attn_layer_norm\.(weight|bias)$/,
    target: "model.decoder.layers.$1.crossAttentionLayerNorm.$2",
  },
  {
    pattern: /^decoder\.layers\.(\d+)\.fc1\.(weight|bias)$/,
    target: "model.decoder.layers.$1.mlp.fc1.$2",
  },
  {
    pattern: /^decoder\.layers\.(\d+)\.fc2\.(weight|bias)$/,
    target: "model.decoder.layers.$1.mlp.fc2.$2",
  },
  {
    pattern: /^decoder\.layers\.(\d+)\.final_layer_norm\.(weight|bias)$/,
    target: "model.decoder.layers.$1.finalLayerNorm.$2",
  },
] as const;

function replacePattern(name: string, pattern: RegExp, target: string): string | null {
  if (!pattern.test(name)) {
    return null;
  }
  return name.replace(pattern, target);
}

function sorted(values: Iterable<string>): string[] {
  return [...values].toSorted((left, right) => left.localeCompare(right));
}

function normalizeCheckpointName(checkpointName: string): string {
  return checkpointName.startsWith("model.")
    ? checkpointName.slice("model.".length)
    : checkpointName;
}

/** Map a Whisper checkpoint tensor name onto the package-owned parameter tree. */
export function whisperWeightPath(checkpointName: string): string | null {
  const name = normalizeCheckpointName(checkpointName);
  const rootMapping: Record<string, string | null> = {
    "encoder.conv1.weight": "model.encoder.conv1.weight",
    "encoder.conv1.bias": "model.encoder.conv1.bias",
    "encoder.conv2.weight": "model.encoder.conv2.weight",
    "encoder.conv2.bias": "model.encoder.conv2.bias",
    "encoder.embed_positions.weight": "model.encoder.positionEmbedding.weight",
    "encoder.layer_norm.weight": "model.encoder.layerNorm.weight",
    "encoder.layer_norm.bias": "model.encoder.layerNorm.bias",
    "decoder.embed_tokens.weight": "model.decoder.tokenEmbedding.weight",
    "decoder.embed_positions.weight": "model.decoder.positionEmbedding.weight",
    "decoder.layer_norm.weight": "model.decoder.layerNorm.weight",
    "decoder.layer_norm.bias": "model.decoder.layerNorm.bias",
    "proj_out.weight": null,
  };

  const rootPath = rootMapping[name];
  if (rootPath !== undefined) {
    return rootPath;
  }

  for (const rule of WEIGHT_PATTERNS) {
    const path = replacePattern(name, rule.pattern, rule.target);
    if (path !== null) {
      return path;
    }
  }
  return null;
}

/** Transform a Hugging Face Whisper tensor into package-owned parameter layout. */
export function transformWhisperWeight(weightPath: string, tensor: MxArray): MxArray {
  if (
    (weightPath === "model.encoder.conv1.weight" || weightPath === "model.encoder.conv2.weight") &&
    tensor.shape.length === 3
  ) {
    using transposed = transpose(tensor, [0, 2, 1]);
    return contiguous(transposed);
  }
  return tensor;
}

function transformedWeight(weightPath: string, tensor: MxArray): MxArray {
  const transformed = transformWhisperWeight(weightPath, tensor);
  if (transformed !== tensor) {
    tensor.free();
  }
  return transformed;
}

function isIgnoredWhisperWeight(checkpointName: string): boolean {
  return (
    checkpointName === "proj_out.weight" ||
    checkpointName === "model.proj_out.weight" ||
    checkpointName.endsWith(".embed_positions.position_ids")
  );
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
  options: WhisperWeightLoadOptions,
): void {
  if (unexpectedWeights.length === 0 || options.strictUnexpectedWeights !== true) {
    return;
  }
  throw new Error(
    `loadWhisperWeights: checkpoint contained unexpected unmapped weights: ${unexpectedWeights.join(", ")}.`,
  );
}

function evalAssignedParameters(model: WhisperForConditionalGeneration): void {
  const parameters = treeFlatten(model.parameters()).map(([, tensor]) => tensor);
  mxEval(...parameters);
}

/** Load Whisper weights into an existing conditional-generation model. */
export async function loadWhisperWeights(
  model: WhisperForConditionalGeneration,
  snapshot: ResolvedSnapshot,
  options: WhisperWeightLoadOptions = {},
): Promise<WhisperWeightLoadResult> {
  const expectedPaths = new Set(listParameterPaths(model.parameters()));
  const assignedPaths = new Set<string>();
  const unexpectedWeights: string[] = [];

  for await (const { name, tensor } of iterateSafetensorWeights(snapshot)) {
    const path = whisperWeightPath(name);
    if (path === null || !expectedPaths.has(path)) {
      tensor.free();
      if (!isIgnoredWhisperWeight(name)) {
        unexpectedWeights.push(name);
      }
      continue;
    }

    let assignedTensor: MxArray | null = tensor;
    try {
      assignedTensor = transformedWeight(path, assignedTensor);
      assignWeightPath(model, path, assignedTensor);
      assignedPaths.add(path);
      assignedTensor = null;
    } finally {
      assignedTensor?.free();
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
