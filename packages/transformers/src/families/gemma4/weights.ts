/**
 * Weight-name mapping for Gemma 4 dense text checkpoints.
 * @module
 */

import {
  concatenate,
  iterateSafetensorTensorChunks,
  type MxArray,
  mxEval,
  retainArray,
} from "@mlxts/core";
import { inspectSnapshot, parseSafetensorIndex, type ResolvedSnapshot } from "@mlxts/hub";

import type { ExceptionalWeightLoaderContext } from "../../types";
import type { Gemma4TextConfig } from "./types";
import { isIgnoredGemma4TextWeightName, sanitizeGemma4TextWeightName } from "./types";

const LANGUAGE_MODEL_PREFIX = "model.language_model.";
const PER_LAYER_EMBEDDING_TARGET_PATH = "model.embedTokensPerLayer.weight";
const EXCEPTIONAL_WEIGHT_TARGET_BYTES = 64 * 1024 * 1024;
const EXCEPTIONAL_WEIGHT_BATCH_SIZE = 8;

function exceptionalCheckpointWeightName(config: Gemma4TextConfig): string {
  return config.modelType === "gemma4"
    ? "model.language_model.embed_tokens_per_layer.weight"
    : "model.embed_tokens_per_layer.weight";
}

export function exceptionalGemma4WeightNames(config: Gemma4TextConfig): readonly string[] {
  return config.hiddenSizePerLayerInput > 0 ? [exceptionalCheckpointWeightName(config)] : [];
}

function shardPathForExceptionalWeight(snapshot: ResolvedSnapshot, checkpointName: string): string {
  const inspection = inspectSnapshot(snapshot);
  const parsedIndex = parseSafetensorIndex(inspection.safetensorsIndex);
  if (parsedIndex !== null) {
    const shardName = parsedIndex.weight_map[checkpointName];
    if (shardName === undefined) {
      throw new Error(
        `Gemma 4 exceptional weight loading: checkpoint index did not contain "${checkpointName}".`,
      );
    }

    const match = snapshot.files.find((file) => file.relativePath === shardName);
    if (match === undefined) {
      throw new Error(
        `Gemma 4 exceptional weight loading: resolved snapshot is missing shard "${shardName}".`,
      );
    }
    return match.localPath;
  }

  const shardPath = inspection.model.safetensorPaths[0];
  if (inspection.model.safetensorPaths.length === 1 && shardPath !== undefined) {
    return shardPath;
  }

  throw new Error(
    `Gemma 4 exceptional weight loading: unable to resolve a shard path for "${checkpointName}".`,
  );
}

function flushChunkBatch(batch: MxArray[], accumulated: MxArray | null): MxArray {
  const firstChunk = batch[0];
  if (firstChunk === undefined) {
    if (accumulated === null) {
      throw new Error(
        "Gemma 4 exceptional weight loading: attempted to flush an empty chunk batch.",
      );
    }
    return accumulated;
  }

  let combinedBatch: MxArray;
  if (batch.length === 1) {
    combinedBatch = retainArray(firstChunk);
  } else {
    using mergedBatch = concatenate(batch, 0);
    mxEval(mergedBatch);
    combinedBatch = retainArray(mergedBatch);
  }

  for (const chunk of batch) {
    chunk.free();
  }
  batch.length = 0;

  if (accumulated === null) {
    return combinedBatch;
  }

  using merged = concatenate([accumulated, combinedBatch], 0);
  mxEval(merged);
  const retainedMerged = retainArray(merged);
  accumulated.free();
  combinedBatch.free();
  return retainedMerged;
}

async function loadExceptionalGemma4Tensor(
  shardPath: string,
  checkpointName: string,
): Promise<MxArray> {
  const batch: MxArray[] = [];
  let accumulated: MxArray | null = null;

  try {
    for await (const entry of iterateSafetensorTensorChunks(shardPath, checkpointName, {
      maxBytesPerChunk: EXCEPTIONAL_WEIGHT_TARGET_BYTES,
    })) {
      batch.push(entry.tensor);
      if (batch.length >= EXCEPTIONAL_WEIGHT_BATCH_SIZE) {
        accumulated = flushChunkBatch(batch, accumulated);
      }
    }

    if (batch.length > 0) {
      accumulated = flushChunkBatch(batch, accumulated);
    }
    if (accumulated === null) {
      throw new Error(
        `Gemma 4 exceptional weight loading: "${checkpointName}" did not yield any tensor chunks.`,
      );
    }

    mxEval(accumulated);
    return accumulated;
  } catch (error) {
    for (const chunk of batch) {
      chunk.free();
    }
    accumulated?.free();
    throw error;
  }
}

export async function loadExceptionalGemma4Weights(
  context: ExceptionalWeightLoaderContext<Gemma4TextConfig>,
): Promise<void> {
  if (context.config.hiddenSizePerLayerInput <= 0) {
    return;
  }

  const checkpointName = exceptionalCheckpointWeightName(context.config);
  const shardPath = shardPathForExceptionalWeight(context.snapshot, checkpointName);
  const tensor = await loadExceptionalGemma4Tensor(shardPath, checkpointName);
  context.assignWeight(PER_LAYER_EMBEDDING_TARGET_PATH, tensor);
}

export function sanitizeGemma4TextWeight(
  config: Gemma4TextConfig,
  checkpointName: string,
): string | null {
  return sanitizeGemma4TextWeightName(config, checkpointName);
}

export function isIgnoredGemma4TextWeight(
  config: Gemma4TextConfig,
  checkpointName: string,
): boolean {
  return isIgnoredGemma4TextWeightName(config, checkpointName);
}

export function sanitizeGemma4Weight(
  config: Gemma4TextConfig,
  checkpointName: string,
): string | null {
  if (!checkpointName.startsWith(LANGUAGE_MODEL_PREFIX)) {
    return null;
  }
  const nestedName = checkpointName.slice(LANGUAGE_MODEL_PREFIX.length);
  return sanitizeGemma4TextWeightName(
    config,
    nestedName.startsWith("lm_head.") ? nestedName : `model.${nestedName}`,
  );
}

export function isIgnoredGemma4Weight(config: Gemma4TextConfig, checkpointName: string): boolean {
  if (!checkpointName.startsWith(LANGUAGE_MODEL_PREFIX)) {
    return true;
  }
  const nestedName = checkpointName.slice(LANGUAGE_MODEL_PREFIX.length);
  return isIgnoredGemma4TextWeightName(
    config,
    nestedName.startsWith("lm_head.") ? nestedName : `model.${nestedName}`,
  );
}
