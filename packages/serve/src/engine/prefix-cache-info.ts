/**
 * Prompt-prefix cache retention summaries for serving observability.
 * @module
 */

import type {
  GenerationPromptPrefixCacheInfo,
  GenerationPromptPrefixCacheModelInfo,
} from "../prompt-cache-observability";
import type { PromptPrefixCacheStats } from "./prefix-cache";

/** Convert one model's prompt-prefix cache counters into engine info. */
export function promptPrefixCacheInfoForModel(
  modelId: string,
  stats: PromptPrefixCacheStats,
): GenerationPromptPrefixCacheInfo {
  const model: GenerationPromptPrefixCacheModelInfo = {
    id: modelId,
    retainedSnapshots: stats.entries,
    retainedSnapshotBytes: stats.retainedSnapshotBytes,
    indexedBlockHashes: stats.indexedBlockHashes,
    tokenBlocks: {
      blockSize: stats.tokenBlocks.blockSize,
      blockCount: stats.tokenBlocks.blockCount,
      blockReferences: stats.tokenBlocks.blockReferences,
      uniqueTokenCount: stats.tokenBlocks.uniqueTokenCount,
      referencedTokenCount: stats.tokenBlocks.referencedTokenCount,
    },
  };
  return aggregatePromptPrefixCacheInfo([model]);
}

/** Aggregate prompt-prefix cache info across loaded model engines. */
export function aggregatePromptPrefixCacheInfo(
  models: readonly GenerationPromptPrefixCacheModelInfo[],
): GenerationPromptPrefixCacheInfo {
  return {
    models: [...models],
    totalRetainedSnapshots: models.reduce((total, model) => total + model.retainedSnapshots, 0),
    totalRetainedSnapshotBytes: models.reduce(
      (total, model) => total + model.retainedSnapshotBytes,
      0,
    ),
    totalIndexedBlockHashes: models.reduce((total, model) => total + model.indexedBlockHashes, 0),
    totalTokenBlocks: models.reduce((total, model) => total + model.tokenBlocks.blockCount, 0),
    totalTokenBlockReferences: models.reduce(
      (total, model) => total + model.tokenBlocks.blockReferences,
      0,
    ),
    totalUniqueTokenCount: models.reduce(
      (total, model) => total + model.tokenBlocks.uniqueTokenCount,
      0,
    ),
    totalReferencedTokenCount: models.reduce(
      (total, model) => total + model.tokenBlocks.referencedTokenCount,
      0,
    ),
  };
}

/** Aggregate prompt-prefix cache info from nested generation engines. */
export function aggregateEnginePromptPrefixCacheInfo(
  infos: readonly (GenerationPromptPrefixCacheInfo | undefined)[],
): GenerationPromptPrefixCacheInfo {
  return aggregatePromptPrefixCacheInfo(infos.flatMap((info) => info?.models ?? []));
}
