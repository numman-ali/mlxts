/**
 * Prompt-prefix cache observability contracts.
 * @module
 */

import type { CacheLayerKind } from "@mlxts/transformers";

export type GenerationPromptCacheMatchType = "exact" | "prefix" | "supersequence" | "lcp";

export type GenerationPromptPrefixTokenBlockInfo = {
  blockSize: number;
  blockCount: number;
  blockReferences: number;
  uniqueTokenCount: number;
  referencedTokenCount: number;
};

export type GenerationPromptPrefixCacheModelInfo = {
  id: string;
  retainedSnapshots: number;
  retainedSnapshotBytes: number;
  indexedBlockHashes: number;
  tokenBlocks: GenerationPromptPrefixTokenBlockInfo;
};

export type GenerationPromptPrefixCacheInfo = {
  models: readonly GenerationPromptPrefixCacheModelInfo[];
  totalRetainedSnapshots: number;
  totalRetainedSnapshotBytes: number;
  totalIndexedBlockHashes: number;
  totalTokenBlocks: number;
  totalTokenBlockReferences: number;
  totalUniqueTokenCount: number;
  totalReferencedTokenCount: number;
};

export type GenerationPromptCacheEventObservability = {
  cacheMatchType?: GenerationPromptCacheMatchType;
  cacheSourceTokenLength?: number;
  cacheSourceSnapshotOffset?: number;
  cacheSourceEstimatedBytes?: number;
  cacheSourceLayerKinds?: readonly CacheLayerKind[];
  cacheSourceTrimmable?: boolean;
  cacheSourceTokenBlockSize?: number;
  cacheSourceTokenBlockCount?: number;
  cacheSourceBlockAlignedTokenLength?: number;
  retainedSnapshots?: number;
  retainedSnapshotBytes?: number;
  indexedBlockHashes?: number;
  tokenBlockSize?: number;
  tokenBlockCount?: number;
  tokenBlockReferences?: number;
  uniqueTokenCount?: number;
  referencedTokenCount?: number;
};
