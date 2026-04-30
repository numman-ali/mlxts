/**
 * Prompt-prefix cache entry metadata and retention helpers.
 * @module
 */

import type { CacheLayerKind, TransformerCacheSnapshot } from "@mlxts/transformers";
import type {
  PromptPrefixTokenBlockHandle,
  PromptPrefixTokenBlockMetadata,
} from "./prefix-cache-blocks";

/** Cache-match shape used by serving policy and future block-prefix stores. */
export type PromptPrefixCacheMatchType = "exact" | "prefix" | "supersequence" | "lcp";

/** Non-token prompt identity required for safe multimodal prefix-cache reuse. */
export type PromptPrefixCacheIdentity = {
  /** Ordered non-token prompt inputs that affect the cached decoder state. */
  contentKeys: readonly string[];
};

/** Source snapshot metadata for a prompt-prefix cache hit. */
export type PromptPrefixCacheEntryMetadata = {
  tokenLength: number;
  estimatedByteSize: number;
  snapshotOffset: number;
  layerKinds: readonly CacheLayerKind[];
  trimmable: boolean;
  identity?: PromptPrefixCacheIdentity;
  tokenBlocks: PromptPrefixTokenBlockMetadata;
};

export type PromptPrefixCacheEntry = {
  tokenIds: number[];
  estimatedByteSize: number;
  identity?: PromptPrefixCacheIdentity;
  snapshot: TransformerCacheSnapshot;
  tokenBlocks: PromptPrefixTokenBlockHandle;
  lastUsed: number;
};

export function commonPrefixLength(left: readonly number[], right: readonly number[]): number {
  const limit = Math.min(left.length, right.length);
  let index = 0;
  while (index < limit && left[index] === right[index]) {
    index += 1;
  }
  return index;
}

export function clonePromptPrefixCacheIdentity(
  identity: PromptPrefixCacheIdentity,
): PromptPrefixCacheIdentity {
  return { contentKeys: [...identity.contentKeys] };
}

function contentKeysMatchPrefix(entryKeys: readonly string[], requestedKeys: readonly string[]) {
  if (entryKeys.length > requestedKeys.length) {
    return false;
  }
  for (let index = 0; index < entryKeys.length; index += 1) {
    if (entryKeys[index] !== requestedKeys[index]) {
      return false;
    }
  }
  return true;
}

export function promptPrefixCacheIdentitiesCompatible(
  entry: PromptPrefixCacheEntry,
  requestedIdentity: PromptPrefixCacheIdentity | undefined,
  sharedTokens: number,
): boolean {
  if (entry.identity === undefined || requestedIdentity === undefined) {
    return entry.identity === undefined && requestedIdentity === undefined;
  }

  return (
    sharedTokens === entry.tokenIds.length &&
    contentKeysMatchPrefix(entry.identity.contentKeys, requestedIdentity.contentKeys)
  );
}

export function disposePromptPrefixCacheEntry(entry: PromptPrefixCacheEntry): void {
  try {
    entry.snapshot[Symbol.dispose]();
  } finally {
    entry.tokenBlocks[Symbol.dispose]();
  }
}

export function shouldEvictPromptPrefixCacheEntry(
  candidate: PromptPrefixCacheEntry,
  current: PromptPrefixCacheEntry,
): boolean {
  if (candidate.tokenIds.length !== current.tokenIds.length) {
    return candidate.tokenIds.length < current.tokenIds.length;
  }
  return candidate.lastUsed < current.lastUsed;
}

export function promptPrefixCacheMatchTypeFor(
  entryTokenLength: number,
  maxReusableTokens: number,
  sharedTokens: number,
): PromptPrefixCacheMatchType {
  if (sharedTokens === entryTokenLength && sharedTokens === maxReusableTokens) {
    return "exact";
  }
  if (sharedTokens === entryTokenLength) {
    return "prefix";
  }
  if (sharedTokens === maxReusableTokens) {
    return "supersequence";
  }
  return "lcp";
}

export function promptPrefixCacheEntryMetadata(
  entry: PromptPrefixCacheEntry,
): PromptPrefixCacheEntryMetadata {
  return {
    tokenLength: entry.tokenIds.length,
    estimatedByteSize: entry.estimatedByteSize,
    snapshotOffset: entry.snapshot.offset,
    layerKinds: [...entry.snapshot.layerKinds],
    trimmable: entry.snapshot.trimmable,
    ...(entry.identity === undefined
      ? {}
      : { identity: clonePromptPrefixCacheIdentity(entry.identity) }),
    tokenBlocks: entry.tokenBlocks.metadata(),
  };
}
