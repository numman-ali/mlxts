/**
 * Block-hash candidate index for prompt-prefix cache entries.
 * @module
 */

import type {
  PromptPrefixTokenBlockHandle,
  PromptPrefixTokenBlockStore,
} from "./prefix-cache-blocks";

export type PromptPrefixCacheBlockIndexedEntry = {
  /** Retained token-block chain used for candidate indexing. */
  tokenBlocks: PromptPrefixTokenBlockHandle;
};

/** Candidate selection returned by the prompt-prefix block index. */
export type PromptPrefixCacheBlockCandidates<T> = {
  /** Candidate entries selected for lookup. */
  entries: readonly T[];
  /** True when the selection already includes every retained entry. */
  includesAllEntries: boolean;
  /** Token depth covered by the selected block hash. */
  coveredTokens: number;
};

/** Index retained prompt-cache entries by their token-block chain hashes. */
export class PromptPrefixCacheBlockIndex<T extends PromptPrefixCacheBlockIndexedEntry> {
  readonly #blockStore: PromptPrefixTokenBlockStore;
  readonly #entriesByBlockHash = new Map<string, Set<T>>();

  constructor(blockStore: PromptPrefixTokenBlockStore) {
    this.#blockStore = blockStore;
  }

  /** Number of block hashes that currently point at retained entries. */
  get size(): number {
    return this.#entriesByBlockHash.size;
  }

  /** Add a retained prompt-cache entry to every block bucket in its chain. */
  add(entry: T): void {
    for (const hash of entry.tokenBlocks.hashes()) {
      const entries = this.#entriesByBlockHash.get(hash);
      if (entries === undefined) {
        this.#entriesByBlockHash.set(hash, new Set([entry]));
      } else {
        entries.add(entry);
      }
    }
  }

  /** Remove a retained prompt-cache entry from every block bucket in its chain. */
  delete(entry: T): void {
    for (const hash of entry.tokenBlocks.hashes()) {
      const entries = this.#entriesByBlockHash.get(hash);
      if (entries === undefined) {
        continue;
      }
      entries.delete(entry);
      if (entries.size === 0) {
        this.#entriesByBlockHash.delete(hash);
      }
    }
  }

  /** Remove all index buckets after all owning entries have been disposed. */
  clear(): void {
    this.#entriesByBlockHash.clear();
  }

  /** Return deepest block-sharing candidates or the complete fallback set. */
  candidates(
    tokenIds: readonly number[],
    maxReusableTokens: number,
    fallbackEntries: readonly T[],
  ): PromptPrefixCacheBlockCandidates<T> {
    if (fallbackEntries.length <= 1 || maxReusableTokens <= 0) {
      return {
        entries: fallbackEntries,
        includesAllEntries: true,
        coveredTokens: 0,
      };
    }

    const reusableTokenIds = tokenIds.slice(0, maxReusableTokens);
    const hashes = this.#blockStore.hashesFor(reusableTokenIds);
    const candidates: T[] = [];
    const seen = new Set<T>();

    for (let index = hashes.length - 1; index >= 0; index -= 1) {
      const hash = hashes[index];
      if (hash === undefined) {
        continue;
      }
      const entries = this.#entriesByBlockHash.get(hash);
      if (entries === undefined) {
        continue;
      }
      for (const entry of entries) {
        if (!seen.has(entry)) {
          seen.add(entry);
          candidates.push(entry);
        }
      }
      if (candidates.length > 0) {
        return {
          entries: candidates,
          includesAllEntries: candidates.length === fallbackEntries.length,
          coveredTokens: Math.min(maxReusableTokens, (index + 1) * this.#blockStore.blockSize),
        };
      }
    }

    return {
      entries: fallbackEntries,
      includesAllEntries: true,
      coveredTokens: 0,
    };
  }
}
