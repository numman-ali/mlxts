/**
 * Prompt-prefix cache snapshots for transformer-backed serving.
 * @module
 */

import type {
  GenerationOptions,
  PromptCacheSnapshotEvent,
  TransformerCache,
  TransformerCacheSnapshot,
} from "@mlxts/transformers";
import {
  type PromptPrefixTokenBlockHandle,
  type PromptPrefixTokenBlockHasher,
  type PromptPrefixTokenBlockStats,
  PromptPrefixTokenBlockStore,
} from "./prefix-cache-blocks";
import {
  clonePromptPrefixCacheIdentity,
  commonPrefixLength,
  disposePromptPrefixCacheEntry,
  type PromptPrefixCacheEntry,
  type PromptPrefixCacheEntryMetadata,
  type PromptPrefixCacheIdentity,
  type PromptPrefixCacheMatchType,
  promptPrefixCacheEntryEstimatedByteSize,
  promptPrefixCacheEntryMetadata,
  promptPrefixCacheIdentitiesCompatible,
  promptPrefixCacheMatchTypeFor,
  shouldEvictPromptPrefixCacheEntry,
} from "./prefix-cache-entry";
import { PromptPrefixCacheBlockIndex } from "./prefix-cache-index";

export type {
  PromptPrefixCacheEntryMetadata,
  PromptPrefixCacheIdentity,
  PromptPrefixCacheMatchType,
} from "./prefix-cache-entry";

export type PromptPrefixCacheHit = {
  /** Forked cache owned by the caller. Dispose it when generation finishes. */
  cache: TransformerCache;
  /** Number of leading prompt tokens restored from cache. */
  readTokens: number;
  /** Relationship between the stored snapshot and requested reusable prefix. */
  matchType: PromptPrefixCacheMatchType;
  /** Metadata for the retained source snapshot that produced this hit. */
  source: PromptPrefixCacheEntryMetadata;
};

export type PromptPrefixCacheUsage = {
  readTokens: number;
  writeTokens: number;
};

/** Retention counters for prompt-prefix cache entries and token blocks. */
export type PromptPrefixCacheStats = {
  /** Retained prompt-boundary snapshots. */
  entries: number;
  /** Estimated retained tensor bytes across all prompt-boundary snapshots. */
  retainedSnapshotBytes: number;
  /** Indexed block hashes that currently point at retained entries. */
  indexedBlockHashes: number;
  /** Shared token-block retention counters. */
  tokenBlocks: PromptPrefixTokenBlockStats;
};

/** Prompt-prefix cache retention options. */
export type PromptPrefixCacheOptions = {
  /** Maximum retained prompt-boundary snapshots. */
  maxEntries?: number;
  /** Maximum estimated retained tensor bytes across retained snapshots. */
  maxBytes?: number;
  /** Token count per retained prompt-token block. */
  blockSize?: number;
  /** Custom token-block hash seam for deterministic cache-index tests. */
  hashBlock?: PromptPrefixTokenBlockHasher;
};

type PromptPrefixCacheSessionOptions = {
  promptCache?: PromptPrefixCache;
  tokenIds: readonly number[];
  identity?: PromptPrefixCacheIdentity;
  enabled: boolean;
  onEvent?: (result: "hit" | "miss" | "write", usage: PromptPrefixCacheUsage) => void;
};

type PromptPrefixGenerationOptions = Pick<
  GenerationOptions,
  "cache" | "samplerHistoryTokenIds" | "onPromptCacheSnapshot"
>;

export type PromptPrefixCacheSession = Disposable & {
  tokenIdsForGeneration(): readonly number[];
  cachedPrefixLength(): number;
  takeCache(): TransformerCache | undefined;
  onPromptCacheSnapshot(event: PromptCacheSnapshotEvent): void;
  generationOptions(): PromptPrefixGenerationOptions;
  usage(): PromptPrefixCacheUsage;
};

const DEFAULT_MAX_ENTRIES = 1;

function maxEntriesFromOptions(options: number | PromptPrefixCacheOptions): number {
  return typeof options === "number" ? options : (options.maxEntries ?? DEFAULT_MAX_ENTRIES);
}

function blockSizeFromOptions(options: number | PromptPrefixCacheOptions): number {
  return typeof options === "number" ? 64 : (options.blockSize ?? 64);
}

function maxBytesFromOptions(options: number | PromptPrefixCacheOptions): number | undefined {
  return typeof options === "number" ? undefined : options.maxBytes;
}

/** Small prompt-boundary snapshot store for repeated local chat turns. */
export class PromptPrefixCache implements Disposable {
  readonly #maxEntries: number;
  readonly #maxBytes: number | undefined;
  readonly #blockStore: PromptPrefixTokenBlockStore;
  readonly #blockIndex: PromptPrefixCacheBlockIndex<PromptPrefixCacheEntry>;
  #entries: PromptPrefixCacheEntry[] = [];
  #clock = 0;

  constructor(options: number | PromptPrefixCacheOptions = DEFAULT_MAX_ENTRIES) {
    const maxEntries = maxEntriesFromOptions(options);
    if (!Number.isInteger(maxEntries) || maxEntries < 0) {
      throw new Error(
        `PromptPrefixCache: maxEntries must be a non-negative integer, got ${maxEntries}.`,
      );
    }
    this.#maxEntries = maxEntries;
    const maxBytes = maxBytesFromOptions(options);
    if (maxBytes !== undefined && (!Number.isInteger(maxBytes) || maxBytes < 0)) {
      throw new Error(
        `PromptPrefixCache: maxBytes must be a non-negative integer when provided, got ${maxBytes}.`,
      );
    }
    this.#maxBytes = maxBytes;
    const blockSize = blockSizeFromOptions(options);
    this.#blockStore =
      typeof options === "number" || options.hashBlock === undefined
        ? new PromptPrefixTokenBlockStore(blockSize)
        : new PromptPrefixTokenBlockStore({ blockSize, hashBlock: options.hashBlock });
    this.#blockIndex = new PromptPrefixCacheBlockIndex(this.#blockStore);
  }

  /** Return the longest reusable prefix cache fork for the provided prompt tokens. */
  lookup(
    tokenIds: readonly number[],
    identity?: PromptPrefixCacheIdentity,
  ): PromptPrefixCacheHit | null {
    if (this.#entries.length === 0 || tokenIds.length <= 1) {
      return null;
    }

    const maxReusableTokens = tokenIds.length - 1;
    const indexedCandidates = this.#blockIndex.candidates(
      tokenIds,
      maxReusableTokens,
      this.#entries,
    );
    const indexedHit = this.#bestHit(
      tokenIds,
      identity,
      maxReusableTokens,
      indexedCandidates.entries,
    );
    const hit =
      indexedHit ??
      (indexedCandidates.includesAllEntries
        ? null
        : this.#bestHit(tokenIds, identity, maxReusableTokens, this.#entries));
    const fallbackHit =
      indexedHit !== null &&
      !indexedCandidates.includesAllEntries &&
      indexedHit.readTokens < indexedCandidates.coveredTokens
        ? this.#bestHit(tokenIds, identity, maxReusableTokens, this.#entries)
        : null;
    const bestHit = fallbackHit ?? hit;

    if (bestHit === null) {
      return null;
    }

    const cache = bestHit.entry.snapshot.fork({ offset: bestHit.readTokens });
    bestHit.entry.lastUsed = ++this.#clock;
    return {
      cache,
      readTokens: bestHit.readTokens,
      matchType: promptPrefixCacheMatchTypeFor(
        bestHit.entry.tokenIds.length,
        maxReusableTokens,
        bestHit.readTokens,
      ),
      source: promptPrefixCacheEntryMetadata(bestHit.entry),
    };
  }

  #bestHit(
    tokenIds: readonly number[],
    identity: PromptPrefixCacheIdentity | undefined,
    maxReusableTokens: number,
    candidates: readonly PromptPrefixCacheEntry[],
  ): { entry: PromptPrefixCacheEntry; readTokens: number } | null {
    let bestEntry: PromptPrefixCacheEntry | undefined;
    let bestReadTokens = 0;

    for (const entry of candidates) {
      const sharedTokens = Math.min(
        commonPrefixLength(entry.tokenIds, tokenIds),
        maxReusableTokens,
      );
      if (
        sharedTokens <= bestReadTokens ||
        !promptPrefixCacheIdentitiesCompatible(entry, identity, sharedTokens) ||
        !entry.snapshot.canFork({ offset: sharedTokens })
      ) {
        continue;
      }
      bestEntry = entry;
      bestReadTokens = sharedTokens;
    }

    if (bestEntry === undefined || bestReadTokens <= 0) {
      return null;
    }

    return { entry: bestEntry, readTokens: bestReadTokens };
  }

  /** Store an owned prompt-boundary snapshot, disposing it if it cannot be retained. */
  store(
    tokenIds: readonly number[],
    snapshot: TransformerCacheSnapshot,
    identity?: PromptPrefixCacheIdentity,
  ): number {
    if (this.#maxEntries === 0 || snapshot.offset <= 0) {
      snapshot[Symbol.dispose]();
      return 0;
    }
    if (snapshot.offset > tokenIds.length - 1) {
      snapshot[Symbol.dispose]();
      throw new Error(
        `PromptPrefixCache.store: snapshot offset ${snapshot.offset} exceeds reusable prompt boundary ${tokenIds.length - 1}.`,
      );
    }
    if (this.#maxBytes !== undefined && snapshot.estimatedByteSize > this.#maxBytes) {
      snapshot[Symbol.dispose]();
      return 0;
    }

    const entryTokenIds = tokenIds.slice(0, snapshot.offset);
    let tokenBlocks: PromptPrefixTokenBlockHandle;
    try {
      tokenBlocks = this.#blockStore.retain(entryTokenIds);
    } catch (error) {
      snapshot[Symbol.dispose]();
      throw error;
    }
    const entry: PromptPrefixCacheEntry = {
      tokenIds: entryTokenIds,
      ...(identity === undefined ? {} : { identity: clonePromptPrefixCacheIdentity(identity) }),
      snapshot,
      tokenBlocks,
      lastUsed: ++this.#clock,
    };
    this.#entries.push(entry);
    this.#blockIndex.add(entry);
    this.#evictOverflow();
    return entry.tokenIds.length;
  }

  [Symbol.dispose](): void {
    for (const entry of this.#entries) {
      this.#disposeEntry(entry);
    }
    this.#entries = [];
    this.#blockIndex.clear();
  }

  /** Return current snapshot and token-block retention counters. */
  stats(): PromptPrefixCacheStats {
    return {
      entries: this.#entries.length,
      retainedSnapshotBytes: this.#retainedSnapshotBytes(),
      indexedBlockHashes: this.#blockIndex.size,
      tokenBlocks: this.#blockStore.stats(),
    };
  }

  #disposeEntry(entry: PromptPrefixCacheEntry): void {
    this.#blockIndex.delete(entry);
    disposePromptPrefixCacheEntry(entry);
  }

  #retainedSnapshotBytes(): number {
    return this.#entries.reduce(
      (total, entry) => total + promptPrefixCacheEntryEstimatedByteSize(entry),
      0,
    );
  }

  #evictOverflow(): void {
    while (
      this.#entries.length > this.#maxEntries ||
      (this.#maxBytes !== undefined && this.#retainedSnapshotBytes() > this.#maxBytes)
    ) {
      let evictionIndex = 0;
      for (let index = 1; index < this.#entries.length; index += 1) {
        const candidate = this.#entries[index];
        const current = this.#entries[evictionIndex];
        if (
          candidate !== undefined &&
          current !== undefined &&
          shouldEvictPromptPrefixCacheEntry(candidate, current)
        ) {
          evictionIndex = index;
        }
      }
      const [removed] = this.#entries.splice(evictionIndex, 1);
      if (removed !== undefined) {
        this.#disposeEntry(removed);
      }
    }
  }
}

class DisabledPromptPrefixCacheSession implements PromptPrefixCacheSession {
  readonly #tokenIds: readonly number[];

  constructor(tokenIds: readonly number[]) {
    this.#tokenIds = tokenIds;
  }

  tokenIdsForGeneration(): readonly number[] {
    return this.#tokenIds;
  }

  cachedPrefixLength(): number {
    return 0;
  }

  takeCache(): TransformerCache | undefined {
    return undefined;
  }

  onPromptCacheSnapshot(_event: PromptCacheSnapshotEvent): void {}

  generationOptions(): PromptPrefixGenerationOptions {
    return {};
  }

  usage(): PromptPrefixCacheUsage {
    return { readTokens: 0, writeTokens: 0 };
  }

  [Symbol.dispose](): void {}
}

class ActivePromptPrefixCacheSession implements PromptPrefixCacheSession {
  readonly #promptCache: PromptPrefixCache;
  readonly #tokenIds: readonly number[];
  readonly #identity: PromptPrefixCacheIdentity | undefined;
  #cache: TransformerCache | null;
  readonly #readTokens: number;
  #writeTokens = 0;
  readonly #onEvent: NonNullable<PromptPrefixCacheSessionOptions["onEvent"]> | undefined;

  constructor(options: PromptPrefixCacheSessionOptions, hit: PromptPrefixCacheHit | null) {
    if (options.promptCache === undefined) {
      throw new Error("PromptPrefixCacheSession: expected a prompt cache.");
    }
    this.#promptCache = options.promptCache;
    this.#tokenIds = options.tokenIds;
    this.#identity =
      options.identity === undefined ? undefined : clonePromptPrefixCacheIdentity(options.identity);
    this.#cache = hit?.cache ?? null;
    this.#readTokens = hit?.readTokens ?? 0;
    this.#onEvent = options.onEvent;
  }

  tokenIdsForGeneration(): readonly number[] {
    return this.#readTokens > 0 ? this.#tokenIds.slice(this.#readTokens) : this.#tokenIds;
  }

  cachedPrefixLength(): number {
    return this.#readTokens;
  }

  takeCache(): TransformerCache | undefined {
    const cache = this.#cache;
    this.#cache = null;
    return cache ?? undefined;
  }

  onPromptCacheSnapshot(event: PromptCacheSnapshotEvent): void {
    const storedTokens = this.#promptCache.store(this.#tokenIds, event.snapshot, this.#identity);
    this.#writeTokens = Math.max(0, storedTokens - this.#readTokens);
    this.#onEvent?.("write", this.usage());
  }

  generationOptions(): PromptPrefixGenerationOptions {
    return {
      ...(this.#cache === null
        ? {}
        : { cache: this.#cache, samplerHistoryTokenIds: this.#tokenIds }),
      onPromptCacheSnapshot: (event: PromptCacheSnapshotEvent) => this.onPromptCacheSnapshot(event),
    };
  }

  usage(): PromptPrefixCacheUsage {
    return { readTokens: this.#readTokens, writeTokens: this.#writeTokens };
  }

  [Symbol.dispose](): void {
    this.#cache?.[Symbol.dispose]();
  }
}

export function createPromptPrefixCacheSession(
  options: PromptPrefixCacheSessionOptions,
): PromptPrefixCacheSession {
  if (options.promptCache === undefined || !options.enabled) {
    return new DisabledPromptPrefixCacheSession(options.tokenIds);
  }

  const hit = options.promptCache.lookup(options.tokenIds, options.identity);
  if (hit === null) {
    options.onEvent?.("miss", { readTokens: 0, writeTokens: 0 });
    return new ActivePromptPrefixCacheSession(options, null);
  }

  const usage = { readTokens: hit.readTokens, writeTokens: 0 };
  options.onEvent?.("hit", usage);
  return new ActivePromptPrefixCacheSession(options, hit);
}
