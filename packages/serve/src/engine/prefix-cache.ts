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

export type PromptPrefixCacheHit = {
  /** Forked cache owned by the caller. Dispose it when generation finishes. */
  cache: TransformerCache;
  readTokens: number;
};

export type PromptPrefixCacheUsage = {
  readTokens: number;
  writeTokens: number;
};

/** Non-token prompt identity required for safe multimodal prefix-cache reuse. */
export type PromptPrefixCacheIdentity = {
  /** Ordered non-token prompt inputs that affect the cached decoder state. */
  contentKeys: readonly string[];
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

type PromptPrefixCacheEntry = {
  tokenIds: number[];
  identity?: PromptPrefixCacheIdentity;
  snapshot: TransformerCacheSnapshot;
  lastUsed: number;
};

const DEFAULT_MAX_ENTRIES = 1;

function commonPrefixLength(left: readonly number[], right: readonly number[]): number {
  const limit = Math.min(left.length, right.length);
  let index = 0;
  while (index < limit && left[index] === right[index]) {
    index += 1;
  }
  return index;
}

function cloneIdentity(identity: PromptPrefixCacheIdentity): PromptPrefixCacheIdentity {
  return { contentKeys: [...identity.contentKeys] };
}

function contentKeysMatchPrefix(
  entryKeys: readonly string[],
  requestedKeys: readonly string[],
): boolean {
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

function identitiesCompatible(
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

function disposeEntry(entry: PromptPrefixCacheEntry): void {
  entry.snapshot[Symbol.dispose]();
}

function shouldEvict(candidate: PromptPrefixCacheEntry, current: PromptPrefixCacheEntry): boolean {
  if (candidate.tokenIds.length !== current.tokenIds.length) {
    return candidate.tokenIds.length < current.tokenIds.length;
  }
  return candidate.lastUsed < current.lastUsed;
}

/** Small prompt-boundary snapshot store for repeated local chat turns. */
export class PromptPrefixCache implements Disposable {
  readonly #maxEntries: number;
  #entries: PromptPrefixCacheEntry[] = [];
  #clock = 0;

  constructor(maxEntries = DEFAULT_MAX_ENTRIES) {
    if (!Number.isInteger(maxEntries) || maxEntries < 0) {
      throw new Error(
        `PromptPrefixCache: maxEntries must be a non-negative integer, got ${maxEntries}.`,
      );
    }
    this.#maxEntries = maxEntries;
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
    let bestEntry: PromptPrefixCacheEntry | undefined;
    let bestReadTokens = 0;

    for (const entry of this.#entries) {
      const sharedTokens = Math.min(
        commonPrefixLength(entry.tokenIds, tokenIds),
        maxReusableTokens,
      );
      if (
        sharedTokens <= bestReadTokens ||
        !identitiesCompatible(entry, identity, sharedTokens) ||
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

    const cache = bestEntry.snapshot.fork({ offset: bestReadTokens });
    bestEntry.lastUsed = ++this.#clock;
    return { cache, readTokens: bestReadTokens };
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

    const entry: PromptPrefixCacheEntry = {
      tokenIds: tokenIds.slice(0, snapshot.offset),
      ...(identity === undefined ? {} : { identity: cloneIdentity(identity) }),
      snapshot,
      lastUsed: ++this.#clock,
    };
    this.#entries.push(entry);
    this.#evictOverflow();
    return entry.tokenIds.length;
  }

  [Symbol.dispose](): void {
    for (const entry of this.#entries) {
      disposeEntry(entry);
    }
    this.#entries = [];
  }

  #evictOverflow(): void {
    while (this.#entries.length > this.#maxEntries) {
      let evictionIndex = 0;
      for (let index = 1; index < this.#entries.length; index += 1) {
        const candidate = this.#entries[index];
        const current = this.#entries[evictionIndex];
        if (candidate !== undefined && current !== undefined && shouldEvict(candidate, current)) {
          evictionIndex = index;
        }
      }
      const [removed] = this.#entries.splice(evictionIndex, 1);
      if (removed !== undefined) {
        disposeEntry(removed);
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
    this.#identity = options.identity === undefined ? undefined : cloneIdentity(options.identity);
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
