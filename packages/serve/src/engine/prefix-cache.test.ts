import { describe, expect, test } from "bun:test";

import type { MxArray } from "@mlxts/core";
import type {
  CacheLayerKind,
  TransformerCache,
  TransformerCacheForkOptions,
  TransformerCacheSnapshot,
} from "@mlxts/transformers";
import { createPromptPrefixCacheSession, PromptPrefixCache } from "./prefix-cache";

class FakeCache implements TransformerCache {
  readonly layerCount = 0;
  readonly layerKinds: readonly CacheLayerKind[];
  readonly offset: number;

  constructor(offset: number, layerKinds: readonly CacheLayerKind[] = []) {
    this.offset = offset;
    this.layerKinds = layerKinds;
  }

  updateAndFetch(): { keys: MxArray; values: MxArray } {
    throw new Error("FakeCache.updateAndFetch should not be called.");
  }

  advance(): void {
    throw new Error("FakeCache.advance should not be called.");
  }

  isEmpty(): boolean {
    return this.offset === 0;
  }

  isTrimmable(): boolean {
    return true;
  }

  snapshot(): TransformerCacheSnapshot {
    return new FakeSnapshot(this.offset, { layerKinds: this.layerKinds });
  }

  arrays(): MxArray[] {
    return [];
  }

  [Symbol.dispose](): void {}
}

type FakeSnapshotOptions = {
  maxForkOffset?: number;
  exactForkOnly?: boolean;
  layerKinds?: readonly CacheLayerKind[];
  trimmable?: boolean;
};

class FakeSnapshot implements TransformerCacheSnapshot {
  readonly offset: number;
  readonly layerKinds: readonly CacheLayerKind[];
  readonly trimmable: boolean;
  readonly maxForkOffset: number;
  readonly exactForkOnly: boolean;
  disposeCount = 0;
  forkOffsets: number[] = [];

  constructor(offset: number, maxForkOffsetOrOptions: number | FakeSnapshotOptions = offset) {
    this.offset = offset;
    if (typeof maxForkOffsetOrOptions === "number") {
      this.maxForkOffset = maxForkOffsetOrOptions;
      this.exactForkOnly = false;
      this.layerKinds = [];
      this.trimmable = true;
      return;
    }

    this.maxForkOffset = maxForkOffsetOrOptions.maxForkOffset ?? offset;
    this.exactForkOnly = maxForkOffsetOrOptions.exactForkOnly ?? false;
    this.layerKinds = maxForkOffsetOrOptions.layerKinds ?? [];
    this.trimmable = maxForkOffsetOrOptions.trimmable ?? true;
  }

  canFork(options: TransformerCacheForkOptions = {}): boolean {
    const offset = options.offset ?? this.offset;
    if (this.disposeCount !== 0) {
      return false;
    }
    if (this.exactForkOnly) {
      return offset === this.offset;
    }
    return offset <= this.maxForkOffset;
  }

  fork(options: TransformerCacheForkOptions = {}): TransformerCache {
    const offset = options.offset ?? this.offset;
    this.forkOffsets.push(offset);
    return new FakeCache(offset, this.layerKinds);
  }

  [Symbol.dispose](): void {
    this.disposeCount += 1;
  }
}

function tokenRange(start: number, length: number): number[] {
  return Array.from({ length }, (_value, index) => start + index);
}

describe("PromptPrefixCache", () => {
  test("misses when empty, prompt is too short, or snapshot rejects the prefix offset", () => {
    using cache = new PromptPrefixCache(2);
    expect(cache.lookup(tokenRange(1, 8))).toBeNull();
    expect(cache.lookup([1])).toBeNull();

    const snapshot = new FakeSnapshot(5, 4);
    expect(cache.store(tokenRange(1, 6), snapshot)).toBe(5);
    expect(cache.lookup(tokenRange(1, 6))).toBeNull();
    expect(snapshot.forkOffsets).toEqual([]);
  });

  test("keeps the longest reusable prefix when capacity is constrained", () => {
    using cache = new PromptPrefixCache(1);
    const longSnapshot = new FakeSnapshot(100);
    const shortSnapshot = new FakeSnapshot(15);
    const longPrompt = tokenRange(1_000, 101);
    const shortPrompt = tokenRange(10, 16);

    expect(cache.store(longPrompt, longSnapshot)).toBe(100);
    expect(cache.store(shortPrompt, shortSnapshot)).toBe(15);

    expect(longSnapshot.disposeCount).toBe(0);
    expect(shortSnapshot.disposeCount).toBe(1);

    const longHit = cache.lookup(longPrompt);
    expect(longHit?.readTokens).toBe(100);
    expect(longHit?.matchType).toBe("exact");
    longHit?.cache[Symbol.dispose]();
    expect(longSnapshot.forkOffsets).toEqual([100]);
    expect(cache.lookup(shortPrompt)).toBeNull();
  });

  test("classifies exact, prefix, supersequence, and LCP matches", () => {
    using exactCache = new PromptPrefixCache(1);
    const exactSnapshot = new FakeSnapshot(4, {
      layerKinds: ["full"],
      trimmable: true,
    });
    expect(exactCache.store([1, 2, 3, 4, 5], exactSnapshot)).toBe(4);
    const exactHit = exactCache.lookup([1, 2, 3, 4, 5]);
    expect(exactHit?.matchType).toBe("exact");
    expect(exactHit?.readTokens).toBe(4);
    expect(exactHit?.source).toEqual({
      tokenLength: 4,
      snapshotOffset: 4,
      layerKinds: ["full"],
      trimmable: true,
    });
    exactHit?.cache[Symbol.dispose]();

    using prefixCache = new PromptPrefixCache(1);
    const prefixSnapshot = new FakeSnapshot(3);
    expect(prefixCache.store([1, 2, 3, 4], prefixSnapshot)).toBe(3);
    const prefixHit = prefixCache.lookup([1, 2, 3, 4, 5, 6]);
    expect(prefixHit?.matchType).toBe("prefix");
    expect(prefixHit?.readTokens).toBe(3);
    prefixHit?.cache[Symbol.dispose]();

    using supersequenceCache = new PromptPrefixCache(1);
    const supersequenceSnapshot = new FakeSnapshot(5);
    expect(supersequenceCache.store([1, 2, 3, 4, 5, 6], supersequenceSnapshot)).toBe(5);
    const supersequenceHit = supersequenceCache.lookup([1, 2, 3, 4]);
    expect(supersequenceHit?.matchType).toBe("supersequence");
    expect(supersequenceHit?.readTokens).toBe(3);
    supersequenceHit?.cache[Symbol.dispose]();

    using lcpCache = new PromptPrefixCache(1);
    const lcpSnapshot = new FakeSnapshot(5);
    expect(lcpCache.store([1, 2, 3, 4, 5, 6], lcpSnapshot)).toBe(5);
    const lcpHit = lcpCache.lookup([1, 2, 3, 9, 10]);
    expect(lcpHit?.matchType).toBe("lcp");
    expect(lcpHit?.readTokens).toBe(3);
    lcpHit?.cache[Symbol.dispose]();
  });

  test("respects exact-boundary snapshots for non-trimmable cache state", () => {
    using cache = new PromptPrefixCache(1);
    const snapshot = new FakeSnapshot(5, {
      exactForkOnly: true,
      layerKinds: ["full", "linear-recurrent"],
      trimmable: false,
    });
    expect(cache.store([1, 2, 3, 4, 5, 6], snapshot)).toBe(5);

    expect(cache.lookup([1, 2, 3, 4])).toBeNull();
    expect(cache.lookup([1, 2, 3, 9, 10])).toBeNull();

    const hit = cache.lookup([1, 2, 3, 4, 5, 6]);
    expect(hit?.matchType).toBe("exact");
    expect(hit?.source).toEqual({
      tokenLength: 5,
      snapshotOffset: 5,
      layerKinds: ["full", "linear-recurrent"],
      trimmable: false,
    });
    hit?.cache[Symbol.dispose]();
  });

  test("requires matching media identities for media-aware entries", () => {
    using cache = new PromptPrefixCache(2);
    const tokenIds = tokenRange(1, 6);
    expect(cache.store(tokenIds, new FakeSnapshot(5), { contentKeys: ["image:first"] })).toBe(5);

    const changedImage = cache.lookup(tokenIds, { contentKeys: ["image:second"] });
    expect(changedImage).toBeNull();

    const missingIdentity = cache.lookup(tokenIds);
    expect(missingIdentity).toBeNull();

    const matchingImage = cache.lookup(tokenIds, { contentKeys: ["image:first"] });
    expect(matchingImage?.readTokens).toBe(5);
    matchingImage?.cache[Symbol.dispose]();
  });

  test("media-aware entries only reuse whole prompt snapshots", () => {
    using cache = new PromptPrefixCache(1);
    expect(
      cache.store(tokenRange(1, 6), new FakeSnapshot(5, 4), { contentKeys: ["image:first"] }),
    ).toBe(5);

    expect(cache.lookup(tokenRange(1, 5), { contentKeys: ["image:first"] })).toBeNull();
  });

  test("disposes snapshots that cannot be retained", () => {
    using disabled = new PromptPrefixCache(0);
    const ignored = new FakeSnapshot(5);
    expect(disabled.store(tokenRange(1, 6), ignored)).toBe(0);
    expect(ignored.disposeCount).toBe(1);

    using cache = new PromptPrefixCache(1);
    const invalid = new FakeSnapshot(5);
    expect(() => cache.store([1, 2], invalid)).toThrow(
      "snapshot offset 5 exceeds reusable prompt boundary 1",
    );
    expect(invalid.disposeCount).toBe(1);
  });

  test("disposes retained snapshots when the cache is disposed", () => {
    const cache = new PromptPrefixCache(2);
    const first = new FakeSnapshot(2);
    const second = new FakeSnapshot(3);
    expect(cache.store(tokenRange(1, 3), first)).toBe(2);
    expect(cache.store(tokenRange(10, 4), second)).toBe(3);

    cache[Symbol.dispose]();

    expect(first.disposeCount).toBe(1);
    expect(second.disposeCount).toBe(1);
    expect(cache.lookup(tokenRange(1, 3))).toBeNull();
  });

  test("records session usage and keeps generated suffix tokens explicit", () => {
    using cache = new PromptPrefixCache(1);
    expect(cache.store(tokenRange(1, 4), new FakeSnapshot(3))).toBe(3);
    const events: string[] = [];
    using session = createPromptPrefixCacheSession({
      promptCache: cache,
      tokenIds: tokenRange(1, 6),
      enabled: true,
      onEvent(result, usage) {
        events.push(`${result}:${usage.readTokens}:${usage.writeTokens}`);
      },
    });

    expect(session.tokenIdsForGeneration()).toEqual([4, 5, 6]);
    expect(events).toEqual(["hit:3:0"]);

    const options = session.generationOptions();
    expect(options.samplerHistoryTokenIds).toEqual(tokenRange(1, 6));
    options.onPromptCacheSnapshot?.({ offset: 5, snapshot: new FakeSnapshot(5) });
    expect(session.usage()).toEqual({ readTokens: 3, writeTokens: 2 });
    expect(events).toEqual(["hit:3:0", "write:3:2"]);
  });
});
