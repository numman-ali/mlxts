import { describe, expect, test } from "bun:test";

import type { MxArray } from "@mlxts/core";
import type {
  TransformerCache,
  TransformerCacheForkOptions,
  TransformerCacheSnapshot,
} from "@mlxts/transformers";
import {
  createPromptPrefixCacheSession,
  PromptPrefixCache,
} from "./transformers-engine-prefix-cache";

class FakeCache implements TransformerCache {
  readonly layerCount = 0;
  readonly offset: number;

  constructor(offset: number) {
    this.offset = offset;
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
    return new FakeSnapshot(this.offset);
  }

  arrays(): MxArray[] {
    return [];
  }

  [Symbol.dispose](): void {}
}

class FakeSnapshot implements TransformerCacheSnapshot {
  readonly offset: number;
  readonly trimmable = true;
  readonly maxForkOffset: number;
  disposeCount = 0;
  forkOffsets: number[] = [];

  constructor(offset: number, maxForkOffset = offset) {
    this.offset = offset;
    this.maxForkOffset = maxForkOffset;
  }

  canFork(options: TransformerCacheForkOptions = {}): boolean {
    return (options.offset ?? this.offset) <= this.maxForkOffset && this.disposeCount === 0;
  }

  fork(options: TransformerCacheForkOptions = {}): TransformerCache {
    const offset = options.offset ?? this.offset;
    this.forkOffsets.push(offset);
    return new FakeCache(offset);
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
    longHit?.cache[Symbol.dispose]();
    expect(longSnapshot.forkOffsets).toEqual([100]);
    expect(cache.lookup(shortPrompt)).toBeNull();
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
