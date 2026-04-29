import { describe, expect, test } from "bun:test";

import type { PromptPrefixTokenBlockHandle } from "./prefix-cache-blocks";
import { PromptPrefixTokenBlockStore } from "./prefix-cache-blocks";
import { PromptPrefixCacheBlockIndex } from "./prefix-cache-index";

type IndexedEntry = {
  id: string;
  tokenBlocks: PromptPrefixTokenBlockHandle;
};

function ids(entries: readonly IndexedEntry[]): string[] {
  return entries.map((entry) => entry.id);
}

describe("PromptPrefixCacheBlockIndex", () => {
  test("returns entries from the deepest shared token block", () => {
    const store = new PromptPrefixTokenBlockStore(2);
    const index = new PromptPrefixCacheBlockIndex<IndexedEntry>(store);
    using shallowBlocks = store.retain([1, 2]);
    using deepBlocks = store.retain([1, 2, 3, 4]);
    using siblingBlocks = store.retain([1, 2, 9, 10]);
    const shallow = { id: "shallow", tokenBlocks: shallowBlocks };
    const deep = { id: "deep", tokenBlocks: deepBlocks };
    const sibling = { id: "sibling", tokenBlocks: siblingBlocks };

    index.add(shallow);
    index.add(deep);
    index.add(sibling);

    expect(ids(index.candidates([1, 2, 3, 4, 5], 4, [shallow, deep, sibling]).entries)).toEqual([
      "deep",
    ]);
    expect(ids(index.candidates([1, 2, 7], 2, [shallow, deep, sibling]).entries)).toEqual([
      "shallow",
      "deep",
      "sibling",
    ]);
  });

  test("falls back to all entries when no indexed block matches", () => {
    const store = new PromptPrefixTokenBlockStore(2);
    const index = new PromptPrefixCacheBlockIndex<IndexedEntry>(store);
    using blocks = store.retain([1, 2]);
    const entry = { id: "entry", tokenBlocks: blocks };
    const fallback = [entry];

    index.add(entry);

    const candidates = index.candidates([9, 10, 11], 2, fallback);
    expect(candidates.entries).toBe(fallback);
    expect(candidates.includesAllEntries).toBe(true);
    expect(candidates.coveredTokens).toBe(0);
  });

  test("does not hash prompts when the retained set cannot be narrowed", () => {
    let hashCalls = 0;
    const store = new PromptPrefixTokenBlockStore({
      blockSize: 2,
      hashBlock: (parentHash, tokenIds) => {
        hashCalls += 1;
        return `${parentHash}:${tokenIds.join(",")}`;
      },
    });
    const index = new PromptPrefixCacheBlockIndex<IndexedEntry>(store);
    using blocks = store.retain([1, 2]);
    const entry = { id: "entry", tokenBlocks: blocks };
    index.add(entry);
    hashCalls = 0;

    const candidates = index.candidates([1, 2, 3, 4, 5], 4, [entry]);

    expect(candidates.entries).toEqual([entry]);
    expect(candidates.includesAllEntries).toBe(true);
    expect(hashCalls).toBe(0);
  });

  test("removes empty block buckets when entries leave the cache", () => {
    const store = new PromptPrefixTokenBlockStore(2);
    const index = new PromptPrefixCacheBlockIndex<IndexedEntry>(store);
    using blocks = store.retain([1, 2, 3, 4]);
    const entry = { id: "entry", tokenBlocks: blocks };

    index.add(entry);
    expect(index.size).toBe(2);

    index.delete(entry);
    expect(index.size).toBe(0);
    expect(ids(index.candidates([1, 2, 3, 4, 5], 4, [entry]).entries)).toEqual(["entry"]);
  });
});
