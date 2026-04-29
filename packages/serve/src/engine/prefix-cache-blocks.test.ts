import { describe, expect, test } from "bun:test";

import { PromptPrefixTokenBlockStore, promptPrefixTokenBlockHash } from "./prefix-cache-blocks";

describe("PromptPrefixTokenBlockStore", () => {
  test("retains full and partial token blocks with block-aligned metadata", () => {
    const store = new PromptPrefixTokenBlockStore({ blockSize: 2 });
    const handle = store.retain([1, 2, 3, 4, 5]);

    expect(handle.metadata()).toEqual({
      blockSize: 2,
      blockCount: 3,
      blockAlignedTokenLength: 4,
    });
    expect(store.stats()).toEqual({
      blockSize: 2,
      blockCount: 3,
      blockReferences: 3,
      uniqueTokenCount: 5,
      referencedTokenCount: 5,
    });

    handle[Symbol.dispose]();
    expect(store.stats()).toEqual({
      blockSize: 2,
      blockCount: 0,
      blockReferences: 0,
      uniqueTokenCount: 0,
      referencedTokenCount: 0,
    });
  });

  test("uses parent-linked hashes so boundary mismatches split block chains", () => {
    const firstHead = promptPrefixTokenBlockHash("root", [1, 2]);
    const sameHead = promptPrefixTokenBlockHash("root", [1, 2]);
    const insideMismatch = promptPrefixTokenBlockHash("root", [1, 9]);
    const firstTail = promptPrefixTokenBlockHash(firstHead, [3, 4]);
    const boundaryMismatchTail = promptPrefixTokenBlockHash(firstHead, [9, 10]);

    expect(firstHead).toBe(sameHead);
    expect(firstHead).not.toBe(insideMismatch);
    expect(firstTail).not.toBe(boundaryMismatchTail);
  });

  test("rolls back retained refs when a later block collides", () => {
    const store = new PromptPrefixTokenBlockStore({
      blockSize: 1,
      hashBlock(parentHash, tokenIds) {
        const [tokenId] = tokenIds;
        if (parentHash !== "root" && (tokenId === 2 || tokenId === 4)) {
          return "forced-tail-collision";
        }
        return promptPrefixTokenBlockHash(parentHash, tokenIds);
      },
    });

    const retained = store.retain([1, 2]);
    expect(store.stats()).toEqual({
      blockSize: 1,
      blockCount: 2,
      blockReferences: 2,
      uniqueTokenCount: 2,
      referencedTokenCount: 2,
    });

    expect(() => store.retain([1, 4])).toThrow("hash collision");
    expect(store.stats()).toEqual({
      blockSize: 1,
      blockCount: 2,
      blockReferences: 2,
      uniqueTokenCount: 2,
      referencedTokenCount: 2,
    });

    retained[Symbol.dispose]();
    expect(store.stats()).toEqual({
      blockSize: 1,
      blockCount: 0,
      blockReferences: 0,
      uniqueTokenCount: 0,
      referencedTokenCount: 0,
    });
  });
});
