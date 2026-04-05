import { describe, expect, test } from "bun:test";

import { loadTekkenJson } from "./tekken";

function base64Token(...bytes: number[]): string {
  return btoa(String.fromCharCode(...bytes));
}

describe("loadTekkenJson", () => {
  test("loads a Tekken tokenizer into the shared BPE surface", () => {
    const tokenizer = loadTekkenJson({
      config: {
        pattern: "\\p{L}+| ?[^\\s\\p{L}\\p{N}]+|\\s+",
        default_num_special_tokens: 4,
      },
      special_tokens: [
        { rank: 0, token_str: "<unk>", is_control: true },
        { rank: 1, token_str: "<s>", is_control: true },
        { rank: 2, token_str: "</s>", is_control: true },
        { rank: 3, token_str: "<pad>", is_control: true },
      ],
      vocab: [
        { rank: 0, token_bytes: base64Token(72), token_str: "H" },
        { rank: 1, token_bytes: base64Token(105), token_str: "i" },
        { rank: 2, token_bytes: base64Token(33), token_str: "!" },
      ],
    });

    expect(tokenizer.bosTokenId).toBe(1);
    expect(tokenizer.eosTokenIds).toEqual([2]);
    expect(tokenizer.padTokenId).toBe(3);
    expect(tokenizer.encode("Hi!")).toEqual([1, 4, 5, 6]);
    expect(tokenizer.decode([1, 4, 5, 6, 2], { skipSpecialTokens: true })).toBe("Hi!");
  });
});
