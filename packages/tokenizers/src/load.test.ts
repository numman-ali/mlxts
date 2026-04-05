import { describe, expect, test } from "bun:test";
import { mkdtempSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { loadTokenizer } from "./load";

function createTempDir(name: string): string {
  return mkdtempSync(join(tmpdir(), `${name}-`));
}

function base64Token(...bytes: number[]): string {
  return btoa(String.fromCharCode(...bytes));
}

describe("loadTokenizer", () => {
  test("prefers supported tokenizer.json", () => {
    const directory = createTempDir("tokenizer-json");
    writeFileSync(
      join(directory, "tokenizer.json"),
      JSON.stringify({
        model: {
          type: "BPE",
          vocab: { H: 0, i: 1, "<|endoftext|>": 2 },
          merges: [],
          unk_token: "<|endoftext|>",
        },
        added_tokens: [{ id: 2, content: "<|endoftext|>", special: true }],
        pre_tokenizer: {
          type: "ByteLevel",
          add_prefix_space: false,
          trim_offsets: true,
          use_regex: true,
        },
        decoder: { type: "ByteLevel", add_prefix_space: true, trim_offsets: true, use_regex: true },
      }),
    );
    writeFileSync(
      join(directory, "tokenizer_config.json"),
      JSON.stringify({ eos_token: "<|endoftext|>", bos_token: "<|endoftext|>" }),
    );

    const tokenizer = loadTokenizer(directory);
    expect(tokenizer.encode("Hi")).toEqual([0, 1]);
  });

  test("auto-detects tekken.json snapshots", () => {
    const directory = createTempDir("tekken-json");
    writeFileSync(
      join(directory, "tekken.json"),
      JSON.stringify({
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
        ],
      }),
    );

    const tokenizer = loadTokenizer(directory);
    expect(tokenizer.encode("Hi")).toEqual([1, 4, 5]);
  });
});
