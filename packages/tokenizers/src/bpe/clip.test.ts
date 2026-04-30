import { describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { UnsupportedTokenizerError } from "../errors";
import { loadTokenizer } from "../load";
import { encodeByteLevelSegment } from "./byte-level";
import { CLIPTokenizer, encodeCLIPTextInput, loadCLIPTokenizer, parseCLIPMergesText } from "./clip";

const END_OF_WORD_SUFFIX = "</w>";

type TestVocabulary = {
  vocab: Record<string, number>;
  merges: Array<[string, string]>;
  ids: Record<string, number>;
};

function withTempDirectory<T>(name: string, run: (directory: string) => T): T {
  const directory = mkdtempSync(join(tmpdir(), `${name}-`));
  try {
    return run(directory);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}

function createTestVocabulary(): TestVocabulary {
  const vocab: Record<string, number> = {};
  const ids: Record<string, number> = {};
  const merges: Array<[string, string]> = [];
  let nextId = 0;

  function addToken(token: string): number {
    const existing = vocab[token];
    if (existing !== undefined) {
      return existing;
    }
    const id = nextId;
    nextId += 1;
    vocab[token] = id;
    return id;
  }

  function addSpecial(token: string, name: string): void {
    ids[name] = addToken(token);
  }

  function addWord(word: string, name: string): void {
    const chars = Array.from(encodeByteLevelSegment(word));
    if (chars.length === 0) {
      throw new Error("test vocabulary cannot add an empty word");
    }

    const initialPieces = chars.map((char, index) =>
      index === chars.length - 1 ? `${char}${END_OF_WORD_SUFFIX}` : char,
    );
    for (const piece of initialPieces) {
      addToken(piece);
    }

    let current = initialPieces[0];
    if (current === undefined) {
      throw new Error("test vocabulary missing first piece");
    }
    for (let index = 1; index < initialPieces.length; index += 1) {
      const right = initialPieces[index];
      if (right === undefined) {
        continue;
      }
      merges.push([current, right]);
      current = `${current}${right}`;
      addToken(current);
    }
    ids[name] = vocab[current] ?? addToken(current);
  }

  addSpecial("<|startoftext|>", "bos");
  addSpecial("<|endoftext|>", "eos");
  addWord("hello", "hello");
  addWord("world", "world");
  addWord("!", "bang");
  addWord("café", "cafe");
  addWord("4", "four");
  addWord("2", "two");
  addWord("long", "long");

  return { vocab, merges, ids };
}

function mergesText(merges: Array<[string, string]>): string {
  return `#version: 0.2\n${merges.map(([left, right]) => `${left} ${right}`).join("\n")}\n`;
}

function createTokenizer(): [CLIPTokenizer, TestVocabulary] {
  const vocabulary = createTestVocabulary();
  const tokenizer = new CLIPTokenizer({
    vocab: vocabulary.vocab,
    merges: vocabulary.merges,
    modelMaxLength: 5,
  });
  return [tokenizer, vocabulary];
}

function tokenId(vocabulary: TestVocabulary, name: string): number {
  const id = vocabulary.ids[name];
  if (id === undefined) {
    throw new Error(`Missing test token id for ${name}`);
  }
  return id;
}

describe("CLIPTokenizer", () => {
  test("encodes CLIP-normalized byte-level BPE with BOS and EOS", () => {
    const [tokenizer, vocabulary] = createTokenizer();

    expect(tokenizer.encode("Hello   WORLD!")).toEqual([
      tokenId(vocabulary, "bos"),
      tokenId(vocabulary, "hello"),
      tokenId(vocabulary, "world"),
      tokenId(vocabulary, "bang"),
      tokenId(vocabulary, "eos"),
    ]);
    expect(
      tokenizer.decode([tokenId(vocabulary, "hello"), tokenId(vocabulary, "world")], {
        skipSpecialTokens: true,
      }),
    ).toBe("hello world");
  });

  test("normalizes unicode to NFC, lowercases text, and splits digits singly", () => {
    const [tokenizer, vocabulary] = createTokenizer();

    expect(tokenizer.encode("CAFE\u0301 42", { addSpecialTokens: false })).toEqual([
      tokenId(vocabulary, "cafe"),
      tokenId(vocabulary, "four"),
      tokenId(vocabulary, "two"),
    ]);
  });

  test("returns offsets and special masks for CLIP prompts", () => {
    const [tokenizer] = createTokenizer();

    const encoding = tokenizer.encodeWithOffsets("Hello!", { returnOffsets: true });

    expect(encoding.specialTokensMask).toEqual([1, 0, 0, 1]);
    expect(encoding.offsets).toEqual([
      { start: 0, end: 0 },
      { start: 0, end: 5 },
      { start: 5, end: 6 },
      { start: 0, end: 0 },
    ]);
  });

  test("prepares fixed-length CLIP text inputs", () => {
    const [tokenizer, vocabulary] = createTokenizer();

    expect(encodeCLIPTextInput(tokenizer, "hello", { maxLength: 5 })).toEqual({
      inputIds: [
        tokenId(vocabulary, "bos"),
        tokenId(vocabulary, "hello"),
        tokenId(vocabulary, "eos"),
        tokenId(vocabulary, "eos"),
        tokenId(vocabulary, "eos"),
      ],
      attentionMask: [1, 1, 1, 0, 0],
      truncated: false,
    });

    expect(encodeCLIPTextInput(tokenizer, "hello world café !", { maxLength: 5 })).toEqual({
      inputIds: [
        tokenId(vocabulary, "bos"),
        tokenId(vocabulary, "hello"),
        tokenId(vocabulary, "world"),
        tokenId(vocabulary, "cafe"),
        tokenId(vocabulary, "eos"),
      ],
      attentionMask: [1, 1, 1, 1, 1],
      truncated: true,
    });
  });

  test("loads Diffusers vocab.json and merges.txt files through tokenizer auto-detection", () => {
    withTempDirectory("mlxts-clip-tokenizer-", (directory) => {
      const vocabulary = createTestVocabulary();
      writeFileSync(join(directory, "vocab.json"), JSON.stringify(vocabulary.vocab));
      writeFileSync(join(directory, "merges.txt"), mergesText(vocabulary.merges));
      writeFileSync(
        join(directory, "tokenizer_config.json"),
        JSON.stringify({ model_max_length: 6, tokenizer_class: "CLIPTokenizer" }),
      );
      writeFileSync(
        join(directory, "tokenizer.json"),
        JSON.stringify({
          model: {
            type: "BPE",
            vocab: { H: 100, i: 101, "<|endoftext|>": tokenId(vocabulary, "eos") },
            merges: [],
            unk_token: "<|endoftext|>",
          },
          pre_tokenizer: {
            type: "ByteLevel",
            add_prefix_space: false,
            trim_offsets: true,
            use_regex: true,
          },
          decoder: { type: "ByteLevel" },
        }),
      );

      const tokenizer = loadTokenizer(directory, { format: "clip-vocab-merges" });
      if (!(tokenizer instanceof CLIPTokenizer)) {
        throw new Error("Expected an explicit CLIP tokenizer load.");
      }
      expect(tokenizer.encode("hello", { addSpecialTokens: false })).toEqual([
        tokenId(vocabulary, "hello"),
      ]);
      expect(encodeCLIPTextInput(tokenizer, "hello").inputIds).toHaveLength(6);

      const autoTokenizer = loadTokenizer(directory);
      expect(autoTokenizer.encode("world", { addSpecialTokens: false })).toEqual([
        tokenId(vocabulary, "world"),
      ]);
    });
  });

  test("parses CLIP merge files and rejects malformed entries", () => {
    expect(parseCLIPMergesText("#version: 0.2\nh e\nhe l\n")).toEqual([
      ["h", "e"],
      ["he", "l"],
    ]);
    expect(() => parseCLIPMergesText("too many pieces\n")).toThrow(UnsupportedTokenizerError);
  });

  test("loads parsed CLIP tokenizer files with configured special tokens", () => {
    const vocabulary = createTestVocabulary();
    const tokenizer = loadCLIPTokenizer(vocabulary.vocab, mergesText(vocabulary.merges), {
      tokenizerConfig: {
        bos_token: "<|startoftext|>",
        eos_token: "<|endoftext|>",
        pad_token: "<|endoftext|>",
        unk_token: "<|endoftext|>",
        model_max_length: 4,
      },
    });

    expect(tokenizer.bosTokenId).toBe(tokenId(vocabulary, "bos"));
    expect(tokenizer.eosTokenIds).toEqual([tokenId(vocabulary, "eos")]);
    expect(tokenizer.padTokenId).toBe(tokenId(vocabulary, "eos"));
    expect(encodeCLIPTextInput(tokenizer, "hello").inputIds).toHaveLength(4);
  });
});
