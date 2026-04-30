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

function encodeVarint(value: number): number[] {
  const bytes: number[] = [];
  let remaining = value >>> 0;
  while (remaining >= 0x80) {
    bytes.push((remaining & 0x7f) | 0x80);
    remaining >>>= 7;
  }
  bytes.push(remaining);
  return bytes;
}

function encodeTag(fieldNumber: number, wireType: number): number[] {
  return encodeVarint((fieldNumber << 3) | wireType);
}

function encodeString(fieldNumber: number, value: string): number[] {
  const data = Array.from(new TextEncoder().encode(value));
  return [...encodeTag(fieldNumber, 2), ...encodeVarint(data.length), ...data];
}

function encodeFloat(fieldNumber: number, value: number): number[] {
  const buffer = new ArrayBuffer(4);
  new DataView(buffer).setFloat32(0, value, true);
  return [...encodeTag(fieldNumber, 5), ...Array.from(new Uint8Array(buffer))];
}

function encodeInt(fieldNumber: number, value: number): number[] {
  return [...encodeTag(fieldNumber, 0), ...encodeVarint(value)];
}

function encodeMessage(fieldNumber: number, bytes: number[]): number[] {
  return [...encodeTag(fieldNumber, 2), ...encodeVarint(bytes.length), ...bytes];
}

function pieceMessage(piece: string, score: number, type = 1): number[] {
  return [...encodeString(1, piece), ...encodeFloat(2, score), ...encodeInt(3, type)];
}

function createSentencePieceModelBytes(): Uint8Array {
  const trainer = [
    ...encodeInt(3, 1),
    ...encodeInt(40, 0),
    ...encodeInt(42, 1),
    ...encodeInt(43, 2),
  ];
  const normalizer = [...encodeInt(3, 1), ...encodeInt(4, 1), ...encodeInt(5, 1)];
  const root = [
    ...encodeMessage(1, pieceMessage("<unk>", 0, 2)),
    ...encodeMessage(1, pieceMessage("</s>", 0, 3)),
    ...encodeMessage(1, pieceMessage("<pad>", 0, 3)),
    ...encodeMessage(1, pieceMessage("▁hi", 5)),
    ...encodeMessage(2, trainer),
    ...encodeMessage(3, normalizer),
  ];
  return new Uint8Array(root);
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

  test("auto-detects Diffusers T5 spiece.model directories", () => {
    const directory = createTempDir("spiece-model");
    writeFileSync(join(directory, "spiece.model"), createSentencePieceModelBytes());

    const tokenizer = loadTokenizer(directory);
    expect(tokenizer.encode("hi")).toEqual([3, 1]);
  });
});
