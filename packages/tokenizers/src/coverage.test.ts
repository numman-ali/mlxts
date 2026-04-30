import { afterEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { loadBPEFromTokenizerJson } from "./bpe/bpe";
import {
  decodeByteLevelTokens,
  encodeByteLevelSegment,
  splitByteLevelText,
} from "./bpe/byte-level";
import { CharTokenizer } from "./char";
import { UnsupportedTokenizerError } from "./errors";
import { loadSentencePiece, loadTokenizer, loadTokenizerJson } from "./load";
import { SentencePieceTokenizer } from "./sentencepiece";
import { parseSentencePieceModel } from "./sentencepiece-proto";

const tempRoots: string[] = [];

function createTempDir(prefix: string): string {
  const directory = mkdtempSync(join(tmpdir(), prefix));
  tempRoots.push(directory);
  return directory;
}

afterEach(() => {
  while (tempRoots.length > 0) {
    const directory = tempRoots.pop();
    if (directory !== undefined) {
      rmSync(directory, { recursive: true, force: true });
    }
  }
});

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

function encodeFixed32(fieldNumber: number, value: number): number[] {
  const buffer = new ArrayBuffer(4);
  new DataView(buffer).setUint32(0, value, true);
  return [...encodeTag(fieldNumber, 5), ...Array.from(new Uint8Array(buffer))];
}

function encodeFixed64(fieldNumber: number, value: bigint): number[] {
  const buffer = new ArrayBuffer(8);
  new DataView(buffer).setBigUint64(0, value, true);
  return [...encodeTag(fieldNumber, 1), ...Array.from(new Uint8Array(buffer))];
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

function createSentencePieceModelBytes(
  options: {
    modelType?: number;
    byteFallback?: boolean;
    includeUnkId?: boolean;
    pieces?: Array<{ piece: string; score: number; type?: number }>;
  } = {},
): Uint8Array {
  const trainer: number[] = [];
  if (options.modelType !== undefined) {
    trainer.push(...encodeInt(3, options.modelType));
  } else {
    trainer.push(...encodeInt(3, 1));
  }
  if (options.byteFallback ?? true) {
    trainer.push(...encodeInt(35, 1));
  }
  if (options.includeUnkId !== false) {
    trainer.push(...encodeInt(40, 0));
  }
  trainer.push(...encodeInt(41, 1), ...encodeInt(42, 2));

  const pieces = options.pieces ?? [
    { piece: "<unk>", score: 0, type: 2 },
    { piece: "<s>", score: 0, type: 3 },
    { piece: "</s>", score: 0, type: 3 },
    { piece: "▁hi", score: 5 },
    { piece: "<0x21>", score: 1, type: 6 },
  ];

  const root = [
    ...pieces.flatMap((entry) =>
      encodeMessage(1, pieceMessage(entry.piece, entry.score, entry.type ?? 1)),
    ),
    ...encodeMessage(2, trainer),
    ...encodeMessage(3, [...encodeInt(3, 1), ...encodeInt(4, 1), ...encodeInt(5, 1)]),
  ];

  return new Uint8Array(root);
}

describe("tokenizer coverage", () => {
  test("UnsupportedTokenizerError keeps a stable name and CharTokenizer covers batches, offsets, and getters", () => {
    const error = new UnsupportedTokenizerError("unsupported");
    expect(error.name).toBe("UnsupportedTokenizerError");

    const tokenizer = CharTokenizer.fromVocab(["a", "🙂", "b"]);
    const single = tokenizer.encodeWithOffsets("a🙂", { returnOffsets: true });
    const batch = tokenizer.encodeBatch(["a", "🙂"], { returnOffsets: true });

    expect(single.ids).toEqual([0, 1]);
    expect(single.offsets).toEqual([
      { start: 0, end: 1 },
      { start: 1, end: 3 },
    ]);
    expect(batch[1]?.offsets).toEqual([{ start: 0, end: 2 }]);
    expect(tokenizer.decodeBatch([[0, 1], [2]])).toEqual(["a🙂", "b"]);
    expect(tokenizer.vocabSize).toBe(3);
    expect(tokenizer.bosTokenId).toBeUndefined();
    expect(tokenizer.eosTokenIds).toEqual([]);
    expect(tokenizer.padTokenId).toBeUndefined();
    expect(tokenizer.vocab).toEqual(["a", "🙂", "b"]);
  });

  test("loadTokenizer resolves char text files and explicit missing-file errors", () => {
    const directory = createTempDir("mlxts-tokenizers-char-");
    const textPath = join(directory, "vocab.txt");
    writeFileSync(textPath, "cab");

    const tokenizer = loadTokenizer(textPath, { format: "char" });
    expect(tokenizer.encode("abc")).toEqual([0, 1, 2]);

    expect(() => loadTokenizer(directory, { format: "char" })).toThrow(
      "char format requires a local text or vocab file",
    );
    expect(() => loadTokenizerJson(directory)).toThrow("tokenizer.json was not found");
    expect(() => loadSentencePiece(directory)).toThrow("tokenizer.model was not found");
    expect(() => loadTokenizer(directory)).toThrow(
      "could not find a supported tokenizer.json, vocab.json + merges.txt, tekken.json, or tokenizer.model",
    );
  });

  test("byte-level helpers round-trip bytes and respect regex splitting controls", () => {
    const encoded = encodeByteLevelSegment("é!");
    expect(decodeByteLevelTokens([encoded])).toBe("é!");
    expect(decodeByteLevelTokens(["<0x48>", "<0x69>"])).toBe("Hi");
    expect(splitByteLevelText(" Hello!", false)).toEqual([[" Hello!", 0, 7]]);
    expect(splitByteLevelText(" Hello!", true).map(([segment]) => segment)).toEqual([
      " Hello",
      "!",
    ]);
  });

  test("loadTokenizer falls back from unsupported tokenizer.json to sentencepiece.model", () => {
    const directory = createTempDir("mlxts-tokenizers-fallback-");
    writeFileSync(
      join(directory, "tokenizer.json"),
      JSON.stringify({
        model: {
          type: "WordPiece",
        },
      }),
    );
    writeFileSync(join(directory, "tokenizer.model"), createSentencePieceModelBytes());

    const tokenizer = loadTokenizer(directory);
    expect(tokenizer.encode("hi!")).toEqual([1, 3, 4, 2]);
    expect(tokenizer.decode([1, 3, 4, 2], { skipSpecialTokens: true })).toBe("hi!");
  });

  test("loadTokenizer surfaces unsupported tokenizer.json when there is no sentencepiece fallback", () => {
    const directory = createTempDir("mlxts-tokenizers-unsupported-");
    writeFileSync(
      join(directory, "tokenizer.json"),
      JSON.stringify({
        model: {
          type: "WordPiece",
        },
      }),
    );

    expect(() => loadTokenizer(directory)).toThrow(UnsupportedTokenizerError);
  });

  test("loadTokenizer accepts explicit tokenizer file paths and malformed JSON objects still fail clearly", () => {
    const directory = createTempDir("mlxts-tokenizers-explicit-");
    const tokenizerJsonPath = join(directory, "tokenizer.json");
    const tokenizerModelPath = join(directory, "tokenizer.model");
    const tokenizerConfigPath = join(directory, "tokenizer_config.json");

    writeFileSync(
      tokenizerJsonPath,
      JSON.stringify({
        model: {
          type: "BPE",
          vocab: { H: 0, i: 1, "<eos>": 2 },
          merges: [],
          unk_token: "<eos>",
        },
        added_tokens: [{ id: 2, content: "<eos>", special: true }],
        pre_tokenizer: {
          type: "ByteLevel",
          add_prefix_space: false,
          trim_offsets: true,
          use_regex: true,
        },
        decoder: { type: "ByteLevel" },
      }),
    );
    writeFileSync(tokenizerModelPath, createSentencePieceModelBytes());
    writeFileSync(tokenizerConfigPath, JSON.stringify(["bad"]));

    expect(loadTokenizer(tokenizerJsonPath, { format: "tokenizer-json" }).encode("Hi")).toEqual([
      0, 1,
    ]);
    expect(
      loadTokenizer(tokenizerModelPath, { format: "sentencepiece-model" }).encode("hi!"),
    ).toEqual([1, 3, 4, 2]);
    expect(() =>
      loadTokenizerJson({
        tokenizerJsonPath,
        tokenizerConfigPath,
      }),
    ).toThrow("tokenizer_config.json must be an object");
    expect(loadTokenizer({ tokenizerModelPath }).encode("hi!")).toEqual([1, 3, 4, 2]);

    const unresolvedSource = join(directory, "remote-like-repo");
    expect(() => loadTokenizer(unresolvedSource)).toThrow(
      "could not find a supported tokenizer.json, vocab.json + merges.txt, tekken.json, or tokenizer.model",
    );

    const brokenDirectory = createTempDir("mlxts-tokenizers-broken-json-");
    writeFileSync(join(brokenDirectory, "tokenizer.json"), "{");
    expect(() => loadTokenizer(brokenDirectory)).toThrow(SyntaxError);
  });

  test("parseSentencePieceModel validates truncated, empty, and malformed protobuf payloads", () => {
    expect(() => parseSentencePieceModel(new Uint8Array([0x0a]))).toThrow("truncated");
    expect(() => parseSentencePieceModel(new Uint8Array([0x0b]))).toThrow(
      "Unsupported protobuf wire type 3",
    );
    expect(() => parseSentencePieceModel(new Uint8Array([0x0a, 0x05, 0x41]))).toThrow(
      "length-delimited field exceeds file size",
    );
    expect(() => parseSentencePieceModel(new Uint8Array())).toThrow("does not contain any pieces");
    expect(() =>
      SentencePieceTokenizer.fromModelBytes(createSentencePieceModelBytes({ modelType: 2 })),
    ).toThrow("only unigram models are supported");
  });

  test("parseSentencePieceModel skips unknown protobuf fields and records optional ids", () => {
    const piece = [
      ...pieceMessage("▁hi", 5),
      ...encodeInt(9, 7),
      ...encodeFixed64(10, 123n),
      ...encodeMessage(11, [1, 2, 3]),
      ...encodeFixed32(12, 99),
    ];
    const trainer = [
      ...encodeInt(3, 1),
      ...encodeInt(35, 1),
      ...encodeInt(40, 0),
      ...encodeInt(41, 1),
      ...encodeInt(42, 2),
      ...encodeInt(43, 9),
      ...encodeInt(99, 1),
      ...encodeMessage(44, [1]),
    ];
    const normalizer = [
      ...encodeInt(3, 0),
      ...encodeInt(4, 0),
      ...encodeInt(5, 0),
      ...encodeMessage(6, [0]),
    ];
    const bytes = new Uint8Array([
      ...encodeMessage(1, pieceMessage("<unk>", 0, 2)),
      ...encodeMessage(1, piece),
      ...encodeMessage(2, trainer),
      ...encodeMessage(3, normalizer),
    ]);

    const model = parseSentencePieceModel(bytes);
    expect(model.padId).toBe(9);
    expect(model.byteFallback).toBe(true);
    expect(model.addDummyPrefix).toBe(false);
    expect(model.removeExtraWhitespaces).toBe(false);
    expect(model.escapeWhitespaces).toBe(false);
    expect(model.pieces[1]?.piece).toBe("▁hi");
  });

  test("SentencePieceTokenizer covers batch helpers, byte fallback, and missing unk handling", () => {
    const tokenizer = SentencePieceTokenizer.fromModelBytes(createSentencePieceModelBytes());
    const batch = tokenizer.encodeBatch(["hi!", "hi"], { returnOffsets: true });

    expect(tokenizer.vocabSize).toBe(5);
    expect(tokenizer.bosTokenId).toBe(1);
    expect(tokenizer.eosTokenIds).toEqual([2]);
    expect(tokenizer.padTokenId).toBeUndefined();
    expect(batch[0]?.ids).toEqual([1, 3, 4, 2]);
    expect(batch[0]?.offsets?.length).toBe(batch[0]?.ids.length);
    expect(
      tokenizer.decodeBatch(
        [
          [1, 3, 4, 2],
          [1, 3, 2],
        ],
        { skipSpecialTokens: true },
      ),
    ).toEqual(["hi!", "hi"]);

    const noUnk = SentencePieceTokenizer.fromModelBytes(
      createSentencePieceModelBytes({
        byteFallback: false,
        includeUnkId: false,
        pieces: [{ piece: "a", score: 1 }],
      }),
    );
    expect(() => noUnk.encode("z", { addSpecialTokens: false })).toThrow("unk_id is missing");
  });

  test("BPE tokenizers cover special tokens, byte fallback, and parser validation", () => {
    const byteLevel = loadBPEFromTokenizerJson(
      {
        model: {
          type: "BPE",
          vocab: {
            H: 0,
            i: 1,
            "<unk>": 2,
            "<bos>": 3,
            "<eos>": 4,
          },
          merges: [],
          unk_token: "<unk>",
          byte_fallback: false,
        },
        added_tokens: [
          { id: 3, content: "<bos>", special: true },
          { id: 4, content: "<eos>", special: true },
        ],
        pre_tokenizer: {
          type: "ByteLevel",
          add_prefix_space: false,
          trim_offsets: true,
          use_regex: true,
        },
        decoder: { type: "ByteLevel" },
      },
      {
        bos_token: "<bos>",
        eos_token: "<eos>",
        add_bos_token: true,
        add_eos_token: true,
      },
    );

    const encoded = byteLevel.encodeBatch(["Hi", "?"], { returnOffsets: true });
    expect(byteLevel.bosTokenId).toBe(3);
    expect(byteLevel.eosTokenIds).toEqual([4]);
    expect(byteLevel.encode("?", { addSpecialTokens: false })).toEqual([2]);
    expect(encoded[0]?.ids).toEqual([3, 0, 1, 4]);
    expect(encoded[0]?.specialTokensMask).toEqual([1, 0, 0, 1]);
    expect(byteLevel.decode([3, 0, 1, 4], { skipSpecialTokens: true })).toBe("Hi");
    expect(byteLevel.decodeBatch([[3, 0, 1, 4]], { skipSpecialTokens: true })).toEqual(["Hi"]);

    const sentencePieceStyle = loadBPEFromTokenizerJson(
      {
        model: {
          type: "BPE",
          vocab: {
            "▁": 0,
            "<0xC3>": 1,
            "<0xA9>": 2,
            "<unk>": 3,
          },
          merges: [],
          unk_token: "<unk>",
          byte_fallback: true,
        },
        added_tokens: [],
        pre_tokenizer: null,
        decoder: {
          type: "Sequence",
          decoders: [
            { type: "Replace", pattern: { String: "▁" }, content: " " },
            { type: "Strip", content: " ", start: 1, stop: 0 },
          ],
        },
      },
      {
        add_bos_token: false,
        add_eos_token: false,
      },
    );
    expect(sentencePieceStyle.encode("é", { addSpecialTokens: false })).toEqual([0, 1, 2]);

    expect(() =>
      loadBPEFromTokenizerJson({
        model: {
          type: "WordPiece",
        },
      }),
    ).toThrow('tokenizer.json.model.type "WordPiece" is not supported');
    expect(() =>
      loadBPEFromTokenizerJson({
        model: {
          type: "BPE",
          vocab: { a: 0 },
          merges: ["broken"],
        },
        pre_tokenizer: {
          type: "ByteLevel",
        },
      }),
    ).toThrow('legacy merge entry "broken" is malformed');
    expect(() =>
      loadBPEFromTokenizerJson({
        model: {
          type: "BPE",
          vocab: { a: 0 },
          merges: [],
        },
        pre_tokenizer: {
          type: "Metaspace",
        },
      }),
    ).toThrow('tokenizer.json.pre_tokenizer.type "Metaspace" is not supported');

    const merged = loadBPEFromTokenizerJson(
      {
        model: {
          type: "BPE",
          vocab: {
            a: 0,
            b: 1,
            ab: 2,
            "<unk>": 3,
            "<pad>": 4,
            "<bos>": 5,
            "<eos>": 6,
          },
          merges: [["a", "b"]],
          unk_token: "<unk>",
        },
        added_tokens: [
          { id: 4, content: "<pad>", special: true },
          { id: 5, content: "<bos>", special: true },
          { id: 6, content: "<eos>", special: true },
        ],
        pre_tokenizer: {
          type: "ByteLevel",
          add_prefix_space: false,
          use_regex: true,
        },
      },
      {
        bos_token: { content: "<bos>" },
        eos_token: "<eos>",
        pad_token: { content: "<pad>" },
        add_bos_token: true,
        add_eos_token: true,
      },
    );
    expect(merged.padTokenId).toBe(4);
    expect(merged.encode("ab")).toEqual([5, 2, 6]);
    expect(merged.decode([5, 2, 6], { skipSpecialTokens: true })).toBe("ab");

    const sequenceWrapped = loadBPEFromTokenizerJson({
      model: {
        type: "BPE",
        vocab: { Hello: 0, "!": 1, "<unk>": 2 },
        merges: [],
        unk_token: "<unk>",
      },
      pre_tokenizer: {
        type: "Sequence",
        pretokenizers: [
          {
            type: "Split",
            pattern: { Regex: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+" },
            behavior: "Isolated",
            invert: false,
          },
          {
            type: "ByteLevel",
            add_prefix_space: false,
            use_regex: false,
          },
        ],
      },
      decoder: { type: "ByteLevel" },
    });
    expect(sequenceWrapped.encode("Hello!", { addSpecialTokens: false })).toEqual([0, 1]);

    expect(() => loadBPEFromTokenizerJson([])).toThrow("tokenizer.json must be an object");
    expect(() =>
      loadBPEFromTokenizerJson({
        model: {
          type: 123,
          vocab: {},
          merges: [],
        },
      }),
    ).toThrow("tokenizer.json.model.type must be a string");
    expect(() =>
      loadBPEFromTokenizerJson({
        model: {
          type: "BPE",
          vocab: { a: "0" },
          merges: [],
        },
        pre_tokenizer: { type: "ByteLevel" },
      }),
    ).toThrow('tokenizer.model.vocab["a"] must be a non-negative integer');
    expect(() =>
      loadBPEFromTokenizerJson({
        model: {
          type: "BPE",
          vocab: { a: 0 },
          merges: [["a", 1]],
        },
        pre_tokenizer: { type: "ByteLevel" },
      }),
    ).toThrow("merge entries must contain two strings");
    expect(() =>
      loadBPEFromTokenizerJson({
        model: {
          type: "BPE",
          vocab: { a: 0 },
          merges: [{}],
        },
        pre_tokenizer: { type: "ByteLevel" },
      }),
    ).toThrow("tokenizer.model.merges must contain string pairs");
    expect(() =>
      loadBPEFromTokenizerJson({
        model: {
          type: "BPE",
          vocab: { a: 0 },
          merges: [],
        },
        added_tokens: [{ id: "bad", content: "<pad>" }],
        pre_tokenizer: { type: "ByteLevel" },
      }),
    ).toThrow('tokenizer.added_tokens["<pad>"].id must be a non-negative integer');
    expect(() =>
      loadBPEFromTokenizerJson({
        model: {
          type: "BPE",
          vocab: { a: 0 },
          merges: [],
        },
        pre_tokenizer: null,
        decoder: { type: "ByteLevel" },
      }),
    ).toThrow("BPE tokenizer without a pre_tokenizer is only supported");
    expect(() =>
      loadBPEFromTokenizerJson({
        model: {
          type: "BPE",
          vocab: { a: 0 },
          merges: [],
        },
        pre_tokenizer: {
          type: "Sequence",
          pretokenizers: {},
        },
      }),
    ).toThrow("tokenizer.json.pre_tokenizer.pretokenizers must be an array");
  });
});
