import { describe, expect, test } from "bun:test";
import { SentencePieceTokenizer } from "./sentencepiece";
import { parseSentencePieceModel } from "./sentencepiece-proto";
import { encodeT5TextInput } from "./t5";

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

function encodeSignedInt(fieldNumber: number, value: number): number[] {
  const bytes: number[] = [];
  let remaining = BigInt.asUintN(64, BigInt(value));
  while (remaining >= 0x80n) {
    bytes.push(Number(remaining & 0x7fn) | 0x80);
    remaining >>= 7n;
  }
  bytes.push(Number(remaining));
  return [...encodeTag(fieldNumber, 0), ...bytes];
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
    ...encodeInt(35, 1),
    ...encodeInt(40, 0),
    ...encodeInt(41, 1),
    ...encodeInt(42, 2),
  ];
  const normalizer = [...encodeInt(3, 1), ...encodeInt(4, 1), ...encodeInt(5, 1)];
  const root = [
    ...encodeMessage(1, pieceMessage("<unk>", 0, 2)),
    ...encodeMessage(1, pieceMessage("<s>", 0, 3)),
    ...encodeMessage(1, pieceMessage("</s>", 0, 3)),
    ...encodeMessage(1, pieceMessage("▁hi", 5)),
    ...encodeMessage(1, pieceMessage("<0x21>", 1, 6)),
    ...encodeMessage(2, trainer),
    ...encodeMessage(3, normalizer),
  ];
  return new Uint8Array(root);
}

function createNegativeSpecialIdModelBytes(): Uint8Array {
  const trainer = [
    ...encodeInt(3, 1),
    ...encodeInt(40, 0),
    ...encodeSignedInt(41, -1),
    ...encodeInt(42, 1),
    ...encodeSignedInt(43, -1),
  ];
  const normalizer = [...encodeInt(3, 1), ...encodeInt(4, 1), ...encodeInt(5, 1)];
  const root = [
    ...encodeMessage(1, pieceMessage("<unk>", 0, 2)),
    ...encodeMessage(1, pieceMessage("</s>", 0, 3)),
    ...encodeMessage(1, pieceMessage("▁hi", 5)),
    ...encodeMessage(2, trainer),
    ...encodeMessage(3, normalizer),
  ];
  return new Uint8Array(root);
}

function t5Tokenizer(): SentencePieceTokenizer {
  return new SentencePieceTokenizer({
    pieces: [
      { piece: "<pad>", score: 0, type: 3 },
      { piece: "</s>", score: 0, type: 3 },
      { piece: "<unk>", score: 0, type: 2 },
      { piece: "▁cat", score: 5, type: 1 },
      { piece: "▁sat", score: 4, type: 1 },
      { piece: "▁on", score: 3, type: 1 },
      { piece: "▁mat", score: 2, type: 1 },
    ],
    byteFallback: false,
    unkId: 2,
    eosId: 1,
    padId: 0,
  });
}

describe("SentencePieceTokenizer", () => {
  test("parses the minimal SentencePiece protobuf subset", () => {
    const model = parseSentencePieceModel(createSentencePieceModelBytes());
    expect(model.pieces.map((entry) => entry.piece)).toEqual([
      "<unk>",
      "<s>",
      "</s>",
      "▁hi",
      "<0x21>",
    ]);
    expect(model.unkId).toBe(0);
    expect(model.bosId).toBe(1);
    expect(model.eosId).toBe(2);
    expect(model.byteFallback).toBe(true);
  });

  test("encodes and decodes a unigram sentencepiece model", () => {
    const tokenizer = SentencePieceTokenizer.fromModelBytes(createSentencePieceModelBytes());
    const ids = tokenizer.encode("hi!");
    expect(ids).toEqual([1, 3, 4, 2]);
    expect(tokenizer.decode(ids, { skipSpecialTokens: true })).toBe("hi!");
  });

  test("parses negative SentencePiece special ids as disabled tokens", () => {
    const model = parseSentencePieceModel(createNegativeSpecialIdModelBytes());
    expect(model.bosId).toBe(-1);
    expect(model.padId).toBe(-1);

    const tokenizer = SentencePieceTokenizer.fromModelBytes(createNegativeSpecialIdModelBytes());
    expect(tokenizer.bosTokenId).toBeUndefined();
    expect(tokenizer.padTokenId).toBeUndefined();
    expect(tokenizer.encode("hi")).toEqual([2, 1]);
  });

  test("encodes fixed-length T5 inputs with EOS truncation and padding", () => {
    const tokenizer = t5Tokenizer();

    expect(encodeT5TextInput(tokenizer, "cat sat", { maxLength: 5 })).toEqual({
      inputIds: [3, 4, 1, 0, 0],
      attentionMask: [1, 1, 1, 0, 0],
      truncated: false,
    });

    expect(encodeT5TextInput(tokenizer, "cat sat on mat", { maxLength: 3 })).toEqual({
      inputIds: [3, 4, 1],
      attentionMask: [1, 1, 1],
      truncated: true,
    });
  });

  test("T5 fixed-length encoding requires usable padding metadata", () => {
    const tokenizer = new SentencePieceTokenizer({
      pieces: [
        { piece: "<unk>", score: 0, type: 2 },
        { piece: "</s>", score: 0, type: 3 },
        { piece: "▁cat", score: 1, type: 1 },
      ],
      byteFallback: false,
      unkId: 0,
      eosId: 1,
    });

    expect(() => encodeT5TextInput(tokenizer, "cat", { maxLength: 0 })).toThrow("maxLength");
    expect(() => encodeT5TextInput(tokenizer, "cat")).toThrow("PAD");
  });
});
