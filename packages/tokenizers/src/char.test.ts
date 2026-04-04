import { describe, expect, test } from "bun:test";
import { CharTokenizer } from "./char";

describe("CharTokenizer", () => {
  const sampleText = "hello world";
  const tokenizer = CharTokenizer.fromText(sampleText);

  test("fromText builds sorted vocabulary", () => {
    const vocab = tokenizer.vocab;
    for (let i = 1; i < vocab.length; i++) {
      const prev = vocab[i - 1];
      const curr = vocab[i];
      if (prev !== undefined && curr !== undefined) {
        expect(prev < curr).toBe(true);
      }
    }
  });

  test("vocabSize matches unique characters", () => {
    const unique = new Set(sampleText).size;
    expect(tokenizer.vocabSize).toBe(unique);
  });

  test("encode/decode roundtrip", () => {
    const tokens = tokenizer.encode(sampleText);
    const decoded = tokenizer.decode(tokens);
    expect(decoded).toBe(sampleText);
  });

  test("encode throws on unknown character", () => {
    expect(() => tokenizer.encode("xyz!")).toThrow("unknown character");
  });

  test("decode throws on out-of-range token ID", () => {
    expect(() => tokenizer.decode([999])).toThrow("out of range");
  });

  test("fromVocab restores tokenizer", () => {
    const restored = CharTokenizer.fromVocab(tokenizer.vocab);
    expect(restored.vocabSize).toBe(tokenizer.vocabSize);
    const tokens = restored.encode(sampleText);
    expect(restored.decode(tokens)).toBe(sampleText);
  });
});
