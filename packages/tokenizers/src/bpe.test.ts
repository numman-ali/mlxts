import { describe, expect, test } from "bun:test";

import { loadBPEFromTokenizerJson } from "./bpe";

function createPhiStyleTokenizer() {
  return loadBPEFromTokenizerJson(
    {
      model: {
        type: "BPE",
        vocab: {
          H: 0,
          i: 1,
          "!": 2,
          Hello: 3,
          world: 4,
          Ġworld: 5,
          "<|endoftext|>": 6,
        },
        merges: [["H", "ello"]],
        unk_token: "<|endoftext|>",
        byte_fallback: false,
      },
      added_tokens: [{ id: 6, content: "<|endoftext|>", special: true }],
      pre_tokenizer: {
        type: "ByteLevel",
        add_prefix_space: false,
        trim_offsets: true,
        use_regex: true,
      },
      decoder: {
        type: "ByteLevel",
      },
    },
    {
      eos_token: "<|endoftext|>",
      bos_token: "<|endoftext|>",
      add_bos_token: false,
      add_eos_token: false,
    },
    {
      eos_token: { content: "<|endoftext|>" },
      bos_token: { content: "<|endoftext|>" },
    },
  );
}

function createSentencePieceStyleBPE() {
  return loadBPEFromTokenizerJson(
    {
      model: {
        type: "BPE",
        vocab: {
          "▁": 0,
          H: 1,
          i: 2,
          "▁Hi": 3,
          "<s>": 4,
        },
        merges: [["▁", "Hi"]],
        unk_token: "<s>",
        byte_fallback: false,
      },
      added_tokens: [{ id: 4, content: "<s>", special: true }],
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
      bos_token: "<s>",
      add_bos_token: true,
      add_eos_token: false,
    },
    {
      bos_token: { content: "<s>" },
    },
  );
}

function createGemmaStyleTokenizer() {
  return loadBPEFromTokenizerJson(
    {
      model: {
        type: "BPE",
        vocab: {
          "<pad>": 0,
          "<eos>": 1,
          "<bos>": 2,
          "<unk>": 3,
          "▁": 4,
          H: 5,
          i: 6,
          "!": 7,
          "▁Hi": 8,
        },
        merges: [["▁", "Hi"]],
        unk_token: "<unk>",
        byte_fallback: true,
      },
      added_tokens: [
        { id: 0, content: "<pad>", special: true },
        { id: 1, content: "<eos>", special: true },
        { id: 2, content: "<bos>", special: true },
        { id: 3, content: "<unk>", special: true },
      ],
      pre_tokenizer: {
        type: "Split",
        pattern: { String: " " },
        behavior: "MergedWithPrevious",
        invert: false,
      },
      normalizer: {
        type: "Replace",
        pattern: { String: " " },
        content: "▁",
      },
      decoder: {
        type: "Sequence",
        decoders: [
          { type: "Replace", pattern: { String: "▁" }, content: " " },
          { type: "ByteFallback" },
          { type: "Fuse" },
        ],
      },
      post_processor: {
        type: "TemplateProcessing",
        single: [
          {
            SpecialToken: {
              id: "<bos>",
              type_id: 0,
            },
          },
          {
            Sequence: {
              id: "A",
              type_id: 0,
            },
          },
        ],
        special_tokens: {
          "<bos>": {
            id: "<bos>",
            ids: [2],
            tokens: ["<bos>"],
          },
        },
      },
    },
    {
      bos_token: "<bos>",
      eos_token: "<eos>",
      pad_token: "<pad>",
      unk_token: "<unk>",
      add_bos_token: true,
      add_eos_token: false,
    },
    {
      bos_token: { content: "<bos>" },
      eos_token: { content: "<eos>" },
      pad_token: { content: "<pad>" },
      unk_token: { content: "<unk>" },
    },
  );
}

function createGemmaTurnTokenizer() {
  return loadBPEFromTokenizerJson(
    {
      model: {
        type: "BPE",
        vocab: {
          "<pad>": 0,
          "<eos>": 1,
          "<bos>": 2,
          "<unk>": 3,
          "\n": 107,
          user: 2364,
          model: 4368,
          Hello: 9259,
          "▁there": 993,
        },
        merges: [],
        unk_token: "<unk>",
        byte_fallback: false,
      },
      added_tokens: [
        { id: 0, content: "<pad>", special: true },
        { id: 1, content: "<eos>", special: true },
        { id: 2, content: "<bos>", special: true },
        { id: 3, content: "<unk>", special: true },
        { id: 105, content: "<|turn>", special: true },
        { id: 106, content: "<turn|>", special: true },
      ],
      pre_tokenizer: {
        type: "Split",
        pattern: { String: " " },
        behavior: "MergedWithPrevious",
        invert: false,
      },
      decoder: {
        type: "Sequence",
        decoders: [
          { type: "Replace", pattern: { String: "▁" }, content: " " },
          { type: "ByteFallback" },
          { type: "Fuse" },
        ],
      },
    },
    {
      bos_token: "<bos>",
      eos_token: "<eos>",
      pad_token: "<pad>",
      unk_token: "<unk>",
      add_bos_token: false,
      add_eos_token: false,
    },
    {
      bos_token: { content: "<bos>" },
      eos_token: { content: "<eos>" },
      pad_token: { content: "<pad>" },
      unk_token: { content: "<unk>" },
    },
  );
}

describe("BPETokenizer", () => {
  test("encodes and decodes a ByteLevel tokenizer", () => {
    const tokenizer = createPhiStyleTokenizer();
    const ids = tokenizer.encode("Hi!");
    expect(ids).toEqual([0, 1, 2]);
    expect(tokenizer.decode(ids)).toBe("Hi!");
  });

  test("returns offsets when requested", () => {
    const tokenizer = createPhiStyleTokenizer();
    const encoded = tokenizer.encodeWithOffsets("Hi!", { returnOffsets: true });
    expect(encoded.offsets).toBeDefined();
    const offsets = encoded.offsets;
    if (offsets === undefined) {
      throw new Error("Expected offsets to be present");
    }
    expect(encoded.ids.length).toBe(offsets.length);
    expect(encoded.specialTokensMask).toEqual([0, 0, 0]);
  });

  test("supports sentencepiece-style BPE tokenizers without a ByteLevel pre-tokenizer", () => {
    const tokenizer = createSentencePieceStyleBPE();
    const ids = tokenizer.encode("Hi");
    expect(ids).toEqual([4, 3]);
    expect(tokenizer.decode(ids, { skipSpecialTokens: true })).toBe("Hi");
  });

  test("supports Gemma-style Split pre-tokenizers for sentencepiece-style BPE", () => {
    const tokenizer = createGemmaStyleTokenizer();
    const ids = tokenizer.encode("Hi!");
    expect(ids).toEqual([2, 8, 7]);
    expect(tokenizer.decode(ids, { skipSpecialTokens: true })).toBe("Hi!");
  });

  test("matches inline special turn tokens before sentencepiece segmentation", () => {
    const tokenizer = createGemmaTurnTokenizer();
    const ids = tokenizer.encode("<bos><|turn>user\nHello there<turn|>\n<|turn>model\n", {
      addSpecialTokens: false,
    });
    expect(ids).toEqual([2, 105, 2364, 107, 9259, 993, 106, 107, 105, 4368, 107]);
  });

  test("matches inline added tokens for ByteLevel tokenizers", () => {
    const tokenizer = createPhiStyleTokenizer();
    const ids = tokenizer.encode("Hi<|endoftext|>Hi", { addSpecialTokens: false });
    expect(ids).toEqual([0, 1, 6, 0, 1]);
  });

  test("decodes added tokens whose ids extend beyond the base vocab range", () => {
    const tokenizer = loadBPEFromTokenizerJson({
      model: {
        type: "BPE",
        vocab: {
          a: 0,
          b: 1,
        },
        merges: [],
        byte_fallback: false,
      },
      added_tokens: [
        { id: 10, content: "<think>", special: false },
        { id: 11, content: "</think>", special: false },
      ],
      pre_tokenizer: {
        type: "ByteLevel",
        add_prefix_space: false,
        trim_offsets: true,
        use_regex: true,
      },
      decoder: {
        type: "ByteLevel",
      },
    });

    expect(tokenizer.vocabSize).toBe(12);
    expect(tokenizer.decode([10, 0, 11], { skipSpecialTokens: false })).toBe("<think>a</think>");
  });
});
