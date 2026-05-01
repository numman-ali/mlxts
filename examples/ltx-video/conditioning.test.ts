import { describe, expect, test } from "bun:test";
import { full, type MxArray } from "@mlxts/core";
import { SentencePieceTokenizer } from "@mlxts/tokenizers";
import type { T5EncoderModelOptions, T5EncoderModelOutput } from "@mlxts/transformers";

import { createLtxVideoPromptConditioner, type LtxVideoT5TextEncoder } from "./conditioning";

type EncoderCall = {
  shape: readonly number[];
  outputHiddenStates: boolean;
};

class FakeT5Encoder implements LtxVideoT5TextEncoder {
  readonly calls: EncoderCall[] = [];
  disposed = false;

  constructor(
    readonly hiddenSize: number,
    readonly hiddenValue: number,
  ) {}

  run(inputIds: MxArray, options: T5EncoderModelOptions = {}): T5EncoderModelOutput {
    this.calls.push({
      shape: [...inputIds.shape],
      outputHiddenStates: options.outputHiddenStates === true,
    });
    const batch = inputIds.shape[0] ?? 1;
    const sequenceLength = inputIds.shape[1] ?? 1;
    return {
      lastHiddenState: full([batch, sequenceLength, this.hiddenSize], this.hiddenValue),
    };
  }

  [Symbol.dispose](): void {
    this.disposed = true;
  }
}

function t5Tokenizer(): SentencePieceTokenizer {
  return new SentencePieceTokenizer({
    pieces: [
      { piece: "<pad>", score: 0, type: 3 },
      { piece: "</s>", score: 0, type: 3 },
      { piece: "<unk>", score: 0, type: 2 },
      { piece: "▁cat", score: 5, type: 1 },
      { piece: "▁dog", score: 4, type: 1 },
    ],
    byteFallback: false,
    unkId: 2,
    eosId: 1,
    padId: 0,
  });
}

function expectTensorValues(actual: ArrayLike<number>, expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index += 1) {
    const actualValue = actual[index];
    const expectedValue = expected[index];
    if (actualValue === undefined || expectedValue === undefined) {
      throw new Error("expectTensorValues: missing value.");
    }
    expect(actualValue).toBeCloseTo(expectedValue, 5);
  }
}

describe("LTX-Video example prompt conditioning", () => {
  test("encodes T5 sequence embeddings and masks for CFG", () => {
    const textEncoder = new FakeT5Encoder(5, 13);
    const conditioner = createLtxVideoPromptConditioner({
      tokenizer: t5Tokenizer(),
      textEncoder,
    });

    const result = conditioner.encodePrompt({
      prompt: ["cat cat cat cat", "dog"],
      negativePrompt: "dog",
      includeNegativePrompt: true,
      numVideosPerPrompt: 2,
      maxSequenceLength: 4,
    });

    try {
      expect(result.batchSize).toBe(4);
      expect(result.promptTruncated).toBe(true);
      expect(result.negativePromptTruncated).toBe(false);
      expect(textEncoder.calls).toEqual([
        { shape: [2, 4], outputHiddenStates: false },
        { shape: [2, 4], outputHiddenStates: false },
      ]);
      expect(result.conditioning.promptEmbeds.shape).toEqual([4, 4, 5]);
      expect(result.conditioning.promptAttentionMask.shape).toEqual([4, 4]);
      expect(result.conditioning.negativePromptEmbeds?.shape).toEqual([4, 4, 5]);
      expect(result.conditioning.negativePromptAttentionMask?.shape).toEqual([4, 4]);

      expectTensorValues(
        Array.from(result.conditioning.promptEmbeds.toTypedArray()).slice(0, 5),
        [13, 13, 13, 13, 13],
      );
      expectTensorValues(
        Array.from(result.conditioning.promptAttentionMask.toTypedArray()).slice(0, 4),
        [1, 1, 1, 1],
      );
    } finally {
      result[Symbol.dispose]();
      conditioner[Symbol.dispose]();
    }
    expect(textEncoder.disposed).toBe(true);
  });

  test("omits negative conditioning when guidance is inactive", () => {
    const conditioner = createLtxVideoPromptConditioner({
      tokenizer: t5Tokenizer(),
      textEncoder: new FakeT5Encoder(5, 13),
    });

    const result = conditioner.encodePrompt({
      prompt: "cat",
      maxSequenceLength: 5,
    });

    try {
      expect(result.batchSize).toBe(1);
      expect(result.promptTruncated).toBe(false);
      expect(result.negativePromptTruncated).toBe(false);
      expect(result.conditioning.promptEmbeds.shape).toEqual([1, 5, 5]);
      expect(result.conditioning.promptAttentionMask.shape).toEqual([1, 5]);
      expect(result.conditioning.negativePromptEmbeds).toBeUndefined();
      expect(result.conditioning.negativePromptAttentionMask).toBeUndefined();
    } finally {
      result[Symbol.dispose]();
      conditioner[Symbol.dispose]();
    }
  });

  test("validates prompt batches and sequence bounds", () => {
    const conditioner = createLtxVideoPromptConditioner({
      tokenizer: t5Tokenizer(),
      textEncoder: new FakeT5Encoder(5, 13),
    });

    expect(() =>
      conditioner.encodePrompt({
        prompt: ["cat", "dog"],
        negativePrompt: ["cat"],
        includeNegativePrompt: true,
      }),
    ).toThrow("negativePrompt batch size");

    expect(() =>
      conditioner.encodePrompt({
        prompt: "cat",
        numVideosPerPrompt: 0,
      }),
    ).toThrow("numVideosPerPrompt");

    expect(() =>
      conditioner.encodePrompt({
        prompt: "cat",
        maxSequenceLength: 129,
      }),
    ).toThrow("maxSequenceLength");

    conditioner[Symbol.dispose]();
    expect(() => conditioner.encodePrompt({ prompt: "cat" })).toThrow("disposed");
  });
});
