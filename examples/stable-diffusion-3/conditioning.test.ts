import { describe, expect, test } from "bun:test";
import { full, type MxArray, zeros } from "@mlxts/core";
import { CLIPTokenizer, SentencePieceTokenizer } from "@mlxts/tokenizers";
import type {
  CLIPTextModelOptions,
  CLIPTextProjectionOutput,
  T5EncoderModelOptions,
  T5EncoderModelOutput,
} from "@mlxts/transformers";

import {
  createStableDiffusion3PromptConditioner,
  type StableDiffusion3CLIPTextEncoder,
  type StableDiffusion3T5TextEncoder,
} from "./conditioning";

type EncoderCall = {
  shape: readonly number[];
  outputHiddenStates: boolean;
};

class FakeCLIPProjectionEncoder implements StableDiffusion3CLIPTextEncoder {
  readonly calls: EncoderCall[] = [];
  readonly tokenRows: number[][] = [];
  disposed = false;

  constructor(
    readonly hiddenSize: number,
    readonly projectionSize: number,
    readonly hiddenBase: number,
    readonly projectionValue: number,
    readonly hiddenStateCount = 4,
  ) {}

  run(inputIds: MxArray, options: CLIPTextModelOptions = {}): CLIPTextProjectionOutput {
    this.tokenRows.push(Array.from(inputIds.toTypedArray()));
    this.calls.push({
      shape: [...inputIds.shape],
      outputHiddenStates: options.outputHiddenStates === true,
    });
    const batch = inputIds.shape[0] ?? 1;
    const sequenceLength = inputIds.shape[1] ?? 1;
    const hiddenStates = options.outputHiddenStates
      ? Array.from({ length: this.hiddenStateCount }, (_unused, index) =>
          full([batch, sequenceLength, this.hiddenSize], this.hiddenBase + index),
        )
      : undefined;
    return {
      lastHiddenState: zeros([batch, sequenceLength, this.hiddenSize]),
      pooledOutput: zeros([batch, this.hiddenSize]),
      textEmbeds: full([batch, this.projectionSize], this.projectionValue),
      ...(hiddenStates === undefined ? {} : { hiddenStates }),
    };
  }

  [Symbol.dispose](): void {
    this.disposed = true;
  }
}

class FakeT5Encoder implements StableDiffusion3T5TextEncoder {
  readonly calls: EncoderCall[] = [];
  readonly tokenRows: number[][] = [];
  disposed = false;

  constructor(
    readonly hiddenSize: number,
    readonly hiddenValue: number,
  ) {}

  run(inputIds: MxArray, options: T5EncoderModelOptions = {}): T5EncoderModelOutput {
    this.tokenRows.push(Array.from(inputIds.toTypedArray()));
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

function clipTokenizer(): CLIPTokenizer {
  return new CLIPTokenizer({
    vocab: {
      "<|startoftext|>": 0,
      "<|endoftext|>": 1,
      c: 2,
      a: 3,
      "t</w>": 4,
      d: 5,
      o: 6,
      "g</w>": 7,
    },
    merges: [],
    modelMaxLength: 6,
  });
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

describe("Stable Diffusion 3 example prompt conditioning", () => {
  test("composes two CLIP branches and one T5 branch for CFG", () => {
    const textEncoder = new FakeCLIPProjectionEncoder(2, 5, 10, 41);
    const textEncoder2 = new FakeCLIPProjectionEncoder(3, 7, 20, 43);
    const textEncoder3 = new FakeT5Encoder(8, 31);
    const conditioner = createStableDiffusion3PromptConditioner({
      tokenizer: clipTokenizer(),
      tokenizer2: clipTokenizer(),
      tokenizer3: t5Tokenizer(),
      textEncoder,
      textEncoder2,
      textEncoder3,
      jointAttentionDim: 8,
      pooledProjectionDim: 12,
    });

    const result = conditioner.encodePrompt({
      prompt: ["cat cat cat cat", "dog"],
      prompt2: "dog",
      prompt3: "cat",
      guidanceScale: 7,
      negativePrompt: "",
      numImagesPerPrompt: 2,
      maxSequenceLength: 4,
      clipSkip: 1,
    });

    try {
      expect(result.batchSize).toBe(4);
      expect(result.promptTruncated).toBe(true);
      expect(result.prompt2Truncated).toBe(false);
      expect(result.prompt3Truncated).toBe(false);
      expect(result.negativePromptTruncated).toBe(false);
      expect(textEncoder.calls).toEqual([
        { shape: [2, 6], outputHiddenStates: true },
        { shape: [2, 6], outputHiddenStates: true },
      ]);
      expect(textEncoder2.calls).toEqual([
        { shape: [2, 6], outputHiddenStates: true },
        { shape: [2, 6], outputHiddenStates: true },
      ]);
      expect(textEncoder3.calls).toEqual([
        { shape: [2, 4], outputHiddenStates: false },
        { shape: [2, 4], outputHiddenStates: false },
      ]);
      expect(result.conditioning.encoderHiddenStates.shape).toEqual([4, 10, 8]);
      expect(result.conditioning.pooledProjections.shape).toEqual([4, 12]);
      expect(result.negativeConditioning?.encoderHiddenStates.shape).toEqual([4, 10, 8]);
      expect(result.negativeConditioning?.pooledProjections.shape).toEqual([4, 12]);

      expectTensorValues(
        Array.from(result.conditioning.encoderHiddenStates.toTypedArray()).slice(0, 8),
        [11, 11, 21, 21, 21, 0, 0, 0],
      );
      expectTensorValues(
        Array.from(result.conditioning.encoderHiddenStates.toTypedArray()).slice(6 * 8, 6 * 8 + 8),
        [31, 31, 31, 31, 31, 31, 31, 31],
      );
      expectTensorValues(
        Array.from(result.conditioning.pooledProjections.toTypedArray()).slice(0, 12),
        [41, 41, 41, 41, 41, 43, 43, 43, 43, 43, 43, 43],
      );
      expectTensorValues(
        Array.from(result.negativeConditioning?.encoderHiddenStates.toTypedArray() ?? []).slice(
          0,
          8,
        ),
        [12, 12, 22, 22, 22, 0, 0, 0],
      );
    } finally {
      result[Symbol.dispose]();
      conditioner[Symbol.dispose]();
    }
    expect(textEncoder.disposed).toBe(true);
    expect(textEncoder2.disposed).toBe(true);
    expect(textEncoder3.disposed).toBe(true);
  });

  test("supports distinct prompts without negative conditioning", () => {
    const textEncoder = new FakeCLIPProjectionEncoder(2, 5, 10, 41);
    const textEncoder2 = new FakeCLIPProjectionEncoder(3, 7, 20, 43);
    const textEncoder3 = new FakeT5Encoder(8, 31);
    const conditioner = createStableDiffusion3PromptConditioner({
      tokenizer: clipTokenizer(),
      tokenizer2: clipTokenizer(),
      tokenizer3: t5Tokenizer(),
      textEncoder,
      textEncoder2,
      textEncoder3,
      jointAttentionDim: 8,
      pooledProjectionDim: 12,
    });

    const result = conditioner.encodePrompt({
      prompt: "cat",
      prompt2: "dog",
      prompt3: "cat",
      negativePrompt: "dog",
      guidanceScale: 1,
      maxSequenceLength: 5,
    });

    try {
      expect(result.batchSize).toBe(1);
      expect(result.conditioning.encoderHiddenStates.shape).toEqual([1, 11, 8]);
      expect(result.conditioning.pooledProjections.shape).toEqual([1, 12]);
      expect(result.negativeConditioning).toBeUndefined();
      expect(result.promptTruncated).toBe(false);
      expect(result.prompt2Truncated).toBe(false);
      expect(result.prompt3Truncated).toBe(false);
      expect(textEncoder.calls).toHaveLength(1);
      expect(textEncoder2.calls).toHaveLength(1);
      expect(textEncoder3.calls).toHaveLength(1);
    } finally {
      result[Symbol.dispose]();
      conditioner[Symbol.dispose]();
    }
  });

  test("matches Diffusers empty secondary prompt fallback rules", () => {
    const textEncoder = new FakeCLIPProjectionEncoder(2, 5, 10, 41);
    const textEncoder2 = new FakeCLIPProjectionEncoder(3, 7, 20, 43);
    const textEncoder3 = new FakeT5Encoder(8, 31);
    const conditioner = createStableDiffusion3PromptConditioner({
      tokenizer: clipTokenizer(),
      tokenizer2: clipTokenizer(),
      tokenizer3: t5Tokenizer(),
      textEncoder,
      textEncoder2,
      textEncoder3,
      jointAttentionDim: 8,
      pooledProjectionDim: 12,
    });

    const result = conditioner.encodePrompt({
      prompt: "cat",
      prompt2: "",
      prompt3: "",
      guidanceScale: 7,
      negativePrompt: "dog",
      negativePrompt2: "",
      negativePrompt3: "",
      maxSequenceLength: 5,
    });

    try {
      expect(result.negativeConditioning).toBeDefined();
      expect(textEncoder2.tokenRows[0]).toEqual(textEncoder.tokenRows[0]);
      expect(textEncoder2.tokenRows[1]).toEqual(textEncoder.tokenRows[1]);
      expect(textEncoder3.tokenRows[0]?.[0]).toBe(3);
      expect(textEncoder3.tokenRows[1]?.[0]).toBe(4);
    } finally {
      result[Symbol.dispose]();
      conditioner[Symbol.dispose]();
    }
  });

  test("validates prompt batches and encoder shapes", () => {
    const conditioner = createStableDiffusion3PromptConditioner({
      tokenizer: clipTokenizer(),
      tokenizer2: clipTokenizer(),
      tokenizer3: t5Tokenizer(),
      textEncoder: new FakeCLIPProjectionEncoder(4, 5, 10, 41),
      textEncoder2: new FakeCLIPProjectionEncoder(5, 7, 20, 43),
      textEncoder3: new FakeT5Encoder(8, 31),
    });

    expect(() =>
      conditioner.encodePrompt({
        prompt: ["cat", "dog"],
        prompt2: ["cat"],
      }),
    ).toThrow("prompt2 batch size");

    expect(() =>
      conditioner.encodePrompt({
        prompt: "cat",
        maxSequenceLength: 513,
      }),
    ).toThrow("maxSequenceLength");

    expect(() =>
      conditioner.encodePrompt({
        prompt: "cat",
        numImagesPerPrompt: 0,
      }),
    ).toThrow("numImagesPerPrompt");

    expect(() =>
      conditioner.encodePrompt({
        prompt: "cat",
        clipSkip: -1,
      }),
    ).toThrow("clipSkip");

    expect(() =>
      conditioner.encodePrompt({
        prompt: "cat",
      }),
    ).toThrow("CLIP hidden size");

    const badPooled = createStableDiffusion3PromptConditioner({
      tokenizer: clipTokenizer(),
      tokenizer2: clipTokenizer(),
      tokenizer3: t5Tokenizer(),
      textEncoder: new FakeCLIPProjectionEncoder(2, 5, 10, 41),
      textEncoder2: new FakeCLIPProjectionEncoder(3, 7, 20, 43),
      textEncoder3: new FakeT5Encoder(8, 31),
      pooledProjectionDim: 99,
    });
    expect(() => badPooled.encodePrompt({ prompt: "cat" })).toThrow("pooled projections");
    badPooled[Symbol.dispose]();

    conditioner[Symbol.dispose]();
    expect(() => conditioner.encodePrompt({ prompt: "cat" })).toThrow("disposed");
  });
});
