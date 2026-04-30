import { describe, expect, test } from "bun:test";
import { full, type MxArray, zeros } from "@mlxts/core";
import { CLIPTokenizer, SentencePieceTokenizer } from "@mlxts/tokenizers";
import type {
  CLIPTextModelOptions,
  CLIPTextModelOutput,
  T5EncoderModelOptions,
  T5EncoderModelOutput,
} from "@mlxts/transformers";

import {
  createFluxPromptConditioner,
  type FluxCLIPTextEncoder,
  type FluxT5TextEncoder,
} from "./conditioning";

type EncoderCall = {
  shape: readonly number[];
  outputHiddenStates: boolean;
};

class FakeCLIPTextEncoder implements FluxCLIPTextEncoder {
  readonly calls: EncoderCall[] = [];
  disposed = false;

  constructor(
    readonly hiddenSize: number,
    readonly pooledSize: number,
    readonly pooledValue: number,
  ) {}

  run(inputIds: MxArray, options: CLIPTextModelOptions = {}): CLIPTextModelOutput {
    this.calls.push({
      shape: [...inputIds.shape],
      outputHiddenStates: options.outputHiddenStates === true,
    });
    const batch = inputIds.shape[0] ?? 1;
    const sequenceLength = inputIds.shape[1] ?? 1;
    return {
      lastHiddenState: zeros([batch, sequenceLength, this.hiddenSize]),
      pooledOutput: full([batch, this.pooledSize], this.pooledValue),
    };
  }

  [Symbol.dispose](): void {
    this.disposed = true;
  }
}

class FakeT5Encoder implements FluxT5TextEncoder {
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

describe("FLUX example prompt conditioning", () => {
  test("encodes CLIP pooled projections and T5 sequence embeddings", () => {
    const textEncoder = new FakeCLIPTextEncoder(3, 7, 23);
    const textEncoder2 = new FakeT5Encoder(5, 13);
    const conditioner = createFluxPromptConditioner({
      tokenizer: clipTokenizer(),
      textEncoder,
      tokenizer2: t5Tokenizer(),
      textEncoder2,
    });

    const result = conditioner.encodePrompt({
      prompt: ["cat cat cat cat", "dog"],
      numImagesPerPrompt: 2,
      maxSequenceLength: 4,
      guidanceScale: 3.5,
    });

    try {
      expect(result.batchSize).toBe(4);
      expect(result.promptTruncated).toBe(true);
      expect(result.prompt2Truncated).toBe(true);
      expect(textEncoder.calls).toEqual([{ shape: [2, 6], outputHiddenStates: false }]);
      expect(textEncoder2.calls).toEqual([{ shape: [2, 4], outputHiddenStates: false }]);
      expect(result.conditioning.encoderHiddenStates.shape).toEqual([4, 4, 5]);
      expect(result.conditioning.pooledProjections.shape).toEqual([4, 7]);
      expect(result.conditioning.textIds?.shape).toEqual([4, 3]);
      expect(result.conditioning.guidance?.shape).toEqual([4]);

      expectTensorValues(
        Array.from(result.conditioning.encoderHiddenStates.toTypedArray()).slice(0, 5),
        [13, 13, 13, 13, 13],
      );
      expectTensorValues(
        Array.from(result.conditioning.pooledProjections.toTypedArray()).slice(0, 7),
        [23, 23, 23, 23, 23, 23, 23],
      );
      expectTensorValues(result.conditioning.guidance?.toTypedArray() ?? [], [3.5, 3.5, 3.5, 3.5]);
    } finally {
      result[Symbol.dispose]();
      conditioner[Symbol.dispose]();
    }
    expect(textEncoder.disposed).toBe(true);
    expect(textEncoder2.disposed).toBe(true);
  });

  test("supports distinct T5 prompts without guidance embeddings", () => {
    const conditioner = createFluxPromptConditioner({
      tokenizer: clipTokenizer(),
      textEncoder: new FakeCLIPTextEncoder(3, 7, 23),
      tokenizer2: t5Tokenizer(),
      textEncoder2: new FakeT5Encoder(5, 13),
    });

    const result = conditioner.encodePrompt({
      prompt: "cat",
      prompt2: "dog",
      maxSequenceLength: 5,
    });

    try {
      expect(result.batchSize).toBe(1);
      expect(result.promptTruncated).toBe(false);
      expect(result.prompt2Truncated).toBe(false);
      expect(result.conditioning.encoderHiddenStates.shape).toEqual([1, 5, 5]);
      expect(result.conditioning.pooledProjections.shape).toEqual([1, 7]);
      expect(result.conditioning.guidance).toBeUndefined();
    } finally {
      result[Symbol.dispose]();
      conditioner[Symbol.dispose]();
    }
  });

  test("validates prompt batches and FLUX sequence bounds", () => {
    const conditioner = createFluxPromptConditioner({
      tokenizer: clipTokenizer(),
      textEncoder: new FakeCLIPTextEncoder(3, 7, 23),
      tokenizer2: t5Tokenizer(),
      textEncoder2: new FakeT5Encoder(5, 13),
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
        numImagesPerPrompt: 0,
      }),
    ).toThrow("numImagesPerPrompt");

    expect(() =>
      conditioner.encodePrompt({
        prompt: "cat",
        maxSequenceLength: 513,
      }),
    ).toThrow("maxSequenceLength");

    conditioner[Symbol.dispose]();
    expect(() => conditioner.encodePrompt({ prompt: "cat" })).toThrow("disposed");
  });
});
