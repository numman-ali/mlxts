import { describe, expect, test } from "bun:test";
import { full, type MxArray, zeros } from "@mlxts/core";
import { CLIPTokenizer } from "@mlxts/tokenizers";
import type {
  CLIPTextModelOptions,
  CLIPTextModelOutput,
  CLIPTextProjectionOutput,
} from "@mlxts/transformers";
import {
  createStableDiffusionPromptConditioner,
  type StableDiffusionProjectedTextEncoder,
  type StableDiffusionTextEncoder,
} from "./conditioning";

type EncoderCall = {
  shape: readonly number[];
  outputHiddenStates: boolean;
};

class FakeTextEncoder implements StableDiffusionTextEncoder {
  readonly calls: EncoderCall[] = [];
  disposed = false;

  constructor(
    readonly hiddenSize: number,
    readonly lastHiddenValue: number,
    readonly penultimateValue: number,
  ) {}

  run(inputIds: MxArray, options: CLIPTextModelOptions = {}): CLIPTextModelOutput {
    this.calls.push({
      shape: [...inputIds.shape],
      outputHiddenStates: options.outputHiddenStates === true,
    });
    const batch = inputIds.shape[0] ?? 1;
    const sequenceLength = inputIds.shape[1] ?? 1;
    const output: CLIPTextModelOutput = {
      lastHiddenState: full([batch, sequenceLength, this.hiddenSize], this.lastHiddenValue),
      pooledOutput: zeros([batch, this.hiddenSize]),
    };
    if (options.outputHiddenStates === true) {
      output.hiddenStates = [
        full([batch, sequenceLength, this.hiddenSize], this.penultimateValue - 1),
        full([batch, sequenceLength, this.hiddenSize], this.penultimateValue),
        full([batch, sequenceLength, this.hiddenSize], this.penultimateValue + 1),
      ];
    }
    return output;
  }

  [Symbol.dispose](): void {
    this.disposed = true;
  }
}

class FakeProjectedTextEncoder implements StableDiffusionProjectedTextEncoder {
  readonly calls: EncoderCall[] = [];
  disposed = false;

  constructor(
    readonly hiddenSize: number,
    readonly projectionSize: number,
    readonly penultimateValue: number,
    readonly textEmbedValue: number,
  ) {}

  run(inputIds: MxArray, options: CLIPTextModelOptions = {}): CLIPTextProjectionOutput {
    this.calls.push({
      shape: [...inputIds.shape],
      outputHiddenStates: options.outputHiddenStates === true,
    });
    const batch = inputIds.shape[0] ?? 1;
    const sequenceLength = inputIds.shape[1] ?? 1;
    return {
      lastHiddenState: full([batch, sequenceLength, this.hiddenSize], this.penultimateValue + 2),
      pooledOutput: zeros([batch, this.hiddenSize]),
      hiddenStates: [
        full([batch, sequenceLength, this.hiddenSize], this.penultimateValue - 1),
        full([batch, sequenceLength, this.hiddenSize], this.penultimateValue),
        full([batch, sequenceLength, this.hiddenSize], this.penultimateValue + 1),
      ],
      textEmbeds: full([batch, this.projectionSize], this.textEmbedValue),
    };
  }

  [Symbol.dispose](): void {
    this.disposed = true;
  }
}

function testTokenizer(): CLIPTokenizer {
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

describe("Stable Diffusion example prompt conditioning", () => {
  test("encodes SD 1.x prompts and CFG negative prompts", () => {
    const tokenizer = testTokenizer();
    const textEncoder = new FakeTextEncoder(4, 9, 7);
    const conditioner = createStableDiffusionPromptConditioner({
      pipelineKind: "stable-diffusion",
      tokenizer,
      textEncoder,
    });

    const result = conditioner.encodePrompt({
      prompt: ["cat cat cat", "dog"],
      negativePrompt: "",
      guidanceScale: 7.5,
      numImagesPerPrompt: 2,
    });

    try {
      expect(result.batchSize).toBe(4);
      expect(result.promptTruncated).toBe(true);
      expect(result.negativePromptTruncated).toBe(false);
      expect(textEncoder.calls).toEqual([
        { shape: [2, 6], outputHiddenStates: false },
        { shape: [2, 6], outputHiddenStates: false },
      ]);
      expect(result.conditioning.encoderHiddenStates.shape).toEqual([4, 6, 4]);
      expect(result.negativeConditioning?.encoderHiddenStates.shape).toEqual([4, 6, 4]);
    } finally {
      result[Symbol.dispose]();
      conditioner[Symbol.dispose]();
    }
    expect(textEncoder.disposed).toBe(true);
  });

  test("encodes SDXL dual CLIP conditioning and zeroes default negative embeddings", () => {
    const textEncoder = new FakeTextEncoder(2, 10, 11);
    const textEncoder2 = new FakeProjectedTextEncoder(3, 4, 21, 31);
    const conditioner = createStableDiffusionPromptConditioner({
      pipelineKind: "stable-diffusion-xl",
      tokenizer: testTokenizer(),
      tokenizer2: testTokenizer(),
      textEncoder,
      textEncoder2,
      forceZerosForEmptyPrompt: true,
    });

    const result = conditioner.encodePrompt({
      prompt: "cat",
      guidanceScale: 7.5,
      targetSize: [128, 160],
      originalSize: [64, 96],
      cropTopLeft: [4, 8],
      negativeTargetSize: [256, 320],
      numImagesPerPrompt: 2,
    });

    try {
      expect(textEncoder.calls).toEqual([{ shape: [1, 6], outputHiddenStates: true }]);
      expect(textEncoder2.calls).toEqual([{ shape: [1, 6], outputHiddenStates: true }]);
      expect(result.conditioning.encoderHiddenStates.shape).toEqual([2, 6, 5]);
      expect(result.conditioning.textTime?.textEmbeds.shape).toEqual([2, 4]);
      expect(result.conditioning.textTime?.timeIds.shape).toEqual([2, 6]);
      expect(result.negativeConditioning?.encoderHiddenStates.shape).toEqual([2, 6, 5]);

      result.conditioning.encoderHiddenStates.eval();
      expectTensorValues(
        Array.from(result.conditioning.encoderHiddenStates.toTypedArray()).slice(0, 5),
        [11, 11, 21, 21, 21],
      );

      result.negativeConditioning?.encoderHiddenStates.eval();
      expectTensorValues(
        Array.from(result.negativeConditioning?.encoderHiddenStates.toTypedArray() ?? []).slice(
          0,
          5,
        ),
        [0, 0, 0, 0, 0],
      );

      result.negativeConditioning?.textTime?.timeIds.eval();
      expectTensorValues(
        result.negativeConditioning?.textTime?.timeIds.toTypedArray() ?? [],
        [64, 96, 4, 8, 256, 320, 64, 96, 4, 8, 256, 320],
      );
    } finally {
      result[Symbol.dispose]();
      conditioner[Symbol.dispose]();
    }
  });

  test("encodes explicit SDXL negative prompts through both encoders", () => {
    const textEncoder = new FakeTextEncoder(2, 10, 11);
    const textEncoder2 = new FakeProjectedTextEncoder(3, 4, 21, 31);
    const conditioner = createStableDiffusionPromptConditioner({
      pipelineKind: "stable-diffusion-xl",
      tokenizer: testTokenizer(),
      tokenizer2: testTokenizer(),
      textEncoder,
      textEncoder2,
      forceZerosForEmptyPrompt: true,
    });

    const result = conditioner.encodePrompt({
      prompt: "cat",
      prompt2: "dog",
      negativePrompt: "",
      negativePrompt2: "cat",
      guidanceScale: 7.5,
      targetSize: [128, 128],
    });

    try {
      expect(textEncoder.calls).toHaveLength(2);
      expect(textEncoder2.calls).toHaveLength(2);
      expect(result.negativeConditioning?.textTime?.textEmbeds.shape).toEqual([1, 4]);
    } finally {
      result[Symbol.dispose]();
      conditioner[Symbol.dispose]();
    }
  });

  test("validates batch sizes and SDXL geometry", () => {
    const conditioner = createStableDiffusionPromptConditioner({
      pipelineKind: "stable-diffusion-xl",
      tokenizer: testTokenizer(),
      tokenizer2: testTokenizer(),
      textEncoder: new FakeTextEncoder(2, 10, 11),
      textEncoder2: new FakeProjectedTextEncoder(3, 4, 21, 31),
    });

    expect(() =>
      conditioner.encodePrompt({
        prompt: ["cat", "dog"],
        negativePrompt: ["cat"],
        guidanceScale: 7.5,
        targetSize: [128, 128],
      }),
    ).toThrow("negativePrompt batch size");

    expect(() =>
      conditioner.encodePrompt({
        prompt: "cat",
        targetSize: [0, 128],
      }),
    ).toThrow("targetSize dimensions");

    conditioner[Symbol.dispose]();
  });
});
