import { describe, expect, test } from "bun:test";
import { array, type MxArray } from "@mlxts/core";
import type { Tokenizer } from "@mlxts/tokenizers";

import { createQwenImagePromptConditioner } from "./conditioning";
import {
  encodeQwenImagePrompt,
  QWEN_IMAGE_PROMPT_DROP_TOKENS,
  QWEN_IMAGE_PROMPT_TEMPLATE_PREFIX,
  QWEN_IMAGE_PROMPT_TEMPLATE_SUFFIX,
} from "./conditioning-runtime";
import type { QwenImageQwen2_5VLTextEncoder, QwenImageTextModelOutput } from "./conditioning-types";

class FakeQwenImageTextEncoder implements QwenImageQwen2_5VLTextEncoder {
  disposed = false;
  lastInputShape: readonly number[] = [];
  lastOutputHiddenStates = false;

  readonly model = {
    runWithHiddenStates: (
      inputIds: MxArray,
      options?: { outputHiddenStates?: boolean },
    ): QwenImageTextModelOutput => {
      const sequenceLength = inputIds.shape[1] ?? 0;
      this.lastInputShape = inputIds.shape;
      this.lastOutputHiddenStates = options?.outputHiddenStates === true;
      return {
        lastHiddenState: array(
          [Array.from({ length: sequenceLength }, (_, index) => [index, index, index, index])],
          "float32",
        ),
      };
    },
  };

  [Symbol.dispose](): void {
    this.disposed = true;
  }
}

function fakeTokenizer(capturedTexts: string[]): Tokenizer {
  return {
    vocabSize: 128,
    bosTokenId: undefined,
    eosTokenIds: [],
    padTokenId: undefined,
    encode: (text) => {
      capturedTexts.push(text);
      const length = text.includes("a red apple") ? QWEN_IMAGE_PROMPT_DROP_TOKENS + 6 : 36;
      return Array.from({ length }, (_, index) => index);
    },
    encodeWithOffsets: (text) => ({ ids: fakeTokenizer(capturedTexts).encode(text) }),
    encodeBatch: () => [],
    decode: () => "",
    decodeBatch: () => [],
  };
}

describe("Qwen-Image example prompt conditioning", () => {
  test("uses the fixed Diffusers prompt wrapper and drops the first 34 final hidden states", () => {
    const capturedTexts: string[] = [];
    const textEncoder = new FakeQwenImageTextEncoder();

    using conditioning = encodeQwenImagePrompt(
      {
        tokenizer: fakeTokenizer(capturedTexts),
        textEncoder,
      },
      {
        prompt: "a red apple",
        negativePrompt: " ",
        trueCfgScale: 4,
        maxSequenceLength: 3,
      },
    );

    expect(capturedTexts[0]).toBe(
      `${QWEN_IMAGE_PROMPT_TEMPLATE_PREFIX}a red apple${QWEN_IMAGE_PROMPT_TEMPLATE_SUFFIX}`,
    );
    expect(capturedTexts[1]).toBe(
      `${QWEN_IMAGE_PROMPT_TEMPLATE_PREFIX} ${QWEN_IMAGE_PROMPT_TEMPLATE_SUFFIX}`,
    );
    expect(textEncoder.lastInputShape).toEqual([1, QWEN_IMAGE_PROMPT_DROP_TOKENS + 2]);
    expect(textEncoder.lastOutputHiddenStates).toBe(false);
    expect(conditioning.batchSize).toBe(1);
    expect(conditioning.promptTruncated).toBe(true);
    expect(conditioning.negativePromptTruncated).toBe(false);
    expect(conditioning.conditioning.trueCfgScale).toBe(4);
    expect(conditioning.conditioning.promptEmbeds.shape).toEqual([1, 3, 4]);
    expect(conditioning.conditioning.promptEmbeds.toList()).toEqual([
      [
        [34, 34, 34, 34],
        [35, 35, 35, 35],
        [36, 36, 36, 36],
      ],
    ]);
    expect(conditioning.conditioning.negativePromptEmbeds?.shape).toEqual([1, 2, 4]);
  });

  test("omits negative conditioning when true CFG is disabled", () => {
    const capturedTexts: string[] = [];

    using conditioning = encodeQwenImagePrompt(
      {
        tokenizer: fakeTokenizer(capturedTexts),
        textEncoder: new FakeQwenImageTextEncoder(),
      },
      {
        prompt: "a red apple",
        trueCfgScale: 1,
        maxSequenceLength: 3,
      },
    );

    expect(capturedTexts).toHaveLength(1);
    expect(conditioning.conditioning.negativePromptEmbeds).toBeUndefined();
    expect(conditioning.conditioning.trueCfgScale).toBe(1);
  });

  test("disposes the loaded text encoder through the conditioner wrapper", () => {
    const capturedTexts: string[] = [];
    const textEncoder = new FakeQwenImageTextEncoder();
    const conditioner = createQwenImagePromptConditioner({
      tokenizer: fakeTokenizer(capturedTexts),
      textEncoder,
    });

    conditioner[Symbol.dispose]();

    expect(textEncoder.disposed).toBe(true);
    expect(() => conditioner.encodePrompt({ prompt: "a red apple" })).toThrow("disposed");
  });

  test("rejects unsupported prompt and hidden-state shapes before generation", () => {
    const capturedTexts: string[] = [];
    const components = {
      tokenizer: fakeTokenizer(capturedTexts),
      textEncoder: new FakeQwenImageTextEncoder(),
    };

    expect(() => encodeQwenImagePrompt(components, { prompt: "", maxSequenceLength: 1 })).toThrow(
      "must not be empty",
    );
    expect(() =>
      encodeQwenImagePrompt(components, { prompt: "a red apple", maxSequenceLength: 1025 }),
    ).toThrow("1024");
    expect(() =>
      encodeQwenImagePrompt(components, { prompt: "a red apple", trueCfgScale: 0 }),
    ).toThrow("positive finite");
  });
});
