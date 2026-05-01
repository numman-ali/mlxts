import { describe, expect, test } from "bun:test";
import { full, type MxArray } from "@mlxts/core";
import type { Tokenizer } from "@mlxts/tokenizers";
import type { InteractionProfile } from "@mlxts/transformers";

import { createZImagePromptConditioner } from "./conditioning";
import { encodeZImagePrompt } from "./conditioning-runtime";
import type { ZImageQwen3TextEncoder, ZImageTextModelOutput } from "./conditioning-types";

type CompileChatPromptOptions = NonNullable<Parameters<InteractionProfile["compileMessages"]>[2]>;

const tokenizer: Tokenizer = {
  vocabSize: 64,
  bosTokenId: undefined,
  eosTokenIds: [],
  padTokenId: undefined,
  encode: () => [],
  encodeWithOffsets: () => ({ ids: [] }),
  encodeBatch: () => [],
  decode: () => "",
  decodeBatch: () => [],
};

class FakeQwen3TextEncoder implements ZImageQwen3TextEncoder {
  disposed = false;
  lastInputShape: readonly number[] = [];
  lastOutputHiddenStates = false;

  readonly model = {
    runWithHiddenStates: (
      inputIds: MxArray,
      options?: { outputHiddenStates?: boolean },
    ): ZImageTextModelOutput => {
      const sequenceLength = inputIds.shape[1] ?? 0;
      this.lastInputShape = inputIds.shape;
      this.lastOutputHiddenStates = options?.outputHiddenStates === true;
      return {
        lastHiddenState: full([1, sequenceLength, 4], 9, "float32"),
        hiddenStates: [
          full([1, sequenceLength, 4], 1, "float32"),
          full([1, sequenceLength, 4], 2, "float32"),
          full([1, sequenceLength, 4], 3, "float32"),
        ],
      };
    },
  };

  [Symbol.dispose](): void {
    this.disposed = true;
  }
}

function fakeInteractionProfile(
  tokenIds: number[],
  captured: CompileChatPromptOptions[],
): InteractionProfile {
  return {
    kind: "chat",
    chatTemplate: null,
    compileTextPrompt: () => ({ text: "", tokenIds: [] }),
    compileMessages: (_tokenizer, messages, options = {}) => {
      expect(messages).toEqual([{ role: "user", content: "a red apple" }]);
      captured.push(options);
      return {
        text: "rendered prompt",
        tokenIds,
      };
    },
  };
}

describe("Z-Image example prompt conditioning", () => {
  test("uses Qwen3 chat-template conditioning and selects the penultimate hidden state", () => {
    const capturedOptions: CompileChatPromptOptions[] = [];
    const textEncoder = new FakeQwen3TextEncoder();

    using conditioning = encodeZImagePrompt(
      {
        tokenizer,
        interactionProfile: fakeInteractionProfile([11, 22, 33, 44], capturedOptions),
        textEncoder,
      },
      {
        prompt: "a red apple",
        maxSequenceLength: 3,
      },
    );

    expect(textEncoder.lastInputShape).toEqual([1, 3]);
    expect(textEncoder.lastOutputHiddenStates).toBe(true);
    expect(capturedOptions).toEqual([{ addGenerationPrompt: true, enableThinking: true }]);
    expect(conditioning.batchSize).toBe(1);
    expect(conditioning.promptTruncated).toBe(true);
    expect(conditioning.conditioning.captionFeatures[0]?.shape).toEqual([3, 4]);
    expect(conditioning.conditioning.captionFeatures[0]?.toList()).toEqual([
      [2, 2, 2, 2],
      [2, 2, 2, 2],
      [2, 2, 2, 2],
    ]);
  });

  test("disposes the loaded text encoder through the conditioner wrapper", () => {
    const capturedOptions: CompileChatPromptOptions[] = [];
    const textEncoder = new FakeQwen3TextEncoder();
    const conditioner = createZImagePromptConditioner({
      tokenizer,
      interactionProfile: fakeInteractionProfile([1], capturedOptions),
      textEncoder,
    });

    conditioner[Symbol.dispose]();

    expect(textEncoder.disposed).toBe(true);
    expect(() => conditioner.encodePrompt({ prompt: "a red apple" })).toThrow("disposed");
  });

  test("rejects unsupported prompt and hidden-state shapes before generation", () => {
    const capturedOptions: CompileChatPromptOptions[] = [];
    const textEncoder = new FakeQwen3TextEncoder();
    expect(() =>
      encodeZImagePrompt(
        {
          tokenizer,
          interactionProfile: fakeInteractionProfile([1], capturedOptions),
          textEncoder,
        },
        { prompt: "", maxSequenceLength: 1 },
      ),
    ).toThrow("must not be empty");
    expect(() =>
      encodeZImagePrompt(
        {
          tokenizer,
          interactionProfile: fakeInteractionProfile([1], capturedOptions),
          textEncoder,
        },
        { prompt: "a red apple", maxSequenceLength: 513 },
      ),
    ).toThrow("512");
  });
});
