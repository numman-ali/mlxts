import { describe, expect, test } from "bun:test";
import { full, type MxArray } from "@mlxts/core";
import type { Tokenizer } from "@mlxts/tokenizers";
import type { InteractionProfile } from "@mlxts/transformers";

import { createFlux2KleinPromptConditioner } from "./conditioning";
import {
  encodeFlux2KleinPrompt,
  FLUX2_KLEIN_DEFAULT_TEXT_ENCODER_OUT_LAYERS,
} from "./conditioning-runtime";
import type { Flux2KleinQwen3TextEncoder, Flux2KleinTextModelOutput } from "./conditioning-types";

type CompileMessagesOptions = NonNullable<Parameters<InteractionProfile["compileMessages"]>[2]>;

const tokenizer: Tokenizer = {
  vocabSize: 64,
  bosTokenId: undefined,
  eosTokenIds: [],
  padTokenId: 0,
  encode: () => [],
  encodeWithOffsets: () => ({ ids: [] }),
  encodeBatch: () => [],
  decode: () => "",
  decodeBatch: () => [],
};

class FakeQwen3TextEncoder implements Flux2KleinQwen3TextEncoder {
  disposed = false;
  inputShapes: number[][] = [];
  outputHiddenStates: boolean[] = [];

  readonly model = {
    runWithHiddenStates: (
      inputIds: MxArray,
      options?: { outputHiddenStates?: boolean },
    ): Flux2KleinTextModelOutput => {
      const sequenceLength = inputIds.shape[1] ?? 0;
      this.inputShapes.push([...inputIds.shape]);
      this.outputHiddenStates.push(options?.outputHiddenStates === true);
      return {
        lastHiddenState: full([1, sequenceLength, 2], -1, "float32"),
        hiddenStates: Array.from({ length: 28 }, (_, layer) =>
          full([1, sequenceLength, 2], layer, "float32"),
        ),
      };
    },
  };

  [Symbol.dispose](): void {
    this.disposed = true;
  }
}

function fakeInteractionProfile(
  captured: { prompt: string; options: CompileMessagesOptions }[],
): InteractionProfile {
  return {
    kind: "chat",
    chatTemplate: null,
    compileTextPrompt: () => ({ text: "", tokenIds: [] }),
    compileMessages: (_tokenizer, messages, options = {}) => {
      const prompt = messages[0]?.content ?? "";
      captured.push({ prompt, options });
      return {
        text: "rendered prompt",
        tokenIds: prompt === "" ? [7, 8] : [11, 22, 33, 44],
      };
    },
  };
}

describe("FLUX.2 Klein example prompt conditioning", () => {
  test("uses Qwen3 chat-template conditioning and concatenates default hidden layers", () => {
    const captured: { prompt: string; options: CompileMessagesOptions }[] = [];
    const textEncoder = new FakeQwen3TextEncoder();

    using conditioning = encodeFlux2KleinPrompt(
      {
        tokenizer,
        interactionProfile: fakeInteractionProfile(captured),
        textEncoder,
      },
      {
        prompt: "a red apple",
        maxSequenceLength: 2,
        guidanceScale: 4,
      },
    );

    expect(textEncoder.inputShapes).toEqual([
      [1, 2],
      [1, 2],
    ]);
    expect(textEncoder.outputHiddenStates).toEqual([true, true]);
    expect(captured).toEqual([
      {
        prompt: "a red apple",
        options: { addGenerationPrompt: true, enableThinking: false },
      },
      {
        prompt: "",
        options: { addGenerationPrompt: true, enableThinking: false },
      },
    ]);
    expect(conditioning.batchSize).toBe(1);
    expect(conditioning.promptTruncated).toBe(true);
    expect(conditioning.negativePromptTruncated).toBe(false);
    expect(conditioning.conditioning.guidanceScale).toBe(4);
    expect(conditioning.conditioning.promptEmbeds.shape).toEqual([1, 2, 6]);
    expect(conditioning.conditioning.promptEmbeds.toList()).toEqual([
      [
        [9, 9, 18, 18, 27, 27],
        [9, 9, 18, 18, 27, 27],
      ],
    ]);
    expect(conditioning.conditioning.negativePromptEmbeds?.shape).toEqual([1, 2, 6]);
    expect(FLUX2_KLEIN_DEFAULT_TEXT_ENCODER_OUT_LAYERS).toEqual([9, 18, 27]);
  });

  test("omits negative conditioning when guidance is disabled", () => {
    const captured: { prompt: string; options: CompileMessagesOptions }[] = [];
    const textEncoder = new FakeQwen3TextEncoder();

    using conditioning = encodeFlux2KleinPrompt(
      {
        tokenizer,
        interactionProfile: fakeInteractionProfile(captured),
        textEncoder,
      },
      {
        prompt: "a red apple",
        guidanceScale: 1,
        maxSequenceLength: 5,
      },
    );

    expect(captured.map((entry) => entry.prompt)).toEqual(["a red apple"]);
    expect(textEncoder.inputShapes).toEqual([[1, 5]]);
    expect(conditioning.promptTruncated).toBe(false);
    expect(conditioning.conditioning.negativePromptEmbeds).toBeUndefined();
    expect(conditioning.conditioning.guidanceScale).toBe(1);
  });

  test("disposes the loaded text encoder through the conditioner wrapper", () => {
    const captured: { prompt: string; options: CompileMessagesOptions }[] = [];
    const textEncoder = new FakeQwen3TextEncoder();
    const conditioner = createFlux2KleinPromptConditioner({
      tokenizer,
      interactionProfile: fakeInteractionProfile(captured),
      textEncoder,
    });

    conditioner[Symbol.dispose]();

    expect(textEncoder.disposed).toBe(true);
    expect(() => conditioner.encodePrompt({ prompt: "a red apple" })).toThrow("disposed");
  });

  test("rejects unsupported prompts and hidden-state selections before generation", () => {
    const captured: { prompt: string; options: CompileMessagesOptions }[] = [];
    const components = {
      tokenizer,
      interactionProfile: fakeInteractionProfile(captured),
      textEncoder: new FakeQwen3TextEncoder(),
    };

    expect(() => encodeFlux2KleinPrompt(components, { prompt: "", maxSequenceLength: 1 })).toThrow(
      "must not be empty",
    );
    expect(() =>
      encodeFlux2KleinPrompt(components, {
        prompt: "a red apple",
        maxSequenceLength: 513,
      }),
    ).toThrow("512");
    expect(() =>
      encodeFlux2KleinPrompt(components, {
        prompt: "a red apple",
        guidanceScale: 0,
      }),
    ).toThrow("positive finite");
    expect(() =>
      encodeFlux2KleinPrompt(components, {
        prompt: "a red apple",
        textEncoderOutLayers: [28],
      }),
    ).toThrow("hidden state 28");
  });
});
