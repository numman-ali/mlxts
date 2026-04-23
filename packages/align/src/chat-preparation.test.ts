import { describe, expect, test } from "bun:test";
import type { Tokenizer } from "@mlxts/tokenizers";
import type { ChatTemplate } from "@mlxts/transformers";

import { prepareChatPreferenceExamples, prepareChatSupervisionExamples } from "./chat-preparation";

const tokenizer: Tokenizer = {
  vocabSize: 256,
  bosTokenId: undefined,
  eosTokenIds: [],
  padTokenId: 0,
  encode(text: string): number[] {
    return [...text].map((char) => char.charCodeAt(0));
  },
  encodeWithOffsets(text: string) {
    return {
      ids: this.encode(text),
      offsets: [...text].map((_, index) => ({ start: index, end: index + 1 })),
    };
  },
  encodeBatch(texts: string[]) {
    return texts.map((text) => ({ ids: this.encode(text) }));
  },
  decode(tokenIds: number[]) {
    return String.fromCharCode(...tokenIds);
  },
  decodeBatch(batch: number[][]) {
    return batch.map((tokenIds) => this.decode(tokenIds));
  },
};

const template: ChatTemplate = {
  template: "",
  format(messages, options) {
    const rendered = messages.map((message) => `${message.role}:${message.content}`).join("|");
    return options?.addGenerationPrompt === true ? `${rendered}|assistant:` : rendered;
  },
};

describe("chat preparation helpers", () => {
  test("prepares supervision examples and reports skipped rows", () => {
    const result = prepareChatSupervisionExamples(
      tokenizer,
      template,
      [
        [
          { role: "user", content: "Hi" },
          { role: "assistant", content: "Hello" },
        ],
        [{ role: "user", content: "Missing assistant" }],
        [
          { role: "user", content: "This prompt is long" },
          { role: "assistant", content: "and this answer makes it longer" },
        ],
        [
          { role: "user", content: "Short again" },
          { role: "assistant", content: "Done" },
        ],
      ],
      {
        limit: 2,
        maxSequenceLength: 40,
      },
    );

    expect(result.examples).toHaveLength(2);
    expect(result.stats.kept).toBe(2);
    expect(result.stats.skippedMalformed).toBe(1);
    expect(result.stats.skippedLong).toBe(1);
  });

  test("prepares preference examples and reports skipped rows", () => {
    const result = prepareChatPreferenceExamples(
      tokenizer,
      template,
      [
        {
          promptMessages: [{ role: "user", content: "Hi" }],
          chosen: { role: "assistant", content: "Hello" },
          rejected: { role: "assistant", content: "No" },
        },
        {
          promptMessages: [{ role: "user", content: "Hi" }],
          chosen: { role: "assistant", content: "Hello" },
          rejected: { role: "user", content: "Not assistant" },
        },
        {
          promptMessages: [{ role: "user", content: "This prompt is very long indeed" }],
          chosen: { role: "assistant", content: "Long answer too" },
          rejected: { role: "assistant", content: "Also long" },
        },
        {
          promptMessages: [{ role: "user", content: "Again" }],
          chosen: { role: "assistant", content: "Yep" },
          rejected: { role: "assistant", content: "Nah" },
        },
      ],
      {
        limit: 2,
        maxSequenceLength: 40,
      },
    );

    expect(result.examples).toHaveLength(2);
    expect(result.stats.kept).toBe(2);
    expect(result.stats.skippedMalformed).toBe(1);
    expect(result.stats.skippedLong).toBe(1);
  });

  test("rejects invalid preparation options", () => {
    expect(() =>
      prepareChatSupervisionExamples(tokenizer, template, [], {
        limit: 0,
        maxSequenceLength: 32,
      }),
    ).toThrow("limit must be a positive integer");

    expect(() =>
      prepareChatPreferenceExamples(tokenizer, template, [], {
        limit: 1,
        maxSequenceLength: 0,
      }),
    ).toThrow("maxSequenceLength must be a positive integer");
  });

  test("rejects undersized prepared slices", () => {
    expect(() =>
      prepareChatSupervisionExamples(
        tokenizer,
        template,
        [
          [
            { role: "user", content: "Hi" },
            { role: "assistant", content: "Hello" },
          ],
        ],
        {
          limit: 2,
          maxSequenceLength: 32,
        },
      ),
    ).toThrow("collected only 1 example(s); expected 2");
  });
});
