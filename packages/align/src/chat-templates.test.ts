import { describe, expect, test } from "bun:test";
import type { Tokenizer } from "@mlxts/tokenizers";
import type { ChatTemplate } from "@mlxts/transformers";

import {
  buildChatPreferenceExample,
  buildChatSupervisionExample,
  renderChatMessages,
} from "./chat-templates";

const tokenizer: Tokenizer = {
  vocabSize: 256,
  bosTokenId: undefined,
  eosTokenIds: [],
  padTokenId: 0,
  encode(text: string): number[] {
    return [...text].map((char) => char.charCodeAt(0));
  },
  encodeWithOffsets(text: string) {
    return { ids: this.encode(text) };
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

describe("chat template helpers", () => {
  test("builds supervised examples from the final assistant turn", () => {
    const example = buildChatSupervisionExample(tokenizer, template, [
      { role: "user", content: "Hi" },
      { role: "assistant", content: "Hello" },
    ]);

    expect(example.inputIds.length).toBe(example.targetIds.length);
    expect(example.lossMask?.some((value) => value === 1)).toBe(true);
  });

  test("builds preference examples from chosen and rejected replies", () => {
    const example = buildChatPreferenceExample(
      tokenizer,
      template,
      [{ role: "user", content: "Hi" }],
      { role: "assistant", content: "Hello" },
      { role: "assistant", content: "No" },
    );

    expect(example.promptIds.length).toBeGreaterThan(0);
    expect(example.chosenIds.length).toBeGreaterThan(0);
    expect(example.rejectedIds.length).toBeGreaterThan(0);
  });

  test("rejects chat supervision without a final assistant turn", () => {
    expect(() =>
      buildChatSupervisionExample(tokenizer, template, [{ role: "user", content: "Hi" }]),
    ).toThrow("must end in an assistant message");
  });

  test("rejects preference examples with non-assistant replies", () => {
    expect(() =>
      buildChatPreferenceExample(
        tokenizer,
        template,
        [{ role: "user", content: "Hi" }],
        { role: "assistant", content: "Hello" },
        { role: "user", content: "No" },
      ),
    ).toThrow("must both be assistant messages");
  });

  test("renders chat messages directly through the provided template", () => {
    expect(
      renderChatMessages(template, [{ role: "user", content: "Hi" }], {
        addGenerationPrompt: false,
      }),
    ).toBe("user:Hi");
  });

  test("rejects empty assistant completions that add no trainable tokens", () => {
    expect(() =>
      buildChatSupervisionExample(tokenizer, template, [
        { role: "user", content: "Hi" },
        { role: "assistant", content: "" },
      ]),
    ).toThrow("produced no trainable tokens");
  });

  test("rejects templates that do not preserve the prompt prefix", () => {
    const brokenTemplate: ChatTemplate = {
      template: "",
      format(messages, options) {
        if (messages.at(-1)?.role === "assistant" && options?.addGenerationPrompt === false) {
          return "broken";
        }
        return template.format(messages, options);
      },
    };

    expect(() =>
      buildChatSupervisionExample(tokenizer, brokenTemplate, [
        { role: "user", content: "Hi" },
        { role: "assistant", content: "Hello" },
      ]),
    ).toThrow("at least as long as the prompt prefix");
  });
});
