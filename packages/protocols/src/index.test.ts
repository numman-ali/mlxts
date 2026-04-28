import { describe, expect, test } from "bun:test";

import { cleanReasoningFromText, createReasoningTagStream, splitReasoningTags } from ".";

describe("reasoning tag normalization", () => {
  test("moves Qwen think-tag text into reasoning content", () => {
    expect(splitReasoningTags("<think>I should answer.</think>\n\nHello!")).toEqual({
      content: "Hello!",
      reasoningContent: "I should answer.",
    });
    expect(cleanReasoningFromText("I should answer.</think>\n\nHello!")).toEqual({
      content: "Hello!",
      reasoningContent: "I should answer.",
    });
  });

  test("moves Anthropic thinking-tag text into reasoning content", () => {
    expect(splitReasoningTags("<antThinking>I should answer.</antThinking>\n\nHello!")).toEqual({
      content: "Hello!",
      reasoningContent: "I should answer.",
    });
  });

  test("cleans Gemma thought-channel markers from visible content", () => {
    expect(splitReasoningTags("<|channel>thought\nI should answer.<channel|>\n\nHello!")).toEqual({
      content: "Hello!",
      reasoningContent: "I should answer.",
    });
  });

  test("streams think, antThinking, and Gemma thought-channel reasoning", () => {
    const qwen = createReasoningTagStream();
    expect(qwen.push("<think>I should ")).toEqual([{ reasoningContent: "I " }]);
    expect(qwen.push("answer.</think>\n\nHel")).toEqual([{ reasoningContent: "should answer." }]);
    expect(qwen.finish()).toEqual([{ content: "Hel" }]);

    const anthropic = createReasoningTagStream();
    expect(anthropic.push("<antThinking>I should ")).toEqual([]);
    expect(anthropic.push("answer.</antThinking>\n\nHel")).toEqual([
      { reasoningContent: "I should answer." },
    ]);
    expect(anthropic.finish()).toEqual([{ content: "Hel" }]);

    const gemma = createReasoningTagStream();
    expect(gemma.push("<|channel>thought\nI should ")).toEqual([]);
    expect(gemma.push("answer.<channel|>\n\nHel")).toEqual([
      { reasoningContent: "I should answer." },
    ]);
    expect(gemma.finish()).toEqual([{ content: "Hel" }]);
  });

  test("buffers markers that are split across stream chunks", () => {
    const qwen = createReasoningTagStream();
    const qwenDeltas = [
      ...qwen.push("<thi"),
      ...qwen.push("nk>I should answer.</thi"),
      ...qwen.push("nk>\n\nHello"),
      ...qwen.finish(),
    ];
    expect(qwenDeltas).toEqual([
      { reasoningContent: "I should answe" },
      { reasoningContent: "r." },
      { content: "Hello" },
    ]);

    const gemma = createReasoningTagStream();
    const gemmaDeltas = [
      ...gemma.push("<|channel>tho"),
      ...gemma.push("ught\nI should answer.<chan"),
      ...gemma.push("nel|>\n\nHello"),
      ...gemma.finish(),
    ];
    expect(gemmaDeltas).toEqual([
      { reasoningContent: "I should ans" },
      { reasoningContent: "wer." },
      { content: "Hello" },
    ]);
  });
});
