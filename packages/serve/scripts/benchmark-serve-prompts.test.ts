import { describe, expect, test } from "bun:test";

import { createBenchmarkPrompt } from "./benchmark-serve-prompts";

describe("serve benchmark prompt builders", () => {
  test("keeps completions prompts token-array exact without text tokenization", () => {
    const prompt = createBenchmarkPrompt(
      4,
      16,
      {
        encode() {
          throw new Error("text tokenization should not run for completions");
        },
      },
      "completions",
    );

    expect(prompt.tokenIds).toHaveLength(4);
    expect(prompt.text).toBe("");
  });

  test("builds text prompts for chat and responses protocol health runs", () => {
    let encodeCalls = 0;
    const tokenizer = {
      encode(text: string) {
        encodeCalls += 1;
        return Array.from({ length: Math.floor(text.length / 8) }, (_, index) => index);
      },
    };

    const prompt = createBenchmarkPrompt(16, 16, tokenizer, "chat");

    expect(prompt.tokenIds).toHaveLength(16);
    expect(prompt.text).not.toBe("");
    expect(encodeCalls).toBeGreaterThan(0);
    expect(tokenizer.encode(prompt.text).length).toBeGreaterThanOrEqual(16);
  });
});
