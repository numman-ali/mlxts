import { describe, expect, test } from "bun:test";

import { createTrainingProofCorpus, parseUltrachatMessagesRow } from "./training-proof";

describe("training proof data helpers", () => {
  test("builds a canonical small proof corpus", () => {
    const corpus = createTrainingProofCorpus();

    expect(corpus.supervisionExamples).toHaveLength(2);
    expect(corpus.supervisionExamples[0]?.at(-1)?.role).toBe("assistant");
    expect(corpus.promptMessages).toHaveLength(2);
    expect(corpus.chosen.role).toBe("assistant");
    expect(corpus.rejected.role).toBe("assistant");
  });

  test("parses UltraChat rows into chat messages", () => {
    const messages = parseUltrachatMessagesRow({
      messages: [
        { role: "user", content: "Hello" },
        { role: "assistant", content: "Hi there" },
      ],
    });

    expect(messages).toEqual([
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi there" },
    ]);
  });

  test("rejects rows that do not end in an assistant message", () => {
    expect(() =>
      parseUltrachatMessagesRow({
        messages: [{ role: "user", content: "Hello" }],
      }),
    ).toThrow("ultrachat row.messages must end in an assistant message.");
  });
});
