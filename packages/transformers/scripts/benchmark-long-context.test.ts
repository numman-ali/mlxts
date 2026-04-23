import { describe, expect, test } from "bun:test";
import type { Tokenizer } from "@mlxts/tokenizers";

import {
  buildNeedlePromptTokenIds,
  defaultContextTargets,
  inferMaxContextTokens,
  normalizeExactResponse,
} from "./benchmark-long-context";

function mockTokenizer(): Tokenizer {
  return {
    vocabSize: 256,
    bosTokenId: 2,
    eosTokenIds: [3],
    padTokenId: 0,
    encode(text: string, options?: { addSpecialTokens?: boolean }) {
      const ids = [...text].map((character) => character.charCodeAt(0));
      return options?.addSpecialTokens === false ? ids : [2, ...ids];
    },
    encodeWithOffsets() {
      throw new Error("not needed");
    },
    encodeBatch() {
      throw new Error("not needed");
    },
    decode(ids: readonly number[]) {
      return ids
        .filter((token) => token !== 2 && token !== 3)
        .map((token) => String.fromCharCode(token))
        .join("");
    },
    decodeBatch() {
      throw new Error("not needed");
    },
  };
}

describe("benchmark-long-context", () => {
  test("defaultContextTargets follows the 32k/64k/128k/256k ladder", () => {
    expect(defaultContextTargets(16_384)).toEqual([]);
    expect(defaultContextTargets(65_536)).toEqual([32_768, 65_536]);
    expect(defaultContextTargets(200_000)).toEqual([32_768, 65_536, 131_072]);
    expect(defaultContextTargets(300_000)).toEqual([32_768, 65_536, 131_072, 262_144]);
  });

  test("inferMaxContextTokens reads the common transformer config fields", () => {
    expect(inferMaxContextTokens({ max_position_embeddings: 131_072 })).toBe(131_072);
    expect(inferMaxContextTokens({ max_sequence_length: 65_536 })).toBe(65_536);
    expect(
      inferMaxContextTokens({
        model_type: "qwen3_5",
        text_config: { max_position_embeddings: 262_144 },
      }),
    ).toBe(262_144);
    expect(() => inferMaxContextTokens({})).toThrow("max context field");
  });

  test("buildNeedlePromptTokenIds fills the exact token budget and keeps the retrieval tail", () => {
    const tokenizer = mockTokenizer();
    const promptTokenIds = buildNeedlePromptTokenIds(tokenizer, 256, "ALBATROSS");

    expect(promptTokenIds).toHaveLength(256);
    expect(tokenizer.decode(promptTokenIds.slice(-80))).toContain("ALBATROSS");
  });

  test("normalizeExactResponse grades the first non-empty generated answer line", () => {
    expect(normalizeExactResponse("\n`MKR-123`,\nassistant\nextra")).toBe("MKR-123");
  });
});
