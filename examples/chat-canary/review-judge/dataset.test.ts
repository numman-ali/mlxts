import { describe, expect, test } from "bun:test";
import { loadJsonlDataset, parseUltrachatMessagesRow } from "@mlxts/data";

type CanaryRow = {
  id: string;
  split: "eval";
  category: string;
  prompt: string;
  ideal_response: string;
  rubric_tags: string[];
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseCanaryRow(value: unknown, lineIndex: number): CanaryRow {
  if (!isRecord(value)) {
    throw new Error(`review-judge canary line ${lineIndex}: expected an object.`);
  }

  const { id, split, category, prompt, ideal_response, rubric_tags } = value;
  if (typeof id !== "string" || id.trim() === "") {
    throw new Error(`review-judge canary line ${lineIndex}: id must be a non-empty string.`);
  }
  if (split !== "eval") {
    throw new Error(`review-judge canary line ${lineIndex}: split must be "eval".`);
  }
  if (typeof category !== "string" || category.trim() === "") {
    throw new Error(`review-judge canary line ${lineIndex}: category must be a non-empty string.`);
  }
  if (typeof prompt !== "string" || prompt.trim() === "") {
    throw new Error(`review-judge canary line ${lineIndex}: prompt must be a non-empty string.`);
  }
  if (typeof ideal_response !== "string" || ideal_response.trim() === "") {
    throw new Error(
      `review-judge canary line ${lineIndex}: ideal_response must be a non-empty string.`,
    );
  }
  if (!Array.isArray(rubric_tags) || rubric_tags.some((tag) => typeof tag !== "string")) {
    throw new Error(`review-judge canary line ${lineIndex}: rubric_tags must be a string array.`);
  }

  return {
    id,
    split,
    category,
    prompt,
    ideal_response,
    rubric_tags,
  };
}

describe("review-judge dataset", () => {
  test("keeps the review-judge SFT JSONL compatible with the LoRA example loader", async () => {
    const dataset = await loadJsonlDataset(
      "examples/chat-canary/review-judge/mlxts-review-judge-sft.jsonl",
      parseUltrachatMessagesRow,
    );
    const records = dataset.items();

    expect(records).toHaveLength(250);
    expect(records.every((messages) => messages.at(-1)?.role === "assistant")).toBe(true);
  });

  test("keeps train rows before eval rows in the review-judge SFT JSONL", async () => {
    const rawRows = await loadJsonlDataset(
      "examples/chat-canary/review-judge/mlxts-review-judge-sft.jsonl",
      (value, lineIndex) => {
        if (!isRecord(value)) {
          throw new Error(`review-judge SFT line ${lineIndex}: expected an object.`);
        }
        return {
          id: value.id,
          split: value.split,
          prompt: value.prompt,
        };
      },
    );

    const rows = rawRows.items();
    expect(rows.slice(0, 200).every((row) => row.split === "train")).toBe(true);
    expect(rows.slice(200).every((row) => row.split === "eval")).toBe(true);
    expect(new Set(rows.map((row) => row.id)).size).toBe(250);
    expect(new Set(rows.map((row) => row.prompt)).size).toBe(250);
  });

  test("keeps the held-out review-judge canary parseable and unique", async () => {
    const dataset = await loadJsonlDataset(
      "examples/chat-canary/review-judge/mlxts-review-judge-canary.jsonl",
      parseCanaryRow,
    );
    const rows = dataset.items();

    expect(rows).toHaveLength(50);
    expect(new Set(rows.map((row) => row.id)).size).toBe(50);
    expect(new Set(rows.map((row) => row.prompt)).size).toBe(50);
    expect(rows.every((row) => row.rubric_tags.length > 0)).toBe(true);
  });
});
