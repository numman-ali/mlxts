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
    throw new Error(`chat-canary line ${lineIndex}: expected an object.`);
  }

  const { id, split, category, prompt, ideal_response, rubric_tags } = value;
  if (typeof id !== "string" || id.trim() === "") {
    throw new Error(`chat-canary line ${lineIndex}: id must be a non-empty string.`);
  }
  if (split !== "eval") {
    throw new Error(`chat-canary line ${lineIndex}: split must be "eval".`);
  }
  if (typeof category !== "string" || category.trim() === "") {
    throw new Error(`chat-canary line ${lineIndex}: category must be a non-empty string.`);
  }
  if (typeof prompt !== "string" || prompt.trim() === "") {
    throw new Error(`chat-canary line ${lineIndex}: prompt must be a non-empty string.`);
  }
  if (typeof ideal_response !== "string" || ideal_response.trim() === "") {
    throw new Error(`chat-canary line ${lineIndex}: ideal_response must be a non-empty string.`);
  }
  if (!Array.isArray(rubric_tags) || rubric_tags.some((tag) => typeof tag !== "string")) {
    throw new Error(`chat-canary line ${lineIndex}: rubric_tags must be a string array.`);
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

describe("chat-canary dataset", () => {
  test("keeps the starter SFT JSONL compatible with the LoRA example loader", async () => {
    const dataset = await loadJsonlDataset(
      "examples/chat-canary/mlxts-chat-sft.jsonl",
      parseUltrachatMessagesRow,
    );
    const records = dataset.items();

    expect(records).toHaveLength(60);
    expect(records.every((messages) => messages.at(-1)?.role === "assistant")).toBe(true);
  });

  test("keeps train rows before eval rows in the starter SFT JSONL", async () => {
    const rawRows = await loadJsonlDataset(
      "examples/chat-canary/mlxts-chat-sft.jsonl",
      (value, lineIndex) => {
        if (!isRecord(value)) {
          throw new Error(`starter SFT line ${lineIndex}: expected an object.`);
        }
        return {
          id: value.id,
          split: value.split,
          messages: value.messages,
        };
      },
    );

    const rows = rawRows.items();
    expect(rows.slice(0, 45).every((row) => row.split === "train")).toBe(true);
    expect(rows.slice(45).every((row) => row.split === "eval")).toBe(true);
  });

  test("keeps the held-out canary file parseable and unique", async () => {
    const dataset = await loadJsonlDataset(
      "examples/chat-canary/mlxts-chat-canary.jsonl",
      parseCanaryRow,
    );
    const rows = dataset.items();

    expect(rows).toHaveLength(15);
    expect(new Set(rows.map((row) => row.id)).size).toBe(15);
    expect(rows.every((row) => row.rubric_tags.length > 0)).toBe(true);
  });

  test("keeps the trainable chat rows in a simple user -> assistant shape", async () => {
    const dataset = await loadJsonlDataset(
      "examples/chat-canary/mlxts-chat-sft.jsonl",
      parseUltrachatMessagesRow,
    );
    const records = dataset.items();

    const allValid = records.every((messages) => {
      const first = messages[0];
      const last = messages.at(-1);
      return (
        first?.role === "user" &&
        typeof first.content === "string" &&
        first.content.trim() !== "" &&
        last?.role === "assistant" &&
        typeof last.content === "string" &&
        last.content.trim() !== ""
      );
    });

    expect(allValid).toBe(true);
  });
});
