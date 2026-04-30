import { describe, expect, test } from "bun:test";

import { parseArgs } from "./index";

describe("parseArgs", () => {
  test("uses cached sources and disabled thinking by default", () => {
    const parsed = parseArgs(["mlx-community/Qwen3.6-27B-4bit", "--image", "photo.jpg"]);

    expect(parsed).toEqual({
      source: "mlx-community/Qwen3.6-27B-4bit",
      imagePath: "photo.jpg",
      prompt: "Describe this image.",
      maxTokens: 128,
      localFilesOnly: true,
      thinking: "disabled",
      json: false,
      overrides: {},
    });
  });

  test("parses generation, download, and thinking flags", () => {
    const parsed = parseArgs([
      "local-model",
      "--image",
      "image.png",
      "--prompt",
      "What colors are visible?",
      "--system-prompt",
      "Be concise.",
      "--max-tokens",
      "32",
      "--temperature",
      "0.2",
      "--top-k",
      "8",
      "--top-p",
      "0.9",
      "--greedy",
      "--allow-download",
      "--enable-thinking",
      "--template-default-thinking",
      "--json",
    ]);

    expect(parsed).toEqual({
      source: "local-model",
      imagePath: "image.png",
      prompt: "What colors are visible?",
      systemPrompt: "Be concise.",
      maxTokens: 32,
      localFilesOnly: false,
      thinking: "template-default",
      json: true,
      overrides: {
        temperature: 0,
        topK: 8,
        topP: 0.9,
      },
    });
  });

  test("rejects missing required inputs", () => {
    expect(() => parseArgs([])).toThrow("Missing model source.");
    expect(() => parseArgs(["model"])).toThrow("Missing required --image <path>.");
  });

  test("rejects non-positive max token counts", () => {
    expect(() => parseArgs(["model", "--image", "image.png", "--max-tokens", "0"])).toThrow(
      'Expected --max-tokens to be a positive integer, got "0".',
    );
  });
});
