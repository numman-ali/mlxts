import { describe, expect, test } from "bun:test";

import {
  formatSuccess,
  formatUsage,
  parseArgs,
  parseCommand,
  runQwenImageExampleCommand,
} from "./index";

describe("parseArgs", () => {
  test("parses help without requiring model or image inputs", () => {
    expect(parseCommand(["--help"])).toEqual({ kind: "help" });
    expect(parseCommand(["local-model", "--help"])).toEqual({ kind: "help" });
  });

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
    expect(() => parseArgs(["model", "--image", "--json"])).toThrow("Missing value for --image.");
  });

  test("rejects non-positive max token counts", () => {
    expect(() => parseArgs(["model", "--image", "image.png", "--max-tokens", "0"])).toThrow(
      'Expected --max-tokens to be a positive integer, got "0".',
    );
  });
});

describe("qwen image example command", () => {
  const report = {
    source: "local-model",
    sourceMode: "cached-local-only" as const,
    imagePath: "image.png",
    prompt: "Describe this image.",
    thinking: "disabled" as const,
    originalSize: { width: 96, height: 96 },
    resizedSize: { width: 256, height: 256 },
    finishReason: "eos",
    generatedTokens: 12,
    text: "A red square.",
    elapsedMs: 12.34,
  };

  test("help is compact AXI stdout and does not acquire the runtime lock", async () => {
    const stdout: string[] = [];
    let lockCalls = 0;

    const exitCode = await runQwenImageExampleCommand(["--help"], {
      stdout: (text) => stdout.push(text),
      acquireLock: () => {
        lockCalls += 1;
        return { [Symbol.dispose]: () => undefined };
      },
    });

    expect(exitCode).toBe(0);
    expect(lockCalls).toBe(0);
    expect(stdout.join("\n")).toBe(formatUsage());
    expect(stdout.join("\n")).toContain("options[14]");
    expect(stdout.join("\n")).toContain("exit_codes[3]");
  });

  test("usage errors use structured stdout before the runtime lock", async () => {
    const stdout: string[] = [];
    let lockCalls = 0;

    const exitCode = await runQwenImageExampleCommand(["local-model", "--image", "--json"], {
      stdout: (text) => stdout.push(text),
      acquireLock: () => {
        lockCalls += 1;
        return { [Symbol.dispose]: () => undefined };
      },
    });

    expect(exitCode).toBe(2);
    expect(lockCalls).toBe(0);
    expect(stdout.join("\n")).toContain('code: "usage"');
    expect(stdout.join("\n")).toContain("Missing value for --image");
  });

  test("success output is structured and progress stays on stderr", async () => {
    const stdout: string[] = [];
    const stderr: string[] = [];
    let lockDepth = 0;

    const exitCode = await runQwenImageExampleCommand(["local-model", "--image", "image.png"], {
      stdout: (text) => stdout.push(text),
      stderr: (text) => stderr.push(text),
      acquireLock: () => {
        lockDepth += 1;
        return {
          [Symbol.dispose]: () => {
            lockDepth -= 1;
          },
        };
      },
      runExample: async (_cli, progress) => {
        expect(lockDepth).toBe(1);
        progress("loading model");
        return report;
      },
    });

    expect(exitCode).toBe(0);
    expect(lockDepth).toBe(0);
    expect(stderr).toEqual(["loading model"]);
    expect(stdout.join("\n")).toBe(formatSuccess(report));
    expect(stdout.join("\n")).toContain("qwen_image_example:");
  });

  test("json mode preserves JSON stdout and runtime errors are structured", async () => {
    const jsonStdout: string[] = [];
    expect(
      await runQwenImageExampleCommand(["local-model", "--image", "image.png", "--json"], {
        stdout: (text) => jsonStdout.push(text),
        stderr: () => undefined,
        acquireLock: () => ({ [Symbol.dispose]: () => undefined }),
        runExample: async () => report,
      }),
    ).toBe(0);
    expect(JSON.parse(jsonStdout.join("\n"))).toMatchObject({
      source: "local-model",
      generatedTokens: 12,
    });

    const runtimeStdout: string[] = [];
    const runtimeStderr: string[] = [];
    expect(
      await runQwenImageExampleCommand(["local-model", "--image", "image.png"], {
        stdout: (text) => runtimeStdout.push(text),
        stderr: (text) => runtimeStderr.push(text),
        acquireLock: () => ({ [Symbol.dispose]: () => undefined }),
        runExample: async () => {
          throw new Error("model load failed");
        },
      }),
    ).toBe(1);
    expect(runtimeStdout.join("\n")).toContain('code: "runtime"');
    expect(runtimeStdout.join("\n")).toContain("model load failed");
    expect(runtimeStderr.join("\n")).toContain("Error: model load failed");
  });
});
