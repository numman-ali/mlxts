import { describe, expect, test } from "bun:test";

import {
  formatSuccess,
  formatUsage,
  parseArgs,
  resolveQwenImageNegativePrompt,
  runQwenImageExampleCommand,
} from "./index";

describe("Qwen-Image example command", () => {
  test("formats compact AXI help", () => {
    const usage = formatUsage();

    expect(usage).toContain("description:");
    expect(usage).toContain("usage[2]:");
    expect(usage).toContain("exit_codes[3]");
  });

  test("parses proof options", () => {
    const parsed = parseArgs([
      "/models/qwen-image",
      "--prompt",
      "a red apple",
      "--negative-prompt",
      " ",
      "--revision",
      "refs/pr/1",
      "--cache-dir",
      "~/.cache/huggingface/hub",
      "--hf-token",
      "hf_test",
      "--variant",
      "fp16",
      "--local-files-only",
      "--output",
      ".tmp/out.bmp",
      "--steps",
      "4",
      "--height",
      "64",
      "--width",
      "96",
      "--true-cfg-scale",
      "3.5",
      "--max-sequence-length",
      "128",
      "--seed",
      "11",
      "--dtype",
      "float32",
      "--json",
    ]);

    expect(parsed).toEqual({
      source: "/models/qwen-image",
      revision: "refs/pr/1",
      cacheDir: "~/.cache/huggingface/hub",
      hfToken: "hf_test",
      variant: "fp16",
      localFilesOnly: true,
      prompt: "a red apple",
      negativePrompt: " ",
      outputPath: ".tmp/out.bmp",
      steps: 4,
      height: 64,
      width: 96,
      trueCfgScale: 3.5,
      maxSequenceLength: 128,
      seed: 11,
      dtype: "float32",
      json: true,
    });
  });

  test("defaults true CFG negative prompt to a single space", () => {
    expect(parseArgs(["/models/qwen-image", "--prompt", "a red apple"]).negativePrompt).toBe(" ");
    expect(
      parseArgs(["/models/qwen-image", "--prompt", "a red apple", "--true-cfg-scale", "1"])
        .negativePrompt,
    ).toBeUndefined();
  });

  test("rejects usage errors before acquiring the runtime lock", async () => {
    const stdout: string[] = [];
    let lockAcquired = false;

    const exitCode = await runQwenImageExampleCommand(["/models/qwen-image"], {
      stdout: (line) => stdout.push(line),
      acquireLock: () => {
        lockAcquired = true;
        return { [Symbol.dispose]: () => {} };
      },
    });

    expect(exitCode).toBe(2);
    expect(lockAcquired).toBe(false);
    expect(stdout.join("\n")).toContain('code: "usage"');
    expect(stdout.join("\n")).toContain("Missing required --prompt");
  });

  test("runs help without acquiring the runtime lock", async () => {
    const stdout: string[] = [];
    let lockAcquired = false;

    const exitCode = await runQwenImageExampleCommand(["--help"], {
      stdout: (line) => stdout.push(line),
      acquireLock: () => {
        lockAcquired = true;
        return { [Symbol.dispose]: () => {} };
      },
    });

    expect(exitCode).toBe(0);
    expect(lockAcquired).toBe(false);
    expect(stdout.join("\n")).toContain("examples/qwen-image/index.ts");
  });

  test("emits structured success and progress on separate channels", async () => {
    const stdout: string[] = [];
    const stderr: string[] = [];
    let disposed = false;

    const exitCode = await runQwenImageExampleCommand(
      ["/models/qwen-image", "--prompt", "a red apple", "--output", ".tmp/out.bmp"],
      {
        stdout: (line) => stdout.push(line),
        stderr: (line) => stderr.push(line),
        acquireLock: () => ({
          [Symbol.dispose]: () => {
            disposed = true;
          },
        }),
        runExample: async (cli, progress) => {
          progress("fake progress");
          return {
            source: cli.source,
            snapshotPath: "/resolved/qwen-image",
            pipeline: "qwen-image",
            prompt: cli.prompt,
            negativePrompt: cli.negativePrompt ?? null,
            outputPath: cli.outputPath,
            imageSize: { width: 64, height: 64 },
            outputBytes: 123,
            steps: cli.steps,
            trueCfgScale: cli.trueCfgScale,
            maxSequenceLength: cli.maxSequenceLength,
            seed: cli.seed,
            dtype: cli.dtype,
            promptTruncated: false,
            negativePromptTruncated: false,
            elapsedMs: 1.5,
          };
        },
      },
    );

    expect(exitCode).toBe(0);
    expect(disposed).toBe(true);
    expect(stderr).toEqual(["fake progress"]);
    expect(stdout.join("\n")).toContain("qwen_image_example:");
    expect(stdout.join("\n")).toContain("status: passed");
    expect(stdout.join("\n")).toContain('output_path: ".tmp/out.bmp"');
    expect(stdout.join("\n")).toContain('negative_prompt: " "');
  });

  test("emits runtime errors on stdout and stack traces on stderr", async () => {
    const stdout: string[] = [];
    const stderr: string[] = [];

    const exitCode = await runQwenImageExampleCommand(
      ["/models/qwen-image", "--prompt", "a red apple"],
      {
        stdout: (line) => stdout.push(line),
        stderr: (line) => stderr.push(line),
        acquireLock: () => ({ [Symbol.dispose]: () => {} }),
        runExample: async () => {
          throw new Error("snapshot missing");
        },
      },
    );

    expect(exitCode).toBe(1);
    expect(stdout.join("\n")).toContain('code: "runtime"');
    expect(stdout.join("\n")).toContain("snapshot missing");
    expect(stderr.join("\n")).toContain("Error: snapshot missing");
  });

  test("validates output, sequence length, dtype, dimensions, and true CFG usage", () => {
    expect(() =>
      parseArgs(["/models/qwen-image", "--prompt", "a", "--output", "sample.png"]),
    ).toThrow(".bmp");
    expect(() =>
      parseArgs(["/models/qwen-image", "--prompt", "a", "--max-sequence-length", "1025"]),
    ).toThrow("1024");
    expect(() => parseArgs(["/models/qwen-image", "--prompt", "a", "--dtype", "float64"])).toThrow(
      "float16",
    );
    expect(() => parseArgs(["/models/qwen-image", "--prompt", "a", "--height", "65"])).toThrow(
      "divisible by 16",
    );
    expect(() =>
      parseArgs([
        "/models/qwen-image",
        "--prompt",
        "a",
        "--true-cfg-scale",
        "1",
        "--negative-prompt",
        "bad",
      ]),
    ).toThrow("greater than 1");
  });

  test("formats default success as compact structured output", () => {
    const formatted = formatSuccess({
      snapshotPath: "/models/qwen-image",
      source: "/models/qwen-image",
      pipeline: "qwen-image",
      prompt: "a small robot",
      negativePrompt: " ",
      outputPath: ".tmp/out.bmp",
      imageSize: { width: 1024, height: 1024 },
      outputBytes: 3_145_782,
      steps: 4,
      trueCfgScale: 4,
      maxSequenceLength: 1024,
      seed: 0,
      dtype: "bfloat16",
      promptTruncated: false,
      negativePromptTruncated: true,
      elapsedMs: 12.34,
    });

    expect(formatted).toContain('pipeline: "qwen-image"');
    expect(formatted).toContain('image_size: "1024x1024"');
    expect(formatted).toContain("negative_prompt_truncated: true");
    expect(formatted).toContain("true_cfg_scale: 4");
  });

  test("resolves Qwen-Image true CFG negative prompt rules", () => {
    expect(resolveQwenImageNegativePrompt(4, undefined)).toBe(" ");
    expect(resolveQwenImageNegativePrompt(4, "low quality")).toBe("low quality");
    expect(resolveQwenImageNegativePrompt(1, undefined)).toBeUndefined();
    expect(() => resolveQwenImageNegativePrompt(1, "low quality")).toThrow("greater than 1");
  });
});
