import { describe, expect, test } from "bun:test";

import { formatSuccess, formatUsage, parseArgs, runStableDiffusionExampleCommand } from "./index";

describe("Stable Diffusion example command", () => {
  test("formats compact AXI help", () => {
    const usage = formatUsage();

    expect(usage).toContain("description:");
    expect(usage).toContain("usage[2]:");
    expect(usage).toContain("exit_codes[3]");
  });

  test("parses proof options", () => {
    const parsed = parseArgs([
      "/models/sd",
      "--prompt",
      "a red apple",
      "--prompt-2",
      "a painted apple",
      "--negative-prompt",
      "blur",
      "--negative-prompt-2",
      "low detail",
      "--revision",
      "refs/pr/1",
      "--cache-dir",
      "~/.cache/huggingface/hub",
      "--hf-token",
      "hf_test",
      "--local-files-only",
      "--output",
      ".tmp/out.bmp",
      "--steps",
      "4",
      "--height",
      "64",
      "--width",
      "96",
      "--guidance-scale",
      "5",
      "--seed",
      "11",
      "--dtype",
      "float32",
      "--json",
    ]);

    expect(parsed).toEqual({
      source: "/models/sd",
      revision: "refs/pr/1",
      cacheDir: "~/.cache/huggingface/hub",
      hfToken: "hf_test",
      localFilesOnly: true,
      prompt: "a red apple",
      prompt2: "a painted apple",
      negativePrompt: "blur",
      negativePrompt2: "low detail",
      outputPath: ".tmp/out.bmp",
      steps: 4,
      height: 64,
      width: 96,
      guidanceScale: 5,
      seed: 11,
      dtype: "float32",
      json: true,
    });
  });

  test("rejects usage errors before acquiring the runtime lock", async () => {
    const stdout: string[] = [];
    let lockAcquired = false;

    const exitCode = await runStableDiffusionExampleCommand(["/models/sd"], {
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

    const exitCode = await runStableDiffusionExampleCommand(["--help"], {
      stdout: (line) => stdout.push(line),
      acquireLock: () => {
        lockAcquired = true;
        return { [Symbol.dispose]: () => {} };
      },
    });

    expect(exitCode).toBe(0);
    expect(lockAcquired).toBe(false);
    expect(stdout.join("\n")).toContain("stable-diffusion/index.ts");
  });

  test("emits structured success and progress on separate channels", async () => {
    const stdout: string[] = [];
    const stderr: string[] = [];
    let disposed = false;

    const exitCode = await runStableDiffusionExampleCommand(
      ["/models/sd", "--prompt", "a red apple", "--output", ".tmp/out.bmp"],
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
            snapshotPath: "/resolved/sd",
            pipeline: "stable-diffusion",
            prompt: cli.prompt,
            negativePrompt: "",
            outputPath: cli.outputPath,
            imageSize: { width: 64, height: 64 },
            outputBytes: 123,
            steps: cli.steps,
            guidanceScale: cli.guidanceScale,
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
    expect(stdout.join("\n")).toContain("stable_diffusion_example:");
    expect(stdout.join("\n")).toContain("status: passed");
    expect(stdout.join("\n")).toContain('output_path: ".tmp/out.bmp"');
  });

  test("emits runtime errors on stdout and stack traces on stderr", async () => {
    const stdout: string[] = [];
    const stderr: string[] = [];

    const exitCode = await runStableDiffusionExampleCommand(
      ["/models/sd", "--prompt", "a red apple"],
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

  test("formats default success as compact structured output", () => {
    const formatted = formatSuccess({
      snapshotPath: "/models/sd",
      source: "/models/sd",
      pipeline: "stable-diffusion-xl",
      prompt: "a small robot",
      negativePrompt: "blur",
      outputPath: ".tmp/out.bmp",
      imageSize: { width: 1024, height: 1024 },
      outputBytes: 3_145_782,
      steps: 20,
      guidanceScale: 7.5,
      seed: 0,
      dtype: "float16",
      promptTruncated: false,
      negativePromptTruncated: true,
      elapsedMs: 12.34,
    });

    expect(formatted).toContain('pipeline: "stable-diffusion-xl"');
    expect(formatted).toContain('image_size: "1024x1024"');
    expect(formatted).toContain("negative_prompt_truncated: true");
  });
});
