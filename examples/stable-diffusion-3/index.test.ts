import { describe, expect, test } from "bun:test";

import { exampleImageProofArtifactReport } from "../image-proof/test-fixtures";
import { formatSuccess, formatUsage, parseArgs, runStableDiffusion3ExampleCommand } from "./index";

describe("Stable Diffusion 3 example command", () => {
  test("formats compact AXI help", () => {
    const usage = formatUsage();

    expect(usage).toContain("description:");
    expect(usage).toContain("usage[2]:");
    expect(usage).toContain("exit_codes[3]");
  });

  test("parses proof options", () => {
    const parsed = parseArgs([
      "/models/sd3",
      "--prompt",
      "a red apple",
      "--prompt-2",
      "a painted apple",
      "--prompt-3",
      "a cinematic apple",
      "--negative-prompt",
      "blur",
      "--negative-prompt-2",
      "",
      "--negative-prompt-3",
      "low detail",
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
      "--guidance-scale",
      "5",
      "--max-sequence-length",
      "128",
      "--clip-skip",
      "1",
      "--seed",
      "11",
      "--dtype",
      "float32",
      "--json",
    ]);

    expect(parsed).toEqual({
      source: "/models/sd3",
      revision: "refs/pr/1",
      cacheDir: "~/.cache/huggingface/hub",
      hfToken: "hf_test",
      variant: "fp16",
      localFilesOnly: true,
      prompt: "a red apple",
      prompt2: "a painted apple",
      prompt3: "a cinematic apple",
      negativePrompt: "blur",
      negativePrompt2: "",
      negativePrompt3: "low detail",
      outputPath: ".tmp/out.bmp",
      steps: 4,
      height: 64,
      width: 96,
      guidanceScale: 5,
      maxSequenceLength: 128,
      clipSkip: 1,
      seed: 11,
      dtype: "float32",
      json: true,
    });
  });

  test("rejects usage errors before acquiring the runtime lock", async () => {
    const stdout: string[] = [];
    let lockAcquired = false;

    const exitCode = await runStableDiffusion3ExampleCommand(["/models/sd3"], {
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

    const exitCode = await runStableDiffusion3ExampleCommand(["--help"], {
      stdout: (line) => stdout.push(line),
      acquireLock: () => {
        lockAcquired = true;
        return { [Symbol.dispose]: () => {} };
      },
    });

    expect(exitCode).toBe(0);
    expect(lockAcquired).toBe(false);
    expect(stdout.join("\n")).toContain("stable-diffusion-3/index.ts");
  });

  test("emits structured success and progress on separate channels", async () => {
    const stdout: string[] = [];
    const stderr: string[] = [];
    let disposed = false;

    const exitCode = await runStableDiffusion3ExampleCommand(
      ["/models/sd3", "--prompt", "a red apple", "--output", ".tmp/out.bmp"],
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
            snapshotPath: "/resolved/sd3",
            pipeline: "stable-diffusion-3",
            prompt: cli.prompt,
            negativePrompt: "",
            outputPath: cli.outputPath,
            imageSize: { width: 64, height: 64 },
            outputBytes: 123,
            artifact: exampleImageProofArtifactReport({
              path: cli.outputPath,
              width: 64,
              height: 64,
              bytes: 123,
            }),
            steps: cli.steps,
            guidanceScale: cli.guidanceScale,
            maxSequenceLength: cli.maxSequenceLength,
            clipSkip: cli.clipSkip,
            seed: cli.seed,
            dtype: cli.dtype,
            promptTruncated: false,
            prompt2Truncated: false,
            prompt3Truncated: false,
            negativePromptTruncated: false,
            negativePrompt2Truncated: false,
            negativePrompt3Truncated: false,
            elapsedMs: 1.5,
          };
        },
      },
    );

    expect(exitCode).toBe(0);
    expect(disposed).toBe(true);
    expect(stderr).toEqual(["fake progress"]);
    expect(stdout.join("\n")).toContain("stable_diffusion_3_example:");
    expect(stdout.join("\n")).toContain("status: passed");
    expect(stdout.join("\n")).toContain('output_path: ".tmp/out.bmp"');
  });

  test("emits runtime errors on stdout and stack traces on stderr", async () => {
    const stdout: string[] = [];
    const stderr: string[] = [];

    const exitCode = await runStableDiffusion3ExampleCommand(
      ["/models/sd3", "--prompt", "a red apple"],
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

  test("validates output, sequence length, clip skip, dtype, and inactive CFG", () => {
    expect(() => parseArgs(["/models/sd3", "--prompt", "cat", "--output", ".tmp/out.png"])).toThrow(
      ".bmp",
    );
    expect(() =>
      parseArgs(["/models/sd3", "--prompt", "cat", "--max-sequence-length", "513"]),
    ).toThrow("max-sequence-length");
    expect(() => parseArgs(["/models/sd3", "--prompt", "cat", "--clip-skip", "-1"])).toThrow(
      "clip-skip",
    );
    expect(() => parseArgs(["/models/sd3", "--prompt", "cat", "--dtype", "int32"])).toThrow(
      "dtype",
    );
    expect(() =>
      parseArgs([
        "/models/sd3",
        "--prompt",
        "cat",
        "--guidance-scale",
        "1",
        "--negative-prompt",
        "blur",
      ]),
    ).toThrow("guidance-scale");
  });

  test("formats default success as compact structured output", () => {
    const formatted = formatSuccess({
      snapshotPath: "/models/sd3",
      source: "/models/sd3",
      pipeline: "stable-diffusion-3",
      prompt: "a small robot",
      negativePrompt: "blur",
      outputPath: ".tmp/out.bmp",
      imageSize: { width: 1024, height: 1024 },
      outputBytes: 3_145_782,
      artifact: exampleImageProofArtifactReport({
        path: ".tmp/out.bmp",
        width: 1024,
        height: 1024,
        bytes: 3_145_782,
      }),
      steps: 28,
      guidanceScale: 7,
      maxSequenceLength: 256,
      clipSkip: 0,
      seed: 0,
      dtype: "bfloat16",
      promptTruncated: false,
      prompt2Truncated: false,
      prompt3Truncated: false,
      negativePromptTruncated: true,
      negativePrompt2Truncated: false,
      negativePrompt3Truncated: false,
      elapsedMs: 12.34,
    });

    expect(formatted).toContain('pipeline: "stable-diffusion-3"');
    expect(formatted).toContain('image_size: "1024x1024"');
    expect(formatted).toContain("artifact_sha256");
    expect(formatted).toContain("negative_prompt_truncated: true");
  });
});
