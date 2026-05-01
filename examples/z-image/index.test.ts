import { describe, expect, test } from "bun:test";

import { exampleImageProofArtifactReport } from "../image-proof/test-fixtures";
import {
  formatSuccess,
  formatUsage,
  parseArgs,
  resolveZImageGuidanceScale,
  runZImageExampleCommand,
} from "./index";

describe("Z-Image example command", () => {
  test("formats compact AXI help", () => {
    const usage = formatUsage();

    expect(usage).toContain("description:");
    expect(usage).toContain("usage[2]:");
    expect(usage).toContain("exit_codes[3]");
  });

  test("parses proof options", () => {
    const parsed = parseArgs([
      "/models/z-image",
      "--prompt",
      "a red apple",
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
      "9",
      "--height",
      "512",
      "--width",
      "768",
      "--guidance-scale",
      "0",
      "--max-sequence-length",
      "128",
      "--seed",
      "11",
      "--dtype",
      "bfloat16",
      "--json",
    ]);

    expect(parsed).toEqual({
      source: "/models/z-image",
      revision: "refs/pr/1",
      cacheDir: "~/.cache/huggingface/hub",
      hfToken: "hf_test",
      variant: "fp16",
      localFilesOnly: true,
      prompt: "a red apple",
      outputPath: ".tmp/out.bmp",
      steps: 9,
      height: 512,
      width: 768,
      guidanceScale: 0,
      maxSequenceLength: 128,
      seed: 11,
      dtype: "bfloat16",
      json: true,
    });
  });

  test("rejects usage errors before acquiring the runtime lock", async () => {
    const stdout: string[] = [];
    let lockAcquired = false;

    const exitCode = await runZImageExampleCommand(["/models/z-image"], {
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

    const exitCode = await runZImageExampleCommand(["--help"], {
      stdout: (line) => stdout.push(line),
      acquireLock: () => {
        lockAcquired = true;
        return { [Symbol.dispose]: () => {} };
      },
    });

    expect(exitCode).toBe(0);
    expect(lockAcquired).toBe(false);
    expect(stdout.join("\n")).toContain("examples/z-image/index.ts");
  });

  test("emits structured success and progress on separate channels", async () => {
    const stdout: string[] = [];
    const stderr: string[] = [];
    let disposed = false;

    const exitCode = await runZImageExampleCommand(
      ["/models/z-image", "--prompt", "a red apple", "--output", ".tmp/out.bmp"],
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
            snapshotPath: "/resolved/z-image",
            pipeline: "z-image",
            prompt: cli.prompt,
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
            seed: cli.seed,
            dtype: cli.dtype,
            promptTruncated: false,
            elapsedMs: 1.5,
          };
        },
      },
    );

    expect(exitCode).toBe(0);
    expect(disposed).toBe(true);
    expect(stderr).toEqual(["fake progress"]);
    expect(stdout.join("\n")).toContain("z_image_example:");
    expect(stdout.join("\n")).toContain("status: passed");
    expect(stdout.join("\n")).toContain('output_path: ".tmp/out.bmp"');
  });

  test("emits runtime errors on stdout and stack traces on stderr", async () => {
    const stdout: string[] = [];
    const stderr: string[] = [];

    const exitCode = await runZImageExampleCommand(["/models/z-image", "--prompt", "a red apple"], {
      stdout: (line) => stdout.push(line),
      stderr: (line) => stderr.push(line),
      acquireLock: () => ({ [Symbol.dispose]: () => {} }),
      runExample: async () => {
        throw new Error("snapshot missing");
      },
    });

    expect(exitCode).toBe(1);
    expect(stdout.join("\n")).toContain('code: "runtime"');
    expect(stdout.join("\n")).toContain("snapshot missing");
    expect(stderr.join("\n")).toContain("Error: snapshot missing");
  });

  test("validates output, geometry, dtype, and guidance usage", () => {
    expect(() => parseArgs(["/models/z-image", "--prompt", "a", "--output", "sample.png"])).toThrow(
      ".bmp",
    );
    expect(() =>
      parseArgs(["/models/z-image", "--prompt", "a", "--max-sequence-length", "513"]),
    ).toThrow("512");
    expect(() => parseArgs(["/models/z-image", "--prompt", "a", "--dtype", "float64"])).toThrow(
      "float16",
    );
    expect(() => parseArgs(["/models/z-image", "--prompt", "a", "--height", "1025"])).toThrow(
      "divisible by 16",
    );
    expect(() => parseArgs(["/models/z-image", "--prompt", "a", "--guidance-scale", "1"])).toThrow(
      "guidance-scale",
    );
  });

  test("formats default success as compact structured output", () => {
    const formatted = formatSuccess({
      snapshotPath: "/models/z-image",
      source: "/models/z-image",
      pipeline: "z-image",
      prompt: "a small robot",
      outputPath: ".tmp/out.bmp",
      imageSize: { width: 1024, height: 1024 },
      outputBytes: 3_145_782,
      artifact: exampleImageProofArtifactReport({
        path: ".tmp/out.bmp",
        width: 1024,
        height: 1024,
        bytes: 3_145_782,
      }),
      steps: 9,
      guidanceScale: 0,
      maxSequenceLength: 512,
      seed: 0,
      dtype: "float32",
      promptTruncated: true,
      elapsedMs: 12.34,
    });

    expect(formatted).toContain('pipeline: "z-image"');
    expect(formatted).toContain('image_size: "1024x1024"');
    expect(formatted).toContain("artifact_sha256");
    expect(formatted).toContain("prompt_truncated: true");
    expect(formatted).toContain("guidance_scale: 0");
  });

  test("keeps current Z-Image proof guidance fixed to zero", () => {
    expect(resolveZImageGuidanceScale(0)).toBe(0);
    expect(() => resolveZImageGuidanceScale(2.5)).toThrow("0 only");
  });
});
