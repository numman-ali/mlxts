import { describe, expect, test } from "bun:test";

import { exampleImageProofArtifactReport } from "../image-proof/test-fixtures";
import {
  formatSuccess,
  formatUsage,
  parseArgs,
  resolveFlux2KleinNegativePrompt,
  runFlux2KleinExampleCommand,
} from "./index";

describe("FLUX.2 Klein example command", () => {
  test("formats compact AXI help", () => {
    const usage = formatUsage();

    expect(usage).toContain("description:");
    expect(usage).toContain("usage[2]:");
    expect(usage).toContain("exit_codes[3]");
  });

  test("parses proof options", () => {
    const parsed = parseArgs([
      "/models/flux2",
      "--prompt",
      "a red apple",
      "--negative-prompt",
      "",
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
      "2",
      "--height",
      "512",
      "--width",
      "768",
      "--guidance-scale",
      "4",
      "--max-sequence-length",
      "128",
      "--seed",
      "11",
      "--dtype",
      "bfloat16",
      "--json",
    ]);

    expect(parsed).toEqual({
      source: "/models/flux2",
      revision: "refs/pr/1",
      cacheDir: "~/.cache/huggingface/hub",
      hfToken: "hf_test",
      variant: "fp16",
      localFilesOnly: true,
      prompt: "a red apple",
      negativePrompt: "",
      outputPath: ".tmp/out.bmp",
      steps: 2,
      height: 512,
      width: 768,
      guidanceScale: 4,
      maxSequenceLength: 128,
      seed: 11,
      dtype: "bfloat16",
      json: true,
    });
  });

  test("rejects usage errors before acquiring the runtime lock", async () => {
    const stdout: string[] = [];
    let lockAcquired = false;

    const exitCode = await runFlux2KleinExampleCommand(["/models/flux2"], {
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

    const exitCode = await runFlux2KleinExampleCommand(["--help"], {
      stdout: (line) => stdout.push(line),
      acquireLock: () => {
        lockAcquired = true;
        return { [Symbol.dispose]: () => {} };
      },
    });

    expect(exitCode).toBe(0);
    expect(lockAcquired).toBe(false);
    expect(stdout.join("\n")).toContain("examples/flux2/index.ts");
  });

  test("emits structured success and progress on separate channels", async () => {
    const stdout: string[] = [];
    const stderr: string[] = [];
    let disposed = false;

    const exitCode = await runFlux2KleinExampleCommand(
      ["/models/flux2", "--prompt", "a red apple", "--output", ".tmp/out.bmp"],
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
            snapshotPath: "/resolved/flux2",
            pipeline: "flux2-klein",
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
    expect(stdout.join("\n")).toContain("flux2_klein_example:");
    expect(stdout.join("\n")).toContain("status: passed");
    expect(stdout.join("\n")).toContain('output_path: ".tmp/out.bmp"');
    expect(stdout.join("\n")).toContain('negative_prompt: ""');
  });

  test("emits runtime errors on stdout and stack traces on stderr", async () => {
    const stdout: string[] = [];
    const stderr: string[] = [];

    const exitCode = await runFlux2KleinExampleCommand(
      ["/models/flux2", "--prompt", "a red apple"],
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

  test("validates output, geometry, dtype, sequence length, and CFG usage", () => {
    expect(() => parseArgs(["/models/flux2", "--prompt", "a", "--output", "sample.png"])).toThrow(
      ".bmp",
    );
    expect(() =>
      parseArgs(["/models/flux2", "--prompt", "a", "--max-sequence-length", "513"]),
    ).toThrow("512");
    expect(() => parseArgs(["/models/flux2", "--prompt", "a", "--dtype", "float64"])).toThrow(
      "float16",
    );
    expect(() => parseArgs(["/models/flux2", "--prompt", "a", "--height", "1025"])).toThrow(
      "divisible by 16",
    );
    expect(() =>
      parseArgs([
        "/models/flux2",
        "--prompt",
        "a",
        "--guidance-scale",
        "1",
        "--negative-prompt",
        "bad",
      ]),
    ).toThrow("greater than 1");
  });

  test("formats default success as compact structured output", () => {
    const formatted = formatSuccess({
      snapshotPath: "/models/flux2",
      source: "/models/flux2",
      pipeline: "flux2-klein",
      prompt: "a small robot",
      negativePrompt: "",
      outputPath: ".tmp/out.bmp",
      imageSize: { width: 1024, height: 1024 },
      outputBytes: 3_145_782,
      artifact: exampleImageProofArtifactReport({
        path: ".tmp/out.bmp",
        width: 1024,
        height: 1024,
        bytes: 3_145_782,
      }),
      steps: 2,
      guidanceScale: 4,
      maxSequenceLength: 512,
      seed: 0,
      dtype: "bfloat16",
      promptTruncated: false,
      negativePromptTruncated: true,
      elapsedMs: 12.34,
    });

    expect(formatted).toContain('pipeline: "flux2-klein"');
    expect(formatted).toContain('image_size: "1024x1024"');
    expect(formatted).toContain("artifact_sha256");
    expect(formatted).toContain("negative_prompt_truncated: true");
    expect(formatted).toContain("guidance_scale: 4");
  });

  test("resolves FLUX.2 Klein negative prompt rules", () => {
    expect(resolveFlux2KleinNegativePrompt(4, false, undefined)).toBe("");
    expect(resolveFlux2KleinNegativePrompt(4, false, "low quality")).toBe("low quality");
    expect(resolveFlux2KleinNegativePrompt(1, false, undefined)).toBeUndefined();
    expect(resolveFlux2KleinNegativePrompt(4, true, undefined)).toBeUndefined();
    expect(() => resolveFlux2KleinNegativePrompt(1, false, "low quality")).toThrow(
      "greater than 1",
    );
    expect(() => resolveFlux2KleinNegativePrompt(4, true, "low quality")).toThrow("distilled");
  });
});
