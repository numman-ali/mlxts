import { describe, expect, test } from "bun:test";

import {
  formatSuccess,
  formatUsage,
  parseArgs,
  resolveFluxGuidanceScale,
  runFluxExampleCommand,
} from "./index";

describe("FLUX example command", () => {
  test("formats compact AXI help", () => {
    const usage = formatUsage();

    expect(usage).toContain("description:");
    expect(usage).toContain("usage[2]:");
    expect(usage).toContain("exit_codes[3]");
  });

  test("parses proof options", () => {
    const parsed = parseArgs([
      "/models/flux",
      "--prompt",
      "a red apple",
      "--prompt-2",
      "a painted apple",
      "--output",
      ".tmp/out.bmp",
      "--steps",
      "4",
      "--height",
      "64",
      "--width",
      "96",
      "--guidance-scale",
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
      snapshotPath: "/models/flux",
      prompt: "a red apple",
      prompt2: "a painted apple",
      outputPath: ".tmp/out.bmp",
      steps: 4,
      height: 64,
      width: 96,
      guidanceScale: 3.5,
      maxSequenceLength: 128,
      seed: 11,
      dtype: "float32",
      json: true,
    });
  });

  test("rejects usage errors before acquiring the runtime lock", async () => {
    const stdout: string[] = [];
    let lockAcquired = false;

    const exitCode = await runFluxExampleCommand(["/models/flux"], {
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

    const exitCode = await runFluxExampleCommand(["--help"], {
      stdout: (line) => stdout.push(line),
      acquireLock: () => {
        lockAcquired = true;
        return { [Symbol.dispose]: () => {} };
      },
    });

    expect(exitCode).toBe(0);
    expect(lockAcquired).toBe(false);
    expect(stdout.join("\n")).toContain("examples/flux/index.ts");
  });

  test("emits structured success and progress on separate channels", async () => {
    const stdout: string[] = [];
    const stderr: string[] = [];
    let disposed = false;

    const exitCode = await runFluxExampleCommand(
      ["/models/flux", "--prompt", "a red apple", "--output", ".tmp/out.bmp"],
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
            snapshotPath: cli.snapshotPath,
            pipeline: "flux",
            prompt: cli.prompt,
            prompt2: cli.prompt2 ?? cli.prompt,
            outputPath: cli.outputPath,
            imageSize: { width: 64, height: 64 },
            outputBytes: 123,
            steps: cli.steps,
            guidanceScale: null,
            maxSequenceLength: cli.maxSequenceLength,
            seed: cli.seed,
            dtype: cli.dtype,
            promptTruncated: false,
            prompt2Truncated: false,
            elapsedMs: 1.5,
          };
        },
      },
    );

    expect(exitCode).toBe(0);
    expect(disposed).toBe(true);
    expect(stderr).toEqual(["fake progress"]);
    expect(stdout.join("\n")).toContain("flux_example:");
    expect(stdout.join("\n")).toContain("status: passed");
    expect(stdout.join("\n")).toContain('output_path: ".tmp/out.bmp"');
  });

  test("emits runtime errors on stdout and stack traces on stderr", async () => {
    const stdout: string[] = [];
    const stderr: string[] = [];

    const exitCode = await runFluxExampleCommand(["/models/flux", "--prompt", "a red apple"], {
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

  test("validates output, sequence length, dtype, and guidance usage", () => {
    expect(() => parseArgs(["/models/flux", "--prompt", "a", "--output", "sample.png"])).toThrow(
      ".bmp",
    );
    expect(() =>
      parseArgs(["/models/flux", "--prompt", "a", "--max-sequence-length", "513"]),
    ).toThrow("512");
    expect(() => parseArgs(["/models/flux", "--prompt", "a", "--dtype", "float64"])).toThrow(
      "float16",
    );
    expect(() => parseArgs(["/models/flux", "--prompt", "a", "--guidance-scale", "-1"])).toThrow(
      "non-negative",
    );
  });

  test("formats default success as compact structured output", () => {
    const formatted = formatSuccess({
      snapshotPath: "/models/flux",
      pipeline: "flux",
      prompt: "a small robot",
      prompt2: "a painted robot",
      outputPath: ".tmp/out.bmp",
      imageSize: { width: 1024, height: 1024 },
      outputBytes: 3_145_782,
      steps: 4,
      guidanceScale: 3.5,
      maxSequenceLength: 512,
      seed: 0,
      dtype: "float16",
      promptTruncated: false,
      prompt2Truncated: true,
      elapsedMs: 12.34,
    });

    expect(formatted).toContain('pipeline: "flux"');
    expect(formatted).toContain('image_size: "1024x1024"');
    expect(formatted).toContain("prompt_2_truncated: true");
    expect(formatted).toContain("guidance_scale: 3.5");
  });

  test("resolves guidance only for guidance-embedded FLUX checkpoints", () => {
    expect(resolveFluxGuidanceScale(undefined, false)).toBeUndefined();
    expect(resolveFluxGuidanceScale(undefined, true)).toBe(3.5);
    expect(resolveFluxGuidanceScale(2.5, true)).toBe(2.5);
    expect(() => resolveFluxGuidanceScale(2.5, false)).toThrow("does not support");
  });
});
