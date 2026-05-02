import { describe, expect, test } from "bun:test";

import { exampleImageProofArtifactReport } from "../image-proof/test-fixtures";
import { formatSuccess, formatUsage, parseArgs, runLtxVideoExampleCommand } from "./index";

describe("LTX-Video example command", () => {
  test("formats compact AXI help", () => {
    const usage = formatUsage();

    expect(usage).toContain("description:");
    expect(usage).toContain("usage[2]:");
    expect(usage).toContain("exit_codes[3]");
  });

  test("parses proof options", () => {
    const parsed = parseArgs([
      "/models/ltx-video",
      "--prompt",
      "a red apple",
      "--negative-prompt",
      "low quality",
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
      "--audio-output",
      ".tmp/out.wav",
      "--steps",
      "4",
      "--height",
      "128",
      "--width",
      "160",
      "--frames",
      "9",
      "--frame-rate",
      "24",
      "--guidance-scale",
      "3.5",
      "--audio-guidance-scale",
      "2.5",
      "--max-sequence-length",
      "64",
      "--seed",
      "11",
      "--dtype",
      "float32",
      "--json",
    ]);

    expect(parsed).toEqual({
      source: "/models/ltx-video",
      revision: "refs/pr/1",
      cacheDir: "~/.cache/huggingface/hub",
      hfToken: "hf_test",
      variant: "fp16",
      localFilesOnly: true,
      prompt: "a red apple",
      negativePrompt: "low quality",
      outputPath: ".tmp/out.bmp",
      audioOutputPath: ".tmp/out.wav",
      steps: 4,
      height: 128,
      width: 160,
      frames: 9,
      frameRate: 24,
      guidanceScale: 3.5,
      audioGuidanceScale: 2.5,
      maxSequenceLength: 64,
      seed: 11,
      dtype: "float32",
      json: true,
    });
  });

  test("rejects usage errors before acquiring the runtime lock", async () => {
    const stdout: string[] = [];
    let lockAcquired = false;

    const exitCode = await runLtxVideoExampleCommand(["/models/ltx-video"], {
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

    const exitCode = await runLtxVideoExampleCommand(["--help"], {
      stdout: (line) => stdout.push(line),
      acquireLock: () => {
        lockAcquired = true;
        return { [Symbol.dispose]: () => {} };
      },
    });

    expect(exitCode).toBe(0);
    expect(lockAcquired).toBe(false);
    expect(stdout.join("\n")).toContain("examples/ltx-video/index.ts");
  });

  test("emits structured success and progress on separate channels", async () => {
    const stdout: string[] = [];
    const stderr: string[] = [];
    let disposed = false;

    const exitCode = await runLtxVideoExampleCommand(
      ["/models/ltx-video", "--prompt", "a red apple", "--output", ".tmp/out.bmp"],
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
            snapshotPath: "/resolved/ltx-video",
            pipeline: "ltx-video",
            prompt: cli.prompt,
            negativePrompt: "",
            outputPath: cli.outputPath,
            imageSize: { width: 128, height: 128 },
            videoSize: { width: 128, height: 128, frames: 9, channels: 3 },
            latentSize: { width: 4, height: 4, frames: 2, channels: 128 },
            outputBytes: 123,
            artifact: exampleImageProofArtifactReport({
              path: cli.outputPath,
              width: 128,
              height: 128,
              bytes: 123,
            }),
            steps: cli.steps,
            guidanceScale: cli.guidanceScale,
            maxSequenceLength: cli.maxSequenceLength,
            requestedFrames: cli.frames,
            decodedFrames: 9,
            frameRate: cli.frameRate,
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
    expect(stdout.join("\n")).toContain("ltx_video_example:");
    expect(stdout.join("\n")).toContain("status: passed");
    expect(stdout.join("\n")).toContain('pipeline: "ltx-video"');
    expect(stdout.join("\n")).toContain('output_path: ".tmp/out.bmp"');
  });

  test("emits runtime errors on stdout and stack traces on stderr", async () => {
    const stdout: string[] = [];
    const stderr: string[] = [];

    const exitCode = await runLtxVideoExampleCommand(
      ["/models/ltx-video", "--prompt", "a red apple"],
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

  test("validates output, geometry, dtype, sequence length, and guidance usage", () => {
    expect(() =>
      parseArgs(["/models/ltx-video", "--prompt", "a", "--output", "sample.png"]),
    ).toThrow(".bmp");
    expect(() => parseArgs(["/models/ltx-video", "--prompt", "a", "--height", "130"])).toThrow(
      "divisible by 32",
    );
    expect(() =>
      parseArgs(["/models/ltx-video", "--prompt", "a", "--audio-output", "sample.mp3"]),
    ).toThrow(".wav");
    expect(() =>
      parseArgs(["/models/ltx-video", "--prompt", "a", "--max-sequence-length", "1025"]),
    ).toThrow("1024");
    expect(() => parseArgs(["/models/ltx-video", "--prompt", "a", "--dtype", "float64"])).toThrow(
      "float16",
    );
    expect(() =>
      parseArgs(["/models/ltx-video", "--prompt", "a", "--guidance-scale", "-1"]),
    ).toThrow("non-negative");
  });

  test("formats default success as compact structured output", () => {
    const formatted = formatSuccess({
      snapshotPath: "/models/ltx-video",
      source: "/models/ltx-video",
      pipeline: "ltx-video",
      prompt: "a small robot",
      negativePrompt: "",
      outputPath: ".tmp/out.bmp",
      imageSize: { width: 512, height: 128 },
      videoSize: { width: 128, height: 128, frames: 9, channels: 3 },
      latentSize: { width: 4, height: 4, frames: 2, channels: 128 },
      outputBytes: 196_662,
      artifact: exampleImageProofArtifactReport({
        path: ".tmp/out.bmp",
        width: 512,
        height: 128,
        bytes: 196_662,
      }),
      steps: 4,
      guidanceScale: 3,
      maxSequenceLength: 128,
      requestedFrames: 9,
      decodedFrames: 9,
      frameRate: 25,
      seed: 0,
      dtype: "float16",
      promptTruncated: false,
      negativePromptTruncated: false,
      elapsedMs: 2,
    });

    expect(formatted).toContain("ltx_video_example:");
    expect(formatted).toContain('pipeline: "ltx-video"');
    expect(formatted).toContain('video_size: "128x128x9"');
    expect(formatted).toContain("requested_frames: 9");
    expect(formatted).toContain("decoded_frames: 9");
    expect(formatted).toContain(`artifact_sha256: "${"a".repeat(64)}"`);
  });

  test("formats LTX-2 audio proof fields when present", () => {
    const formatted = formatSuccess({
      snapshotPath: "/models/ltx2",
      source: "/models/ltx2",
      pipeline: "ltx2",
      prompt: "a small robot",
      negativePrompt: "",
      outputPath: ".tmp/out.bmp",
      audioOutputPath: ".tmp/out.wav",
      imageSize: { width: 512, height: 128 },
      videoSize: { width: 128, height: 128, frames: 9, channels: 3 },
      latentSize: { width: 4, height: 4, frames: 2, channels: 128 },
      outputBytes: 196_662,
      artifact: exampleImageProofArtifactReport({
        path: ".tmp/out.bmp",
        width: 512,
        height: 128,
        bytes: 196_662,
      }),
      audioSize: {
        sampleRate: 16000,
        channels: 1,
        samples: 8000,
        durationSeconds: 0.5,
      },
      audioOutputBytes: 16_044,
      audioArtifact: {
        path: ".tmp/out.wav",
        format: "wav",
        sampleRate: 16000,
        channels: 1,
        samples: 8000,
        durationSeconds: 0.5,
        bitsPerSample: 16,
        bytes: 16_044,
        sha256: "b".repeat(64),
        peakAbs: 0.75,
        meanAbs: 0.12,
        checks: {
          riffHeaderValid: true,
          byteLengthMatches: true,
          sampleRateMatches: true,
          sha256Present: true,
          finiteTensor: true,
          waveformHasSamples: true,
        },
        status: "passed",
      },
      steps: 4,
      guidanceScale: 3,
      audioGuidanceScale: 2,
      maxSequenceLength: 128,
      requestedFrames: 9,
      decodedFrames: 9,
      frameRate: 25,
      seed: 0,
      dtype: "float16",
      promptTruncated: false,
      negativePromptTruncated: false,
      elapsedMs: 2,
    });

    expect(formatted).toContain('pipeline: "ltx2"');
    expect(formatted).toContain('audio_output_path: ".tmp/out.wav"');
    expect(formatted).toContain("audio_sample_rate: 16000");
    expect(formatted).toContain('audio_artifact_sha256: "bbbb');
    expect(formatted).toContain("audio_guidance_scale: 2");
  });
});
