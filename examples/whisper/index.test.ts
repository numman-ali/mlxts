import { describe, expect, test } from "bun:test";

import { formatSuccess, formatUsage, parseArgs, runWhisperExampleCommand } from "./index";

describe("Whisper example command", () => {
  test("formats compact AXI help", () => {
    const usage = formatUsage();

    expect(usage).toContain("description:");
    expect(usage).toContain("usage[2]:");
    expect(usage).toContain("exit_codes[3]");
  });

  test("parses proof options", () => {
    const parsed = parseArgs([
      "/models/whisper",
      "--audio",
      "./speech.wav",
      "--revision",
      "refs/pr/1",
      "--cache-dir",
      "~/.cache/huggingface/hub",
      "--hf-token",
      "hf_test",
      "--local-files-only",
      "--task",
      "translate",
      "--language",
      "fr",
      "--timestamps",
      "--max-tokens",
      "12",
      "--json",
    ]);

    expect(parsed).toEqual({
      source: "/models/whisper",
      audioPath: "./speech.wav",
      revision: "refs/pr/1",
      cacheDir: "~/.cache/huggingface/hub",
      hfToken: "hf_test",
      localFilesOnly: true,
      task: "translate",
      language: "fr",
      withoutTimestamps: false,
      maxTokens: 12,
      json: true,
    });
  });

  test("rejects usage errors before acquiring the runtime lock", async () => {
    const stdout: string[] = [];
    let lockAcquired = false;

    const exitCode = await runWhisperExampleCommand(["/models/whisper"], {
      stdout: (line) => stdout.push(line),
      acquireLock: () => {
        lockAcquired = true;
        return { [Symbol.dispose]: () => {} };
      },
    });

    expect(exitCode).toBe(2);
    expect(lockAcquired).toBe(false);
    expect(stdout.join("\n")).toContain('code: "usage"');
    expect(stdout.join("\n")).toContain("Missing required --audio");
  });

  test("runs help without acquiring the runtime lock", async () => {
    const stdout: string[] = [];
    let lockAcquired = false;

    const exitCode = await runWhisperExampleCommand(["--help"], {
      stdout: (line) => stdout.push(line),
      acquireLock: () => {
        lockAcquired = true;
        return { [Symbol.dispose]: () => {} };
      },
    });

    expect(exitCode).toBe(0);
    expect(lockAcquired).toBe(false);
    expect(stdout.join("\n")).toContain("whisper/index.ts");
  });

  test("emits structured success and progress on separate channels", async () => {
    const stdout: string[] = [];
    const stderr: string[] = [];
    let disposed = false;

    const exitCode = await runWhisperExampleCommand(
      ["/models/whisper", "--audio", "./speech.wav"],
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
            snapshotPath: "/resolved/whisper",
            audioPath: cli.audioPath,
            sampleRate: 16000,
            channels: 1,
            frames: 16000,
            durationSeconds: 1,
            task: cli.task,
            language: "en",
            withoutTimestamps: cli.withoutTimestamps,
            maxTokens: cli.maxTokens,
            promptTokens: 4,
            generatedTokens: 2,
            stoppedReason: "eos",
            text: "hello",
            elapsedMs: 1.5,
          };
        },
      },
    );

    expect(exitCode).toBe(0);
    expect(disposed).toBe(true);
    expect(stderr).toEqual(["fake progress"]);
    expect(stdout.join("\n")).toContain("whisper_example:");
    expect(stdout.join("\n")).toContain("status: passed");
    expect(stdout.join("\n")).toContain('text: "hello"');
  });

  test("emits runtime errors on stdout and stack traces on stderr", async () => {
    const stdout: string[] = [];
    const stderr: string[] = [];

    const exitCode = await runWhisperExampleCommand(
      ["/models/whisper", "--audio", "./speech.wav"],
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
      source: "/models/whisper",
      snapshotPath: "/models/whisper",
      audioPath: "./speech.wav",
      sampleRate: 16000,
      channels: 1,
      frames: 16000,
      durationSeconds: 1,
      task: "transcribe",
      language: "en",
      withoutTimestamps: true,
      maxTokens: 64,
      promptTokens: 4,
      generatedTokens: 3,
      stoppedReason: "eos",
      text: "hello there",
      elapsedMs: 12.34,
    });

    expect(formatted).toContain('task: "transcribe"');
    expect(formatted).toContain("sample_rate: 16000");
    expect(formatted).toContain('text: "hello there"');
  });
});
