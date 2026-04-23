import { describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

import { buildAcceptanceArgs, readPreset, main as soakMain } from "./soak";

describe("soak runner", () => {
  test("readPreset defaults to gpt-small", () => {
    expect(readPreset([])).toBe("gpt-small");
  });

  test("readPreset accepts gpt-tiny and gpt-small", () => {
    expect(readPreset(["--preset", "gpt-tiny"])).toBe("gpt-tiny");
    expect(readPreset(["--preset", "gpt-small"])).toBe("gpt-small");
  });

  test("buildAcceptanceArgs injects soak mode and preset defaults", () => {
    const args = buildAcceptanceArgs(["--preset", "gpt-small"]);

    expect(args).toContain("--mode");
    expect(args).toContain("soak");
    expect(args).toContain("--max-steps");
    expect(args).toContain("50");
    expect(args).toContain("--grad-accum");
    expect(args).toContain("8");
    expect(args).toContain("--max-slope-mb-per-event");
    expect(args).toContain("8");
    expect(args).toContain("--stall-timeout-sec");
    expect(args).toContain("600");
    expect(args).toContain("--early-stop-patience");
    expect(args).toContain("none");
    expect(args).toContain("--throughput-window");
    expect(args).toContain("5");
  });

  test("buildAcceptanceArgs preserves explicit overrides", () => {
    const args = buildAcceptanceArgs([
      "--preset",
      "gpt-tiny",
      "--gradient-checkpointing",
      "true",
      "--early-stop-patience",
      "4",
      "--early-stop-min-delta",
      "0.05",
      "--max-steps",
      "1000",
      "--log-interval",
      "20",
      "--throughput-window",
      "7",
      "--min-throughput-ratio",
      "0.8",
    ]);

    const maxStepsIndex = args.indexOf("--max-steps");
    const gradientCheckpointingIndex = args.indexOf("--gradient-checkpointing");
    const logIntervalIndex = args.indexOf("--log-interval");
    const throughputWindowIndex = args.indexOf("--throughput-window");
    const minRatioIndex = args.indexOf("--min-throughput-ratio");
    const earlyStopPatienceIndex = args.indexOf("--early-stop-patience");
    const earlyStopMinDeltaIndex = args.indexOf("--early-stop-min-delta");

    expect(maxStepsIndex).toBeGreaterThan(-1);
    expect(args[maxStepsIndex + 1]).toBe("1000");
    expect(gradientCheckpointingIndex).toBeGreaterThan(-1);
    expect(args[gradientCheckpointingIndex + 1]).toBe("true");
    expect(logIntervalIndex).toBeGreaterThan(-1);
    expect(args[logIntervalIndex + 1]).toBe("20");
    expect(earlyStopPatienceIndex).toBeGreaterThan(-1);
    expect(args[earlyStopPatienceIndex + 1]).toBe("4");
    expect(earlyStopMinDeltaIndex).toBeGreaterThan(-1);
    expect(args[earlyStopMinDeltaIndex + 1]).toBe("0.05");
    expect(throughputWindowIndex).toBeGreaterThan(-1);
    expect(args[throughputWindowIndex + 1]).toBe("7");
    expect(minRatioIndex).toBeGreaterThan(-1);
    expect(args[minRatioIndex + 1]).toBe("0.8");
  });

  test("buildAcceptanceArgs rejects unknown flags", () => {
    expect(() => buildAcceptanceArgs(["--preset", "gpt-small", "--mystery"])).toThrow(
      "unknown flag",
    );
  });

  test("main runs the soak wrapper and returns the child status", () => {
    const directory = mkdtempSync(join(tmpdir(), "nanogpt-soak-main-"));
    const dataPath = join(directory, "tiny.txt");
    writeFileSync(dataPath, "abcdefghijklmnopqrstuvwxyz ".repeat(120), "utf-8");

    try {
      const status = soakMain([
        "--preset",
        "gpt-tiny",
        "--name",
        `soak-main-${Date.now()}`,
        "--max-steps",
        "2",
        "--poll-seconds",
        "1",
        "--throughput-window",
        "1",
        "--min-throughput-ratio",
        "0.01",
        "--max-slope-mb-per-event",
        "1024",
        "--log-interval",
        "1",
        "--eval-interval",
        "1",
        "--eval-steps",
        "1",
        "--snapshot-interval",
        "1",
        "--resume-interval",
        "1",
        "--warmup-steps",
        "1",
        "--data",
        dataPath,
      ]);

      expect(status).toBe(0);
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  }, 60_000);
});
