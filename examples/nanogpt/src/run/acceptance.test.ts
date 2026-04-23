import { afterEach, describe, expect, test } from "bun:test";
import { CharTokenizer } from "@mlxts/tokenizers";
import { spawnSync } from "child_process";
import { existsSync, mkdtempSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join, resolve } from "path";
import {
  main as acceptanceMain,
  assertCompletedStatus,
  assertSoakStabilityForEvents,
  buildManagerArgs,
  checkpointPathFromStatus,
  finalLossFromStatus,
  getNumberFlag,
  parseArgs,
  readMode,
  readPresetName,
  readRunOptions,
  readStepEvents,
  type StepEventRecord,
  samplePrompt,
  waitForTerminalState,
} from "./acceptance";
import { ensureRunDir, eventsPath, repoRootFromPackageRoot, runDir, writeRunStatus } from "./files";

const CREATED_RUNS = new Set<string>();

function stepEvent(
  step: number,
  tokensPerSec: number,
  activeMemoryBytes?: number,
): StepEventRecord {
  return {
    type: "step",
    step,
    tokensPerSec,
    activeMemoryBytes,
  };
}

function packageRoot(): string {
  return resolve(import.meta.dir, "../..");
}

function repoRoot(): string {
  return repoRootFromPackageRoot(packageRoot());
}

function withRunDirectory(runId: string): string {
  const directory = runDir(repoRoot(), runId);
  ensureRunDir(directory);
  CREATED_RUNS.add(runId);
  return directory;
}

afterEach(() => {
  for (const runId of CREATED_RUNS) {
    rmSync(runDir(repoRoot(), runId), { recursive: true, force: true });
  }
  CREATED_RUNS.clear();
});

describe("acceptance soak stability", () => {
  test("parseArgs captures valued and boolean flags", () => {
    const flags = parseArgs([
      "bun",
      "acceptance.ts",
      "--preset",
      "gpt-small",
      "--json",
      "--max-steps",
      "250",
    ]);

    expect(flags.get("preset")).toBe("gpt-small");
    expect(flags.get("json")).toBe("true");
    expect(flags.get("max-steps")).toBe("250");
  });

  test("preset and mode readers validate supported values", () => {
    expect(readPresetName(new Map())).toBe("gpt-tiny");
    expect(readMode(new Map())).toBe("acceptance");
    expect(() => readPresetName(new Map([["preset", "mystery"]]))).toThrow("Unknown preset");
    expect(() => readMode(new Map([["mode", "mystery"]]))).toThrow("Unknown mode");
  });

  test("getNumberFlag rejects non-numeric values", () => {
    expect(() => getNumberFlag(new Map([["max-steps", "oops"]]), "max-steps", 10)).toThrow(
      "finite number",
    );
  });

  test("readRunOptions rejects unknown flags", () => {
    expect(() =>
      readRunOptions(
        new Map([
          ["preset", "gpt-small"],
          ["mystery", "true"],
        ]),
      ),
    ).toThrow("unknown flag");
  });

  test("buildManagerArgs applies defaults and resolves runtime flags", () => {
    const dataDirectory = mkdtempSync(join(tmpdir(), "acceptance-args-"));
    const dataPath = join(dataDirectory, "train.txt");
    writeFileSync(dataPath, "hello", "utf-8");

    const args = buildManagerArgs(
      "gpt-small",
      "run-1",
      {
        maxSteps: 5000,
        batchSize: 1,
        gradAccumSteps: 8,
        evalInterval: 250,
        evalSteps: 20,
        learningRate: 3e-4,
        weightDecay: 0.1,
        maxGradNorm: 1,
        warmupSteps: 250,
        minLearningRate: 3e-5,
        logInterval: 25,
        lossTarget: 1.5,
        snapshotInterval: 250,
        resumeInterval: 1000,
        stallTimeoutSeconds: 600,
      },
      new Map([
        ["data", dataPath],
        ["memory-limit-mb", "2048"],
        ["gradient-checkpointing", "false"],
        ["early-stop-patience", "6"],
        ["early-stop-min-delta", "0.05"],
      ]),
    );

    try {
      expect(args).toContain("--data");
      expect(args).toContain(resolve(process.cwd(), dataPath));
      expect(args).toContain("--memory-limit-mb");
      expect(args).toContain("2048");
      expect(args).toContain("--gradient-checkpointing");
      expect(args).toContain("false");
      expect(args).toContain("--early-stop-patience");
      expect(args).toContain("6");
      expect(args).toContain("--early-stop-min-delta");
      expect(args).toContain("0.05");
      expect(args).toContain("--stall-timeout-sec");
      expect(args).toContain("600");
      expect(args).toContain("--max-steps");
      expect(args).toContain("5000");
    } finally {
      rmSync(dataDirectory, { recursive: true, force: true });
    }
  });

  test("readRunOptions resolves canonical defaults", () => {
    const options = readRunOptions(
      new Map([
        ["mode", "soak"],
        ["preset", "gpt-small"],
      ]),
    );
    expect(options.mode).toBe("soak");
    expect(options.presetName).toBe("gpt-small");
    expect(options.parameterCount).toBeGreaterThan(0);
    expect(options.maxSlopeMbPerEvent).toBe(8);
    expect(options.stallTimeoutSeconds).toBe(600);
    expect(options.lossTarget).toBe(1.8);
    expect(options.args).toContain("--early-stop-patience");
    expect(options.args).toContain("none");
  });

  test("acceptance defaults enable early stopping and prefer best-checkpoint metrics", () => {
    const options = readRunOptions(new Map([["preset", "gpt-tiny"]]));
    expect(options.args).toContain("--early-stop-patience");
    expect(options.args).toContain("8");
    expect(options.args).toContain("--early-stop-min-delta");
    expect(options.args).toContain("0.02");

    expect(
      finalLossFromStatus({
        runId: "run-1",
        state: "stopped",
        startedAt: "now",
        updatedAt: "now",
        supervisorHeartbeatAt: "now",
        bestValLoss: 1.23,
        earlyStopReason: "plateaued",
      }),
    ).toBe(1.23);

    expect(
      checkpointPathFromStatus({
        runId: "run-1",
        state: "stopped",
        startedAt: "now",
        updatedAt: "now",
        supervisorHeartbeatAt: "now",
        bestCheckpoint: "/tmp/best",
        earlyStopReason: "plateaued",
      }),
    ).toBe("/tmp/best");
  });

  test("samplePrompt prefers newline and otherwise falls back to the first vocab item", () => {
    expect(samplePrompt(CharTokenizer.fromText("abc"))).toBe("a");
    expect(samplePrompt(CharTokenizer.fromText("a\nb"))).toBe("\n");
  });

  test("readStepEvents parses canonical step events from the run directory", () => {
    const runId = `acceptance-events-${Date.now()}`;
    const directory = withRunDirectory(runId);
    writeFileSync(
      eventsPath(directory),
      `${JSON.stringify(stepEvent(1, 1000, 100))}\n${JSON.stringify({ type: "eval", step: 1 })}\n`,
      "utf-8",
    );

    expect(readStepEvents(runId)).toEqual([stepEvent(1, 1000, 100)]);
  });

  test("readStepEvents returns an empty array for an empty log", () => {
    const runId = `acceptance-events-empty-${Date.now()}`;
    const directory = withRunDirectory(runId);
    writeFileSync(eventsPath(directory), "", "utf-8");

    expect(readStepEvents(runId)).toEqual([]);
  });

  test("readStepEvents ignores a partial trailing line", () => {
    const runId = `acceptance-events-partial-${Date.now()}`;
    const directory = withRunDirectory(runId);
    writeFileSync(
      eventsPath(directory),
      `${JSON.stringify(stepEvent(1, 1000, 100))}\n{"type":"step"`,
      "utf-8",
    );

    expect(readStepEvents(runId)).toEqual([stepEvent(1, 1000, 100)]);
  });

  test("waitForTerminalState returns immediately for a completed run", () => {
    const runId = `acceptance-status-${Date.now()}`;
    const directory = withRunDirectory(runId);
    writeRunStatus(directory, {
      runId,
      state: "completed",
      startedAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      supervisorHeartbeatAt: new Date().toISOString(),
    });

    expect(waitForTerminalState(runId, 1).state).toBe("completed");
  });

  test("waitForTerminalState returns stalled runs and assertCompletedStatus explains why", () => {
    const runId = `acceptance-stalled-${Date.now()}`;
    const directory = withRunDirectory(runId);
    writeRunStatus(directory, {
      runId,
      state: "stalled",
      startedAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      supervisorHeartbeatAt: new Date().toISOString(),
      stallReason: "No progress event for 180s",
    });

    const status = waitForTerminalState(runId, 1);
    expect(status.state).toBe("stalled");
    expect(() => assertCompletedStatus(runId, status)).toThrow(
      "stalled: No progress event for 180s",
    );
  });

  test("waitForTerminalState fails clearly when the operator becomes unhealthy", () => {
    const runId = `acceptance-dead-${Date.now()}`;
    const directory = withRunDirectory(runId);
    writeRunStatus(directory, {
      runId,
      state: "running",
      startedAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      supervisorHeartbeatAt: new Date().toISOString(),
      trainerHeartbeatAt: new Date().toISOString(),
      lastProgressAt: new Date().toISOString(),
      supervisorPid: 9_999_999,
      trainerPid: process.pid,
    });

    expect(() => waitForTerminalState(runId, 1)).toThrow("became unhealthy");
  });

  test("status helpers enforce completion, final loss, and checkpoint presence", () => {
    expect(() =>
      assertCompletedStatus("run-1", {
        runId: "run-1",
        state: "failed",
        startedAt: "now",
        updatedAt: "now",
        supervisorHeartbeatAt: "now",
      }),
    ).toThrow("ended in state failed");

    expect(() =>
      assertCompletedStatus("run-plateau", {
        runId: "run-plateau",
        state: "stopped",
        startedAt: "now",
        updatedAt: "now",
        supervisorHeartbeatAt: "now",
        earlyStopReason: "validation loss plateaued",
      }),
    ).not.toThrow();

    expect(
      finalLossFromStatus({
        runId: "run-2",
        state: "completed",
        startedAt: "now",
        updatedAt: "now",
        supervisorHeartbeatAt: "now",
        lastValLoss: 1.25,
      }),
    ).toBe(1.25);
    expect(
      finalLossFromStatus({
        runId: "run-2b",
        state: "completed",
        startedAt: "now",
        updatedAt: "now",
        supervisorHeartbeatAt: "now",
        bestValLoss: 1.74,
        lastValLoss: 1.73,
      }),
    ).toBe(1.73);
    expect(() =>
      finalLossFromStatus({
        runId: "run-3",
        state: "completed",
        startedAt: "now",
        updatedAt: "now",
        supervisorHeartbeatAt: "now",
      }),
    ).toThrow("final recorded loss");

    expect(
      checkpointPathFromStatus({
        runId: "run-4",
        state: "completed",
        startedAt: "now",
        updatedAt: "now",
        supervisorHeartbeatAt: "now",
        latestResumeCheckpoint: "/tmp/checkpoint",
      }),
    ).toBe("/tmp/checkpoint");
    expect(() =>
      checkpointPathFromStatus({
        runId: "run-5",
        state: "completed",
        startedAt: "now",
        updatedAt: "now",
        supervisorHeartbeatAt: "now",
      }),
    ).toThrow("without a checkpoint");
  });

  test("accepts stable throughput and memory behavior", () => {
    const metrics = assertSoakStabilityForEvents(
      "stable-run",
      [
        stepEvent(1, 1000, 100 * 1024 * 1024),
        stepEvent(2, 980, 101 * 1024 * 1024),
        stepEvent(3, 970, 102 * 1024 * 1024),
        stepEvent(4, 960, 103 * 1024 * 1024),
      ],
      2,
      0.9,
      2,
    );

    expect(metrics.throughputRatio).toBeGreaterThanOrEqual(0.9);
    expect(metrics.slopeMbPerEvent).toBeLessThanOrEqual(2);
  });

  test("rejects severe throughput collapse", () => {
    expect(() =>
      assertSoakStabilityForEvents(
        "slow-run",
        [
          stepEvent(1, 1000, 100 * 1024 * 1024),
          stepEvent(2, 980, 101 * 1024 * 1024),
          stepEvent(3, 300, 102 * 1024 * 1024),
          stepEvent(4, 280, 103 * 1024 * 1024),
        ],
        2,
        0.8,
        2,
      ),
    ).toThrow("throughput ratio");
  });

  test("rejects steep active-memory slope", () => {
    expect(() =>
      assertSoakStabilityForEvents(
        "leaky-run",
        [
          stepEvent(1, 1000, 100 * 1024 * 1024),
          stepEvent(2, 980, 200 * 1024 * 1024),
          stepEvent(3, 970, 400 * 1024 * 1024),
          stepEvent(4, 960, 700 * 1024 * 1024),
        ],
        2,
        0.8,
        50,
      ),
    ).toThrow("active memory slope");
  });

  test("rejects soak runs with too few step or memory samples", () => {
    expect(() =>
      assertSoakStabilityForEvents("short-run", [stepEvent(1, 1000, 100 * 1024 * 1024)], 1, 0.8, 2),
    ).toThrow("need at least 2");

    expect(() =>
      assertSoakStabilityForEvents(
        "no-memory-run",
        [stepEvent(1, 1000), stepEvent(2, 980)],
        1,
        0.8,
        2,
      ),
    ).toThrow("enough memory-bearing step events");
  });

  test("soak mode completes a tiny supervised run", () => {
    const directory = mkdtempSync(join(tmpdir(), "nanogpt-acceptance-"));
    const dataPath = join(directory, "tiny.txt");
    const runId = `acceptance-soak-${Date.now()}`;
    const runDirectory = join(import.meta.dir, "..", "..", "..", "..", ".nanogpt-runs", runId);
    writeFileSync(dataPath, "abcdefghijklmnopqrstuvwxyz ".repeat(120), "utf-8");

    try {
      expect(existsSync(join(import.meta.dir, "soak.ts"))).toBe(true);
      const result = spawnSync(
        "bun",
        [
          "run",
          "src/run/acceptance.ts",
          "--mode",
          "soak",
          "--preset",
          "gpt-tiny",
          "--max-steps",
          "2",
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
          "--name",
          runId,
        ],
        {
          cwd: join(import.meta.dir, "..", ".."),
          encoding: "utf-8",
        },
      );

      expect(result.status).toBe(0);
      expect(result.stdout).toContain("mode=soak status=completed");
    } finally {
      rmSync(runDirectory, { recursive: true, force: true });
      rmSync(directory, { recursive: true, force: true });
    }
  }, 60_000);

  test("acceptance mode completes a tiny supervised run and emits a sample", async () => {
    const directory = mkdtempSync(join(tmpdir(), "nanogpt-acceptance-mode-"));
    const dataPath = join(directory, "tiny.txt");
    const runId = `acceptance-mode-${Date.now()}`;
    CREATED_RUNS.add(runId);
    writeFileSync(dataPath, "abcdefghijklmnopqrstuvwxyz ".repeat(120), "utf-8");

    try {
      await expect(
        acceptanceMain([
          "bun",
          "acceptance.ts",
          "--mode",
          "acceptance",
          "--preset",
          "gpt-tiny",
          "--name",
          runId,
          "--max-steps",
          "2",
          "--poll-seconds",
          "1",
          "--loss-target",
          "10",
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
        ]),
      ).resolves.toBeUndefined();
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  }, 60_000);
});
