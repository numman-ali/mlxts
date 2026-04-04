import { describe, expect, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join, resolve } from "path";
import {
  appendEvent,
  checkpointsDir,
  clearRunControl,
  ensureRunDir,
  eventsPath,
  packageRootFromRunDir,
  pidPath,
  readLatestCheckpoint,
  readRunControl,
  readRunSpec,
  readRunStatus,
  repoRootFromPackageRoot,
  runControlPath,
  runDir,
  runSpecPath,
  runStatusPath,
  runsRoot,
  stderrPath,
  writePid,
  writeRunControl,
  writeRunSpec,
  writeRunStatus,
} from "./files";

function tempRunDirectory(name: string): string {
  return mkdtempSync(join(tmpdir(), `${name}-`));
}

function withTempRunDirectory(
  name: string,
  fn: (directory: string) => Promise<void> | void,
): Promise<void> {
  const directory = tempRunDirectory(name);
  return Promise.resolve()
    .then(() => fn(directory))
    .finally(() => {
      rmSync(directory, { recursive: true, force: true });
    });
}

describe("run files", () => {
  test("path helpers and round-trip readers/writers behave canonically", async () => {
    await withTempRunDirectory("nanogpt-runs-root", async (repoRoot) => {
      const directory = runDir(repoRoot, "demo-run");
      const packageRoot = join(repoRoot, "packages", "nanogpt");

      ensureRunDir(directory);
      expect(existsSync(checkpointsDir(directory))).toBe(true);
      expect(runsRoot(repoRoot)).toBe(join(repoRoot, ".nanogpt-runs"));
      expect(runSpecPath(directory)).toBe(join(directory, "run.json"));
      expect(runStatusPath(directory)).toBe(join(directory, "status.json"));
      expect(runControlPath(directory)).toBe(join(directory, "control.json"));
      expect(eventsPath(directory)).toBe(join(directory, "events.jsonl"));
      expect(stderrPath(directory)).toBe(join(directory, "stderr.log"));
      expect(pidPath(directory)).toBe(join(directory, "pid"));
      expect(packageRootFromRunDir(directory)).toBe(
        resolve(directory, "..", "..", "packages", "nanogpt"),
      );
      expect(repoRootFromPackageRoot(packageRoot)).toBe(resolve(packageRoot, "..", ".."));

      writeRunSpec(directory, {
        runId: "demo-run",
        createdAt: "2026-04-04T00:00:00.000Z",
        repoRoot,
        packageRoot,
        checkpointDir: checkpointsDir(directory),
        stallTimeoutSeconds: 600,
        trainerArgs: ["--preset", "gpt-tiny"],
        resumedFrom: "previous-run",
      });
      writeRunStatus(directory, {
        runId: "demo-run",
        state: "running",
        startedAt: "2026-04-04T00:00:00.000Z",
        updatedAt: "2026-04-04T00:00:01.000Z",
        supervisorHeartbeatAt: "2026-04-04T00:00:01.000Z",
        trainerHeartbeatAt: "2026-04-04T00:00:01.000Z",
        lastProgressAt: "2026-04-04T00:00:01.000Z",
        stallTimeoutSeconds: 600,
        supervisorPid: 100,
        trainerPid: 200,
        preset: "gpt-tiny",
        config: {
          nLayer: 6,
          nHead: 6,
          nEmbd: 384,
          blockSize: 256,
          dropout: 0.2,
          gradientCheckpointing: false,
          vocabSize: 65,
        },
        parameterCount: 10_770_816,
        step: 7,
        maxSteps: 500,
        batchSize: 4,
        gradAccumSteps: 1,
        warmupSteps: 100,
        lastStepLoss: 2.3,
        lastTrainLoss: 2.2,
        lastValLoss: 2.1,
        lastTokensPerSec: 1234,
        latestCheckpoint: join(checkpointsDir(directory), "gpt-tiny-snapshot-step-7"),
        latestSnapshotCheckpoint: join(checkpointsDir(directory), "gpt-tiny-snapshot-step-7"),
        latestResumeCheckpoint: join(checkpointsDir(directory), "gpt-tiny-resume-step-7"),
        latestCheckpointKind: "resume",
        activeMemoryBytes: 1024,
        cacheMemoryBytes: 2048,
        peakMemoryBytes: 4096,
        memoryLimitBytes: 8192,
        exitCode: null,
        signal: null,
        resumeFrom: "previous-run",
        controlCommand: "stop",
        controlRequestedAt: "2026-04-04T00:00:02.000Z",
        stallReason: "none",
      });
      writeRunControl(directory, {
        command: "cancel",
        requestedAt: "2026-04-04T00:00:03.000Z",
      });
      appendEvent(directory, { type: "step", step: 7 });
      writePid(directory, 12345);

      expect(readRunSpec(directory)).toEqual({
        runId: "demo-run",
        createdAt: "2026-04-04T00:00:00.000Z",
        repoRoot,
        packageRoot,
        checkpointDir: checkpointsDir(directory),
        stallTimeoutSeconds: 600,
        trainerArgs: ["--preset", "gpt-tiny"],
        resumedFrom: "previous-run",
      });
      expect(readRunStatus(directory)).toEqual({
        runId: "demo-run",
        state: "running",
        startedAt: "2026-04-04T00:00:00.000Z",
        updatedAt: "2026-04-04T00:00:01.000Z",
        supervisorHeartbeatAt: "2026-04-04T00:00:01.000Z",
        trainerHeartbeatAt: "2026-04-04T00:00:01.000Z",
        lastProgressAt: "2026-04-04T00:00:01.000Z",
        stallTimeoutSeconds: 600,
        supervisorPid: 100,
        trainerPid: 200,
        preset: "gpt-tiny",
        config: {
          nLayer: 6,
          nHead: 6,
          nEmbd: 384,
          blockSize: 256,
          dropout: 0.2,
          gradientCheckpointing: false,
          vocabSize: 65,
        },
        parameterCount: 10_770_816,
        step: 7,
        maxSteps: 500,
        batchSize: 4,
        gradAccumSteps: 1,
        warmupSteps: 100,
        lastStepLoss: 2.3,
        lastTrainLoss: 2.2,
        lastValLoss: 2.1,
        lastTokensPerSec: 1234,
        latestCheckpoint: join(checkpointsDir(directory), "gpt-tiny-snapshot-step-7"),
        latestSnapshotCheckpoint: join(checkpointsDir(directory), "gpt-tiny-snapshot-step-7"),
        latestResumeCheckpoint: join(checkpointsDir(directory), "gpt-tiny-resume-step-7"),
        latestCheckpointKind: "resume",
        activeMemoryBytes: 1024,
        cacheMemoryBytes: 2048,
        peakMemoryBytes: 4096,
        memoryLimitBytes: 8192,
        exitCode: null,
        signal: null,
        resumeFrom: "previous-run",
        controlCommand: "stop",
        controlRequestedAt: "2026-04-04T00:00:02.000Z",
        stallReason: "none",
      });
      expect(readRunControl(directory)).toEqual({
        command: "cancel",
        requestedAt: "2026-04-04T00:00:03.000Z",
      });
      expect(await Bun.file(eventsPath(directory)).text()).toContain('"step":7');
      expect(await Bun.file(pidPath(directory)).text()).toBe("12345\n");
      expect(readdirSync(directory).some((entry) => entry.endsWith(".tmp"))).toBe(false);
    });
  });

  test("readRunControl returns undefined when no control file exists and clearRunControl removes it", () => {
    return withTempRunDirectory("nanogpt-run-control", (directory) => {
      ensureRunDir(directory);
      expect(readRunControl(directory)).toBeUndefined();

      writeRunControl(directory, {
        command: "stop",
        requestedAt: "2026-04-04T00:00:00.000Z",
      });
      expect(readRunControl(directory)?.command).toBe("stop");
      clearRunControl(directory);
      expect(readRunControl(directory)).toBeUndefined();
    });
  });

  test("readLatestCheckpoint returns the latest checkpoint directory and handles missing roots", () => {
    return withTempRunDirectory("nanogpt-latest-checkpoint", (directory) => {
      expect(readLatestCheckpoint(directory)).toBeUndefined();

      ensureRunDir(directory);
      expect(readLatestCheckpoint(directory)).toBeUndefined();

      mkdirSync(join(checkpointsDir(directory), "checkpoint-a"));
      mkdirSync(join(checkpointsDir(directory), "checkpoint-z"));
      writeFileSync(join(checkpointsDir(directory), "note.txt"), "ignore me", "utf-8");

      expect(readLatestCheckpoint(directory)).toBe(join(checkpointsDir(directory), "checkpoint-z"));
    });
  });

  test("invalid status and control payloads throw clear errors", () => {
    return withTempRunDirectory("nanogpt-run-invalid", (directory) => {
      ensureRunDir(directory);

      writeFileSync(
        runStatusPath(directory),
        JSON.stringify({
          runId: "broken",
          state: "mystery",
          startedAt: "2026-04-04T00:00:00.000Z",
          updatedAt: "2026-04-04T00:00:00.000Z",
          supervisorHeartbeatAt: "2026-04-04T00:00:00.000Z",
        }),
        "utf-8",
      );
      expect(() => readRunStatus(directory)).toThrow("known run state");

      writeFileSync(
        runStatusPath(directory),
        JSON.stringify({
          runId: "broken",
          state: "running",
          startedAt: "2026-04-04T00:00:00.000Z",
          updatedAt: "2026-04-04T00:00:00.000Z",
          supervisorHeartbeatAt: "2026-04-04T00:00:00.000Z",
          controlCommand: "pause",
        }),
        "utf-8",
      );
      expect(() => readRunStatus(directory)).toThrow("expected stop/cancel");

      writeFileSync(
        runControlPath(directory),
        JSON.stringify({
          command: "pause",
          requestedAt: "2026-04-04T00:00:00.000Z",
        }),
        "utf-8",
      );
      expect(() => readRunControl(directory)).toThrow("expected stop/cancel");

      writeFileSync(runSpecPath(directory), JSON.stringify(["nope"]), "utf-8");
      expect(() => readRunSpec(directory)).toThrow("expected a JSON object");

      writeFileSync(
        runSpecPath(directory),
        JSON.stringify({
          runId: "broken",
          createdAt: "2026-04-04T00:00:00.000Z",
          repoRoot: "/repo",
          packageRoot: "/repo/packages/nanogpt",
          checkpointDir: checkpointsDir(directory),
          trainerArgs: "not-an-array",
        }),
        "utf-8",
      );
      expect(() => readRunSpec(directory)).toThrow("expected a string array");

      writeFileSync(
        runControlPath(directory),
        JSON.stringify({
          command: "stop",
          requestedAt: 123,
        }),
        "utf-8",
      );
      expect(() => readRunControl(directory)).toThrow("expected a string");
    });
  });
});
