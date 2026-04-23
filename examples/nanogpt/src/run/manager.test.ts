import { afterEach, describe, expect, test } from "bun:test";
import { spawnSync } from "child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join, resolve } from "path";
import {
  checkpointsDir,
  ensureRunDir,
  readRunStatus,
  repoRootFromPackageRoot,
  runDir,
  writeRunSpec,
  writeRunStatus,
} from "./files";

const CREATED_RUNS = new Set<string>();

function packageRoot(): string {
  return resolve(import.meta.dir, "../..");
}

function repoRoot(): string {
  return repoRootFromPackageRoot(packageRoot());
}

function runManager(args: string[]) {
  return spawnSync("bun", ["run", "manager", ...args], {
    cwd: packageRoot(),
    encoding: "utf-8",
  });
}

function parseJsonLines(content: string): Record<string, unknown>[] {
  const lines = content.split("\n").filter((line) => line.trim().length > 0);
  const parsed: Record<string, unknown>[] = [];
  for (const line of lines) {
    const value: unknown = JSON.parse(line);
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
      continue;
    }
    parsed.push(Object.fromEntries(Object.entries(value)));
  }
  return parsed;
}

async function waitForStatus(
  runId: string,
  predicate: (status: ReturnType<typeof readRunStatus>) => boolean,
  timeoutMs = 60_000,
): Promise<ReturnType<typeof readRunStatus>> {
  const directory = runDir(repoRoot(), runId);
  const deadline = Date.now() + timeoutMs;

  while (Date.now() < deadline) {
    if (existsSync(join(directory, "status.json"))) {
      const status = readRunStatus(directory);
      if (predicate(status)) {
        return status;
      }
    }
    await Bun.sleep(500);
  }

  throw new Error(`Timed out waiting for run ${runId}`);
}

function cleanupRun(runId: string): void {
  const directory = runDir(repoRoot(), runId);
  if (!existsSync(directory)) {
    return;
  }

  try {
    const status = readRunStatus(directory);
    for (const pid of [status.trainerPid, status.supervisorPid]) {
      if (pid === undefined) {
        continue;
      }
      try {
        process.kill(pid, "SIGKILL");
      } catch {
        // Process already exited.
      }
    }
  } catch {
    // Ignore status read failures during cleanup.
  }

  rmSync(directory, { recursive: true, force: true });
}

afterEach(() => {
  for (const runId of CREATED_RUNS) {
    cleanupRun(runId);
  }
  CREATED_RUNS.clear();
});

describe("run manager", () => {
  test("supervised runs can stop and resume from resume checkpoints", async () => {
    const directory = mkdtempSync(join(tmpdir(), "nanogpt-run-manager-"));
    const dataPath = join(directory, "train.txt");
    writeFileSync(dataPath, "abcdefghijklmnopqrstuvwxyz ".repeat(120), "utf-8");

    const runId = `manager-stop-${Date.now()}`;
    CREATED_RUNS.add(runId);

    const startResult = runManager([
      "start",
      "--name",
      runId,
      "--preset",
      "gpt-tiny",
      "--data",
      dataPath,
      "--max-steps",
      "200",
      "--batch-size",
      "1",
      "--grad-accum",
      "1",
      "--eval-interval",
      "1",
      "--eval-steps",
      "1",
      "--log-interval",
      "1",
      "--snapshot-interval",
      "1",
      "--resume-interval",
      "1",
    ]);
    expect(startResult.status).toBe(0);

    await waitForStatus(runId, (status) => status.step !== undefined && status.step >= 1);
    const stopResult = runManager(["stop", "--name", runId]);
    expect(stopResult.status).toBe(0);

    const stopped = await waitForStatus(runId, (status) => status.state === "stopped");
    expect(stopped.latestResumeCheckpoint).toBeDefined();

    const resumedRunId = `${runId}-resume`;
    CREATED_RUNS.add(resumedRunId);
    const resumeResult = runManager([
      "resume",
      "--from",
      runId,
      "--name",
      resumedRunId,
      "--max-steps",
      "12",
    ]);
    expect(resumeResult.status).toBe(0);

    const completed = await waitForStatus(
      resumedRunId,
      (status) => status.state === "completed" && status.step === 12,
    );
    expect(completed.latestResumeCheckpoint).toBeDefined();

    const eventLog = readFileSync(join(runDir(repoRoot(), runId), "events.jsonl"), "utf-8");
    const seqs = parseJsonLines(eventLog)
      .map((line) => line.seq)
      .filter((seq): seq is number => typeof seq === "number");
    expect(seqs.length).toBeGreaterThan(0);
    expect(seqs[0]).toBe(1);
    for (let index = 1; index < seqs.length; index++) {
      expect(seqs[index]).toBeGreaterThan(seqs[index - 1] ?? 0);
    }
  }, 60_000);

  test("cancel requests terminate a run without forcing a final resume checkpoint", async () => {
    const directory = mkdtempSync(join(tmpdir(), "nanogpt-run-cancel-"));
    const dataPath = join(directory, "train.txt");
    writeFileSync(dataPath, "abcdefghijklmnopqrstuvwxyz ".repeat(120), "utf-8");

    const runId = `manager-cancel-${Date.now()}`;
    CREATED_RUNS.add(runId);

    const startResult = runManager([
      "start",
      "--name",
      runId,
      "--preset",
      "gpt-tiny",
      "--data",
      dataPath,
      "--max-steps",
      "1000",
      "--batch-size",
      "1",
      "--grad-accum",
      "8",
      "--eval-interval",
      "10",
      "--eval-steps",
      "1",
      "--log-interval",
      "1",
      "--snapshot-interval",
      "0",
      "--resume-interval",
      "0",
    ]);
    expect(startResult.status).toBe(0);

    await waitForStatus(runId, (status) => status.step !== undefined && status.step >= 1);
    const cancelResult = runManager(["cancel", "--name", runId]);
    expect(cancelResult.status).toBe(0);
    expect(cancelResult.stdout).toContain("latest resume checkpoint");

    const cancelled = await waitForStatus(runId, (status) => status.state === "cancelled");
    expect(cancelled.latestResumeCheckpoint).toBeUndefined();

    const statusResult = runManager(["status", "--name", runId, "--json"]);
    expect(statusResult.status).toBe(0);
    const statusPayload: unknown = JSON.parse(statusResult.stdout.trim());
    expect(typeof statusPayload).toBe("object");
    expect(statusPayload).not.toBeNull();
    if (typeof statusPayload !== "object" || statusPayload === null) {
      throw new Error("expected manager status JSON object");
    }
    expect("state" in statusPayload ? statusPayload.state : undefined).toBe("cancelled");
    expect("exitCode" in statusPayload).toBe(true);
    expect("signal" in statusPayload).toBe(true);
  }, 60_000);

  test("status reports best-checkpoint and early-stop details for auto-stopped runs", async () => {
    const directory = mkdtempSync(join(tmpdir(), "nanogpt-run-early-stop-"));
    const dataPath = join(directory, "train.txt");
    writeFileSync(dataPath, "abcdefghijklmnopqrstuvwxyz ".repeat(120), "utf-8");

    const runId = `manager-early-stop-${Date.now()}`;
    CREATED_RUNS.add(runId);

    const startResult = runManager([
      "start",
      "--name",
      runId,
      "--preset",
      "gpt-tiny",
      "--data",
      dataPath,
      "--max-steps",
      "10",
      "--batch-size",
      "1",
      "--grad-accum",
      "1",
      "--eval-interval",
      "1",
      "--eval-steps",
      "1",
      "--log-interval",
      "1",
      "--snapshot-interval",
      "0",
      "--resume-interval",
      "0",
      "--early-stop-patience",
      "1",
      "--early-stop-min-delta",
      "10",
    ]);
    expect(startResult.status).toBe(0);

    const stopped = await waitForStatus(
      runId,
      (status) => status.state === "stopped" && status.earlyStopReason !== undefined,
    );
    expect(stopped.bestCheckpoint).toBeDefined();
    expect(stopped.bestCheckpointStep).toBe(1);
    expect(stopped.bestValLoss).toBeDefined();

    const statusResult = runManager(["status", "--name", runId, "--json"]);
    expect(statusResult.status).toBe(0);
    const payload: unknown = JSON.parse(statusResult.stdout.trim());
    if (typeof payload !== "object" || payload === null || Array.isArray(payload)) {
      throw new Error("expected manager status JSON object");
    }
    expect("bestCheckpoint" in payload ? payload.bestCheckpoint : undefined).toBeDefined();
    expect("bestCheckpointStep" in payload ? payload.bestCheckpointStep : undefined).toBe(1);
    expect("earlyStopReason" in payload ? payload.earlyStopReason : undefined).toBeDefined();
  }, 60_000);

  test("status reports operator health for dead supervisor runs", () => {
    const runId = `manager-health-${Date.now()}`;
    CREATED_RUNS.add(runId);
    const directory = runDir(repoRoot(), runId);
    ensureRunDir(directory);
    const trainer = Bun.spawn(["sleep", "30"], {
      stdin: "ignore",
      stdout: "ignore",
      stderr: "ignore",
    });
    trainer.unref();
    writeRunSpec(directory, {
      runId,
      createdAt: new Date().toISOString(),
      repoRoot: repoRoot(),
      packageRoot: packageRoot(),
      checkpointDir: checkpointsDir(directory),
      stallTimeoutSeconds: 600,
      trainerArgs: ["--preset", "gpt-tiny"],
    });
    writeRunStatus(directory, {
      runId,
      state: "running",
      startedAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      supervisorHeartbeatAt: new Date().toISOString(),
      trainerHeartbeatAt: new Date().toISOString(),
      lastProgressAt: new Date().toISOString(),
      supervisorPid: 9_999_999,
      trainerPid: trainer.pid,
    });

    const result = runManager(["status", "--name", runId, "--json"]);
    expect(result.status).toBe(0);
    const payload: unknown = JSON.parse(result.stdout.trim());
    expect(typeof payload).toBe("object");
    if (typeof payload !== "object" || payload === null) {
      throw new Error("expected status payload object");
    }
    expect("operatorHealth" in payload ? payload.operatorHealth : undefined).toBe(
      "dead-supervisor",
    );
    expect("supervisorAlive" in payload ? payload.supervisorAlive : undefined).toBe(false);
    expect("trainerAlive" in payload ? payload.trainerAlive : undefined).toBe(true);
    expect("stallTimeoutSeconds" in payload ? payload.stallTimeoutSeconds : undefined).toBe(600);
  });

  test("manager rejects unknown flags", () => {
    const runId = `manager-unknown-${Date.now()}`;
    CREATED_RUNS.add(runId);
    const directory = runDir(repoRoot(), runId);
    ensureRunDir(directory);
    writeRunSpec(directory, {
      runId,
      createdAt: new Date().toISOString(),
      repoRoot: repoRoot(),
      packageRoot: packageRoot(),
      checkpointDir: checkpointsDir(directory),
      stallTimeoutSeconds: 600,
      trainerArgs: ["--preset", "gpt-tiny"],
    });
    writeRunStatus(directory, {
      runId,
      state: "running",
      startedAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      supervisorHeartbeatAt: new Date().toISOString(),
      trainerHeartbeatAt: new Date().toISOString(),
      lastProgressAt: new Date().toISOString(),
      supervisorPid: 9_999_999,
      trainerPid: 9_999_998,
    });

    const result = runManager(["status", "--name", runId, "--mystery"]);
    expect(result.status).toBe(1);
    expect(result.stderr).toContain("unknown flag");
  });

  test("start accepts gradient checkpointing overrides and surfaces them in status", async () => {
    const directory = mkdtempSync(join(tmpdir(), "nanogpt-run-gc-"));
    const dataPath = join(directory, "train.txt");
    writeFileSync(dataPath, "abcdefghijklmnopqrstuvwxyz ".repeat(120), "utf-8");

    const runId = `manager-gc-${Date.now()}`;
    CREATED_RUNS.add(runId);

    const startResult = runManager([
      "start",
      "--name",
      runId,
      "--preset",
      "gpt-tiny",
      "--gradient-checkpointing",
      "true",
      "--data",
      dataPath,
      "--max-steps",
      "2",
      "--batch-size",
      "1",
      "--grad-accum",
      "1",
      "--eval-interval",
      "1",
      "--eval-steps",
      "1",
      "--log-interval",
      "1",
      "--snapshot-interval",
      "1",
      "--resume-interval",
      "1",
    ]);
    expect(startResult.status).toBe(0);

    const completed = await waitForStatus(
      runId,
      (status) => status.state === "completed" && status.step === 2,
    );
    expect(completed.config?.gradientCheckpointing).toBe(true);
  }, 60_000);
});
