import { describe, expect, test } from "bun:test";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join, resolve } from "path";
import { writeJsonFile } from "./files-json";
import type {
  RunState,
  RunStatus,
  SupervisedRunManagerCliOptions,
  SupervisedRunManagerRunOptions,
  SupervisedRunStatusOptions,
} from "./index";
import {
  activePid,
  appendEvent,
  appendSupervisorEvent,
  applyPendingControl,
  checkpointsDir,
  clearRunControl,
  createStatusPayload,
  deriveOperatorHealth,
  ensureRunDir,
  eventsPath,
  finalState,
  finishSupervisorRun,
  formatStatusPayload,
  generateRunId,
  getFlag,
  hasFlag,
  managerEvent,
  maybeEscalateTrainer,
  maybeMarkStalled,
  packageRootFromRunDir,
  parseArgs,
  pidPath,
  pipeTextStream,
  pumpTrainerStdout,
  readEvent,
  readLatestCheckpoint,
  readRunControl,
  readRunSpec,
  readRunStatus,
  repoRootFromPackageRoot,
  runControlPath,
  runDir,
  runSpecPath,
  runStatusPath,
  runSupervisedManagerCli,
  runSupervisedManagerCliCommand,
  runSupervisedSupervisor,
  runsRoot,
  stderrPath,
  stripFlag,
  trainerArgsFrom,
  updateStatusFromEvent,
  validateAllowedFlags,
  writeControl,
  writePid,
  writeRunControl,
  writeRunSpec,
  writeRunStatus,
} from "./index";
import { createStderrStream } from "./supervisor-streams";

function withTempDirectory(
  name: string,
  fn: (directory: string) => Promise<void> | void,
): Promise<void> {
  const directory = mkdtempSync(join(tmpdir(), `${name}-`));
  return Promise.resolve()
    .then(() => fn(directory))
    .finally(() => {
      rmSync(directory, { recursive: true, force: true });
    });
}

async function captureStdout(fn: () => Promise<void> | void): Promise<string> {
  const originalWrite = process.stdout.write;
  let output = "";
  process.stdout.write = ((chunk: string | Uint8Array) => {
    output += typeof chunk === "string" ? chunk : new TextDecoder().decode(chunk);
    return true;
  }) as typeof process.stdout.write;
  try {
    await fn();
  } finally {
    process.stdout.write = originalWrite;
  }
  return output;
}

function createRunOptions(repoRoot: string): SupervisedRunManagerRunOptions {
  return {
    repoRoot,
    packageRoot: repoRoot,
    runsDirectoryName: ".test-runs",
    supervisorCommand: () => ["bun", "-e", ""],
    statusCommand: (runId) => `status ${runId}`,
  };
}

function createStatusOptions(repoRoot: string): SupervisedRunStatusOptions {
  return {
    repoRoot,
    runsDirectoryName: ".test-runs",
  };
}

function createCliOptions(repoRoot: string): SupervisedRunManagerCliOptions {
  return {
    usage: "test manager usage\n",
    startFlagAllowlist: new Set(["name", "stall-timeout-sec", "max-steps", "help"]),
    resumeFlagAllowlist: new Set(["name", "from", "stall-timeout-sec", "max-steps", "help"]),
    statusFlagAllowlist: new Set(["name", "json", "help"]),
    watchFlagAllowlist: new Set(["name", "json", "interval", "help"]),
    controlFlagAllowlist: new Set(["name", "help"]),
    run: createRunOptions(repoRoot),
    status: createStatusOptions(repoRoot),
  };
}

function baseStatus(state: RunState = "running"): RunStatus {
  return {
    runId: "demo-run",
    state,
    startedAt: "2026-04-04T00:00:00.000Z",
    updatedAt: "2026-04-04T00:00:01.000Z",
    supervisorHeartbeatAt: "2026-04-04T00:00:01.000Z",
    trainerHeartbeatAt: "2026-04-04T00:00:01.000Z",
    lastProgressAt: "2026-04-04T00:00:01.000Z",
    supervisorPid: process.pid,
    trainerPid: process.pid,
  };
}

function textStream(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }
      controller.close();
    },
  });
}

describe("supervised run files", () => {
  test("path helpers and file readers preserve the run layout", async () => {
    await withTempDirectory("mlxts-supervised-run", (repoRoot) => {
      const directory = runDir(repoRoot, "demo-run", ".test-runs");
      const packageRoot = join(repoRoot, "examples", "trainer");

      ensureRunDir(directory);
      expect(existsSync(checkpointsDir(directory))).toBe(true);
      expect(runsRoot(repoRoot, ".test-runs")).toBe(join(repoRoot, ".test-runs"));
      expect(packageRootFromRunDir(directory, "examples/trainer")).toBe(
        resolve(directory, "..", "..", "examples", "trainer"),
      );
      expect(repoRootFromPackageRoot(packageRoot)).toBe(resolve(packageRoot, "..", ".."));
      expect(pidPath(directory)).toBe(join(directory, "pid"));
      expect(stderrPath(directory)).toBe(join(directory, "stderr.log"));

      writeRunSpec(directory, {
        runId: "demo-run",
        createdAt: "2026-04-04T00:00:00.000Z",
        repoRoot,
        packageRoot,
        checkpointDir: checkpointsDir(directory),
        stallTimeoutSeconds: 600,
        trainerArgs: ["--preset", "tiny"],
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
        supervisorPid: process.pid,
        trainerPid: process.pid,
        preset: "tiny",
        config: { gradientCheckpointing: true },
        step: 7,
        maxSteps: 10,
        latestResumeCheckpoint: join(checkpointsDir(directory), "resume-step-7"),
        earlyStopPatience: null,
        exitCode: null,
        signal: null,
      });
      writePid(directory, 12345);
      appendEvent(directory, { type: "step", step: 7 });

      expect(readRunSpec(directory).trainerArgs).toEqual(["--preset", "tiny"]);
      expect(readRunStatus(directory)).toMatchObject({
        runId: "demo-run",
        state: "running",
        preset: "tiny",
        config: { gradientCheckpointing: true },
        step: 7,
      });
      expect(readFileSync(join(directory, "pid"), "utf-8")).toBe("12345\n");
      expect(readFileSync(eventsPath(directory), "utf-8")).toContain('"step":7');
    });
  });

  test("control, checkpoint, and invalid payload readers keep operators honest", async () => {
    await withTempDirectory("mlxts-supervised-invalid", (directory) => {
      ensureRunDir(directory);
      expect(readRunControl(directory)).toBeUndefined();
      clearRunControl(directory);
      writeRunControl(directory, {
        command: "cancel",
        requestedAt: "2026-04-04T00:00:00.000Z",
      });
      expect(readRunControl(directory)).toEqual({
        command: "cancel",
        requestedAt: "2026-04-04T00:00:00.000Z",
      });
      clearRunControl(directory);
      expect(readRunControl(directory)).toBeUndefined();
      expect(readLatestCheckpoint(directory)).toBeUndefined();
      expect(readLatestCheckpoint(join(directory, "missing-run"))).toBeUndefined();
      mkdirSync(join(checkpointsDir(directory), "aaa"), { recursive: true });
      mkdirSync(join(checkpointsDir(directory), "checkpoint-step-2"), { recursive: true });
      mkdirSync(join(checkpointsDir(directory), "checkpoint-step-10"), { recursive: true });
      expect(readLatestCheckpoint(directory)).toBe(
        join(checkpointsDir(directory), "checkpoint-step-10"),
      );

      expect(() => writeJsonFile(join(directory, "missing", "payload.json"), {})).toThrow();

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
        runStatusPath(directory),
        JSON.stringify({
          runId: "broken",
          state: "running",
          startedAt: "2026-04-04T00:00:00.000Z",
          updatedAt: "2026-04-04T00:00:00.000Z",
          supervisorHeartbeatAt: "2026-04-04T00:00:00.000Z",
          earlyStopPatience: -1,
        }),
        "utf-8",
      );
      expect(() => readRunStatus(directory)).toThrow("non-negative integer");

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
          runId: 123,
          createdAt: "2026-04-04T00:00:00.000Z",
          repoRoot: directory,
          packageRoot: directory,
          checkpointDir: checkpointsDir(directory),
          trainerArgs: [],
        }),
        "utf-8",
      );
      expect(() => readRunSpec(directory)).toThrow("run.runId");

      writeFileSync(
        runSpecPath(directory),
        JSON.stringify({
          runId: "broken",
          createdAt: "2026-04-04T00:00:00.000Z",
          repoRoot: directory,
          packageRoot: directory,
          checkpointDir: checkpointsDir(directory),
          stallTimeoutSeconds: 0,
          trainerArgs: [],
        }),
        "utf-8",
      );
      expect(() => readRunSpec(directory)).toThrow("positive integer");

      writeFileSync(
        runSpecPath(directory),
        JSON.stringify({
          runId: "broken",
          createdAt: "2026-04-04T00:00:00.000Z",
          repoRoot: directory,
          packageRoot: directory,
          checkpointDir: checkpointsDir(directory),
          trainerArgs: "nope",
        }),
        "utf-8",
      );
      expect(() => readRunSpec(directory)).toThrow("string array");
    });
  });
});

describe("supervised run manager helpers", () => {
  test("manager arg helpers reject drift and absolutize detached path flags", () => {
    expect(parseArgs(["bun", "manager", "status", "--name", "demo"])).toEqual({
      command: "status",
      args: ["--name", "demo"],
    });
    expect(getFlag(["--name", "demo"], "name")).toBe("demo");
    expect(hasFlag(["--json"], "json")).toBe(true);
    expect(() => validateAllowedFlags(["--name", "demo"], new Set(["json"]), "status")).toThrow(
      "unknown flag",
    );
    expect(stripFlag(["--name", "demo", "--json"], "json", false)).toEqual(["--name", "demo"]);
    expect(trainerArgsFrom(["--name", "demo", "--data", "input.txt", "--json"])).toEqual([
      "--data",
      resolve(process.cwd(), "input.txt"),
    ]);
    expect(generateRunId(["--resume", "checkpoint"])).toMatch(/-resume$/);
    expect(generateRunId(["--preset", "tiny"], () => "unit")).toMatch(/-unit$/);
  });

  test("status and control helpers preserve operator-facing semantics", async () => {
    await withTempDirectory("mlxts-supervised-manager", async (repoRoot) => {
      const directory = runDir(repoRoot, "demo-run", ".test-runs");
      ensureRunDir(directory);
      writeRunSpec(directory, {
        runId: "demo-run",
        createdAt: "2026-04-04T00:00:00.000Z",
        repoRoot,
        packageRoot: repoRoot,
        checkpointDir: checkpointsDir(directory),
        stallTimeoutSeconds: 600,
        trainerArgs: ["--max-steps", "10"],
      });
      writeRunStatus(directory, {
        runId: "demo-run",
        state: "running",
        startedAt: "2026-04-04T00:00:00.000Z",
        updatedAt: "2026-04-04T00:00:01.000Z",
        supervisorHeartbeatAt: "2026-04-04T00:00:01.000Z",
        trainerHeartbeatAt: "2026-04-04T00:00:01.000Z",
        lastProgressAt: "2026-04-04T00:00:01.000Z",
        supervisorPid: process.pid,
        trainerPid: process.pid,
        step: 3,
        maxSteps: 10,
      });

      await captureStdout(() => writeControl("demo-run", "stop", createRunOptions(repoRoot)));
      expect(readRunControl(directory)).toMatchObject({ command: "stop" });

      const payload = createStatusPayload("demo-run", {
        repoRoot,
        runsDirectoryName: ".test-runs",
      });
      expect(payload.operatorHealth).toBe("healthy");
      expect(formatStatusPayload(payload, { repoRoot, runsDirectoryName: ".test-runs" })).toContain(
        "state: stopping",
      );
      expect(
        formatStatusPayload(
          {
            ...payload,
            earlyStopPatience: null,
          },
          { repoRoot, runsDirectoryName: ".test-runs" },
        ),
      ).toContain("early stop: disabled");
      expect(
        formatStatusPayload(
          {
            ...payload,
            earlyStopPatience: 3,
          },
          { repoRoot, runsDirectoryName: ".test-runs" },
        ),
      ).toContain("early stop: 3");

      expect(() =>
        createStatusPayload("missing-run", {
          repoRoot,
          runsDirectoryName: ".test-runs",
        }),
      ).toThrow("Unknown run");

      const noPidDirectory = runDir(repoRoot, "no-pid", ".test-runs");
      ensureRunDir(noPidDirectory);
      writeRunSpec(noPidDirectory, {
        runId: "no-pid",
        createdAt: "2026-04-04T00:00:00.000Z",
        repoRoot,
        packageRoot: repoRoot,
        checkpointDir: checkpointsDir(noPidDirectory),
        trainerArgs: [],
      });
      writeRunStatus(noPidDirectory, {
        runId: "no-pid",
        state: "running",
        startedAt: "2026-04-04T00:00:00.000Z",
        updatedAt: "2026-04-04T00:00:01.000Z",
        supervisorHeartbeatAt: "2026-04-04T00:00:01.000Z",
      });
      expect(
        createStatusPayload("no-pid", { repoRoot, runsDirectoryName: ".test-runs" }).rssMb,
      ).toBe(undefined);

      writeRunStatus(noPidDirectory, {
        ...readRunStatus(noPidDirectory),
        trainerPid: 9_999_999,
      });
      expect(
        createStatusPayload("no-pid", { repoRoot, runsDirectoryName: ".test-runs" }).processState,
      ).toBe(undefined);
    });
  });

  test("manager CLI covers start, status, stop, cancel, resume, help, and validation", async () => {
    await withTempDirectory("mlxts-supervised-cli", async (repoRoot) => {
      const options = createCliOptions(repoRoot);
      expect(
        await captureStdout(() => runSupervisedManagerCli(options, ["bun", "manager", "help"])),
      ).toBe("test manager usage\n");
      expect(
        await captureStdout(() =>
          runSupervisedManagerCli(options, [
            "bun",
            "manager",
            "start",
            "--name",
            "cli-run",
            "--max-steps",
            "4",
          ]),
        ),
      ).toContain("Started run cli-run");

      const directory = runDir(repoRoot, "cli-run", ".test-runs");
      writeRunStatus(directory, {
        ...readRunStatus(directory),
        state: "running",
        supervisorPid: process.pid,
        trainerPid: process.pid,
        latestResumeCheckpoint: join(checkpointsDir(directory), "resume-step-1"),
      });
      const statusJson = await captureStdout(() =>
        runSupervisedManagerCli(options, [
          "bun",
          "manager",
          "status",
          "--name",
          "cli-run",
          "--json",
        ]),
      );
      expect(JSON.parse(statusJson).runId).toBe("cli-run");

      expect(
        await captureStdout(() =>
          runSupervisedManagerCli(options, ["bun", "manager", "stop", "--name", "cli-run"]),
        ),
      ).toContain("Requested graceful stop");

      writeRunStatus(directory, {
        ...readRunStatus(directory),
        state: "stopped",
        latestResumeCheckpoint: join(checkpointsDir(directory), "resume-step-1"),
      });
      expect(
        await captureStdout(() =>
          runSupervisedManagerCli(options, [
            "bun",
            "manager",
            "resume",
            "--from",
            "cli-run",
            "--name",
            "cli-resume",
            "--max-steps",
            "5",
          ]),
        ),
      ).toContain("Started run cli-resume");

      const cancelDirectory = runDir(repoRoot, "cli-cancel", ".test-runs");
      ensureRunDir(cancelDirectory);
      writeRunSpec(cancelDirectory, {
        runId: "cli-cancel",
        createdAt: "2026-04-04T00:00:00.000Z",
        repoRoot,
        packageRoot: repoRoot,
        checkpointDir: checkpointsDir(cancelDirectory),
        trainerArgs: [],
      });
      writeRunStatus(cancelDirectory, baseStatus());
      expect(
        await captureStdout(() =>
          runSupervisedManagerCli(options, ["bun", "manager", "cancel", "--name", "cli-cancel"]),
        ),
      ).toContain("Requested cancel");

      expect(
        await captureStdout(() =>
          runSupervisedManagerCli(options, ["bun", "manager", "unknown-command"]),
        ),
      ).toBe("test manager usage\n");
      await expect(
        runSupervisedManagerCli(options, [
          "bun",
          "manager",
          "start",
          "--name",
          "cli-run",
          "--max-steps",
          "4",
        ]),
      ).rejects.toThrow("already exists");
      await expect(
        runSupervisedManagerCli(options, [
          "bun",
          "manager",
          "start",
          "--name",
          "bad-timeout",
          "--stall-timeout-sec",
          "0",
        ]),
      ).rejects.toThrow("positive number");
      await expect(runSupervisedManagerCli(options, ["bun", "manager", "resume"])).rejects.toThrow(
        "resume requires",
      );
      await expect(
        runSupervisedManagerCli(options, [
          "bun",
          "manager",
          "watch",
          "--name",
          "cli-run",
          "--interval",
          "0",
        ]),
      ).rejects.toThrow("positive number");
      await expect(runSupervisedManagerCli(options, ["bun", "manager", "watch"])).rejects.toThrow(
        "watch requires",
      );
      await expect(runSupervisedManagerCli(options, ["bun", "manager", "stop"])).rejects.toThrow(
        "stop requires",
      );
      await expect(runSupervisedManagerCli(options, ["bun", "manager", "cancel"])).rejects.toThrow(
        "cancel requires",
      );
    });
  });

  test("manager command runner emits AXI stdout errors and exit codes", async () => {
    await withTempDirectory("mlxts-supervised-cli-command", async (repoRoot) => {
      const options = createCliOptions(repoRoot);
      const stdout: string[] = [];
      const helpCode = await runSupervisedManagerCliCommand(options, ["bun", "manager", "help"], {
        stdout: (text) => stdout.push(text),
      });
      expect(helpCode).toBe(0);
      expect(stdout.join("")).toBe("test manager usage\n");

      stdout.length = 0;
      const usageCode = await runSupervisedManagerCliCommand(
        options,
        ["bun", "manager", "status"],
        { stdout: (text) => stdout.push(text) },
      );
      expect(usageCode).toBe(2);
      expect(stdout.join("\n")).toContain("error:");
      expect(stdout.join("\n")).toContain('code: "usage"');
      expect(stdout.join("\n")).toContain("status requires --name");

      stdout.length = 0;
      const unknownCode = await runSupervisedManagerCliCommand(
        options,
        ["bun", "manager", "unknown-command"],
        { stdout: (text) => stdout.push(text) },
      );
      expect(unknownCode).toBe(2);
      expect(stdout.join("\n")).toContain("unknown-command");

      stdout.length = 0;
      const startCode = await runSupervisedManagerCliCommand(
        options,
        ["bun", "manager", "start", "--name", "cli-command-run", "--max-steps", "4"],
        { stdout: (text) => stdout.push(text) },
      );
      expect(startCode).toBe(0);
      expect(stdout.join("\n")).toContain("manager_run:");
      expect(stdout.join("\n")).toContain("Started run cli-command-run");

      stdout.length = 0;
      const runtimeCode = await runSupervisedManagerCliCommand(
        options,
        ["bun", "manager", "start", "--name", "cli-command-run", "--max-steps", "4"],
        { stdout: (text) => stdout.push(text) },
      );
      expect(runtimeCode).toBe(1);
      expect(stdout.join("\n")).toContain('code: "runtime"');
      expect(stdout.join("\n")).toContain("already exists");
    });
  });

  test("manager run helpers reject unsafe lifecycle transitions", async () => {
    await withTempDirectory("mlxts-supervised-run-errors", async (repoRoot) => {
      const options = createRunOptions(repoRoot);
      const directory = runDir(repoRoot, "finished-run", ".test-runs");
      ensureRunDir(directory);
      writeRunSpec(directory, {
        runId: "finished-run",
        createdAt: "2026-04-04T00:00:00.000Z",
        repoRoot,
        packageRoot: repoRoot,
        checkpointDir: checkpointsDir(directory),
        trainerArgs: [],
      });
      writeRunStatus(directory, {
        ...baseStatus("completed"),
        runId: "finished-run",
      });
      expect(() => writeControl("finished-run", "stop", options)).toThrow("already finished");

      writeRunStatus(directory, {
        ...baseStatus(),
        runId: "finished-run",
        supervisorPid: 9_999_999,
        trainerPid: 9_999_998,
      });
      expect(() => writeControl("finished-run", "stop", options)).toThrow("dead-both");
      await expect(
        runSupervisedManagerCli(createCliOptions(repoRoot), [
          "bun",
          "manager",
          "resume",
          "--from",
          "finished-run",
        ]),
      ).rejects.toThrow("no resume checkpoint");

      writeRunStatus(directory, {
        ...readRunStatus(directory),
        latestResumeCheckpoint: join(checkpointsDir(directory), "resume-step-1"),
      });
      await expect(
        runSupervisedManagerCli(createCliOptions(repoRoot), [
          "bun",
          "manager",
          "resume",
          "--from",
          "finished-run",
          "--stall-timeout-sec",
          "0",
        ]),
      ).rejects.toThrow("positive number");
      await expect(
        runSupervisedManagerCli(createCliOptions(repoRoot), ["bun", "manager", "status"]),
      ).rejects.toThrow("status requires");
    });
  });
});

describe("supervised run status events", () => {
  test("operator health classifies live, dead, starting, and terminal process states", () => {
    expect(activePid(undefined)).toBe(false);
    expect(deriveOperatorHealth(baseStatus("starting")).operatorHealth).toBe("healthy");
    expect(
      deriveOperatorHealth({
        ...baseStatus("running"),
        supervisorPid: 9_999_999,
        trainerPid: 9_999_998,
      }).operatorHealth,
    ).toBe("dead-both");
    expect(
      deriveOperatorHealth({
        ...baseStatus("completed"),
        supervisorPid: 9_999_999,
        trainerPid: 9_999_998,
      }).operatorHealth,
    ).toBe("healthy");
    expect(
      deriveOperatorHealth({
        ...baseStatus("running"),
        supervisorPid: 9_999_999,
        trainerPid: process.pid,
      }).operatorHealth,
    ).toBe("dead-supervisor");
    expect(
      deriveOperatorHealth({
        ...baseStatus("running"),
        supervisorPid: process.pid,
        trainerPid: 9_999_998,
      }).operatorHealth,
    ).toBe("dead-trainer");
    expect(updateStatusFromEvent(baseStatus(), { type: "progress" }).lastProgressAt).toBeDefined();
    expect(createStatusPayload).toBeDefined();
  });

  test("event state machine preserves start, progress, checkpoint, early-stop, and control data", async () => {
    await withTempDirectory("mlxts-supervised-events", (directory) => {
      ensureRunDir(directory);
      let status = updateStatusFromEvent(baseStatus("starting"), {
        type: "start",
        timestamp: "2026-04-04T00:00:02.000Z",
        preset: "tiny",
        config: { gradientCheckpointing: true },
        params: 10,
        maxSteps: 5,
        batchSize: 2,
        gradAccumSteps: 3,
        warmupSteps: 1,
        startStep: 0,
        resumeFrom: "/checkpoint",
        earlyStopPatience: null,
        earlyStopMinDelta: 0.1,
        activeMemoryBytes: 1,
        cacheMemoryBytes: 2,
        peakMemoryBytes: 3,
        memoryLimitBytes: 4,
      });
      expect(status).toMatchObject({
        state: "running",
        preset: "tiny",
        parameterCount: 10,
        maxSteps: 5,
        earlyStopPatience: null,
      });

      status = updateStatusFromEvent(status, {
        type: "step",
        timestamp: "2026-04-04T00:00:03.000Z",
        step: 1,
        loss: 0.9,
        tokensPerSec: 123,
      });
      status = updateStatusFromEvent(status, {
        type: "eval",
        timestamp: "2026-04-04T00:00:04.000Z",
        step: 1,
        trainLoss: 0.8,
        valLoss: 0.7,
      });
      status = updateStatusFromEvent(status, {
        type: "checkpoint",
        timestamp: "2026-04-04T00:00:05.000Z",
        step: 1,
        path: "/snapshot",
        kind: "snapshot",
      });
      status = updateStatusFromEvent(status, {
        type: "checkpoint",
        timestamp: "2026-04-04T00:00:06.000Z",
        step: 2,
        path: "/resume",
        kind: "resume",
      });
      status = updateStatusFromEvent(status, {
        type: "best-checkpoint",
        timestamp: "2026-04-04T00:00:07.000Z",
        step: 2,
        valLoss: 0.6,
        path: "/best",
      });
      status = updateStatusFromEvent(status, {
        type: "early-stop",
        timestamp: "2026-04-04T00:00:08.000Z",
        step: 2,
        bestValLoss: 0.6,
        bestCheckpointStep: 2,
        bestCheckpointPath: "/best",
        patience: 1,
        minDelta: 0.1,
        consecutiveBadEvals: 1,
        reason: "plateau",
      });
      status = updateStatusFromEvent(status, {
        type: "control",
        timestamp: "2026-04-04T00:00:09.000Z",
        command: "stop",
        requestedAt: "2026-04-04T00:00:09.000Z",
      });
      status = updateStatusFromEvent(status, {
        type: "done",
        timestamp: "2026-04-04T00:00:10.000Z",
        totalSteps: 5,
      });

      expect(status).toMatchObject({
        state: "stopping",
        step: 5,
        lastStepLoss: 0.9,
        lastTrainLoss: 0.8,
        lastValLoss: 0.7,
        latestSnapshotCheckpoint: "/snapshot",
        latestResumeCheckpoint: "/resume",
        bestCheckpoint: "/best",
        earlyStopReason: "plateau",
      });

      writeRunControl(directory, {
        command: "cancel",
        requestedAt: "2026-04-04T00:00:11.000Z",
      });
      const controlled = applyPendingControl(status, directory);
      expect(controlled.state).toBe("cancelling");
      clearRunControl(directory);
      expect(finishSupervisorRun(directory, controlled, 0, null).state).toBe("cancelled");
    });
  });

  test("stalls, final states, escalation, and event parsing cover edge cases", () => {
    expect(readEvent('{"type":"step"}')).toEqual({ type: "step" });
    expect(readEvent("not-json")).toBeNull();
    expect(managerEvent("unit", { ok: true })).toMatchObject({ type: "manager", event: "unit" });

    const stalled = maybeMarkStalled(
      {
        ...baseStatus(),
        lastProgressAt: "2000-01-01T00:00:00.000Z",
      },
      1,
    );
    expect(stalled.state).toBe("stalled");
    expect(maybeMarkStalled(baseStatus("starting"), 1).state).toBe("starting");
    expect(finalState({ ...baseStatus(), step: 5, maxSteps: 5 }, 0, null)).toBe("completed");
    expect(finalState({ ...baseStatus(), step: 4, maxSteps: 5 }, 0, null)).toBe("stopped");
    expect(finalState(baseStatus("stopping"), 1, null)).toBe("failed");
    expect(finalState(baseStatus(), null, "SIGKILL")).toBe("failed");

    const child = Bun.spawn(["sleep", "5"], {
      stdin: "ignore",
      stdout: "ignore",
      stderr: "ignore",
    });
    child.unref();
    maybeEscalateTrainer(
      {
        ...baseStatus("stopping"),
        controlRequestedAt: "2000-01-01T00:00:00.000Z",
      },
      child,
    );
    child.kill("SIGKILL");
  });

  test("supervisor event appends monotonically increasing sequence numbers", async () => {
    await withTempDirectory("mlxts-supervised-seq", (directory) => {
      ensureRunDir(directory);
      appendSupervisorEvent(directory, managerEvent("first"));
      appendSupervisorEvent(directory, managerEvent("second"));
      const events = readFileSync(eventsPath(directory), "utf-8");
      expect(events).toContain('"seq":1');
      expect(events).toContain('"seq":2');
    });
  });

  test("trainer stream pumps keep non-json lines and tail events visible", async () => {
    await withTempDirectory("mlxts-supervised-streams", async (directory) => {
      ensureRunDir(directory);
      const chunks: string[] = [];
      await pipeTextStream(textStream(["std", "err"]), (chunk) => {
        chunks.push(chunk);
      });
      expect(chunks.join("")).toBe("stderr");

      const seen: Record<string, unknown>[] = [];
      await pumpTrainerStdout(
        textStream(['not-json\n{"type":"step","step":3}\n', '{"type":"done","totalSteps":3}']),
        directory,
        (event) => {
          seen.push(event);
        },
      );

      const events = readFileSync(eventsPath(directory), "utf-8");
      expect(events).toContain('"event":"trainer-nonjson"');
      expect(events).toContain('"type":"done"');
      expect(seen).toEqual([
        { type: "step", step: 3 },
        { type: "done", totalSteps: 3 },
      ]);

      const stderr = createStderrStream(directory);
      await new Promise<void>((resolvePromise, reject) => {
        stderr.once("finish", resolvePromise);
        stderr.once("error", reject);
        stderr.end("trainer stderr");
      });
      expect(readFileSync(stderrPath(directory), "utf-8")).toBe("trainer stderr");
    });
  });
});

describe("supervised run supervisor", () => {
  test("requires an explicit run directory", async () => {
    await expect(
      runSupervisedSupervisor({ trainerCommand: () => ["bun", "-e", ""] }, ["bun", "supervisor"]),
    ).rejects.toThrow("Missing required flag");
  });

  test("runs a trainer command and materializes terminal status/events", async () => {
    await withTempDirectory("mlxts-supervised-supervisor", async (repoRoot) => {
      const directory = runDir(repoRoot, "demo-run", ".test-runs");
      ensureRunDir(directory);
      writeRunSpec(directory, {
        runId: "demo-run",
        createdAt: "2026-04-04T00:00:00.000Z",
        repoRoot,
        packageRoot: repoRoot,
        checkpointDir: checkpointsDir(directory),
        stallTimeoutSeconds: 600,
        trainerArgs: [],
      });

      const trainerSource = `
        console.error("trainer stderr");
        console.log(JSON.stringify({ type: "start", timestamp: "2026-04-04T00:00:01.000Z", maxSteps: 1, startStep: 0 }));
        console.log(JSON.stringify({ type: "step", timestamp: "2026-04-04T00:00:02.000Z", step: 1, loss: 0.5, tokensPerSec: 12 }));
        console.log(JSON.stringify({ type: "done", timestamp: "2026-04-04T00:00:03.000Z", totalSteps: 1 }));
      `;

      await runSupervisedSupervisor({ trainerCommand: () => ["bun", "-e", trainerSource] }, [
        "bun",
        "supervisor",
        "--run-dir",
        directory,
      ]);

      expect(readRunStatus(directory)).toMatchObject({
        state: "completed",
        step: 1,
        maxSteps: 1,
        lastStepLoss: 0.5,
      });
      expect(readFileSync(eventsPath(directory), "utf-8")).toContain('"event":"trainer-exited"');
      expect(readFileSync(stderrPath(directory), "utf-8")).toContain("trainer stderr");
    });
  });

  test("applies a pending control file while the trainer is still running", async () => {
    await withTempDirectory("mlxts-supervised-control", async (repoRoot) => {
      const directory = runDir(repoRoot, "control-run", ".test-runs");
      ensureRunDir(directory);
      writeRunSpec(directory, {
        runId: "control-run",
        createdAt: "2026-04-04T00:00:00.000Z",
        repoRoot,
        packageRoot: repoRoot,
        checkpointDir: checkpointsDir(directory),
        stallTimeoutSeconds: 600,
        trainerArgs: [],
      });

      const run = runSupervisedSupervisor(
        { trainerCommand: () => ["bun", "-e", "await Bun.sleep(6500);"] },
        ["bun", "supervisor", "--run-dir", directory],
      );
      await Bun.sleep(100);
      writeRunControl(directory, {
        command: "stop",
        requestedAt: new Date().toISOString(),
      });
      await run;

      expect(readRunStatus(directory).state).toBe("stopped");
      const events = readFileSync(eventsPath(directory), "utf-8");
      expect(events).toContain('"event":"stop-requested"');
      expect(events).toContain('"event":"trainer-exited"');
    });
  }, 10_000);
});
