#!/usr/bin/env bun

import { acquireRuntimeCommandLock } from "../../../../scripts/runtime-command-lock";
import {
  DEFAULT_STALL_TIMEOUT_SECONDS,
  type RunStatus,
  readRunSpec,
  writePid,
  writeRunStatus,
} from "./files";
import {
  appendSupervisorEvent,
  applyPendingControl,
  CONTROL_CHECK_INTERVAL_MS,
  finishSupervisorRun,
  HEARTBEAT_INTERVAL_MS,
  managerEvent,
  maybeEscalateTrainer,
  maybeMarkStalled,
  updateStatusFromEvent,
} from "./supervisor-events";
import { createStderrStream, pipeTextStream, pumpTrainerStdout } from "./supervisor-streams";

function readRunDirectory(argv: string[]): string {
  const index = argv.indexOf("--run-dir");
  const runDirectory = index >= 0 ? argv[index + 1] : undefined;
  if (runDirectory === undefined) {
    throw new Error("Missing required flag --run-dir");
  }
  return runDirectory;
}

export async function main(argv = process.argv): Promise<void> {
  using _runtimeLock = acquireRuntimeCommandLock("train:nanogpt-supervisor");
  const runDirectory = readRunDirectory(argv);
  const spec = readRunSpec(runDirectory);
  const stallTimeoutMs = (spec.stallTimeoutSeconds ?? DEFAULT_STALL_TIMEOUT_SECONDS) * 1000;
  const stderrStream = createStderrStream(runDirectory);

  let status: RunStatus = {
    runId: spec.runId,
    state: "starting",
    startedAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    supervisorHeartbeatAt: new Date().toISOString(),
    trainerHeartbeatAt: new Date().toISOString(),
    lastProgressAt: new Date().toISOString(),
    stallTimeoutSeconds: spec.stallTimeoutSeconds ?? DEFAULT_STALL_TIMEOUT_SECONDS,
    supervisorPid: process.pid,
    resumeFrom: spec.resumedFrom,
  };

  writePid(runDirectory, process.pid);
  writeRunStatus(runDirectory, status);
  appendSupervisorEvent(runDirectory, managerEvent("supervisor-started", { pid: process.pid }));

  const trainer = Bun.spawn(["bun", "run", "src/cli.ts", "train", ...spec.trainerArgs], {
    cwd: spec.packageRoot,
    stdin: "ignore",
    stdout: "pipe",
    stderr: "pipe",
  });

  status = {
    ...status,
    trainerPid: trainer.pid,
  };
  writeRunStatus(runDirectory, status);
  appendSupervisorEvent(runDirectory, managerEvent("trainer-started", { pid: trainer.pid }));

  const stdoutStream = trainer.stdout;
  const stderrPipe = trainer.stderr;
  if (stdoutStream === null || stderrPipe === null) {
    throw new Error("trainer process did not expose piped stdout/stderr streams");
  }

  const stdoutPump = pumpTrainerStdout(stdoutStream, runDirectory, (event) => {
    status = updateStatusFromEvent(status, event);
    writeRunStatus(runDirectory, status);
  }).catch((error) => {
    appendSupervisorEvent(
      runDirectory,
      managerEvent("trainer-stdout-error", {
        message: error instanceof Error ? error.message : String(error),
      }),
    );
    throw error;
  });
  const stderrPump = pipeTextStream(stderrPipe, (chunk) => {
    stderrStream.write(chunk);
  });
  const exitPromise = trainer.exited.then(() => ({
    code: trainer.exitCode ?? null,
    signal: trainer.signalCode ?? null,
  }));

  const heartbeat = setInterval(() => {
    status = maybeMarkStalled(
      {
        ...status,
        supervisorHeartbeatAt: new Date().toISOString(),
      },
      stallTimeoutMs,
    );
    writeRunStatus(runDirectory, status);
  }, HEARTBEAT_INTERVAL_MS);

  const controlTimer = setInterval(() => {
    const nextStatus = applyPendingControl(status, runDirectory);
    if (nextStatus !== status) {
      status = nextStatus;
      writeRunStatus(runDirectory, status);
      appendSupervisorEvent(
        runDirectory,
        managerEvent(status.controlCommand === "cancel" ? "cancel-requested" : "stop-requested", {
          requestedAt: status.controlRequestedAt,
        }),
      );
    }
    maybeEscalateTrainer(status, trainer);
  }, CONTROL_CHECK_INTERVAL_MS);

  const exitResult = await exitPromise;
  await Promise.all([stdoutPump, stderrPump]);

  clearInterval(heartbeat);
  clearInterval(controlTimer);
  stderrStream.end();

  status = finishSupervisorRun(
    runDirectory,
    applyPendingControl(status, runDirectory),
    exitResult.code,
    exitResult.signal,
  );
  writeRunStatus(runDirectory, status);
  appendSupervisorEvent(
    runDirectory,
    managerEvent("trainer-exited", {
      exitCode: exitResult.code,
      signal: exitResult.signal,
      state: status.state,
    }),
  );
}

await main();
