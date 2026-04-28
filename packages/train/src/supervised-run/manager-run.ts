import { existsSync } from "fs";

import {
  checkpointsDir,
  DEFAULT_STALL_TIMEOUT_SECONDS,
  deriveOperatorHealth,
  ensureRunDir,
  type RunControlCommand,
  type RunSpec,
  readRunSpec,
  readRunStatus,
  runDir,
  writeRunControl,
  writeRunSpec,
  writeRunStatus,
} from "./files";
import { generateRunId, getFlag, nowIso, stripFlag, trainerArgsFrom } from "./manager-args";

export type SupervisedRunManagerRunOptions = {
  repoRoot: string;
  packageRoot: string;
  runsDirectoryName?: string | undefined;
  pathFlags?: ReadonlySet<string> | undefined;
  runIdLabel?: ((args: string[]) => string) | undefined;
  supervisorCommand: (runDirectory: string) => string[];
  supervisorCwd?: string | undefined;
  statusCommand: (runId: string) => string;
};

function runDirectoryFor(options: SupervisedRunManagerRunOptions, runId: string): string {
  return runDir(options.repoRoot, runId, options.runsDirectoryName);
}

function writeStartFiles(
  runId: string,
  trainerArgs: string[],
  stallTimeoutSeconds: number,
  options: SupervisedRunManagerRunOptions,
  resumedFrom?: string,
): string {
  const directory = runDirectoryFor(options, runId);
  ensureRunDir(directory);
  const spec: RunSpec = {
    runId,
    createdAt: nowIso(),
    repoRoot: options.repoRoot,
    packageRoot: options.packageRoot,
    checkpointDir: checkpointsDir(directory),
    stallTimeoutSeconds,
    trainerArgs,
    resumedFrom,
  };
  writeRunSpec(directory, spec);
  writeRunStatus(directory, {
    runId,
    state: "starting",
    startedAt: spec.createdAt,
    updatedAt: spec.createdAt,
    supervisorHeartbeatAt: spec.createdAt,
    trainerHeartbeatAt: spec.createdAt,
    lastProgressAt: spec.createdAt,
    stallTimeoutSeconds,
    resumeFrom: resumedFrom,
  });
  return directory;
}

function startSupervisor(runDirectory: string, options: SupervisedRunManagerRunOptions): number {
  const child = Bun.spawn(options.supervisorCommand(runDirectory), {
    cwd: options.supervisorCwd ?? options.packageRoot,
    detached: true,
    env: process.env,
    stdin: "ignore",
    stdout: "ignore",
    stderr: "ignore",
  });
  child.unref();
  return child.pid ?? 0;
}

export function startRun(
  args: string[],
  options: SupervisedRunManagerRunOptions,
  resumedFrom?: string,
): void {
  const requestedName = getFlag(args, "name");
  const runId = requestedName ?? generateRunId(args, options.runIdLabel);
  const directory = runDirectoryFor(options, runId);
  if (existsSync(directory)) {
    throw new Error(`Run "${runId}" already exists`);
  }

  const stallTimeoutSeconds = Number(
    getFlag(args, "stall-timeout-sec", String(DEFAULT_STALL_TIMEOUT_SECONDS)),
  );
  if (!Number.isFinite(stallTimeoutSeconds) || stallTimeoutSeconds <= 0) {
    throw new Error("start requires --stall-timeout-sec to be a positive number");
  }

  const trainerBaseArgs = trainerArgsFrom(args, options.pathFlags);
  const trainerArgs = [
    ...trainerBaseArgs,
    "--json",
    "--run-dir",
    directory,
    "--checkpoint-dir",
    checkpointsDir(directory),
  ];
  writeStartFiles(runId, trainerArgs, stallTimeoutSeconds, options, resumedFrom);
  const supervisorPid = startSupervisor(directory, options);
  writeRunStatus(directory, {
    ...readRunStatus(directory),
    supervisorPid,
    updatedAt: nowIso(),
  });
  process.stdout.write(
    `Started run ${runId}\n  dir: ${directory}\n  supervisor pid: ${supervisorPid}\n  status: ${options.statusCommand(runId)}\n`,
  );
}

function resolveExistingRun(runId: string, options: SupervisedRunManagerRunOptions): string {
  const directory = runDirectoryFor(options, runId);
  if (!existsSync(directory)) {
    throw new Error(`Unknown run "${runId}"`);
  }
  return directory;
}

export function writeControl(
  runId: string,
  command: RunControlCommand,
  options: SupervisedRunManagerRunOptions,
): void {
  const directory = resolveExistingRun(runId, options);
  const status = readRunStatus(directory);
  if (
    status.state === "stopped" ||
    status.state === "completed" ||
    status.state === "failed" ||
    status.state === "cancelled"
  ) {
    throw new Error(`Run "${runId}" is already finished in state ${status.state}`);
  }

  const health = deriveOperatorHealth(status);
  if (health.operatorHealth !== "healthy") {
    throw new Error(`Run "${runId}" is ${health.operatorHealth}; cannot deliver ${command}`);
  }

  const requestedAt = nowIso();
  writeRunControl(directory, { command, requestedAt });
  writeRunStatus(directory, {
    ...status,
    state: command === "stop" ? "stopping" : "cancelling",
    updatedAt: requestedAt,
    controlCommand: command,
    controlRequestedAt: requestedAt,
  });

  if (command === "cancel") {
    const checkpoint = status.latestResumeCheckpoint ?? status.latestCheckpoint;
    process.stdout.write(
      `Requested cancel for run ${runId}. Work since the latest resume checkpoint may be lost${checkpoint === undefined ? "" : ` (${checkpoint})`}.\n`,
    );
    return;
  }

  process.stdout.write(`Requested graceful stop for run ${runId}\n`);
}

export function resumeRun(args: string[], options: SupervisedRunManagerRunOptions): void {
  const fromRunId = getFlag(args, "from");
  if (fromRunId === undefined) {
    throw new Error("resume requires --from <run-id>");
  }

  const requestedName = getFlag(args, "name");
  const directory = resolveExistingRun(fromRunId, options);
  const spec = readRunSpec(directory);
  const status = readRunStatus(directory);
  const latestCheckpoint = status.latestResumeCheckpoint ?? status.latestCheckpoint;
  if (latestCheckpoint === undefined) {
    throw new Error(`Run "${fromRunId}" has no resume checkpoint to continue from`);
  }

  const stallTimeoutSeconds = Number(
    getFlag(
      args,
      "stall-timeout-sec",
      String(spec.stallTimeoutSeconds ?? DEFAULT_STALL_TIMEOUT_SECONDS),
    ),
  );
  if (!Number.isFinite(stallTimeoutSeconds) || stallTimeoutSeconds <= 0) {
    throw new Error("resume requires --stall-timeout-sec to be a positive number");
  }

  let trainerArgs = stripFlag(spec.trainerArgs, "json", false);
  trainerArgs = stripFlag(trainerArgs, "checkpoint-dir");
  trainerArgs = stripFlag(trainerArgs, "run-dir");
  trainerArgs = stripFlag(trainerArgs, "resume");
  trainerArgs = stripFlag(trainerArgs, "warm-start");
  trainerArgs = stripFlag(trainerArgs, "preset");

  const overrides = trainerArgsFrom(args, options.pathFlags);
  const mergedArgs = [...trainerArgs, ...overrides, "--resume", latestCheckpoint];
  if (requestedName !== undefined) {
    mergedArgs.push("--name", requestedName);
  }
  startRun(
    [...mergedArgs, "--stall-timeout-sec", String(stallTimeoutSeconds)],
    options,
    latestCheckpoint,
  );
}
