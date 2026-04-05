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
import {
  generateRunId,
  getFlag,
  nowIso,
  packageRoot,
  repoRoot,
  trainerArgsFrom,
} from "./manager-args";

function stripFlag(args: string[], key: string, takesValue = true): string[] {
  const stripped: string[] = [];
  for (let index = 0; index < args.length; index++) {
    if (args[index] !== `--${key}`) {
      const value = args[index];
      if (value !== undefined) {
        stripped.push(value);
      }
      continue;
    }
    if (takesValue) {
      index += 1;
    }
  }
  return stripped;
}

function writeStartFiles(
  runId: string,
  trainerArgs: string[],
  stallTimeoutSeconds: number,
  resumedFrom?: string,
): string {
  const root = repoRoot();
  const directory = runDir(root, runId);
  ensureRunDir(directory);
  const spec: RunSpec = {
    runId,
    createdAt: nowIso(),
    repoRoot: root,
    packageRoot: packageRoot(),
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

function startSupervisor(runDirectory: string): number {
  const child = Bun.spawn(["bun", "run", "src/run/supervisor.ts", "--run-dir", runDirectory], {
    cwd: packageRoot(),
    detached: true,
    stdin: "ignore",
    stdout: "ignore",
    stderr: "ignore",
  });
  child.unref();
  return child.pid ?? 0;
}

export function startRun(args: string[], resumedFrom?: string): void {
  const requestedName = getFlag(args, "name");
  const runId = requestedName ?? generateRunId(args);
  const directory = runDir(repoRoot(), runId);
  if (existsSync(directory)) {
    throw new Error(`Run "${runId}" already exists`);
  }

  const stallTimeoutSeconds = Number(
    getFlag(args, "stall-timeout-sec", String(DEFAULT_STALL_TIMEOUT_SECONDS)),
  );
  if (!Number.isFinite(stallTimeoutSeconds) || stallTimeoutSeconds <= 0) {
    throw new Error("start requires --stall-timeout-sec to be a positive number");
  }

  const trainerBaseArgs = trainerArgsFrom(args);
  const trainerArgs = [
    ...trainerBaseArgs,
    "--json",
    "--run-dir",
    directory,
    "--checkpoint-dir",
    checkpointsDir(directory),
  ];
  writeStartFiles(runId, trainerArgs, stallTimeoutSeconds, resumedFrom);
  const supervisorPid = startSupervisor(directory);
  process.stdout.write(
    `Started run ${runId}\n  dir: ${directory}\n  supervisor pid: ${supervisorPid}\n  status: bun run packages/nanogpt/src/run/manager.ts status --name ${runId}\n`,
  );
}

function resolveExistingRun(runId: string): string {
  const directory = runDir(repoRoot(), runId);
  if (!existsSync(directory)) {
    throw new Error(`Unknown run "${runId}"`);
  }
  return directory;
}

export function writeControl(runId: string, command: RunControlCommand): void {
  const directory = resolveExistingRun(runId);
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

export function resumeRun(args: string[]): void {
  const fromRunId = getFlag(args, "from");
  if (fromRunId === undefined) {
    throw new Error("resume requires --from <run-id>");
  }

  const requestedName = getFlag(args, "name");
  const directory = resolveExistingRun(fromRunId);
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

  const overrides = trainerArgsFrom(args);
  const mergedArgs = [...trainerArgs, ...overrides, "--resume", latestCheckpoint];
  if (requestedName !== undefined) {
    mergedArgs.push("--name", requestedName);
  }
  startRun([...mergedArgs, "--stall-timeout-sec", String(stallTimeoutSeconds)], latestCheckpoint);
}
