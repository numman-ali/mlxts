import {
  closeSync,
  fsyncSync,
  openSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "fs";
import { dirname } from "path";

import { eventsPath, pidPath, runControlPath, runSpecPath, runStatusPath } from "./files-paths";
import type { RunControl, RunControlCommand, RunSpec, RunState, RunStatus } from "./files-types";

type JsonRecord = Record<string, unknown>;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readJsonFile(path: string): JsonRecord {
  const parsed: unknown = JSON.parse(readFileSync(path, "utf-8"));
  if (!isRecord(parsed)) {
    throw new Error(`${path}: expected a JSON object`);
  }
  return Object.fromEntries(Object.entries(parsed));
}

function isNotFoundError(error: unknown): boolean {
  return typeof error === "object" && error !== null && "code" in error && error.code === "ENOENT";
}

export function writeJsonFile(path: string, value: JsonRecord): void {
  const tempPath = `${path}.${process.pid}.${crypto.randomUUID()}.tmp`;
  let tempFd: number | undefined;

  try {
    tempFd = openSync(tempPath, "w");
    writeFileSync(tempFd, `${JSON.stringify(value, null, 2)}\n`, "utf-8");
    fsyncSync(tempFd);
    closeSync(tempFd);
    tempFd = undefined;
    renameSync(tempPath, path);

    try {
      const directoryFd = openSync(dirname(path), "r");
      try {
        fsyncSync(directoryFd);
      } finally {
        closeSync(directoryFd);
      }
    } catch {
      // Directory fsync is best-effort across platforms.
    }
  } catch (error) {
    if (tempFd !== undefined) {
      try {
        closeSync(tempFd);
      } catch {
        // Ignore cleanup failures.
      }
    }
    rmSync(tempPath, { force: true });
    throw error;
  }
}

function readString(value: unknown, context: string): string {
  if (typeof value !== "string") {
    throw new Error(`${context}: expected a string`);
  }
  return value;
}

function readOptionalString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function readOptionalNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function readOptionalPositiveInteger(value: unknown, context: string): number | undefined {
  const number = readOptionalNumber(value);
  if (number === undefined) {
    return undefined;
  }
  if (!Number.isInteger(number) || number <= 0) {
    throw new Error(`${context}: expected a positive integer`);
  }
  return number;
}

function readOptionalNullableNumber(value: unknown): number | null | undefined {
  if (value === null) {
    return null;
  }
  return readOptionalNumber(value);
}

function readOptionalNullableNonNegativeInteger(
  value: unknown,
  context: string,
): number | null | undefined {
  if (value === null) {
    return null;
  }

  const number = readOptionalNumber(value);
  if (number === undefined) {
    return undefined;
  }
  if (!Number.isInteger(number) || number < 0) {
    throw new Error(`${context}: expected a non-negative integer`);
  }
  return number;
}

function readStringArray(value: unknown, context: string): string[] {
  if (!Array.isArray(value)) {
    throw new Error(`${context}: expected a string array`);
  }
  return value.map((entry, index) => readString(entry, `${context}[${index}]`));
}

function readOptionalConfig(value: unknown): RunStatus["config"] {
  if (!isRecord(value)) {
    return undefined;
  }
  return Object.fromEntries(Object.entries(value));
}

function isRunState(value: unknown): value is RunState {
  return (
    value === "starting" ||
    value === "running" ||
    value === "stopping" ||
    value === "cancelling" ||
    value === "stalled" ||
    value === "stopped" ||
    value === "completed" ||
    value === "failed" ||
    value === "cancelled"
  );
}

function isRunControlCommand(value: unknown): value is RunControlCommand {
  return value === "stop" || value === "cancel";
}

export function readRunSpec(runDirectory: string): RunSpec {
  const raw = readJsonFile(runSpecPath(runDirectory));
  return {
    runId: readString(raw.runId, "run.runId"),
    createdAt: readString(raw.createdAt, "run.createdAt"),
    repoRoot: readString(raw.repoRoot, "run.repoRoot"),
    packageRoot: readString(raw.packageRoot, "run.packageRoot"),
    checkpointDir: readString(raw.checkpointDir, "run.checkpointDir"),
    stallTimeoutSeconds: readOptionalPositiveInteger(
      raw.stallTimeoutSeconds,
      "run.stallTimeoutSeconds",
    ),
    trainerArgs: readStringArray(raw.trainerArgs, "run.trainerArgs"),
    resumedFrom: readOptionalString(raw.resumedFrom),
  };
}

export function readRunStatus(runDirectory: string): RunStatus {
  const raw = readJsonFile(runStatusPath(runDirectory));
  if (!isRunState(raw.state)) {
    throw new Error(`status.state: expected a known run state, got ${String(raw.state)}`);
  }

  const controlCommand = raw.controlCommand;
  if (controlCommand !== undefined && !isRunControlCommand(controlCommand)) {
    throw new Error(`status.controlCommand: expected stop/cancel, got ${String(controlCommand)}`);
  }

  return {
    runId: readString(raw.runId, "status.runId"),
    state: raw.state,
    startedAt: readString(raw.startedAt, "status.startedAt"),
    updatedAt: readString(raw.updatedAt, "status.updatedAt"),
    supervisorHeartbeatAt: readString(raw.supervisorHeartbeatAt, "status.supervisorHeartbeatAt"),
    trainerHeartbeatAt: readOptionalString(raw.trainerHeartbeatAt),
    lastProgressAt: readOptionalString(raw.lastProgressAt),
    stallTimeoutSeconds: readOptionalPositiveInteger(
      raw.stallTimeoutSeconds,
      "status.stallTimeoutSeconds",
    ),
    supervisorPid: readOptionalNumber(raw.supervisorPid),
    trainerPid: readOptionalNumber(raw.trainerPid),
    preset: readOptionalString(raw.preset),
    config: readOptionalConfig(raw.config),
    parameterCount: readOptionalNumber(raw.parameterCount),
    step: readOptionalNumber(raw.step),
    maxSteps: readOptionalNumber(raw.maxSteps),
    batchSize: readOptionalNumber(raw.batchSize),
    gradAccumSteps: readOptionalNumber(raw.gradAccumSteps),
    warmupSteps: readOptionalNumber(raw.warmupSteps),
    lastStepLoss: readOptionalNumber(raw.lastStepLoss),
    lastTrainLoss: readOptionalNumber(raw.lastTrainLoss),
    lastValLoss: readOptionalNumber(raw.lastValLoss),
    bestValLoss: readOptionalNumber(raw.bestValLoss),
    lastTokensPerSec: readOptionalNumber(raw.lastTokensPerSec),
    latestCheckpoint: readOptionalString(raw.latestCheckpoint),
    latestSnapshotCheckpoint: readOptionalString(raw.latestSnapshotCheckpoint),
    latestResumeCheckpoint: readOptionalString(raw.latestResumeCheckpoint),
    bestCheckpoint: readOptionalString(raw.bestCheckpoint),
    bestCheckpointStep: readOptionalNumber(raw.bestCheckpointStep),
    latestCheckpointKind: readOptionalString(raw.latestCheckpointKind),
    activeMemoryBytes: readOptionalNumber(raw.activeMemoryBytes),
    cacheMemoryBytes: readOptionalNumber(raw.cacheMemoryBytes),
    peakMemoryBytes: readOptionalNumber(raw.peakMemoryBytes),
    memoryLimitBytes: readOptionalNumber(raw.memoryLimitBytes),
    earlyStopPatience: readOptionalNullableNonNegativeInteger(
      raw.earlyStopPatience,
      "status.earlyStopPatience",
    ),
    earlyStopMinDelta: readOptionalNumber(raw.earlyStopMinDelta),
    earlyStopConsecutiveBadEvals: readOptionalNumber(raw.earlyStopConsecutiveBadEvals),
    earlyStopReason: readOptionalString(raw.earlyStopReason),
    exitCode: readOptionalNullableNumber(raw.exitCode),
    signal: raw.signal === null ? null : readOptionalString(raw.signal),
    resumeFrom: readOptionalString(raw.resumeFrom),
    controlCommand,
    controlRequestedAt: readOptionalString(raw.controlRequestedAt),
    stallReason: readOptionalString(raw.stallReason),
  };
}

export function readRunControl(runDirectory: string): RunControl | undefined {
  try {
    const raw = readJsonFile(runControlPath(runDirectory));
    if (!isRunControlCommand(raw.command)) {
      throw new Error(`control.command: expected stop/cancel, got ${String(raw.command)}`);
    }

    return {
      command: raw.command,
      requestedAt: readString(raw.requestedAt, "control.requestedAt"),
    };
  } catch (error) {
    if (isNotFoundError(error)) {
      return undefined;
    }
    throw error;
  }
}

export function writeRunSpec(runDirectory: string, spec: RunSpec): void {
  writeJsonFile(runSpecPath(runDirectory), spec);
}

export function writeRunStatus(runDirectory: string, status: RunStatus): void {
  writeJsonFile(runStatusPath(runDirectory), status);
}

export function writeRunControl(runDirectory: string, control: RunControl): void {
  writeJsonFile(runControlPath(runDirectory), control);
}

export function clearRunControl(runDirectory: string): void {
  rmSync(runControlPath(runDirectory), { force: true });
}

export function appendEvent(runDirectory: string, event: JsonRecord): void {
  writeFileSync(eventsPath(runDirectory), `${JSON.stringify(event)}\n`, {
    encoding: "utf-8",
    flag: "a",
  });
}

export function writePid(runDirectory: string, pid: number): void {
  writeFileSync(pidPath(runDirectory), `${pid}\n`, "utf-8");
}
