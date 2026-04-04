import {
  closeSync,
  existsSync,
  fsyncSync,
  mkdirSync,
  openSync,
  readdirSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "fs";
import { dirname, join, resolve } from "path";
import type { GPTConfig } from "../config";

export const DEFAULT_STALL_TIMEOUT_SECONDS = 600;

export type RunState =
  | "starting"
  | "running"
  | "stopping"
  | "cancelling"
  | "stalled"
  | "stopped"
  | "completed"
  | "failed"
  | "cancelled";

export type RunControlCommand = "stop" | "cancel";
export type OperatorHealth = "healthy" | "dead-supervisor" | "dead-trainer" | "dead-both";

export type RunSpec = {
  runId: string;
  createdAt: string;
  repoRoot: string;
  packageRoot: string;
  checkpointDir: string;
  stallTimeoutSeconds?: number | undefined;
  trainerArgs: string[];
  resumedFrom?: string | undefined;
};

export type RunControl = {
  command: RunControlCommand;
  requestedAt: string;
};

export type RunStatus = {
  runId: string;
  state: RunState;
  startedAt: string;
  updatedAt: string;
  supervisorHeartbeatAt: string;
  trainerHeartbeatAt?: string | undefined;
  lastProgressAt?: string | undefined;
  stallTimeoutSeconds?: number | undefined;
  supervisorPid?: number | undefined;
  trainerPid?: number | undefined;
  preset?: string | undefined;
  config?: GPTConfig | undefined;
  parameterCount?: number | undefined;
  step?: number | undefined;
  maxSteps?: number | undefined;
  batchSize?: number | undefined;
  gradAccumSteps?: number | undefined;
  warmupSteps?: number | undefined;
  lastStepLoss?: number | undefined;
  lastTrainLoss?: number | undefined;
  lastValLoss?: number | undefined;
  lastTokensPerSec?: number | undefined;
  latestCheckpoint?: string | undefined;
  latestSnapshotCheckpoint?: string | undefined;
  latestResumeCheckpoint?: string | undefined;
  latestCheckpointKind?: string | undefined;
  activeMemoryBytes?: number | undefined;
  cacheMemoryBytes?: number | undefined;
  peakMemoryBytes?: number | undefined;
  memoryLimitBytes?: number | undefined;
  exitCode?: number | null | undefined;
  signal?: string | null | undefined;
  resumeFrom?: string | undefined;
  controlCommand?: RunControlCommand | undefined;
  controlRequestedAt?: string | undefined;
  stallReason?: string | undefined;
};

export type RunHealth = {
  supervisorAlive: boolean;
  trainerAlive: boolean;
  operatorHealth: OperatorHealth;
};

type JsonRecord = Record<string, unknown>;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function packageRootFromRunDir(runDir: string): string {
  return resolve(runDir, "..", "..", "packages", "nanogpt");
}

export function repoRootFromPackageRoot(packageRoot: string): string {
  return resolve(packageRoot, "..", "..");
}

export function runsRoot(repoRoot: string): string {
  return join(repoRoot, ".nanogpt-runs");
}

export function runDir(repoRoot: string, runId: string): string {
  return join(runsRoot(repoRoot), runId);
}

export function checkpointsDir(runDirectory: string): string {
  return join(runDirectory, "checkpoints");
}

export function runSpecPath(runDirectory: string): string {
  return join(runDirectory, "run.json");
}

export function runStatusPath(runDirectory: string): string {
  return join(runDirectory, "status.json");
}

export function runControlPath(runDirectory: string): string {
  return join(runDirectory, "control.json");
}

export function eventsPath(runDirectory: string): string {
  return join(runDirectory, "events.jsonl");
}

export function stderrPath(runDirectory: string): string {
  return join(runDirectory, "stderr.log");
}

export function pidPath(runDirectory: string): string {
  return join(runDirectory, "pid");
}

export function ensureRunDir(runDirectory: string): void {
  mkdirSync(runDirectory, { recursive: true });
  mkdirSync(checkpointsDir(runDirectory), { recursive: true });
}

function readJsonFile(path: string): JsonRecord {
  const parsed: unknown = JSON.parse(readFileSync(path, "utf-8"));
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
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

function readStringArray(value: unknown, context: string): string[] {
  if (!Array.isArray(value)) {
    throw new Error(`${context}: expected a string array`);
  }
  return value.map((entry, index) => readString(entry, `${context}[${index}]`));
}

function readOptionalBoolean(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

function readOptionalConfig(value: unknown): GPTConfig | undefined {
  if (!isRecord(value)) {
    return undefined;
  }

  const nLayer = readOptionalNumber(value.nLayer);
  const nHead = readOptionalNumber(value.nHead);
  const nEmbd = readOptionalNumber(value.nEmbd);
  const blockSize = readOptionalNumber(value.blockSize);
  const dropout = readOptionalNumber(value.dropout);
  const gradientCheckpointing = readOptionalBoolean(value.gradientCheckpointing);
  const vocabSize = readOptionalNumber(value.vocabSize);

  if (
    nLayer === undefined ||
    nHead === undefined ||
    nEmbd === undefined ||
    blockSize === undefined ||
    dropout === undefined ||
    gradientCheckpointing === undefined ||
    vocabSize === undefined
  ) {
    return undefined;
  }

  return {
    nLayer,
    nHead,
    nEmbd,
    blockSize,
    dropout,
    gradientCheckpointing,
    vocabSize,
  };
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
    lastTokensPerSec: readOptionalNumber(raw.lastTokensPerSec),
    latestCheckpoint: readOptionalString(raw.latestCheckpoint),
    latestSnapshotCheckpoint: readOptionalString(raw.latestSnapshotCheckpoint),
    latestResumeCheckpoint: readOptionalString(raw.latestResumeCheckpoint),
    latestCheckpointKind: readOptionalString(raw.latestCheckpointKind),
    activeMemoryBytes: readOptionalNumber(raw.activeMemoryBytes),
    cacheMemoryBytes: readOptionalNumber(raw.cacheMemoryBytes),
    peakMemoryBytes: readOptionalNumber(raw.peakMemoryBytes),
    memoryLimitBytes: readOptionalNumber(raw.memoryLimitBytes),
    exitCode: readOptionalNullableNumber(raw.exitCode),
    signal: raw.signal === null ? null : readOptionalString(raw.signal),
    resumeFrom: readOptionalString(raw.resumeFrom),
    controlCommand,
    controlRequestedAt: readOptionalString(raw.controlRequestedAt),
    stallReason: readOptionalString(raw.stallReason),
  };
}

export function readRunControl(runDirectory: string): RunControl | undefined {
  const path = runControlPath(runDirectory);
  let raw: JsonRecord;
  try {
    raw = readJsonFile(path);
  } catch (error) {
    if (isNotFoundError(error)) {
      return undefined;
    }
    throw error;
  }
  if (!isRunControlCommand(raw.command)) {
    throw new Error(`control.command: expected stop/cancel, got ${String(raw.command)}`);
  }

  return {
    command: raw.command,
    requestedAt: readString(raw.requestedAt, "control.requestedAt"),
  };
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

function runStateIsTerminal(state: RunState): boolean {
  return (
    state === "stopped" || state === "completed" || state === "failed" || state === "cancelled"
  );
}

export function activePid(pid: number | undefined): boolean {
  if (pid === undefined) {
    return false;
  }
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

export function deriveOperatorHealth(status: RunStatus): RunHealth {
  const supervisorAlive = activePid(status.supervisorPid);
  const trainerAlive = activePid(status.trainerPid);

  if (status.state === "starting") {
    return {
      supervisorAlive,
      trainerAlive,
      operatorHealth: supervisorAlive ? "healthy" : "dead-supervisor",
    };
  }

  if (supervisorAlive && trainerAlive) {
    return { supervisorAlive, trainerAlive, operatorHealth: "healthy" };
  }

  if (!supervisorAlive && !trainerAlive) {
    return {
      supervisorAlive,
      trainerAlive,
      operatorHealth: runStateIsTerminal(status.state) ? "healthy" : "dead-both",
    };
  }

  if (!supervisorAlive) {
    return { supervisorAlive, trainerAlive, operatorHealth: "dead-supervisor" };
  }

  return { supervisorAlive, trainerAlive, operatorHealth: "dead-trainer" };
}

export function writePid(runDirectory: string, pid: number): void {
  writeFileSync(pidPath(runDirectory), `${pid}\n`, "utf-8");
}

function checkpointStep(name: string): number | undefined {
  const match = /-step-(\d+)$/.exec(name);
  if (match === null) {
    return undefined;
  }
  const value = Number(match[1]);
  return Number.isInteger(value) ? value : undefined;
}

export function readLatestCheckpoint(runDirectory: string): string | undefined {
  const checkpointDirectory = checkpointsDir(runDirectory);
  if (!existsSync(checkpointDirectory)) {
    return undefined;
  }

  const entries = readdirSync(checkpointDirectory, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort((left, right) => {
      const leftStep = checkpointStep(left);
      const rightStep = checkpointStep(right);
      if (leftStep !== undefined && rightStep !== undefined && leftStep !== rightStep) {
        return leftStep - rightStep;
      }
      return left.localeCompare(right);
    });
  const latest = entries.at(-1);
  return latest === undefined ? undefined : join(checkpointDirectory, latest);
}
