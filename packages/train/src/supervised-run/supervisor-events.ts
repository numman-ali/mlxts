import {
  appendEvent,
  clearRunControl,
  type RunState,
  type RunStatus,
  readRunControl,
} from "./files";

export const HEARTBEAT_INTERVAL_MS = 30_000;
export const CONTROL_CHECK_INTERVAL_MS = 5_000;
export const STOP_ESCALATE_AFTER_MS = 60_000;
export const KILL_ESCALATE_AFTER_MS = 90_000;

function nowIso(): string {
  return new Date().toISOString();
}

function readNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function readNullableNumber(value: unknown): number | null | undefined {
  if (value === null) {
    return null;
  }
  return readNumber(value);
}

function readString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readConfig(value: unknown): RunStatus["config"] {
  if (!isRecord(value)) {
    return undefined;
  }
  return Object.fromEntries(Object.entries(value));
}

export function readEvent(line: string): Record<string, unknown> | null {
  try {
    const parsed: unknown = JSON.parse(line);
    if (!isRecord(parsed)) {
      return null;
    }
    return Object.fromEntries(Object.entries(parsed));
  } catch {
    return null;
  }
}

export function managerEvent(
  event: string,
  details: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    timestamp: nowIso(),
    type: "manager",
    event,
    ...details,
  };
}

let eventSeq = 0;

export function appendSupervisorEvent(runDirectory: string, event: Record<string, unknown>): void {
  appendEvent(runDirectory, {
    ...event,
    seq: ++eventSeq,
  });
}

function checkpointKindFromEvent(event: Record<string, unknown>): string | undefined {
  const kind = readString(event.kind);
  return kind === "snapshot" || kind === "resume" ? kind : undefined;
}

function supervisorStatusUpdate(status: RunStatus, event: Record<string, unknown>): RunStatus {
  const updatedAt = readString(event.timestamp) ?? nowIso();
  return {
    ...status,
    updatedAt,
    supervisorHeartbeatAt: updatedAt,
  };
}

function trainerStatusUpdate(status: RunStatus, event: Record<string, unknown>): RunStatus {
  const updatedAt = readString(event.timestamp) ?? nowIso();
  return {
    ...status,
    updatedAt,
    trainerHeartbeatAt: updatedAt,
  };
}

function markProgress(status: RunStatus, event: Record<string, unknown>): RunStatus {
  const updatedAt = readString(event.timestamp) ?? nowIso();
  return {
    ...status,
    lastProgressAt: updatedAt,
    stallReason: undefined,
    state: status.state === "stalled" ? "running" : status.state,
  };
}

function applyStartEvent(status: RunStatus, event: Record<string, unknown>): RunStatus {
  const next = markProgress(trainerStatusUpdate(status, event), event);
  const earlyStopPatience = readNullableNumber(event.earlyStopPatience);
  return {
    ...next,
    state: "running",
    preset: readString(event.preset) ?? status.preset,
    config: readConfig(event.config) ?? status.config,
    parameterCount: readNumber(event.params) ?? status.parameterCount,
    maxSteps: readNumber(event.maxSteps) ?? status.maxSteps,
    batchSize: readNumber(event.batchSize) ?? status.batchSize,
    gradAccumSteps: readNumber(event.gradAccumSteps) ?? status.gradAccumSteps,
    warmupSteps: readNumber(event.warmupSteps) ?? status.warmupSteps,
    step: readNumber(event.startStep) ?? status.step,
    resumeFrom: readString(event.resumeFrom) ?? status.resumeFrom,
    earlyStopPatience:
      earlyStopPatience === undefined ? status.earlyStopPatience : earlyStopPatience,
    earlyStopMinDelta: readNumber(event.earlyStopMinDelta) ?? status.earlyStopMinDelta,
    activeMemoryBytes: readNumber(event.activeMemoryBytes) ?? status.activeMemoryBytes,
    cacheMemoryBytes: readNumber(event.cacheMemoryBytes) ?? status.cacheMemoryBytes,
    peakMemoryBytes: readNumber(event.peakMemoryBytes) ?? status.peakMemoryBytes,
    memoryLimitBytes: readNumber(event.memoryLimitBytes) ?? status.memoryLimitBytes,
  };
}

function applyStepEvent(status: RunStatus, event: Record<string, unknown>): RunStatus {
  const next = markProgress(trainerStatusUpdate(status, event), event);
  return {
    ...next,
    step: readNumber(event.step) ?? status.step,
    lastStepLoss: readNumber(event.loss) ?? status.lastStepLoss,
    lastTokensPerSec: readNumber(event.tokensPerSec) ?? status.lastTokensPerSec,
    activeMemoryBytes: readNumber(event.activeMemoryBytes) ?? status.activeMemoryBytes,
    cacheMemoryBytes: readNumber(event.cacheMemoryBytes) ?? status.cacheMemoryBytes,
    peakMemoryBytes: readNumber(event.peakMemoryBytes) ?? status.peakMemoryBytes,
    memoryLimitBytes: readNumber(event.memoryLimitBytes) ?? status.memoryLimitBytes,
  };
}

function applyEvalEvent(status: RunStatus, event: Record<string, unknown>): RunStatus {
  const next = markProgress(trainerStatusUpdate(status, event), event);
  return {
    ...next,
    step: readNumber(event.step) ?? status.step,
    lastTrainLoss: readNumber(event.trainLoss) ?? status.lastTrainLoss,
    lastValLoss: readNumber(event.valLoss) ?? status.lastValLoss,
    activeMemoryBytes: readNumber(event.activeMemoryBytes) ?? status.activeMemoryBytes,
    cacheMemoryBytes: readNumber(event.cacheMemoryBytes) ?? status.cacheMemoryBytes,
    peakMemoryBytes: readNumber(event.peakMemoryBytes) ?? status.peakMemoryBytes,
    memoryLimitBytes: readNumber(event.memoryLimitBytes) ?? status.memoryLimitBytes,
  };
}

function applyCheckpointEvent(status: RunStatus, event: Record<string, unknown>): RunStatus {
  const next = markProgress(trainerStatusUpdate(status, event), event);
  const path = readString(event.path) ?? status.latestCheckpoint;
  const kind = checkpointKindFromEvent(event);
  return {
    ...next,
    step: readNumber(event.step) ?? status.step,
    latestCheckpoint: path,
    latestCheckpointKind: kind ?? status.latestCheckpointKind,
    latestSnapshotCheckpoint:
      kind === "snapshot" ? path : (status.latestSnapshotCheckpoint ?? undefined),
    latestResumeCheckpoint: kind === "resume" ? path : (status.latestResumeCheckpoint ?? undefined),
  };
}

function applyEarlyStopEvent(status: RunStatus, event: Record<string, unknown>): RunStatus {
  const next = markProgress(trainerStatusUpdate(status, event), event);
  const earlyStopPatience = readNullableNumber(event.patience);
  return {
    ...next,
    step: readNumber(event.step) ?? status.step,
    bestValLoss: readNumber(event.bestValLoss) ?? status.bestValLoss,
    bestCheckpointStep: readNumber(event.bestCheckpointStep) ?? status.bestCheckpointStep,
    bestCheckpoint: readString(event.bestCheckpointPath) ?? status.bestCheckpoint,
    earlyStopPatience:
      earlyStopPatience === undefined ? status.earlyStopPatience : earlyStopPatience,
    earlyStopMinDelta: readNumber(event.minDelta) ?? status.earlyStopMinDelta,
    earlyStopConsecutiveBadEvals:
      readNumber(event.consecutiveBadEvals) ?? status.earlyStopConsecutiveBadEvals,
    earlyStopReason: readString(event.reason) ?? status.earlyStopReason,
  };
}

function applyControlEvent(status: RunStatus, event: Record<string, unknown>): RunStatus {
  const next = markProgress(supervisorStatusUpdate(status, event), event);
  const command = readString(event.command);
  if (command !== "stop" && command !== "cancel") {
    return next;
  }
  return {
    ...next,
    state: command === "stop" ? "stopping" : "cancelling",
    controlCommand: command,
    controlRequestedAt: readString(event.requestedAt) ?? status.controlRequestedAt,
  };
}

function applyDoneEvent(status: RunStatus, event: Record<string, unknown>): RunStatus {
  const next = markProgress(trainerStatusUpdate(status, event), event);
  return {
    ...next,
    step: readNumber(event.totalSteps) ?? status.step,
  };
}

export function updateStatusFromEvent(
  status: RunStatus,
  event: Record<string, unknown>,
): RunStatus {
  switch (readString(event.type)) {
    case "start":
      return applyStartEvent(status, event);
    case "step":
      return applyStepEvent(status, event);
    case "eval":
      return applyEvalEvent(status, event);
    case "progress":
      return markProgress(trainerStatusUpdate(status, event), event);
    case "checkpoint":
      return applyCheckpointEvent(status, event);
    case "best-checkpoint":
      return {
        ...markProgress(trainerStatusUpdate(status, event), event),
        step: readNumber(event.step) ?? status.step,
        bestValLoss: readNumber(event.valLoss) ?? status.bestValLoss,
        bestCheckpointStep: readNumber(event.step) ?? status.bestCheckpointStep,
        bestCheckpoint: readString(event.path) ?? status.bestCheckpoint,
        earlyStopConsecutiveBadEvals: 0,
        earlyStopReason: undefined,
      };
    case "control":
      return applyControlEvent(status, event);
    case "early-stop":
      return applyEarlyStopEvent(status, event);
    case "done":
      return applyDoneEvent(status, event);
    default:
      return markProgress(trainerStatusUpdate(status, event), event);
  }
}

export function finalState(
  status: RunStatus,
  exitCode: number | null,
  signal: string | null,
): RunState {
  if (status.state === "cancelling") {
    return "cancelled";
  }
  if (status.state === "stopping") {
    return exitCode === 0 ? "stopped" : "failed";
  }
  if (exitCode === 0) {
    if (
      status.step !== undefined &&
      status.maxSteps !== undefined &&
      status.step >= status.maxSteps
    ) {
      return "completed";
    }
    return "stopped";
  }
  if (signal !== null) {
    return "failed";
  }
  return "failed";
}

export function applyPendingControl(status: RunStatus, runDirectory: string): RunStatus {
  const control = readRunControl(runDirectory);
  if (control === undefined) {
    return status;
  }
  if (
    status.controlCommand === control.command &&
    status.controlRequestedAt === control.requestedAt
  ) {
    return status;
  }
  return {
    ...status,
    state: control.command === "stop" ? "stopping" : "cancelling",
    controlCommand: control.command,
    controlRequestedAt: control.requestedAt,
    updatedAt: nowIso(),
  };
}

function controlAgeMs(status: RunStatus): number | null {
  if (status.controlRequestedAt === undefined) {
    return null;
  }
  return Date.now() - new Date(status.controlRequestedAt).getTime();
}

export function maybeEscalateTrainer(
  status: RunStatus,
  trainer: ReturnType<typeof Bun.spawn>,
): void {
  const ageMs = controlAgeMs(status);
  if (ageMs === null) {
    return;
  }
  if (ageMs >= KILL_ESCALATE_AFTER_MS) {
    trainer.kill("SIGKILL");
    return;
  }
  if (ageMs >= STOP_ESCALATE_AFTER_MS) {
    trainer.kill("SIGTERM");
  }
}

export function maybeMarkStalled(status: RunStatus, stallTimeoutMs: number): RunStatus {
  if (status.state !== "running" && status.state !== "stopping") {
    return status;
  }
  if (status.lastProgressAt === undefined) {
    return status;
  }

  const ageMs = Date.now() - new Date(status.lastProgressAt).getTime();
  if (ageMs < stallTimeoutMs) {
    return status;
  }

  return {
    ...status,
    state: "stalled",
    updatedAt: nowIso(),
    stallReason: `No progress event for ${Math.round(ageMs / 1000)}s`,
  };
}

export function finishSupervisorRun(
  runDirectory: string,
  status: RunStatus,
  exitCode: number | null,
  signal: string | null,
): RunStatus {
  clearRunControl(runDirectory);
  return {
    ...status,
    state: finalState(status, exitCode, signal),
    updatedAt: nowIso(),
    supervisorHeartbeatAt: nowIso(),
    exitCode,
    signal,
  };
}
