#!/usr/bin/env bun

import { createWriteStream } from "fs";
import type { GPTConfig } from "../config";
import {
  appendEvent,
  clearRunControl,
  DEFAULT_STALL_TIMEOUT_SECONDS,
  type RunState,
  type RunStatus,
  readRunControl,
  readRunSpec,
  stderrPath,
  writePid,
  writeRunStatus,
} from "./files";

const HEARTBEAT_INTERVAL_MS = 30_000;
const CONTROL_CHECK_INTERVAL_MS = 5_000;
const STOP_ESCALATE_AFTER_MS = 60_000;
const KILL_ESCALATE_AFTER_MS = 90_000;

function nowIso(): string {
  return new Date().toISOString();
}

function readEvent(line: string): Record<string, unknown> | null {
  try {
    const parsed: unknown = JSON.parse(line);
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      return null;
    }
    return Object.fromEntries(Object.entries(parsed));
  } catch {
    return null;
  }
}

function readNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function readString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readBoolean(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

function readConfig(value: unknown): GPTConfig | undefined {
  if (!isRecord(value)) {
    return undefined;
  }

  const nLayer = readNumber(value.nLayer);
  const nHead = readNumber(value.nHead);
  const nEmbd = readNumber(value.nEmbd);
  const blockSize = readNumber(value.blockSize);
  const dropout = readNumber(value.dropout);
  const gradientCheckpointing = readBoolean(value.gradientCheckpointing);
  const vocabSize = readNumber(value.vocabSize);

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

function managerEvent(
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

function appendSupervisorEvent(runDirectory: string, event: Record<string, unknown>): void {
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

function applyProgressEvent(status: RunStatus, event: Record<string, unknown>): RunStatus {
  const next = markProgress(trainerStatusUpdate(status, event), event);
  return {
    ...next,
    step: readNumber(event.step) ?? status.step,
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

function updateStatusFromEvent(status: RunStatus, event: Record<string, unknown>): RunStatus {
  switch (readString(event.type)) {
    case "start":
      return applyStartEvent(status, event);
    case "step":
      return applyStepEvent(status, event);
    case "eval":
      return applyEvalEvent(status, event);
    case "progress":
      return applyProgressEvent(status, event);
    case "checkpoint":
      return applyCheckpointEvent(status, event);
    case "control":
      return applyControlEvent(status, event);
    case "done":
      return applyDoneEvent(status, event);
    default:
      return markProgress(trainerStatusUpdate(status, event), event);
  }
}

function finalState(status: RunStatus, exitCode: number | null, signal: string | null): RunState {
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

function applyPendingControl(status: RunStatus, runDirectory: string): RunStatus {
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

function maybeEscalateTrainer(status: RunStatus, trainer: ReturnType<typeof Bun.spawn>): void {
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

function maybeMarkStalled(status: RunStatus, stallTimeoutMs: number): RunStatus {
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

function appendTrainerEventLine(
  runDirectory: string,
  line: string,
  onEvent: (event: Record<string, unknown>) => void,
): void {
  const event = readEvent(line);
  if (event === null) {
    appendSupervisorEvent(runDirectory, managerEvent("trainer-nonjson", { line }));
    return;
  }

  appendSupervisorEvent(runDirectory, event);
  onEvent(event);
}

function drainTrainerBuffer(
  runDirectory: string,
  buffer: string,
  onEvent: (event: Record<string, unknown>) => void,
): string {
  let newlineIndex = buffer.indexOf("\n");
  while (newlineIndex >= 0) {
    const line = buffer.slice(0, newlineIndex).replace(/\r$/, "");
    buffer = buffer.slice(newlineIndex + 1);
    if (line.length > 0) {
      appendTrainerEventLine(runDirectory, line, onEvent);
    }
    newlineIndex = buffer.indexOf("\n");
  }

  return buffer;
}

async function pipeTextStream(
  stream: ReadableStream<Uint8Array>,
  sink: (chunk: string) => void,
): Promise<void> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        const tail = decoder.decode();
        if (tail.length > 0) {
          sink(tail);
        }
        return;
      }
      if (value.byteLength > 0) {
        sink(decoder.decode(value, { stream: true }));
      }
    }
  } finally {
    reader.releaseLock();
  }
}

async function pumpTrainerStdout(
  stream: ReadableStream<Uint8Array>,
  runDirectory: string,
  onEvent: (event: Record<string, unknown>) => void,
): Promise<void> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        buffer += decoder.decode();
        break;
      }
      if (value.byteLength > 0) {
        buffer += decoder.decode(value, { stream: true });
      }
      buffer = drainTrainerBuffer(runDirectory, buffer, onEvent);
    }

    const tail = buffer.trim();
    if (tail.length > 0) {
      appendTrainerEventLine(runDirectory, tail, onEvent);
    }
  } finally {
    reader.releaseLock();
  }
}

export async function main(argv = process.argv): Promise<void> {
  const runDirectory = (() => {
    const index = argv.indexOf("--run-dir");
    return index >= 0 ? argv[index + 1] : undefined;
  })();
  if (runDirectory === undefined) {
    throw new Error("Missing required flag --run-dir");
  }

  const spec = readRunSpec(runDirectory);
  const stallTimeoutMs = (spec.stallTimeoutSeconds ?? DEFAULT_STALL_TIMEOUT_SECONDS) * 1000;
  const stderrStream = createWriteStream(stderrPath(runDirectory), { flags: "a" });
  let status: RunStatus = {
    runId: spec.runId,
    state: "starting",
    startedAt: nowIso(),
    updatedAt: nowIso(),
    supervisorHeartbeatAt: nowIso(),
    trainerHeartbeatAt: nowIso(),
    lastProgressAt: nowIso(),
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
        supervisorHeartbeatAt: nowIso(),
      },
      stallTimeoutMs,
    );
    writeRunStatus(runDirectory, status);
  }, HEARTBEAT_INTERVAL_MS);

  const controlTimer = setInterval(() => {
    const control = readRunControl(runDirectory);
    if (control !== undefined) {
      if (
        status.controlCommand !== control.command ||
        status.controlRequestedAt !== control.requestedAt
      ) {
        status = {
          ...status,
          state: control.command === "stop" ? "stopping" : "cancelling",
          controlCommand: control.command,
          controlRequestedAt: control.requestedAt,
          updatedAt: nowIso(),
        };
        writeRunStatus(runDirectory, status);
        appendSupervisorEvent(
          runDirectory,
          managerEvent(control.command === "stop" ? "stop-requested" : "cancel-requested", {
            requestedAt: control.requestedAt,
          }),
        );
      }
      maybeEscalateTrainer(status, trainer);
    }
  }, CONTROL_CHECK_INTERVAL_MS);

  const exitResult = await exitPromise;
  await Promise.all([stdoutPump, stderrPump]);

  clearInterval(heartbeat);
  clearInterval(controlTimer);
  stderrStream.end();
  status = applyPendingControl(status, runDirectory);
  clearRunControl(runDirectory);

  status = {
    ...status,
    state: finalState(status, exitResult.code, exitResult.signal),
    updatedAt: nowIso(),
    supervisorHeartbeatAt: nowIso(),
    exitCode: exitResult.code,
    signal: exitResult.signal,
  };
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
