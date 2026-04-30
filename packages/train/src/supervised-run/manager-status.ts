import { existsSync } from "fs";

import type { RunStatus } from "./files";
import {
  deriveOperatorHealth,
  type RunControlCommand,
  readLatestCheckpoint,
  readRunControl,
  readRunSpec,
  readRunStatus,
  runDir,
} from "./files";

export type SupervisedRunStatusOptions = {
  repoRoot: string;
  runsDirectoryName?: string | undefined;
  formatBatchLine?: ((payload: StatusPayload) => string) | undefined;
  stdout?: ((text: string) => void) | undefined;
};

function writeStdout(options: Pick<SupervisedRunStatusOptions, "stdout">, text: string): void {
  const stdout = options.stdout ?? ((chunk: string) => process.stdout.write(chunk));
  stdout(text);
}

function processMetrics(pid: number | undefined): {
  processState?: string | undefined;
  rssMb?: number | undefined;
} {
  if (pid === undefined) {
    return {};
  }

  const result = Bun.spawnSync(["ps", "-o", "state=,rss=", "-p", String(pid)], {
    stdout: "pipe",
    stderr: "pipe",
  });
  if (result.exitCode !== 0) {
    return {};
  }

  const [stateRaw = "", rssRaw = ""] = new TextDecoder()
    .decode(result.stdout)
    .trim()
    .split(/\s+/, 2);
  const rssKb = Number(rssRaw);
  return {
    processState: stateRaw.trim() || undefined,
    rssMb: Number.isFinite(rssKb) ? rssKb / 1024 : undefined,
  };
}

function resolveExistingRun(runId: string, options: SupervisedRunStatusOptions): string {
  const directory = runDir(options.repoRoot, runId, options.runsDirectoryName);
  if (!existsSync(directory)) {
    throw new Error(`Unknown run "${runId}"`);
  }
  return directory;
}

export type StatusPayload = {
  runId: string;
  directory: string;
  state: RunStatus["state"];
  startedAt: string;
  updatedAt: string;
  supervisorHeartbeatAt: string;
  trainerHeartbeatAt?: string | undefined;
  lastProgressAt?: string | undefined;
  supervisorPid?: number | undefined;
  trainerPid?: number | undefined;
  preset?: string | undefined;
  config?: RunStatus["config"] | undefined;
  parameterCount?: number | undefined;
  step?: number | undefined;
  maxSteps?: number | undefined;
  batchSize?: number | undefined;
  gradAccumSteps?: number | undefined;
  warmupSteps?: number | undefined;
  stallTimeoutSeconds?: number | undefined;
  lastStepLoss?: number | undefined;
  lastTrainLoss?: number | undefined;
  lastValLoss?: number | undefined;
  bestValLoss?: number | undefined;
  lastTokensPerSec?: number | undefined;
  latestCheckpoint?: string | undefined;
  latestSnapshotCheckpoint?: string | undefined;
  latestResumeCheckpoint?: string | undefined;
  bestCheckpoint?: string | undefined;
  bestCheckpointStep?: number | undefined;
  latestCheckpointKind?: string | undefined;
  activeMemoryBytes?: number | undefined;
  cacheMemoryBytes?: number | undefined;
  peakMemoryBytes?: number | undefined;
  memoryLimitBytes?: number | undefined;
  earlyStopPatience?: number | null | undefined;
  earlyStopMinDelta?: number | undefined;
  earlyStopConsecutiveBadEvals?: number | undefined;
  earlyStopReason?: string | undefined;
  exitCode?: number | null | undefined;
  signal?: string | null | undefined;
  supervisorAlive?: boolean | undefined;
  trainerAlive?: boolean | undefined;
  operatorHealth?: string | undefined;
  processState?: string | undefined;
  rssMb?: number | undefined;
  trainerArgs: string[];
  resumeFrom?: string | undefined;
  controlCommand?: RunControlCommand | undefined;
  controlRequestedAt?: string | undefined;
  stallReason?: string | undefined;
};

export function createStatusPayload(
  runId: string,
  options: SupervisedRunStatusOptions,
): StatusPayload {
  const directory = resolveExistingRun(runId, options);
  const spec = readRunSpec(directory);
  const control = readRunControl(directory);
  const status = readRunStatus(directory);
  const health = deriveOperatorHealth(status);
  const metrics = processMetrics(status.trainerPid ?? status.supervisorPid);

  return {
    runId,
    directory,
    state: status.state,
    startedAt: status.startedAt,
    updatedAt: status.updatedAt,
    supervisorHeartbeatAt: status.supervisorHeartbeatAt,
    trainerHeartbeatAt: status.trainerHeartbeatAt,
    lastProgressAt: status.lastProgressAt,
    supervisorPid: status.supervisorPid,
    trainerPid: status.trainerPid,
    preset: status.preset,
    config: status.config,
    parameterCount: status.parameterCount,
    step: status.step,
    maxSteps: status.maxSteps,
    batchSize: status.batchSize,
    gradAccumSteps: status.gradAccumSteps,
    warmupSteps: status.warmupSteps,
    stallTimeoutSeconds: status.stallTimeoutSeconds ?? spec.stallTimeoutSeconds,
    lastStepLoss: status.lastStepLoss,
    lastTrainLoss: status.lastTrainLoss,
    lastValLoss: status.lastValLoss,
    bestValLoss: status.bestValLoss,
    lastTokensPerSec: status.lastTokensPerSec,
    latestCheckpoint: status.latestCheckpoint ?? readLatestCheckpoint(directory),
    latestSnapshotCheckpoint: status.latestSnapshotCheckpoint,
    latestResumeCheckpoint: status.latestResumeCheckpoint,
    bestCheckpoint: status.bestCheckpoint,
    bestCheckpointStep: status.bestCheckpointStep,
    latestCheckpointKind: status.latestCheckpointKind,
    activeMemoryBytes: status.activeMemoryBytes,
    cacheMemoryBytes: status.cacheMemoryBytes,
    peakMemoryBytes: status.peakMemoryBytes,
    memoryLimitBytes: status.memoryLimitBytes,
    earlyStopPatience: status.earlyStopPatience,
    earlyStopMinDelta: status.earlyStopMinDelta,
    earlyStopConsecutiveBadEvals: status.earlyStopConsecutiveBadEvals,
    earlyStopReason: status.earlyStopReason,
    exitCode: status.exitCode,
    signal: status.signal,
    supervisorAlive: health.supervisorAlive,
    trainerAlive: health.trainerAlive,
    operatorHealth: health.operatorHealth,
    processState: metrics.processState,
    rssMb: metrics.rssMb,
    trainerArgs: spec.trainerArgs,
    resumeFrom: status.resumeFrom ?? spec.resumedFrom,
    controlCommand: control?.command ?? status.controlCommand,
    controlRequestedAt: control?.requestedAt ?? status.controlRequestedAt,
    stallReason: status.stallReason,
  };
}

function formatMemoryBytes(bytes: number | undefined): string {
  return bytes === undefined ? "-" : `${Math.round(bytes / (1024 * 1024)).toLocaleString()} MB`;
}

function formatOptionalNumber(value: number | undefined, digits = 4): string {
  return value === undefined ? "-" : value.toFixed(digits);
}

function formatTokensPerSec(value: number | undefined): string {
  return value === undefined ? "-" : Math.round(value).toLocaleString();
}

function formatEarlyStopPatience(value: number | null | undefined): string {
  if (value === undefined) {
    return "-";
  }
  return value === null ? "disabled" : String(value);
}

function formatOperatorHealthLine(payload: StatusPayload): string {
  const supervisor = payload.supervisorAlive ? "alive" : "dead";
  const trainer = payload.trainerAlive ? "alive" : "dead";
  return `  operator: ${payload.operatorHealth ?? "-"} (supervisor ${supervisor}, trainer ${trainer})`;
}

function formatBestValLine(payload: StatusPayload): string {
  return `  best val: ${formatOptionalNumber(payload.bestValLoss)}  best step: ${payload.bestCheckpointStep ?? "-"}`;
}

function formatEarlyStopLine(payload: StatusPayload): string {
  return `  early stop: ${formatEarlyStopPatience(payload.earlyStopPatience)}  min delta: ${formatOptionalNumber(payload.earlyStopMinDelta)}`;
}

function defaultBatchLine(payload: StatusPayload): string {
  return `  batch: ${payload.batchSize ?? "-"}  grad accum: ${payload.gradAccumSteps ?? "-"}`;
}

function statusLines(payload: StatusPayload, options: SupervisedRunStatusOptions): string[] {
  return [
    `Run ${payload.runId}`,
    `  state: ${payload.state}`,
    formatOperatorHealthLine(payload),
    `  step: ${payload.step ?? "-"} / ${payload.maxSteps ?? "-"}`,
    options.formatBatchLine?.(payload) ?? defaultBatchLine(payload),
    `  stall timeout: ${payload.stallTimeoutSeconds ?? "-"}s`,
    `  loss: ${formatOptionalNumber(payload.lastStepLoss)}  val: ${formatOptionalNumber(payload.lastValLoss)}`,
    formatBestValLine(payload),
    `  tokens/sec: ${formatTokensPerSec(payload.lastTokensPerSec)}`,
    formatEarlyStopLine(payload),
    `  active memory: ${formatMemoryBytes(payload.activeMemoryBytes)}`,
    `  cache memory: ${formatMemoryBytes(payload.cacheMemoryBytes)}`,
    `  peak memory: ${formatMemoryBytes(payload.peakMemoryBytes)}`,
    `  rss: ${payload.rssMb === undefined ? "-" : payload.rssMb.toFixed(1)} MB`,
    `  process: ${payload.processState ?? "-"}`,
    `  exit: ${payload.exitCode ?? "-"}  signal: ${payload.signal ?? "-"}`,
    `  latest checkpoint: ${payload.latestCheckpoint ?? "-"}`,
    `  best checkpoint: ${payload.bestCheckpoint ?? "-"}`,
    `  early-stop reason: ${payload.earlyStopReason ?? "-"}`,
  ];
}

export function formatStatusPayload(
  payload: StatusPayload,
  options: SupervisedRunStatusOptions,
): string {
  return `${statusLines(payload, options).join("\n")}\n`;
}

export function printStatus(
  runId: string,
  asJson: boolean,
  options: SupervisedRunStatusOptions,
): void {
  const payload = createStatusPayload(runId, options);
  if (asJson) {
    writeStdout(options, `${JSON.stringify(payload)}\n`);
    return;
  }
  writeStdout(options, formatStatusPayload(payload, options));
}

export async function watchRun(
  runId: string,
  intervalSeconds: number,
  asJson: boolean,
  options: SupervisedRunStatusOptions,
): Promise<void> {
  while (true) {
    printStatus(runId, asJson, options);
    await Bun.sleep(intervalSeconds * 1000);
  }
}
