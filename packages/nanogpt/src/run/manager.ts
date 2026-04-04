#!/usr/bin/env bun

import { existsSync } from "fs";
import { resolve } from "path";
import type { GPTConfig } from "../config";
import {
  checkpointsDir,
  DEFAULT_STALL_TIMEOUT_SECONDS,
  deriveOperatorHealth,
  ensureRunDir,
  type RunControlCommand,
  type RunSpec,
  type RunStatus,
  readLatestCheckpoint,
  readRunControl,
  readRunSpec,
  readRunStatus,
  runDir,
  writeRunControl,
  writeRunSpec,
  writeRunStatus,
} from "./files";

const TRAIN_FLAG_ALLOWLIST = new Set([
  "preset",
  "gradient-checkpointing",
  "data",
  "max-steps",
  "batch-size",
  "grad-accum",
  "eval-interval",
  "eval-steps",
  "log-interval",
  "lr",
  "weight-decay",
  "max-grad-norm",
  "warmup-steps",
  "min-lr",
  "seed",
  "resume",
  "warm-start",
  "checkpoint-dir",
  "snapshot-interval",
  "resume-interval",
  "sample-interval",
  "sample-tokens",
  "early-stop-patience",
  "early-stop-min-delta",
  "memory-limit-mb",
  "cache-limit-mb",
  "wired-limit-mb",
  "json",
  "help",
]);

const START_FLAG_ALLOWLIST = new Set([...TRAIN_FLAG_ALLOWLIST, "name", "stall-timeout-sec"]);
const RESUME_FLAG_ALLOWLIST = new Set([
  ...TRAIN_FLAG_ALLOWLIST,
  "name",
  "from",
  "stall-timeout-sec",
]);
const STATUS_FLAG_ALLOWLIST = new Set(["name", "json", "help"]);
const WATCH_FLAG_ALLOWLIST = new Set(["name", "json", "interval", "help"]);
const CONTROL_FLAG_ALLOWLIST = new Set(["name", "help"]);

function usage(): string {
  return `nanogpt run manager

Usage:
  bun run packages/nanogpt/src/run/manager.ts start [train flags...]
  bun run packages/nanogpt/src/run/manager.ts resume --from <run-id> [train flags...]
  bun run packages/nanogpt/src/run/manager.ts status --name <run-id> [--json]
  bun run packages/nanogpt/src/run/manager.ts watch --name <run-id> [--interval <seconds>] [--json]
  bun run packages/nanogpt/src/run/manager.ts stop --name <run-id>
  bun run packages/nanogpt/src/run/manager.ts cancel --name <run-id>

Notes:
  start/resume accept --stall-timeout-sec <seconds> (default 600)
  train flags also accept --early-stop-patience <n|none> and --early-stop-min-delta <n>
  cancel is best-effort and may lose work since the latest resume checkpoint
`;
}

function nowIso(): string {
  return new Date().toISOString();
}

function nowStamp(): string {
  return nowIso().replace(/[:.]/g, "-");
}

function repoRoot(): string {
  return resolve(import.meta.dir, "../../../../");
}

function packageRoot(): string {
  return resolve(import.meta.dir, "../..");
}

function parseArgs(argv: string[]): { command: string; args: string[] } {
  return {
    command: argv[2] ?? "help",
    args: argv.slice(3),
  };
}

function validateAllowedFlags(args: string[], allowed: ReadonlySet<string>, context: string): void {
  for (let index = 0; index < args.length; index++) {
    const value = args[index];
    if (value === undefined || !value.startsWith("--")) {
      continue;
    }
    const key = value.slice(2);
    if (!allowed.has(key)) {
      throw new Error(`${context}: unknown flag --${key}`);
    }
    if (index + 1 < args.length && !args[index + 1]?.startsWith("--")) {
      index += 1;
    }
  }
}

function getFlag(args: string[], key: string, fallback?: string): string | undefined {
  for (let index = 0; index < args.length; index++) {
    if (args[index] === `--${key}`) {
      return args[index + 1] ?? fallback;
    }
  }
  return fallback;
}

function hasFlag(args: string[], key: string): boolean {
  return args.includes(`--${key}`);
}

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

function absolutizeTrainerPaths(args: string[]): string[] {
  const normalized = [...args];
  for (let index = 0; index < normalized.length; index++) {
    const key = normalized[index];
    const value = normalized[index + 1];
    if ((key === "--data" || key === "--resume" || key === "--warm-start") && value !== undefined) {
      normalized[index + 1] = resolve(process.cwd(), value);
    }
  }
  return normalized;
}

function trainerArgsFrom(args: string[]): string[] {
  let trainerArgs = stripFlag(args, "name");
  trainerArgs = stripFlag(trainerArgs, "from");
  trainerArgs = stripFlag(trainerArgs, "interval");
  trainerArgs = stripFlag(trainerArgs, "json", false);
  trainerArgs = stripFlag(trainerArgs, "run-dir");
  trainerArgs = stripFlag(trainerArgs, "stall-timeout-sec");
  trainerArgs = stripFlag(trainerArgs, "help", false);
  return absolutizeTrainerPaths(trainerArgs);
}

function generateRunId(args: string[]): string {
  const preset = getFlag(args, "preset");
  const label = preset ?? (getFlag(args, "resume") === undefined ? "gpt" : "resume");
  return `${nowStamp()}-${label}`;
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

function startRun(args: string[], resumedFrom?: string): void {
  validateAllowedFlags(args, START_FLAG_ALLOWLIST, "start");
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

function readStatusSummary(runId: string): {
  directory: string;
  spec: RunSpec;
  status: RunStatus;
  metrics: { processState?: string | undefined; rssMb?: number | undefined };
} {
  const directory = resolveExistingRun(runId);
  const spec = readRunSpec(directory);
  const status = readRunStatus(directory);
  const metrics = processMetrics(status.trainerPid ?? status.supervisorPid);
  return { directory, spec, status, metrics };
}

type StatusPayload = {
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
  config?: GPTConfig | undefined;
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

function createStatusPayload(runId: string): StatusPayload {
  const { directory, spec, status, metrics } = readStatusSummary(runId);
  const control = readRunControl(directory);
  const health = deriveOperatorHealth(status);
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

function formatGradientCheckpointing(config: GPTConfig | undefined): string {
  if (config?.gradientCheckpointing === undefined) {
    return "-";
  }
  return String(config.gradientCheckpointing);
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

function statusLines(payload: StatusPayload): string[] {
  return [
    `Run ${payload.runId}`,
    `  state: ${payload.state}`,
    formatOperatorHealthLine(payload),
    `  step: ${payload.step ?? "-"} / ${payload.maxSteps ?? "-"}`,
    `  batch: ${payload.batchSize ?? "-"}  grad accum: ${payload.gradAccumSteps ?? "-"}  gradient checkpointing: ${formatGradientCheckpointing(payload.config)}`,
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

function formatStatusPayload(payload: StatusPayload): string {
  return `${statusLines(payload).join("\n")}\n`;
}

function printStatus(runId: string, asJson: boolean): void {
  const payload = createStatusPayload(runId);
  if (asJson) {
    process.stdout.write(`${JSON.stringify(payload)}\n`);
    return;
  }
  process.stdout.write(formatStatusPayload(payload));
}

async function watchRun(runId: string, intervalSeconds: number, asJson: boolean): Promise<void> {
  while (true) {
    printStatus(runId, asJson);
    await Bun.sleep(intervalSeconds * 1000);
  }
}

function handleStart(args: string[]): void {
  validateAllowedFlags(args, START_FLAG_ALLOWLIST, "start");
  if (hasFlag(args, "help")) {
    process.stdout.write(usage());
    return;
  }
  startRun(args);
}

function handleResume(args: string[]): void {
  validateAllowedFlags(args, RESUME_FLAG_ALLOWLIST, "resume");
  if (hasFlag(args, "help")) {
    process.stdout.write(usage());
    return;
  }
  resumeRun(args);
}

function handleStatus(args: string[]): void {
  validateAllowedFlags(args, STATUS_FLAG_ALLOWLIST, "status");
  if (hasFlag(args, "help")) {
    process.stdout.write(usage());
    return;
  }
  const runId = getFlag(args, "name");
  if (runId === undefined) {
    throw new Error("status requires --name <run-id>");
  }
  printStatus(runId, hasFlag(args, "json"));
}

async function handleWatch(args: string[]): Promise<void> {
  validateAllowedFlags(args, WATCH_FLAG_ALLOWLIST, "watch");
  if (hasFlag(args, "help")) {
    process.stdout.write(usage());
    return;
  }
  const runId = getFlag(args, "name");
  if (runId === undefined) {
    throw new Error("watch requires --name <run-id>");
  }
  const interval = Number(getFlag(args, "interval") ?? "10");
  if (!Number.isFinite(interval) || interval <= 0) {
    throw new Error("watch requires --interval to be a positive number");
  }
  await watchRun(runId, interval, hasFlag(args, "json"));
}

function handleStop(args: string[]): void {
  validateAllowedFlags(args, CONTROL_FLAG_ALLOWLIST, "stop");
  if (hasFlag(args, "help")) {
    process.stdout.write(usage());
    return;
  }
  const runId = getFlag(args, "name");
  if (runId === undefined) {
    throw new Error("stop requires --name <run-id>");
  }
  writeControl(runId, "stop");
}

function handleCancel(args: string[]): void {
  validateAllowedFlags(args, CONTROL_FLAG_ALLOWLIST, "cancel");
  if (hasFlag(args, "help")) {
    process.stdout.write(usage());
    return;
  }
  const runId = getFlag(args, "name");
  if (runId === undefined) {
    throw new Error("cancel requires --name <run-id>");
  }
  writeControl(runId, "cancel");
}

function writeControl(runId: string, command: RunControlCommand): void {
  const { directory, status } = readStatusSummary(runId);
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

function resumeRun(args: string[]): void {
  validateAllowedFlags(args, RESUME_FLAG_ALLOWLIST, "resume");
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

export async function main(argv = process.argv): Promise<void> {
  const { command, args } = parseArgs(argv);
  if (command === "help" || hasFlag(args, "help")) {
    process.stdout.write(usage());
    return;
  }

  const handlers: Record<string, (commandArgs: string[]) => void | Promise<void>> = {
    start: handleStart,
    resume: handleResume,
    status: handleStatus,
    watch: handleWatch,
    stop: handleStop,
    cancel: handleCancel,
  };

  const handler = handlers[command];
  if (handler === undefined) {
    process.stdout.write(usage());
    return;
  }

  await handler(args);
}

await main();
