#!/usr/bin/env bun

import { readFileSync } from "fs";
import { resolve } from "path";
import { applyCheckpoint, loadCheckpoint } from "../checkpoint";
import { estimateParameterCount, GPT_SMALL, GPT_TINY, resolveConfig } from "../config";
import { generate } from "../generate";
import { GPT } from "../model/gpt";
import { CharTokenizer } from "../tokenizer";
import {
  DEFAULT_STALL_TIMEOUT_SECONDS,
  deriveOperatorHealth,
  eventsPath,
  type RunStatus,
  readRunStatus,
  repoRootFromPackageRoot,
  runDir,
} from "./files";

type PresetName = "gpt-tiny" | "gpt-small";
type RunMode = "acceptance" | "soak";
type AcceptanceDefaults = {
  gradAccumSteps: number;
  batchSize: number;
  maxSteps: number;
  evalInterval: number;
  evalSteps: number;
  learningRate: number;
  weightDecay: number;
  maxGradNorm: number | null;
  warmupSteps: number;
  minLearningRate: number;
  logInterval: number;
  lossTarget: number;
  snapshotInterval: number;
  resumeInterval: number;
  stallTimeoutSeconds: number;
};

type AcceptanceRunOptions = {
  presetName: PresetName;
  mode: RunMode;
  runId: string;
  pollSeconds: number;
  lossTarget: number;
  throughputWindow: number;
  minThroughputRatio: number;
  maxSlopeMbPerEvent: number;
  stallTimeoutSeconds: number;
  parameterCount: number;
  args: string[];
};

export type StepEventRecord = {
  type: "step";
  step: number;
  tokensPerSec: number;
  activeMemoryBytes?: number | undefined;
};

const ACCEPTANCE_DEFAULTS: Record<PresetName, AcceptanceDefaults> = {
  "gpt-tiny": {
    maxSteps: 5000,
    batchSize: 4,
    gradAccumSteps: 1,
    evalInterval: 250,
    evalSteps: 20,
    logInterval: 25,
    learningRate: 3e-4,
    weightDecay: 0.1,
    maxGradNorm: 1,
    warmupSteps: 250,
    minLearningRate: 3e-5,
    lossTarget: 1.5,
    snapshotInterval: 250,
    resumeInterval: 1000,
    stallTimeoutSeconds: DEFAULT_STALL_TIMEOUT_SECONDS,
  },
  "gpt-small": {
    maxSteps: 5000,
    batchSize: 1,
    gradAccumSteps: 8,
    evalInterval: 250,
    evalSteps: 20,
    logInterval: 25,
    learningRate: 3e-4,
    weightDecay: 0.1,
    maxGradNorm: 1,
    warmupSteps: 250,
    minLearningRate: 3e-5,
    lossTarget: 1.5,
    snapshotInterval: 250,
    resumeInterval: 1000,
    stallTimeoutSeconds: DEFAULT_STALL_TIMEOUT_SECONDS,
  },
};

const ACCEPTANCE_FLAG_ALLOWLIST = new Set([
  "mode",
  "preset",
  "gradient-checkpointing",
  "name",
  "poll-seconds",
  "loss-target",
  "throughput-window",
  "min-throughput-ratio",
  "max-slope-mb-per-event",
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
  "snapshot-interval",
  "resume-interval",
  "early-stop-patience",
  "early-stop-min-delta",
  "stall-timeout-sec",
  "data",
  "memory-limit-mb",
  "cache-limit-mb",
  "wired-limit-mb",
  "json",
  "help",
]);

function packageRoot(): string {
  return resolve(import.meta.dir, "../..");
}

function repoRoot(): string {
  return repoRootFromPackageRoot(packageRoot());
}

export function parseArgs(argv: string[]): Map<string, string> {
  const flags = new Map<string, string>();
  for (let index = 2; index < argv.length; index++) {
    const argument = argv[index];
    if (argument === undefined || !argument.startsWith("--")) {
      continue;
    }
    const key = argument.slice(2);
    const next = argv[index + 1];
    if (next !== undefined && !next.startsWith("--")) {
      flags.set(key, next);
      index += 1;
      continue;
    }
    flags.set(key, "true");
  }
  return flags;
}

function validateAllowedFlags(
  flags: Map<string, string>,
  allowed: ReadonlySet<string>,
  context: string,
): void {
  for (const key of flags.keys()) {
    if (!allowed.has(key)) {
      throw new Error(`${context}: unknown flag --${key}`);
    }
  }
}

function getFlag(flags: Map<string, string>, key: string, fallback?: string): string | undefined {
  return flags.get(key) ?? fallback;
}

export function getNumberFlag(flags: Map<string, string>, key: string, fallback: number): number {
  const raw = flags.get(key);
  if (raw === undefined) {
    return fallback;
  }
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) {
    throw new Error(`Flag --${key} must be a finite number`);
  }
  return parsed;
}

export function readPresetName(flags: Map<string, string>): PresetName {
  const value = getFlag(flags, "preset", "gpt-tiny");
  if (value === "gpt-tiny" || value === "gpt-small") {
    return value;
  }
  throw new Error(`Unknown preset "${value}". Expected gpt-tiny or gpt-small.`);
}

export function readMode(flags: Map<string, string>): RunMode {
  const value = getFlag(flags, "mode", "acceptance");
  if (value === "acceptance" || value === "soak") {
    return value;
  }
  throw new Error(`Unknown mode "${value}". Expected acceptance or soak.`);
}

function parseJsonLine(line: string): unknown {
  return JSON.parse(line);
}

function readJsonLines(content: string): unknown[] {
  if (content.trim() === "") {
    return [];
  }

  const lines = content.split("\n");
  const parsed: unknown[] = [];
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index];
    if (line === undefined || line.trim() === "") {
      continue;
    }
    try {
      parsed.push(parseJsonLine(line));
    } catch (error) {
      if (index === lines.length - 1) {
        break;
      }
      throw error;
    }
  }
  return parsed;
}

function isStepEventRecord(value: unknown): value is StepEventRecord {
  return (
    typeof value === "object" &&
    value !== null &&
    "type" in value &&
    value.type === "step" &&
    "step" in value &&
    typeof value.step === "number" &&
    "tokensPerSec" in value &&
    typeof value.tokensPerSec === "number"
  );
}

export function readStepEvents(runId: string): StepEventRecord[] {
  const eventLog = eventsPath(runDir(repoRoot(), runId));
  const stepEvents: StepEventRecord[] = [];
  for (const parsed of readJsonLines(readFileSync(eventLog, "utf-8"))) {
    if (isStepEventRecord(parsed)) {
      stepEvents.push(parsed);
    }
  }
  return stepEvents;
}

function average(values: readonly number[]): number {
  if (values.length === 0) {
    throw new Error("Cannot average an empty sequence");
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function decodeOutput(bytes: Uint8Array): string {
  return new TextDecoder().decode(bytes);
}

export function assertSoakStabilityForEvents(
  runId: string,
  stepEvents: readonly StepEventRecord[],
  sampleSize: number,
  minRatio: number,
  maxSlopeMbPerEvent: number,
): {
  throughputRatio: number;
  firstAverage: number;
  lastAverage: number;
  slopeMbPerEvent: number;
} {
  if (stepEvents.length < sampleSize * 2) {
    throw new Error(
      `Soak run ${runId} recorded only ${stepEvents.length} step events; need at least ${
        sampleSize * 2
      } to judge throughput drift.`,
    );
  }

  const firstAverage = average(stepEvents.slice(0, sampleSize).map((event) => event.tokensPerSec));
  const lastAverage = average(stepEvents.slice(-sampleSize).map((event) => event.tokensPerSec));
  const ratio = lastAverage / firstAverage;

  if (ratio < minRatio) {
    throw new Error(
      `Soak failed: throughput ratio ${ratio.toFixed(3)} is below the minimum ${minRatio.toFixed(3)} ` +
        `(first ${Math.round(firstAverage).toLocaleString()} tok/s, last ${Math.round(lastAverage).toLocaleString()} tok/s).`,
    );
  }

  const memoryEvents = stepEvents.filter((event) => event.activeMemoryBytes !== undefined);
  if (memoryEvents.length < 2) {
    throw new Error(`Soak run ${runId} did not record enough memory-bearing step events.`);
  }

  const firstMemory = memoryEvents[0];
  const lastMemory = memoryEvents[memoryEvents.length - 1];
  if (firstMemory?.activeMemoryBytes === undefined || lastMemory?.activeMemoryBytes === undefined) {
    throw new Error(`Soak run ${runId} did not record valid active-memory samples.`);
  }

  const slopeMbPerEvent =
    (lastMemory.activeMemoryBytes - firstMemory.activeMemoryBytes) /
    Math.max(memoryEvents.length - 1, 1) /
    (1024 * 1024);
  if (slopeMbPerEvent > maxSlopeMbPerEvent) {
    throw new Error(
      `Soak failed: active memory slope ${slopeMbPerEvent.toFixed(3)} MB/event exceeded ${maxSlopeMbPerEvent.toFixed(3)} MB/event.`,
    );
  }

  return {
    throughputRatio: ratio,
    firstAverage,
    lastAverage,
    slopeMbPerEvent,
  };
}

function assertSoakStability(
  runId: string,
  sampleSize: number,
  minRatio: number,
  maxSlopeMbPerEvent: number,
): void {
  const metrics = assertSoakStabilityForEvents(
    runId,
    readStepEvents(runId),
    sampleSize,
    minRatio,
    maxSlopeMbPerEvent,
  );

  process.stdout.write(
    `throughputRatio=${metrics.throughputRatio.toFixed(3)} firstAvg=${Math.round(metrics.firstAverage).toLocaleString()} lastAvg=${Math.round(metrics.lastAverage).toLocaleString()} activeSlopeMbPerEvent=${metrics.slopeMbPerEvent.toFixed(3)}\n`,
  );
}

export function samplePrompt(tokenizer: CharTokenizer): string {
  const vocab = tokenizer.vocab;
  return vocab.includes("\n") ? "\n" : (vocab[0] ?? "");
}

export function waitForTerminalState(
  runId: string,
  pollSeconds: number,
): ReturnType<typeof readRunStatus> {
  const directory = runDir(repoRoot(), runId);
  while (true) {
    const status = readRunStatus(directory);
    const health = deriveOperatorHealth(status);
    if (
      status.state === "completed" ||
      status.state === "stopped" ||
      status.state === "stalled" ||
      status.state === "failed" ||
      status.state === "cancelled"
    ) {
      return status;
    }
    if (health.operatorHealth !== "healthy") {
      // Fast supervised runs can finish between two immediate status reads:
      // the supervisor may already be gone while the last non-terminal status
      // is still what we observed. Re-read once after a short grace before
      // classifying the operator as unhealthy.
      Bun.sleepSync(250);
      const refreshedStatus = readRunStatus(directory);
      const refreshedHealth = deriveOperatorHealth(refreshedStatus);
      if (
        refreshedStatus.state === "completed" ||
        refreshedStatus.state === "stopped" ||
        refreshedStatus.state === "stalled" ||
        refreshedStatus.state === "failed" ||
        refreshedStatus.state === "cancelled"
      ) {
        return refreshedStatus;
      }
      if (refreshedHealth.operatorHealth !== "healthy") {
        throw new Error(
          `Acceptance run ${runId} became unhealthy (${refreshedHealth.operatorHealth}) before reaching a terminal state`,
        );
      }
    }
    Bun.sleepSync(pollSeconds * 1000);
  }
}

function appendOptionalArg(
  args: string[],
  flags: Map<string, string>,
  key: string,
  value?: string,
): void {
  const resolved = value ?? flags.get(key);
  if (resolved !== undefined) {
    args.push(`--${key}`, resolved);
  }
}

export function buildManagerArgs(
  presetName: PresetName,
  runId: string,
  defaults: AcceptanceDefaults,
  flags: Map<string, string>,
): string[] {
  const args = [
    "run",
    "src/run/manager.ts",
    "start",
    "--name",
    runId,
    "--preset",
    presetName,
    "--max-steps",
    String(getNumberFlag(flags, "max-steps", defaults.maxSteps)),
    "--batch-size",
    String(getNumberFlag(flags, "batch-size", defaults.batchSize)),
    "--grad-accum",
    String(getNumberFlag(flags, "grad-accum", defaults.gradAccumSteps)),
    "--eval-interval",
    String(getNumberFlag(flags, "eval-interval", defaults.evalInterval)),
    "--eval-steps",
    String(getNumberFlag(flags, "eval-steps", defaults.evalSteps)),
    "--log-interval",
    String(getNumberFlag(flags, "log-interval", defaults.logInterval)),
    "--lr",
    String(getNumberFlag(flags, "lr", defaults.learningRate)),
    "--weight-decay",
    String(getNumberFlag(flags, "weight-decay", defaults.weightDecay)),
    "--max-grad-norm",
    String(getNumberFlag(flags, "max-grad-norm", defaults.maxGradNorm ?? 1)),
    "--warmup-steps",
    String(getNumberFlag(flags, "warmup-steps", defaults.warmupSteps)),
    "--min-lr",
    String(getNumberFlag(flags, "min-lr", defaults.minLearningRate)),
    "--snapshot-interval",
    String(getNumberFlag(flags, "snapshot-interval", defaults.snapshotInterval)),
    "--resume-interval",
    String(getNumberFlag(flags, "resume-interval", defaults.resumeInterval)),
    "--stall-timeout-sec",
    String(getNumberFlag(flags, "stall-timeout-sec", defaults.stallTimeoutSeconds)),
  ];

  const gradientCheckpointing = getFlag(flags, "gradient-checkpointing");
  if (gradientCheckpointing !== undefined) {
    args.push("--gradient-checkpointing", gradientCheckpointing);
  }
  appendOptionalArg(args, flags, "early-stop-patience");
  appendOptionalArg(args, flags, "early-stop-min-delta");

  const dataPath = flags.get("data");
  if (dataPath !== undefined) {
    args.push("--data", resolve(process.cwd(), dataPath));
  }

  appendOptionalArg(args, flags, "memory-limit-mb");
  appendOptionalArg(args, flags, "cache-limit-mb");
  appendOptionalArg(args, flags, "wired-limit-mb");
  return args;
}

export function readRunOptions(flags: Map<string, string>): AcceptanceRunOptions {
  validateAllowedFlags(flags, ACCEPTANCE_FLAG_ALLOWLIST, "acceptance");
  const presetName = readPresetName(flags);
  const mode = readMode(flags);
  const defaults = ACCEPTANCE_DEFAULTS[presetName];
  const runId =
    getFlag(flags, "name") ?? `${new Date().toISOString().replace(/[:.]/g, "-")}-${presetName}`;
  const pollSeconds = getNumberFlag(flags, "poll-seconds", 10);
  const lossTarget = getNumberFlag(flags, "loss-target", defaults.lossTarget);
  const throughputWindow = getNumberFlag(flags, "throughput-window", 25);
  const minThroughputRatio = getNumberFlag(flags, "min-throughput-ratio", 0.5);
  const maxSlopeMbPerEvent = getNumberFlag(flags, "max-slope-mb-per-event", 8);
  const stallTimeoutSeconds = getNumberFlag(
    flags,
    "stall-timeout-sec",
    defaults.stallTimeoutSeconds,
  );
  const modelPreset = presetName === "gpt-small" ? GPT_SMALL : GPT_TINY;
  const parameterCount = estimateParameterCount(resolveConfig(modelPreset, 65));
  const managerFlags = new Map(flags);
  if (!managerFlags.has("early-stop-patience")) {
    managerFlags.set("early-stop-patience", mode === "acceptance" ? "8" : "none");
  }
  if (mode === "acceptance" && !managerFlags.has("early-stop-min-delta")) {
    managerFlags.set("early-stop-min-delta", "0.02");
  }

  return {
    presetName,
    mode,
    runId,
    pollSeconds,
    lossTarget,
    throughputWindow,
    minThroughputRatio,
    maxSlopeMbPerEvent,
    stallTimeoutSeconds,
    parameterCount,
    args: buildManagerArgs(presetName, runId, defaults, managerFlags),
  };
}

export function assertCompletedStatus(runId: string, status: RunStatus): void {
  if (status.state === "stalled") {
    const reason = status.stallReason === undefined ? "unknown stall reason" : status.stallReason;
    throw new Error(`Acceptance run ${runId} stalled: ${reason}`);
  }
  if (status.state === "stopped" && status.earlyStopReason !== undefined) {
    return;
  }
  if (status.state !== "completed") {
    throw new Error(`Acceptance run ${runId} ended in state ${status.state}`);
  }
}

export function finalLossFromStatus(status: RunStatus): number {
  const finalLoss = status.bestValLoss ?? status.lastValLoss ?? status.lastStepLoss;
  if (finalLoss === undefined) {
    throw new Error("Acceptance run completed without a final recorded loss");
  }
  return finalLoss;
}

export function checkpointPathFromStatus(status: RunStatus): string {
  const checkpointPath =
    status.bestCheckpoint ?? status.latestResumeCheckpoint ?? status.latestCheckpoint;
  if (checkpointPath === undefined) {
    throw new Error("Acceptance run completed without a checkpoint");
  }
  return checkpointPath;
}

export async function main(argv = process.argv): Promise<void> {
  const flags = parseArgs(argv);
  const {
    presetName,
    mode,
    runId,
    pollSeconds,
    lossTarget,
    throughputWindow,
    minThroughputRatio,
    maxSlopeMbPerEvent,
    parameterCount,
    args,
  } = readRunOptions(flags);

  process.stdout.write(
    `Acceptance run: ${presetName} (${parameterCount.toLocaleString()} params)\n`,
  );
  const startResult = Bun.spawnSync(["bun", ...args], {
    cwd: packageRoot(),
    stdout: "pipe",
    stderr: "pipe",
  });
  if (startResult.exitCode !== 0) {
    process.stderr.write(decodeOutput(startResult.stderr));
    process.exit(startResult.exitCode);
  }

  process.stdout.write(decodeOutput(startResult.stdout));
  const status = waitForTerminalState(runId, pollSeconds);
  assertCompletedStatus(runId, status);

  if (mode === "soak") {
    assertSoakStability(runId, throughputWindow, minThroughputRatio, maxSlopeMbPerEvent);
    process.stdout.write(`run=${runId} preset=${presetName} mode=soak status=completed\n`);
    return;
  }

  const finalLoss = finalLossFromStatus(status);
  if (finalLoss >= lossTarget) {
    throw new Error(
      `Acceptance failed: ${presetName} final loss ${finalLoss.toFixed(4)} did not beat ${lossTarget.toFixed(4)}`,
    );
  }

  const checkpointPath = checkpointPathFromStatus(status);

  const checkpoint = loadCheckpoint(checkpointPath);
  const tokenizer = CharTokenizer.fromVocab(checkpoint.tokenizer.chars);
  const model = new GPT(checkpoint.config);

  try {
    model.eval();
    applyCheckpoint(model, checkpoint);
    const sample = generate(model, checkpoint.config, tokenizer, samplePrompt(tokenizer), {
      maxNewTokens: 200,
      temperature: 0.8,
    });
    process.stdout.write(`checkpoint=${checkpointPath}\n`);
    process.stdout.write(`finalLoss=${finalLoss.toFixed(4)} target=${lossTarget.toFixed(4)}\n`);
    process.stdout.write(`sample:\n${sample}\n`);
  } finally {
    model[Symbol.dispose]();
  }
}

if (import.meta.main) {
  await main();
}
