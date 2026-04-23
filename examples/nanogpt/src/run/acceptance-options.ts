import { resolve } from "path";
import { estimateParameterCount, GPT_SMALL, GPT_TINY, resolveConfig } from "../config";
import { DEFAULT_STALL_TIMEOUT_SECONDS } from "./files";

export type PresetName = "gpt-tiny" | "gpt-small";
export type RunMode = "acceptance" | "soak";

export type AcceptanceDefaults = {
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

export type AcceptanceRunOptions = {
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
    lossTarget: 1.8,
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
    lossTarget: 1.8,
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
    "manager",
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
