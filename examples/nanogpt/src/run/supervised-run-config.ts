import type {
  StatusPayload,
  SupervisedRunManagerCliOptions,
  SupervisedRunManagerRunOptions,
  SupervisedRunStatusOptions,
  SupervisedRunSupervisorOptions,
} from "@mlxts/train/supervised-run";
import { resolve } from "path";

export const NANOGPT_RUNS_DIRECTORY = ".nanogpt-runs";

export const TRAIN_FLAG_ALLOWLIST = new Set([
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

export const START_FLAG_ALLOWLIST = new Set([...TRAIN_FLAG_ALLOWLIST, "name", "stall-timeout-sec"]);
export const RESUME_FLAG_ALLOWLIST = new Set([
  ...TRAIN_FLAG_ALLOWLIST,
  "name",
  "from",
  "stall-timeout-sec",
]);
export const STATUS_FLAG_ALLOWLIST = new Set(["name", "json", "help"]);
export const WATCH_FLAG_ALLOWLIST = new Set(["name", "json", "interval", "help"]);
export const CONTROL_FLAG_ALLOWLIST = new Set(["name", "help"]);

export const USAGE = `nanogpt run manager

Run from examples/nanogpt/:

Usage:
  bun run manager start [train flags...]
  bun run manager resume --from <run-id> [train flags...]
  bun run manager status --name <run-id> [--json]
  bun run manager watch --name <run-id> [--interval <seconds>] [--json]
  bun run manager stop --name <run-id>
  bun run manager cancel --name <run-id>

Notes:
  start/resume accept --stall-timeout-sec <seconds> (default 600)
  train flags also accept --early-stop-patience <n|none> and --early-stop-min-delta <n>
  cancel is best-effort and may lose work since the latest resume checkpoint
`;

export function repoRoot(): string {
  return resolve(import.meta.dir, "../../../../");
}

export function packageRoot(): string {
  return resolve(import.meta.dir, "../..");
}

function nanogptRunIdLabel(args: string[]): string {
  const presetIndex = args.indexOf("--preset");
  const preset = presetIndex >= 0 ? args[presetIndex + 1] : undefined;
  if (preset !== undefined) {
    return preset;
  }
  return args.includes("--resume") ? "resume" : "gpt";
}

function formatGradientCheckpointing(config: StatusPayload["config"]): string {
  if (typeof config !== "object" || config === null || Array.isArray(config)) {
    return "-";
  }
  const value = Object.fromEntries(Object.entries(config)).gradientCheckpointing;
  return typeof value === "boolean" ? String(value) : "-";
}

function formatNanogptBatchLine(payload: StatusPayload): string {
  return `  batch: ${payload.batchSize ?? "-"}  grad accum: ${payload.gradAccumSteps ?? "-"}  gradient checkpointing: ${formatGradientCheckpointing(payload.config)}`;
}

export const nanogptManagerRunOptions: SupervisedRunManagerRunOptions = {
  repoRoot: repoRoot(),
  packageRoot: packageRoot(),
  runsDirectoryName: NANOGPT_RUNS_DIRECTORY,
  runIdLabel: nanogptRunIdLabel,
  supervisorCommand: (runDirectory) => [
    "bun",
    "run",
    "src/run/supervisor.ts",
    "--run-dir",
    runDirectory,
  ],
  statusCommand: (runId) => `(from examples/nanogpt/) bun run manager status --name ${runId}`,
};

export const nanogptStatusOptions: SupervisedRunStatusOptions = {
  repoRoot: repoRoot(),
  runsDirectoryName: NANOGPT_RUNS_DIRECTORY,
  formatBatchLine: formatNanogptBatchLine,
};

export const nanogptManagerCliOptions: SupervisedRunManagerCliOptions = {
  usage: USAGE,
  startFlagAllowlist: START_FLAG_ALLOWLIST,
  resumeFlagAllowlist: RESUME_FLAG_ALLOWLIST,
  statusFlagAllowlist: STATUS_FLAG_ALLOWLIST,
  watchFlagAllowlist: WATCH_FLAG_ALLOWLIST,
  controlFlagAllowlist: CONTROL_FLAG_ALLOWLIST,
  run: nanogptManagerRunOptions,
  status: nanogptStatusOptions,
};

export const nanogptSupervisorOptions: SupervisedRunSupervisorOptions = {
  trainerCommand: (spec) => ["bun", "run", "src/cli.ts", "train", ...spec.trainerArgs],
};
