import { resolve } from "path";

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

export function usage(): string {
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

export function nowIso(): string {
  return new Date().toISOString();
}

function nowStamp(): string {
  return nowIso().replace(/[:.]/g, "-");
}

export function repoRoot(): string {
  return resolve(import.meta.dir, "../../../../");
}

export function packageRoot(): string {
  return resolve(import.meta.dir, "../..");
}

export function parseArgs(argv: string[]): { command: string; args: string[] } {
  return {
    command: argv[2] ?? "help",
    args: argv.slice(3),
  };
}

export function validateAllowedFlags(
  args: string[],
  allowed: ReadonlySet<string>,
  context: string,
): void {
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

export function getFlag(args: string[], key: string, fallback?: string): string | undefined {
  for (let index = 0; index < args.length; index++) {
    if (args[index] === `--${key}`) {
      return args[index + 1] ?? fallback;
    }
  }
  return fallback;
}

export function hasFlag(args: string[], key: string): boolean {
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

export function trainerArgsFrom(args: string[]): string[] {
  let trainerArgs = stripFlag(args, "name");
  trainerArgs = stripFlag(trainerArgs, "from");
  trainerArgs = stripFlag(trainerArgs, "interval");
  trainerArgs = stripFlag(trainerArgs, "json", false);
  trainerArgs = stripFlag(trainerArgs, "run-dir");
  trainerArgs = stripFlag(trainerArgs, "stall-timeout-sec");
  trainerArgs = stripFlag(trainerArgs, "help", false);
  return absolutizeTrainerPaths(trainerArgs);
}

export function generateRunId(args: string[]): string {
  const preset = getFlag(args, "preset");
  const label = preset ?? (getFlag(args, "resume") === undefined ? "gpt" : "resume");
  return `${nowStamp()}-${label}`;
}
