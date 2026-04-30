import { resolve } from "path";

export const DEFAULT_PATH_FLAGS = new Set(["data", "resume", "warm-start"]);

export class SupervisedRunManagerUsageError extends Error {}

export function nowIso(): string {
  return new Date().toISOString();
}

function nowStamp(): string {
  return nowIso().replace(/[:.]/g, "-");
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
      throw new SupervisedRunManagerUsageError(`${context}: unknown flag --${key}`);
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

export function stripFlag(args: string[], key: string, takesValue = true): string[] {
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

function absolutizeTrainerPaths(args: string[], pathFlags: ReadonlySet<string>): string[] {
  const normalized = [...args];
  for (let index = 0; index < normalized.length; index++) {
    const key = normalized[index];
    const value = normalized[index + 1];
    if (key === undefined || value === undefined || !key.startsWith("--")) {
      continue;
    }
    if (pathFlags.has(key.slice(2))) {
      normalized[index + 1] = resolve(process.cwd(), value);
    }
  }
  return normalized;
}

export function trainerArgsFrom(
  args: string[],
  pathFlags: ReadonlySet<string> = DEFAULT_PATH_FLAGS,
): string[] {
  let trainerArgs = stripFlag(args, "name");
  trainerArgs = stripFlag(trainerArgs, "from");
  trainerArgs = stripFlag(trainerArgs, "interval");
  trainerArgs = stripFlag(trainerArgs, "json", false);
  trainerArgs = stripFlag(trainerArgs, "run-dir");
  trainerArgs = stripFlag(trainerArgs, "stall-timeout-sec");
  trainerArgs = stripFlag(trainerArgs, "help", false);
  return absolutizeTrainerPaths(trainerArgs, pathFlags);
}

export function defaultRunIdLabel(args: string[]): string {
  return getFlag(args, "preset") ?? (getFlag(args, "resume") === undefined ? "run" : "resume");
}

export function generateRunId(
  args: string[],
  runIdLabel: (args: string[]) => string = defaultRunIdLabel,
): string {
  return `${nowStamp()}-${runIdLabel(args)}`;
}
