import { setCacheLimitBytes, setMemoryLimitBytes, setWiredLimitBytes } from "@mlxts/core";

export const USAGE_ERROR_EXIT_CODE = 2;
export const RUNTIME_ERROR_EXIT_CODE = 1;
export const USER_ERROR_EXIT_CODE = USAGE_ERROR_EXIT_CODE;
export const SYSTEM_ERROR_EXIT_CODE = RUNTIME_ERROR_EXIT_CODE;

export class UserError extends Error {}

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
  "run-dir",
  "json",
  "help",
]);

export const GENERATE_FLAG_ALLOWLIST = new Set([
  "checkpoint",
  "prompt",
  "max-tokens",
  "temperature",
  "json",
  "help",
]);

export const EXPORT_FLAG_ALLOWLIST = new Set(["checkpoint", "output", "help"]);

export const TRAIN_VALUE_FLAGS = new Set(
  [...TRAIN_FLAG_ALLOWLIST].filter((key) => key !== "json" && key !== "help"),
);
export const GENERATE_VALUE_FLAGS = new Set(
  [...GENERATE_FLAG_ALLOWLIST].filter((key) => key !== "json" && key !== "help"),
);
export const EXPORT_VALUE_FLAGS = new Set(
  [...EXPORT_FLAG_ALLOWLIST].filter((key) => key !== "help"),
);

type JsonEnvelope = {
  timestamp: string;
  [key: string]: unknown;
};

export type ParsedArgs = {
  command: string;
  flags: Map<string, string>;
  valuedFlags: ReadonlySet<string>;
};

export type NanogptCliErrorCode = "usage" | "runtime";

export function quoteScalar(value: string | number | boolean | null | undefined): string {
  return JSON.stringify(value ?? null);
}

export function getBooleanFlag(flags: Map<string, string>, key: string): boolean | undefined {
  const raw = flags.get(key);
  if (raw === undefined) {
    return undefined;
  }
  if (raw === "true") {
    return true;
  }
  if (raw === "false") {
    return false;
  }
  throw new UserError(`Flag --${key} must be "true" or "false"`);
}

export function parseArgs(argv: string[]): ParsedArgs {
  const command = argv[2] ?? "help";
  const flags = new Map<string, string>();
  const valuedFlags = new Set<string>();

  for (let index = 3; index < argv.length; index++) {
    const argument = argv[index];
    if (argument === undefined || !argument.startsWith("--")) {
      continue;
    }

    const equalsIndex = argument.indexOf("=");
    if (equalsIndex > 2) {
      const key = argument.slice(2, equalsIndex);
      flags.set(key, argument.slice(equalsIndex + 1));
      valuedFlags.add(key);
      continue;
    }

    const key = argument.slice(2);
    const next = argv[index + 1];
    if (next !== undefined && !next.startsWith("--")) {
      flags.set(key, next);
      valuedFlags.add(key);
      index += 1;
      continue;
    }
    flags.set(key, "true");
  }

  return { command, flags, valuedFlags };
}

export function validateKnownFlags(
  flags: Map<string, string>,
  allowed: ReadonlySet<string>,
  context: string,
): void {
  for (const key of flags.keys()) {
    if (!allowed.has(key)) {
      throw new UserError(`${context}: unknown flag --${key}`);
    }
  }
}

export function validateFlagValues(
  flags: Map<string, string>,
  valuedFlags: ReadonlySet<string>,
  valueFlags: ReadonlySet<string>,
  context: string,
): void {
  for (const key of flags.keys()) {
    const hasValue = valuedFlags.has(key);
    if (valueFlags.has(key)) {
      if (!hasValue) {
        throw new UserError(`${context}: flag --${key} requires a value`);
      }
      continue;
    }
    if (hasValue) {
      throw new UserError(`${context}: flag --${key} does not accept a value`);
    }
  }
}

export function getFlag(
  flags: Map<string, string>,
  key: string,
  fallback?: string,
): string | undefined {
  return flags.get(key) ?? fallback;
}

export function getNumberFlag(flags: Map<string, string>, key: string, fallback: number): number {
  const raw = flags.get(key);
  if (raw === undefined) {
    return fallback;
  }

  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) {
    throw new UserError(`Flag --${key} must be a finite number`);
  }
  return parsed;
}

export function getNullablePositiveNumberFlag(
  flags: Map<string, string>,
  key: string,
  fallback: number | null,
): number | null {
  const raw = flags.get(key);
  if (raw === undefined) {
    return fallback;
  }
  if (raw === "none" || raw === "null") {
    return null;
  }

  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new UserError(`Flag --${key} must be a positive number or "none"`);
  }
  return parsed;
}

export function getNullableNonNegativeIntegerFlag(
  flags: Map<string, string>,
  key: string,
  fallback: number | null,
): number | null {
  const raw = flags.get(key);
  if (raw === undefined) {
    return fallback;
  }
  if (raw === "none" || raw === "null") {
    return null;
  }

  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new UserError(`Flag --${key} must be a non-negative integer or "none"`);
  }
  return parsed;
}

export function getNonNegativeNumberFlag(
  flags: Map<string, string>,
  key: string,
  fallback: number,
): number {
  const raw = flags.get(key);
  if (raw === undefined) {
    return fallback;
  }

  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new UserError(`Flag --${key} must be a non-negative number`);
  }
  return parsed;
}

export function emitJson(value: Record<string, unknown>): void {
  const envelope: JsonEnvelope = {
    timestamp: new Date().toISOString(),
    ...value,
  };
  process.stdout.write(`${JSON.stringify(envelope)}\n`);
}

export function formatNanogptCliError(
  message: string,
  code: NanogptCliErrorCode,
  helpCommand: string,
): string {
  return [
    "error:",
    `  code: ${quoteScalar(code)}`,
    `  message: ${quoteScalar(message)}`,
    "help[1]:",
    `  ${quoteScalar(`Run \`${helpCommand}\` for nanoGPT CLI help`)}`,
  ].join("\n");
}

function toBytesFromMb(flagKey: string, value: string | undefined): number | undefined {
  if (value === undefined) {
    return undefined;
  }

  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new UserError(`Flag --${flagKey} must be a positive number`);
  }
  return Math.round(parsed * 1024 * 1024);
}

export function applyRuntimeLimits(flags: Map<string, string>): void {
  const cacheLimitBytes = toBytesFromMb("cache-limit-mb", flags.get("cache-limit-mb"));
  const memoryLimitBytes = toBytesFromMb("memory-limit-mb", flags.get("memory-limit-mb"));
  const wiredLimitBytes = toBytesFromMb("wired-limit-mb", flags.get("wired-limit-mb"));

  if (cacheLimitBytes !== undefined) {
    setCacheLimitBytes(cacheLimitBytes);
  }
  if (memoryLimitBytes !== undefined) {
    setMemoryLimitBytes(memoryLimitBytes);
  }
  if (wiredLimitBytes !== undefined) {
    setWiredLimitBytes(wiredLimitBytes);
  }
}
