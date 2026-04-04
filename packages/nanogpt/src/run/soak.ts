#!/usr/bin/env bun

import { resolve } from "path";

export type PresetName = "gpt-tiny" | "gpt-small";
type SoakDefaults = {
  maxSteps: string;
  batchSize: string;
  gradAccum: string;
  evalInterval: string;
  evalSteps: string;
  logInterval: string;
  warmupSteps: string;
  snapshotInterval: string;
  resumeInterval: string;
  minThroughputRatio: string;
  maxSlopeMbPerEvent: string;
  stallTimeoutSeconds: string;
};

const SOAK_DEFAULTS: Record<PresetName, SoakDefaults> = {
  "gpt-tiny": {
    maxSteps: "250",
    batchSize: "4",
    gradAccum: "1",
    evalInterval: "50",
    evalSteps: "10",
    logInterval: "10",
    warmupSteps: "25",
    snapshotInterval: "50",
    resumeInterval: "250",
    minThroughputRatio: "0.5",
    maxSlopeMbPerEvent: "8",
    stallTimeoutSeconds: "600",
  },
  "gpt-small": {
    maxSteps: "50",
    batchSize: "1",
    gradAccum: "8",
    evalInterval: "25",
    evalSteps: "5",
    logInterval: "5",
    warmupSteps: "10",
    snapshotInterval: "25",
    resumeInterval: "50",
    minThroughputRatio: "0.5",
    maxSlopeMbPerEvent: "8",
    stallTimeoutSeconds: "600",
  },
};

function packageRoot(): string {
  return resolve(import.meta.dir, "../..");
}

function readFlag(args: string[], key: string): string | undefined {
  for (let index = 0; index < args.length; index++) {
    if (args[index] === `--${key}`) {
      return args[index + 1];
    }
  }
  return undefined;
}

function hasFlag(args: string[], key: string): boolean {
  return args.includes(`--${key}`);
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

export function readPreset(args: string[]): PresetName {
  const preset = readFlag(args, "preset") ?? "gpt-small";
  if (preset === "gpt-tiny" || preset === "gpt-small") {
    return preset;
  }
  throw new Error(`Unknown preset "${preset}"`);
}

function withDefaultFlag(args: string[], key: string, value: string): void {
  if (!hasFlag(args, key)) {
    args.push(`--${key}`, value);
  }
}

export function buildAcceptanceArgs(args: string[]): string[] {
  validateAllowedFlags(
    args,
    new Set([
      "preset",
      "name",
      "poll-seconds",
      "throughput-window",
      "max-steps",
      "batch-size",
      "grad-accum",
      "eval-interval",
      "eval-steps",
      "log-interval",
      "warmup-steps",
      "snapshot-interval",
      "resume-interval",
      "min-throughput-ratio",
      "max-slope-mb-per-event",
      "stall-timeout-sec",
      "data",
      "memory-limit-mb",
      "cache-limit-mb",
      "wired-limit-mb",
    ]),
    "soak",
  );
  const preset = readPreset(args);
  const defaults = SOAK_DEFAULTS[preset];
  const forwarded = [...args];

  withDefaultFlag(forwarded, "max-steps", defaults.maxSteps);
  withDefaultFlag(forwarded, "batch-size", defaults.batchSize);
  withDefaultFlag(forwarded, "grad-accum", defaults.gradAccum);
  withDefaultFlag(forwarded, "eval-interval", defaults.evalInterval);
  withDefaultFlag(forwarded, "eval-steps", defaults.evalSteps);
  withDefaultFlag(forwarded, "log-interval", defaults.logInterval);
  withDefaultFlag(forwarded, "warmup-steps", defaults.warmupSteps);
  withDefaultFlag(forwarded, "snapshot-interval", defaults.snapshotInterval);
  withDefaultFlag(forwarded, "resume-interval", defaults.resumeInterval);
  withDefaultFlag(forwarded, "min-throughput-ratio", defaults.minThroughputRatio);
  withDefaultFlag(forwarded, "max-slope-mb-per-event", defaults.maxSlopeMbPerEvent);
  withDefaultFlag(forwarded, "stall-timeout-sec", defaults.stallTimeoutSeconds);

  return ["run", "src/run/acceptance.ts", "--mode", "soak", ...forwarded];
}

export function main(argv = process.argv.slice(2)): number {
  const result = Bun.spawnSync(["bun", ...buildAcceptanceArgs(argv)], {
    cwd: packageRoot(),
    stdout: "inherit",
    stderr: "inherit",
  });
  return result.exitCode;
}

if (import.meta.main) {
  process.exit(main());
}
