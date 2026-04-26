#!/usr/bin/env bun

import { join } from "path";
import { acquireRuntimeCommandLock } from "./runtime-command-lock";

type RegressionProfile = "quick" | "real" | "substantial";

type CliOptions = {
  profile: RegressionProfile;
  qwenModel: string;
  gemma4Model: string;
  reportDir: string;
  requestTimeoutMs: number;
  allowDownload: boolean;
};

function usage(): string {
  return [
    "Usage: bun run regression:qwen-gemma -- [options]",
    "",
    "Options:",
    "  --profile <quick|real|substantial>  Regression tier, default quick.",
    "  --qwen-model <id>                  Qwen model id/path.",
    "  --gemma4-model <id>                Gemma 4 model id/path.",
    "  --report-dir <path>                Directory for benchmark JSON evidence.",
    "  --request-timeout-ms <n>           Client timeout for endpoint requests.",
    "  --allow-download                   Allow Hub downloads where supported.",
  ].join("\n");
}

function readStringFlag(args: readonly string[], index: number, flag: string): string {
  const value = args[index + 1];
  if (value === undefined || value.startsWith("--")) {
    throw new Error(`${flag} requires a value.\n\n${usage()}`);
  }
  return value;
}

function readPositiveIntegerFlag(args: readonly string[], index: number, flag: string): number {
  const value = Number.parseInt(readStringFlag(args, index, flag), 10);
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${flag} must be a positive integer.`);
  }
  return value;
}

function readProfile(args: readonly string[], index: number): RegressionProfile {
  const value = readStringFlag(args, index, "--profile");
  if (value === "quick" || value === "real" || value === "substantial") {
    return value;
  }
  throw new Error('--profile must be "quick", "real", or "substantial".');
}

export function parseQwenGemmaRegressionArgs(argv: readonly string[]): CliOptions {
  const options: CliOptions = {
    profile: "quick",
    qwenModel: "mlx-community/Qwen3.6-27B-4bit",
    gemma4Model: "google/gemma-4-E2B-it",
    reportDir: ".tmp/qwen-gemma-regression",
    requestTimeoutMs: 3_600_000,
    allowDownload: false,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
      case "--help":
      case "-h":
        console.log(usage());
        process.exit(0);
        return options;
      case "--profile":
        options.profile = readProfile(argv, index);
        index += 1;
        break;
      case "--qwen-model":
        options.qwenModel = readStringFlag(argv, index, arg);
        index += 1;
        break;
      case "--gemma4-model":
        options.gemma4Model = readStringFlag(argv, index, arg);
        index += 1;
        break;
      case "--report-dir":
        options.reportDir = readStringFlag(argv, index, arg);
        index += 1;
        break;
      case "--request-timeout-ms":
        options.requestTimeoutMs = readPositiveIntegerFlag(argv, index, arg);
        index += 1;
        break;
      case "--allow-download":
        options.allowDownload = true;
        break;
      default:
        throw new Error(`Unknown option: ${arg}\n\n${usage()}`);
    }
  }

  return options;
}

function inheritedStringEnv(): Record<string, string> {
  return Object.fromEntries(
    Object.entries(process.env).filter((entry): entry is [string, string] => {
      const value = entry[1];
      return typeof value === "string";
    }),
  );
}

async function runCommand(label: string, args: readonly string[]): Promise<void> {
  console.log(`[qwen-gemma-regression] ${label}: ${args.join(" ")}`);
  const child = Bun.spawn([...args], {
    cwd: new URL("..", import.meta.url).pathname,
    env: inheritedStringEnv(),
    stdout: "inherit",
    stderr: "inherit",
  });
  const exitCode = await child.exited;
  if (exitCode !== 0) {
    throw new Error(`[qwen-gemma-regression] ${label} failed with exit code ${exitCode}.`);
  }
}

async function runQuickProfile(): Promise<void> {
  await runCommand("transformer focused regressions", [
    "bun",
    "run",
    "--filter",
    "@mlxts/transformers",
    "regression:models",
  ]);
  await runCommand("serve focused regressions", [
    "bun",
    "run",
    "--filter",
    "@mlxts/serve",
    "regression:serve",
  ]);
}

async function runRealProfile(options: CliOptions): Promise<void> {
  await runCommand("transformer real decode smoke", [
    "bun",
    "run",
    "packages/transformers/scripts/regression-model-matrix.ts",
    "--decode-smoke",
    "--qwen-model",
    options.qwenModel,
    "--gemma4-model",
    options.gemma4Model,
  ]);
  await runCommand("serve real endpoint smoke", [
    "bun",
    "run",
    "packages/serve/scripts/regression-serve-matrix.ts",
    "--real-models",
    "--qwen-model",
    options.qwenModel,
    "--gemma4-model",
    options.gemma4Model,
    "--report-dir",
    join(options.reportDir, "serve"),
    "--request-timeout-ms",
    String(options.requestTimeoutMs),
    ...(options.allowDownload ? ["--allow-download"] : []),
  ]);
}

async function runSubstantialProfile(options: CliOptions): Promise<void> {
  await runCommand("transformer real decode smoke", [
    "bun",
    "run",
    "packages/transformers/scripts/regression-model-matrix.ts",
    "--decode-smoke",
    "--qwen-model",
    options.qwenModel,
    "--gemma4-model",
    options.gemma4Model,
  ]);
  await runCommand("serve capability smoke", [
    "bun",
    "run",
    "packages/serve/scripts/regression-serve-matrix.ts",
    "--capability-smoke",
    "--qwen-model",
    options.qwenModel,
    "--gemma4-model",
    options.gemma4Model,
    "--report-dir",
    join(options.reportDir, "serve"),
    "--request-timeout-ms",
    String(options.requestTimeoutMs),
    ...(options.allowDownload ? ["--allow-download"] : []),
  ]);
  await runCommand("Qwen long-context retrieval smoke", [
    "bun",
    "run",
    "bench:generation:context",
    "--model",
    options.qwenModel,
    "--rungs",
    "32768",
    "--needle-placements",
    "all",
    "--generation-tokens",
    "24",
    "--fail-on-mismatch",
    "--max-active-slope-mb-per-token",
    "1",
    "--report-json",
    join(options.reportDir, "qwen36-context-32k-all.json"),
  ]);
}

export async function runQwenGemmaRegression(options: CliOptions): Promise<void> {
  if (options.profile === "quick") {
    await runQuickProfile();
    return;
  }

  using _runtimeLock = acquireRuntimeCommandLock(`regression:qwen-gemma:${options.profile}`);
  if (options.profile === "real") {
    await runRealProfile(options);
    return;
  }
  await runSubstantialProfile(options);
}

if (import.meta.main) {
  runQwenGemmaRegression(parseQwenGemmaRegressionArgs(Bun.argv.slice(2))).catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  });
}
