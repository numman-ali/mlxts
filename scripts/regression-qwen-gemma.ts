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

type QwenGemmaRegressionCommand = { kind: "help" } | { kind: "run"; options: CliOptions };

type QwenGemmaRegressionStage = {
  label: string;
  status: "passed";
};

type QwenGemmaRegressionStageSpec = {
  label: string;
  args: readonly string[];
};

type QwenGemmaRegressionResult = {
  profile: RegressionProfile;
  reportDir: string;
  stages: QwenGemmaRegressionStage[];
};

type QwenGemmaRegressionRuntime = {
  stdout?: (text: string) => void;
  stderr?: (text: string) => void;
  runRegression?: (
    options: CliOptions,
    progress: (text: string) => void,
  ) => Promise<QwenGemmaRegressionResult>;
};

class QwenGemmaRegressionUsageError extends Error {}

export function formatQwenGemmaRegressionUsage(): string {
  return [
    "description: Run the Qwen/Gemma transformer and serve regression profiles",
    "usage[3]:",
    "  bun run regression:qwen-gemma",
    "  bun run regression:qwen-gemma -- --profile real",
    "  bun run regression:qwen-gemma -- --profile substantial --report-dir .tmp/qwen-gemma-regression",
    "options[7]{flag,description}:",
    '  "--profile <quick|real|substantial>","Regression tier; default quick; real includes mixed fairness"',
    '  "--qwen-model <id>","Qwen model id/path"',
    '  "--gemma4-model <id>","Gemma 4 model id/path"',
    '  "--report-dir <path>","Directory for benchmark JSON evidence"',
    '  "--request-timeout-ms <n>","Client timeout for endpoint requests; default 3600000"',
    '  "--allow-download","Allow Hub downloads where supported"',
    '  "--help","Show this help"',
    "profiles[3]{name,meaning}:",
    '  "quick","Focused unit regressions only"',
    '  "real","Real Qwen/Gemma decode and endpoint smoke"',
    '  "substantial","Real smoke plus capability and long-context checks"',
    "exit_codes[3]{code,meaning}:",
    '  0,"regression passed"',
    '  1,"runtime or regression failure"',
    '  2,"usage error"',
  ].join("\n");
}

function readStringFlag(args: readonly string[], index: number, flag: string): string {
  const value = args[index + 1];
  if (value === undefined || value.trim() === "" || value.startsWith("-")) {
    throw new QwenGemmaRegressionUsageError(`${flag} requires a value.`);
  }
  return value;
}

function readPositiveIntegerFlag(args: readonly string[], index: number, flag: string): number {
  const rawValue = args[index + 1]?.trim();
  if (rawValue === undefined || rawValue === "" || rawValue.startsWith("--")) {
    throw new QwenGemmaRegressionUsageError(`${flag} requires a value.`);
  }
  const value = /^\d+$/.test(rawValue) ? Number(rawValue) : Number.NaN;
  if (!Number.isInteger(value) || value <= 0) {
    throw new QwenGemmaRegressionUsageError(`${flag} must be a positive integer.`);
  }
  return value;
}

function readProfile(args: readonly string[], index: number): RegressionProfile {
  const value = readStringFlag(args, index, "--profile");
  if (value === "quick" || value === "real" || value === "substantial") {
    return value;
  }
  throw new QwenGemmaRegressionUsageError('--profile must be "quick", "real", or "substantial".');
}

export function parseQwenGemmaRegressionArgs(argv: readonly string[]): QwenGemmaRegressionCommand {
  if (argv.length === 1 && (argv[0] === "--help" || argv[0] === "-h")) {
    return { kind: "help" };
  }
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
    if (arg === undefined) {
      throw new QwenGemmaRegressionUsageError("argument parsing reached an empty slot.");
    }
    switch (arg) {
      case "--help":
      case "-h":
        return { kind: "help" };
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
        throw new QwenGemmaRegressionUsageError(
          arg.startsWith("-") ? `unknown option "${arg}".` : `unexpected argument "${arg}".`,
        );
    }
  }

  return { kind: "run", options };
}

function inheritedStringEnv(): Record<string, string> {
  return Object.fromEntries(
    Object.entries(process.env).filter((entry): entry is [string, string] => {
      const value = entry[1];
      return typeof value === "string";
    }),
  );
}

async function pipeReadableToProgress(
  stream: ReadableStream<Uint8Array> | null,
  progress: (text: string) => void,
): Promise<void> {
  if (stream === null) {
    return;
  }
  const decoder = new TextDecoder();
  let pending = "";
  for await (const chunk of stream) {
    pending += decoder.decode(chunk, { stream: true });
    let newline = pending.indexOf("\n");
    while (newline !== -1) {
      const line = pending.slice(0, newline);
      if (line !== "") {
        progress(line);
      }
      pending = pending.slice(newline + 1);
      newline = pending.indexOf("\n");
    }
  }
  pending += decoder.decode();
  if (pending !== "") {
    progress(pending);
  }
}

async function runCommand(
  label: string,
  args: readonly string[],
  progress: (text: string) => void,
): Promise<QwenGemmaRegressionStage> {
  progress(`[qwen-gemma-regression] ${label}: ${args.join(" ")}`);
  const child = Bun.spawn([...args], {
    cwd: new URL("..", import.meta.url).pathname,
    env: inheritedStringEnv(),
    stdout: "pipe",
    stderr: "pipe",
  });
  const stdout = pipeReadableToProgress(child.stdout, progress);
  const stderr = pipeReadableToProgress(child.stderr, progress);
  const exitCode = await child.exited;
  await Promise.all([stdout, stderr]);
  if (exitCode !== 0) {
    throw new Error(`[qwen-gemma-regression] ${label} failed with exit code ${exitCode}.`);
  }
  return { label, status: "passed" };
}

export function qwenGemmaRegressionStageSpecs(options: CliOptions): QwenGemmaRegressionStageSpec[] {
  if (options.profile === "quick") {
    return [
      {
        label: "transformer focused regressions",
        args: ["bun", "run", "--filter", "@mlxts/transformers", "regression:models"],
      },
      {
        label: "serve focused regressions",
        args: ["bun", "run", "--filter", "@mlxts/serve", "regression:serve"],
      },
    ];
  }

  if (options.profile === "real") {
    return [
      {
        label: "transformer real decode smoke",
        args: [
          "bun",
          "run",
          "packages/transformers/scripts/regression-model-matrix.ts",
          "--decode-smoke",
          "--qwen-model",
          options.qwenModel,
          "--gemma4-model",
          options.gemma4Model,
        ],
      },
      {
        label: "serve real endpoint smoke",
        args: [
          "bun",
          "run",
          "packages/serve/scripts/regression-serve-matrix.ts",
          "--real-models",
          "--qwen-model",
          options.qwenModel,
          "--gemma4-model",
          options.gemma4Model,
          "--fairness-smoke",
          "--report-dir",
          join(options.reportDir, "serve"),
          "--request-timeout-ms",
          String(options.requestTimeoutMs),
          ...(options.allowDownload ? ["--allow-download"] : []),
        ],
      },
    ];
  }

  return [
    {
      label: "transformer real decode smoke",
      args: [
        "bun",
        "run",
        "packages/transformers/scripts/regression-model-matrix.ts",
        "--decode-smoke",
        "--qwen-model",
        options.qwenModel,
        "--gemma4-model",
        options.gemma4Model,
      ],
    },
    {
      label: "serve capability smoke",
      args: [
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
      ],
    },
    {
      label: "Qwen long-context retrieval smoke",
      args: [
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
      ],
    },
  ];
}

async function runStageSpecs(
  specs: readonly QwenGemmaRegressionStageSpec[],
  progress: (text: string) => void,
): Promise<QwenGemmaRegressionStage[]> {
  const stages: QwenGemmaRegressionStage[] = [];
  for (const spec of specs) {
    stages.push(await runCommand(spec.label, spec.args, progress));
  }
  return stages;
}

export async function runQwenGemmaRegression(
  options: CliOptions,
  progress: (text: string) => void = console.error,
): Promise<QwenGemmaRegressionResult> {
  const specs = qwenGemmaRegressionStageSpecs(options);
  if (options.profile === "quick") {
    const stages = await runStageSpecs(specs, progress);
    return { profile: options.profile, reportDir: options.reportDir, stages };
  }

  using _runtimeLock = acquireRuntimeCommandLock(`regression:qwen-gemma:${options.profile}`);
  const stages = await runStageSpecs(specs, progress);
  return { profile: options.profile, reportDir: options.reportDir, stages };
}

function toon(value: string | number | boolean | null): string {
  return typeof value === "string" ? JSON.stringify(value) : String(value);
}

function formatMultilineField(name: string, value: string): string[] {
  const lines = value.split(/\r?\n/);
  if (lines.length === 1) {
    return [`  ${name}: ${toon(value)}`];
  }
  return [`  ${name}: |`, ...lines.map((line) => `    ${line}`)];
}

export function formatQwenGemmaRegressionSuccess(result: QwenGemmaRegressionResult): string {
  return [
    "qwen_gemma_regression:",
    "  status: passed",
    `  profile: ${toon(result.profile)}`,
    `  report_dir: ${toon(result.reportDir)}`,
    `  stages: ${result.stages.length}`,
    `stages[${result.stages.length}]{label,status}:`,
    ...result.stages.map((stage) => `  ${toon(stage.label)},${toon(stage.status)}`),
  ].join("\n");
}

export function formatQwenGemmaRegressionError(message: string, help: string): string {
  return ["error:", ...formatMultilineField("message", message), `help: ${toon(help)}`].join("\n");
}

export async function runQwenGemmaRegressionCommand(
  argv: readonly string[],
  runtime: QwenGemmaRegressionRuntime = {},
): Promise<number> {
  const stdout = runtime.stdout ?? console.log;
  const stderr = runtime.stderr ?? console.error;
  const runRegression = runtime.runRegression ?? runQwenGemmaRegression;
  let command: QwenGemmaRegressionCommand;

  try {
    command = parseQwenGemmaRegressionArgs(argv);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    stdout(
      formatQwenGemmaRegressionError(message, "bun run regression:qwen-gemma -- --profile quick"),
    );
    return error instanceof QwenGemmaRegressionUsageError ? 2 : 1;
  }

  if (command.kind === "help") {
    stdout(formatQwenGemmaRegressionUsage());
    return 0;
  }

  try {
    const result = await runRegression(command.options, stderr);
    stdout(formatQwenGemmaRegressionSuccess(result));
    return 0;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    stdout(
      formatQwenGemmaRegressionError(
        message,
        "review stderr and rerun the Qwen/Gemma regression after fixing the failure",
      ),
    );
    return 1;
  }
}

if (import.meta.main) {
  const exitCode = await runQwenGemmaRegressionCommand(Bun.argv.slice(2));
  process.exit(exitCode);
}
