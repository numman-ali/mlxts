#!/usr/bin/env bun

import { acquireRuntimeCommandLock } from "../../../scripts/runtime-command-lock";
import {
  clearMemoryCache,
  getMemoryStats,
  getPeakMemoryBytes,
  mxEval,
  resetPeakMemory,
  treeFlatten,
} from "../../core/src/index";
import { loadCausalLM } from "../src/load";

type CliOptions = {
  realModels: boolean;
  decodeSmoke: boolean;
  qwenModel: string;
  gemma4Model: string;
  qwenMaxActiveGb: number;
  gemma4MaxActiveGb: number;
};

type TransformersModelRegressionCommand = { kind: "help" } | { kind: "run"; options: CliOptions };

type RuntimeLock = {
  [Symbol.dispose](): void;
};

type TransformersModelRegressionRuntime = {
  stdout?: (text: string) => void;
  stderr?: (text: string) => void;
  acquireLock?: () => RuntimeLock;
  runRegression?: (
    options: CliOptions,
    progress: (text: string) => void,
  ) => Promise<TransformersModelRegressionResult>;
};

type RegressionStage = {
  label: string;
  status: "passed";
};

type LoadedMemoryReport = {
  model: string;
  modelType: string;
  activeGb: number;
  cacheGb: number;
  peakGb: number;
  parameterCount: number;
};

type DecodeSmokeMetrics = {
  promptTps: number;
  generationTps: number;
  peakMemoryGb: number;
  activeSlopeMbPerToken: number;
  explicitEvalCountPerToken: number;
};

type DecodeSmokeBudget = {
  minPromptTps: number;
  minGenerationTps: number;
  maxPeakMemoryGb: number;
  maxActiveSlopeMbPerToken: number;
  maxExplicitEvalCountPerToken: number;
};

type DecodeSmokeReport = {
  model: string;
  metrics: DecodeSmokeMetrics;
};

type TransformersModelRegressionResult = {
  focusedChecks: "passed";
  realModels: boolean;
  decodeSmoke: boolean;
  stages: RegressionStage[];
  memoryReports: LoadedMemoryReport[];
  decodeReports: DecodeSmokeReport[];
};

export const TRANSFORMERS_MODEL_REGRESSION_FOCUSED_TESTS = [
  "packages/nn/src/quantized/quantized-embedding.test.ts",
  "packages/quantize/src/quantize-module.test.ts",
  "packages/quantize/src/setup-quantized-module.test.ts",
  "packages/transformers/scripts/regression-model-matrix.test.ts",
  "packages/transformers/src/families/qwen3_5/model.test.ts",
  "packages/transformers/src/families/qwen3_5/weights.test.ts",
  "packages/transformers/src/families/gemma4/model.test.ts",
  "packages/transformers/src/families/gemma4/weights.test.ts",
  "packages/transformers/src/load.test.ts",
] as const;

class TransformersModelRegressionUsageError extends Error {}

export function formatTransformersModelRegressionUsage(): string {
  return [
    "description: Run the @mlxts/transformers Qwen/Gemma regression matrix",
    "usage[3]:",
    "  bun run --filter '@mlxts/transformers' regression:models",
    "  bun run packages/transformers/scripts/regression-model-matrix.ts --real-models",
    "  bun run packages/transformers/scripts/regression-model-matrix.ts --decode-smoke",
    "options[7]{flag,description}:",
    '  "--real-models","Load cached Qwen and Gemma 4 checkpoints"',
    '  "--decode-smoke","Run short local decode benchmarks; implies --real-models"',
    '  "--qwen-model <id>","Qwen model id/path"',
    '  "--gemma4-model <id>","Gemma 4 model id/path"',
    '  "--qwen-max-active-gb <n>","Fail if Qwen load active memory exceeds this; default 16.5"',
    '  "--gemma4-max-active-gb <n>","Fail if Gemma 4 load active memory exceeds this; default 10.5"',
    '  "--help","Show this help"',
    "exit_codes[3]{code,meaning}:",
    '  0,"regression passed"',
    '  1,"runtime or regression failure"',
    '  2,"usage error"',
  ].join("\n");
}

function defaultOptions(): CliOptions {
  return {
    realModels: false,
    decodeSmoke: false,
    qwenModel: "mlx-community/Qwen3.6-27B-4bit",
    gemma4Model: "google/gemma-4-E2B-it",
    qwenMaxActiveGb: 16.5,
    gemma4MaxActiveGb: 10.5,
  };
}

function readStringFlag(args: readonly string[], index: number, flag: string): string {
  const value = args[index + 1];
  if (value === undefined || value.trim() === "" || value.startsWith("-")) {
    throw new TransformersModelRegressionUsageError(`${flag} requires a value.`);
  }
  return value;
}

function readNumberFlag(args: readonly string[], index: number, flag: string): number {
  const rawValue = args[index + 1]?.trim();
  if (rawValue === undefined || rawValue === "" || rawValue.startsWith("--")) {
    throw new TransformersModelRegressionUsageError(`${flag} requires a value.`);
  }
  const value = Number(rawValue);
  if (!Number.isFinite(value) || value <= 0) {
    throw new TransformersModelRegressionUsageError(`${flag} must be a positive number.`);
  }
  return value;
}

export function parseTransformersModelRegressionArgs(
  argv: readonly string[],
): TransformersModelRegressionCommand {
  if (argv.length === 1 && (argv[0] === "--help" || argv[0] === "-h")) {
    return { kind: "help" };
  }
  const options = defaultOptions();

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === undefined) {
      throw new TransformersModelRegressionUsageError("argument parsing reached an empty slot.");
    }
    switch (arg) {
      case "--help":
      case "-h":
        return { kind: "help" };
      case "--real-models":
        options.realModels = true;
        break;
      case "--decode-smoke":
        options.decodeSmoke = true;
        options.realModels = true;
        break;
      case "--qwen-model":
        options.qwenModel = readStringFlag(argv, index, arg);
        index += 1;
        break;
      case "--gemma4-model":
        options.gemma4Model = readStringFlag(argv, index, arg);
        index += 1;
        break;
      case "--qwen-max-active-gb":
        options.qwenMaxActiveGb = readNumberFlag(argv, index, arg);
        index += 1;
        break;
      case "--gemma4-max-active-gb":
        options.gemma4MaxActiveGb = readNumberFlag(argv, index, arg);
        index += 1;
        break;
      default:
        throw new TransformersModelRegressionUsageError(
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
  capture: string[],
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
        capture.push(line);
      }
      pending = pending.slice(newline + 1);
      newline = pending.indexOf("\n");
    }
  }
  pending += decoder.decode();
  if (pending !== "") {
    progress(pending);
    capture.push(pending);
  }
}

async function runCommand(
  label: string,
  args: readonly string[],
  progress: (text: string) => void,
): Promise<RegressionStage> {
  progress(`[regression] ${label}: ${args.join(" ")}`);
  const child = Bun.spawn(args, {
    cwd: new URL("../../..", import.meta.url).pathname,
    env: inheritedStringEnv(),
    stdout: "pipe",
    stderr: "pipe",
  });
  const captured: string[] = [];
  const stdout = pipeReadableToProgress(child.stdout, progress, captured);
  const stderr = pipeReadableToProgress(child.stderr, progress, captured);
  const exitCode = await child.exited;
  await Promise.all([stdout, stderr]);
  if (exitCode !== 0) {
    throw new Error(`[regression] ${label} failed with exit code ${exitCode}.`);
  }
  return { label, status: "passed" };
}

async function runCapturedCommand(
  label: string,
  args: readonly string[],
  progress: (text: string) => void,
): Promise<string> {
  progress(`[regression] ${label}: ${args.join(" ")}`);
  const child = Bun.spawn(args, {
    cwd: new URL("../../..", import.meta.url).pathname,
    env: inheritedStringEnv(),
    stdout: "pipe",
    stderr: "pipe",
  });
  const captured: string[] = [];
  const stdout = pipeReadableToProgress(child.stdout, progress, captured);
  const stderr = pipeReadableToProgress(child.stderr, progress, captured);
  const exitCode = await child.exited;
  await Promise.all([stdout, stderr]);
  if (exitCode !== 0) {
    throw new Error(`[regression] ${label} failed with exit code ${exitCode}.`);
  }
  return captured.join("\n");
}

async function runFocusedUnitChecks(progress: (text: string) => void): Promise<RegressionStage> {
  return await runCommand(
    "focused unit checks",
    ["bun", "test", ...TRANSFORMERS_MODEL_REGRESSION_FOCUSED_TESTS],
    progress,
  );
}

function extractMetric(line: string, key: string): number {
  const match = new RegExp(`${key}=(-?\\d+(?:\\.\\d+)?)`).exec(line);
  const value = match?.[1];
  if (value === undefined) {
    throw new Error(`[regression] decode smoke missing ${key} in: ${line}`);
  }
  return Number(value);
}

export function parseDecodeSmokeMetrics(output: string, label: string): DecodeSmokeMetrics {
  const averagesLine = output
    .split(/\r?\n/)
    .findLast((line) => line.trimStart().startsWith("Averages:"));
  if (averagesLine === undefined) {
    throw new Error(`[regression] ${label} did not print benchmark averages.`);
  }
  return {
    promptTps: extractMetric(averagesLine, "prompt_tps"),
    generationTps: extractMetric(averagesLine, "generation_tps"),
    peakMemoryGb: extractMetric(averagesLine, "peak_memory"),
    activeSlopeMbPerToken: extractMetric(averagesLine, "active_slope_mb_per_token"),
    explicitEvalCountPerToken: extractMetric(averagesLine, "evals_per_token"),
  };
}

function assertDecodeSmokeBudget(
  model: string,
  metrics: DecodeSmokeMetrics,
  budget: DecodeSmokeBudget,
): void {
  const failures: string[] = [];
  if (metrics.promptTps < budget.minPromptTps) {
    failures.push(`prompt_tps ${metrics.promptTps.toFixed(3)} < ${budget.minPromptTps.toFixed(3)}`);
  }
  if (metrics.generationTps < budget.minGenerationTps) {
    failures.push(
      `generation_tps ${metrics.generationTps.toFixed(3)} < ${budget.minGenerationTps.toFixed(3)}`,
    );
  }
  if (metrics.peakMemoryGb > budget.maxPeakMemoryGb) {
    failures.push(
      `peak_memory ${metrics.peakMemoryGb.toFixed(3)}GB > ${budget.maxPeakMemoryGb.toFixed(3)}GB`,
    );
  }
  if (metrics.activeSlopeMbPerToken > budget.maxActiveSlopeMbPerToken) {
    failures.push(
      `active_slope ${metrics.activeSlopeMbPerToken.toFixed(
        3,
      )} MB/token > ${budget.maxActiveSlopeMbPerToken.toFixed(3)} MB/token`,
    );
  }
  if (metrics.explicitEvalCountPerToken > budget.maxExplicitEvalCountPerToken) {
    failures.push(
      `evals_per_token ${metrics.explicitEvalCountPerToken.toFixed(
        3,
      )} > ${budget.maxExplicitEvalCountPerToken.toFixed(3)}`,
    );
  }

  if (failures.length > 0) {
    throw new Error(`[regression] ${model} decode smoke failed: ${failures.join("; ")}.`);
  }
}

function decodeBudgetForModel(model: string, options: CliOptions): DecodeSmokeBudget {
  if (model === options.gemma4Model) {
    return {
      minPromptTps: 4_000,
      minGenerationTps: 60,
      maxPeakMemoryGb: 12,
      maxActiveSlopeMbPerToken: 1,
      maxExplicitEvalCountPerToken: 1.05,
    };
  }

  return {
    minPromptTps: 150,
    minGenerationTps: 20,
    maxPeakMemoryGb: 20,
    maxActiveSlopeMbPerToken: 2,
    maxExplicitEvalCountPerToken: 1.05,
  };
}

async function loadMemoryReport(modelSource: string): Promise<LoadedMemoryReport> {
  clearMemoryCache();
  resetPeakMemory();

  using model = await loadCausalLM(modelSource, { localFilesOnly: true });
  const parameters = treeFlatten(model.parameters()).map(([, tensor]) => tensor);
  mxEval(...parameters);
  const memory = getMemoryStats();
  return {
    model: modelSource,
    modelType: model.config.modelType,
    activeGb: memory.activeBytes / 1e9,
    cacheGb: memory.cacheBytes / 1e9,
    peakGb: getPeakMemoryBytes() / 1e9,
    parameterCount: parameters.length,
  };
}

function assertMemoryBudget(
  report: LoadedMemoryReport,
  maxActiveGb: number,
  progress: (text: string) => void,
): void {
  progress(
    `[regression] ${report.model} model_type=${report.modelType} active=${report.activeGb.toFixed(
      3,
    )}GB cache=${report.cacheGb.toFixed(3)}GB peak=${report.peakGb.toFixed(3)}GB params=${
      report.parameterCount
    }`,
  );
  if (report.activeGb > maxActiveGb) {
    throw new Error(
      `[regression] ${report.model} active memory ${report.activeGb.toFixed(
        3,
      )}GB exceeded budget ${maxActiveGb.toFixed(3)}GB.`,
    );
  }
}

async function runRealModelLoads(
  options: CliOptions,
  progress: (text: string) => void,
): Promise<LoadedMemoryReport[]> {
  const reports = [
    await loadMemoryReport(options.qwenModel),
    await loadMemoryReport(options.gemma4Model),
  ] as const;
  const [qwenReport, gemma4Report] = reports;
  assertMemoryBudget(qwenReport, options.qwenMaxActiveGb, progress);
  assertMemoryBudget(gemma4Report, options.gemma4MaxActiveGb, progress);
  return [...reports];
}

async function runDecodeSmoke(
  options: CliOptions,
  progress: (text: string) => void,
): Promise<DecodeSmokeReport[]> {
  const reports: DecodeSmokeReport[] = [];
  for (const model of [options.qwenModel, options.gemma4Model]) {
    const output = await runCapturedCommand(
      `decode smoke ${model}`,
      [
        "bun",
        "run",
        "bench:generation:parity",
        "--model",
        model,
        "--prompt-tokens",
        "1024",
        "--generation-tokens",
        "128",
        "--trials",
        "1",
        "--memory-sample-interval",
        "16",
        "--skip-mlx-lm-reference",
      ],
      progress,
    );
    const metrics = parseDecodeSmokeMetrics(output, `decode smoke ${model}`);
    assertDecodeSmokeBudget(model, metrics, decodeBudgetForModel(model, options));
    reports.push({ model, metrics });
  }
  return reports;
}

export async function runTransformersModelRegression(
  options: CliOptions,
  progress: (text: string) => void = console.error,
): Promise<TransformersModelRegressionResult> {
  const stages = [await runFocusedUnitChecks(progress)];
  const memoryReports = options.realModels ? await runRealModelLoads(options, progress) : [];
  const decodeReports = options.decodeSmoke ? await runDecodeSmoke(options, progress) : [];
  return {
    focusedChecks: "passed",
    realModels: options.realModels,
    decodeSmoke: options.decodeSmoke,
    stages,
    memoryReports,
    decodeReports,
  };
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

export function formatTransformersModelRegressionSuccess(
  result: TransformersModelRegressionResult,
): string {
  const lines = [
    "transformers_model_regression:",
    "  status: passed",
    `  real_models: ${toon(result.realModels)}`,
    `  decode_smoke: ${toon(result.decodeSmoke)}`,
    `  stages: ${result.stages.length}`,
    `  memory_reports: ${result.memoryReports.length}`,
    `  decode_reports: ${result.decodeReports.length}`,
    `stages[${result.stages.length}]{label,status}:`,
    ...result.stages.map((stage) => `  ${toon(stage.label)},${toon(stage.status)}`),
  ];
  if (result.memoryReports.length > 0) {
    lines.push(
      `memory_reports[${result.memoryReports.length}]{model,model_type,active_gb,cache_gb,peak_gb,parameter_count}:`,
      ...result.memoryReports.map((report) =>
        [
          toon(report.model),
          toon(report.modelType),
          report.activeGb.toFixed(3),
          report.cacheGb.toFixed(3),
          report.peakGb.toFixed(3),
          String(report.parameterCount),
        ].join(","),
      ),
    );
  }
  if (result.decodeReports.length > 0) {
    lines.push(
      `decode_reports[${result.decodeReports.length}]{model,prompt_tps,generation_tps,peak_memory_gb,active_slope_mb_per_token,evals_per_token}:`,
      ...result.decodeReports.map((report) =>
        [
          toon(report.model),
          report.metrics.promptTps.toFixed(3),
          report.metrics.generationTps.toFixed(3),
          report.metrics.peakMemoryGb.toFixed(3),
          report.metrics.activeSlopeMbPerToken.toFixed(3),
          report.metrics.explicitEvalCountPerToken.toFixed(3),
        ].join(","),
      ),
    );
  }
  return lines.join("\n");
}

export function formatTransformersModelRegressionError(message: string, help: string): string {
  return ["error:", ...formatMultilineField("message", message), `help: ${toon(help)}`].join("\n");
}

async function runWithOptionalRuntimeLock(
  options: CliOptions,
  progress: (text: string) => void,
  acquireLock: () => RuntimeLock,
  runRegression: (
    options: CliOptions,
    progress: (text: string) => void,
  ) => Promise<TransformersModelRegressionResult>,
): Promise<TransformersModelRegressionResult> {
  let lock: RuntimeLock | undefined;
  try {
    if (options.realModels || options.decodeSmoke) {
      lock = acquireLock();
    }
    return await runRegression(options, progress);
  } finally {
    lock?.[Symbol.dispose]();
  }
}

export async function runTransformersModelRegressionCommand(
  argv: readonly string[],
  runtime: TransformersModelRegressionRuntime = {},
): Promise<number> {
  const stdout = runtime.stdout ?? console.log;
  const stderr = runtime.stderr ?? console.error;
  const acquireLock = runtime.acquireLock ?? (() => acquireRuntimeCommandLock("regression:models"));
  const runRegression = runtime.runRegression ?? runTransformersModelRegression;
  let command: TransformersModelRegressionCommand;

  try {
    command = parseTransformersModelRegressionArgs(argv);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    stdout(
      formatTransformersModelRegressionError(
        message,
        "bun run packages/transformers/scripts/regression-model-matrix.ts --help",
      ),
    );
    return error instanceof TransformersModelRegressionUsageError ? 2 : 1;
  }

  if (command.kind === "help") {
    stdout(formatTransformersModelRegressionUsage());
    return 0;
  }

  try {
    const result = await runWithOptionalRuntimeLock(
      command.options,
      stderr,
      acquireLock,
      runRegression,
    );
    stdout(formatTransformersModelRegressionSuccess(result));
    return 0;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    stdout(
      formatTransformersModelRegressionError(
        message,
        "review stderr and rerun the transformers model regression after fixing the failure",
      ),
    );
    return 1;
  }
}

if (import.meta.main) {
  const exitCode = await runTransformersModelRegressionCommand(Bun.argv.slice(2));
  process.exit(exitCode);
}
