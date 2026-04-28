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

function usage(): string {
  return [
    "Usage: bun run packages/transformers/scripts/regression-model-matrix.ts [options]",
    "",
    "Options:",
    "  --real-models                 Load real cached Qwen and Gemma 4 checkpoints.",
    "  --decode-smoke                Also run a short local decode benchmark for real models.",
    "  --qwen-model <id>             Qwen model id/path.",
    "  --gemma4-model <id>           Gemma 4 model id/path.",
    "  --qwen-max-active-gb <n>      Fail if Qwen load active memory exceeds this.",
    "  --gemma4-max-active-gb <n>    Fail if Gemma 4 load active memory exceeds this.",
  ].join("\n");
}

function readStringFlag(args: string[], index: number, flag: string): string {
  const value = args[index + 1];
  if (value === undefined || value.startsWith("--")) {
    throw new Error(`${flag} requires a value.\n\n${usage()}`);
  }
  return value;
}

function readNumberFlag(args: string[], index: number, flag: string): number {
  const value = Number(readStringFlag(args, index, flag));
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${flag} must be a positive number.`);
  }
  return value;
}

function parseArgs(argv: string[]): CliOptions {
  const options: CliOptions = {
    realModels: false,
    decodeSmoke: false,
    qwenModel: "mlx-community/Qwen3.6-27B-4bit",
    gemma4Model: "google/gemma-4-E2B-it",
    qwenMaxActiveGb: 16.5,
    gemma4MaxActiveGb: 10.5,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
      case "--help":
      case "-h":
        console.log(usage());
        Bun.exit(0);
        return options;
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
  console.log(`[regression] ${label}: ${args.join(" ")}`);
  const child = Bun.spawn(args, {
    cwd: new URL("../../..", import.meta.url).pathname,
    env: inheritedStringEnv(),
    stdout: "inherit",
    stderr: "inherit",
  });
  const exitCode = await child.exited;
  if (exitCode !== 0) {
    throw new Error(`[regression] ${label} failed with exit code ${exitCode}.`);
  }
}

async function runCapturedCommand(label: string, args: readonly string[]): Promise<string> {
  console.log(`[regression] ${label}: ${args.join(" ")}`);
  const child = Bun.spawn(args, {
    cwd: new URL("../../..", import.meta.url).pathname,
    env: inheritedStringEnv(),
    stdout: "pipe",
    stderr: "pipe",
  });
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(child.stdout).text(),
    new Response(child.stderr).text(),
    child.exited,
  ]);
  if (stdout.length > 0) {
    console.log(stdout.trimEnd());
  }
  if (stderr.length > 0) {
    console.error(stderr.trimEnd());
  }
  if (exitCode !== 0) {
    throw new Error(`[regression] ${label} failed with exit code ${exitCode}.`);
  }
  return `${stdout}\n${stderr}`;
}

async function runFocusedUnitChecks(): Promise<void> {
  await runCommand("focused unit checks", [
    "bun",
    "test",
    "packages/nn/src/quantized/quantized-embedding.test.ts",
    "packages/quantize/src/quantize-module.test.ts",
    "packages/quantize/src/setup-quantized-module.test.ts",
    "packages/transformers/src/families/qwen3_5/model.test.ts",
    "packages/transformers/src/families/qwen3_5/weights.test.ts",
    "packages/transformers/src/families/gemma4/model.test.ts",
    "packages/transformers/src/families/gemma4/weights.test.ts",
    "packages/transformers/src/load.test.ts",
  ]);
}

function extractMetric(line: string, key: string): number {
  const match = new RegExp(`${key}=(-?\\d+(?:\\.\\d+)?)`).exec(line);
  const value = match?.[1];
  if (value === undefined) {
    throw new Error(`[regression] decode smoke missing ${key} in: ${line}`);
  }
  return Number(value);
}

function parseDecodeSmokeMetrics(output: string, label: string): DecodeSmokeMetrics {
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

function assertMemoryBudget(report: LoadedMemoryReport, maxActiveGb: number): void {
  console.log(
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

async function runRealModelLoads(options: CliOptions): Promise<void> {
  assertMemoryBudget(await loadMemoryReport(options.qwenModel), options.qwenMaxActiveGb);
  assertMemoryBudget(await loadMemoryReport(options.gemma4Model), options.gemma4MaxActiveGb);
}

async function runDecodeSmoke(options: CliOptions): Promise<void> {
  for (const model of [options.qwenModel, options.gemma4Model]) {
    const output = await runCapturedCommand(`decode smoke ${model}`, [
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
    ]);
    assertDecodeSmokeBudget(
      model,
      parseDecodeSmokeMetrics(output, `decode smoke ${model}`),
      decodeBudgetForModel(model, options),
    );
  }
}

async function main(): Promise<void> {
  const options = parseArgs(Bun.argv.slice(2));
  if (options.realModels || options.decodeSmoke) {
    using _runtimeLock = acquireRuntimeCommandLock("regression:models");
    await runRegression(options);
    return;
  }

  await runRegression(options);
}

async function runRegression(options: CliOptions): Promise<void> {
  await runFocusedUnitChecks();
  if (options.realModels) {
    await runRealModelLoads(options);
  }
  if (options.decodeSmoke) {
    await runDecodeSmoke(options);
  }
}

await main();
