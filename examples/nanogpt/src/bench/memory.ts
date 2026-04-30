#!/usr/bin/env bun

import {
  array,
  clearMemoryCache,
  getMemoryStats,
  mxEval,
  random,
  resetPeakMemory,
  reshape,
  synchronize,
  transpose,
} from "@mlxts/core";
import { crossEntropy } from "@mlxts/nn";
import { acquireRuntimeCommandLock } from "../../../../scripts/runtime-command-lock";

import { GPT_SMALL, GPT_TINY, type ModelPreset, resolveConfig } from "../config";
import { CausalSelfAttention } from "../model/causal-self-attention";
import { GPT } from "../model/gpt";
import { initializeGPT } from "../model/init";

type Scenario = "reshape-transpose" | "attention" | "gpt-loss";
type PresetName = "gpt-tiny" | "gpt-small";

class MemoryBenchmarkUsageError extends Error {}

const MEMORY_BENCHMARK_FLAGS = new Set([
  "scenario",
  "preset",
  "sequence-length",
  "iterations",
  "warmup",
  "max-end-growth-mb",
  "json",
  "help",
]);

const MEMORY_BENCHMARK_VALUE_FLAGS = new Set([
  "scenario",
  "preset",
  "sequence-length",
  "iterations",
  "warmup",
  "max-end-growth-mb",
]);

type MemorySnapshot = {
  activeBytes: number;
  cacheBytes: number;
  peakBytes: number;
  limitBytes: number;
};

type ScenarioRunner = {
  runOnce: () => MemorySnapshot;
  dispose: () => void;
};

type MemoryBenchmarkOptions = {
  maxEndGrowthMb: number;
  measuredIterations: number;
  presetName: PresetName;
  scenario: Scenario;
  sequenceLength: number;
  useJson: boolean;
  warmupIterations: number;
};

function usage(): string {
  return [
    "description: Measure nanoGPT active-memory drift for leak-prone scenarios",
    "usage[1]:",
    "  bun run bench:memory [options]",
    "options[8]{flag,description}:",
    '  "--scenario <reshape-transpose|attention|gpt-loss>","Scenario to run; default attention"',
    '  "--preset <gpt-tiny|gpt-small>","Model preset for attention/gpt-loss; default gpt-small"',
    '  "--sequence-length <n>","Sequence length; default 64"',
    '  "--iterations <n>","Measured iterations; default 20"',
    '  "--warmup <n>","Warmup iterations; default 5"',
    '  "--max-end-growth-mb <n>","Fail when end growth exceeds this; default 64"',
    '  "--json","Emit a JSON result object to stdout"',
    '  "--help","Show this help"',
    "exit_codes[3]{code,meaning}:",
    '  0,"benchmark passed or help"',
    '  1,"runtime or benchmark failure"',
    '  2,"usage error"',
  ].join("\n");
}

type ParsedArgs = {
  flags: Map<string, string>;
  valuedFlags: ReadonlySet<string>;
};

function parseArgs(argv: string[]): ParsedArgs {
  const flags = new Map<string, string>();
  const valuedFlags = new Set<string>();
  for (let index = 2; index < argv.length; index++) {
    const argument = argv[index];
    if (argument === undefined || argument === "--") {
      continue;
    }
    if (!argument.startsWith("--")) {
      throw new MemoryBenchmarkUsageError(`Unexpected positional argument "${argument}"`);
    }
    const equalsIndex = argument.indexOf("=");
    if (equalsIndex > 2) {
      const key = argument.slice(2, equalsIndex);
      flags.set(key, argument.slice(equalsIndex + 1));
      valuedFlags.add(key);
      continue;
    }

    const key = argument.slice(2);
    const value = argv[index + 1];
    if (value !== undefined && !value.startsWith("--")) {
      flags.set(key, value);
      valuedFlags.add(key);
      index += 1;
      continue;
    }
    flags.set(key, "true");
  }
  return { flags, valuedFlags };
}

function validateFlags(flags: Map<string, string>, valuedFlags: ReadonlySet<string>): void {
  for (const key of flags.keys()) {
    if (!MEMORY_BENCHMARK_FLAGS.has(key)) {
      throw new MemoryBenchmarkUsageError(`Unknown flag --${key}`);
    }
    const hasValue = valuedFlags.has(key);
    if (MEMORY_BENCHMARK_VALUE_FLAGS.has(key)) {
      if (!hasValue) {
        throw new MemoryBenchmarkUsageError(`Flag --${key} requires a value`);
      }
      continue;
    }
    if (hasValue) {
      throw new MemoryBenchmarkUsageError(`Flag --${key} does not accept a value`);
    }
  }
}

function readScenario(flags: Map<string, string>): Scenario {
  const value = flags.get("scenario") ?? "attention";
  if (value === "reshape-transpose" || value === "attention" || value === "gpt-loss") {
    return value;
  }
  throw new MemoryBenchmarkUsageError(`Unknown scenario "${value}"`);
}

function readPreset(flags: Map<string, string>): PresetName {
  const value = flags.get("preset") ?? "gpt-small";
  if (value === "gpt-tiny" || value === "gpt-small") {
    return value;
  }
  throw new MemoryBenchmarkUsageError(`Unknown preset "${value}"`);
}

function readPositiveInteger(flags: Map<string, string>, key: string, fallback: number): number {
  const value = flags.get(key);
  if (value === undefined) {
    return fallback;
  }
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new MemoryBenchmarkUsageError(`Flag --${key} must be a positive integer`);
  }
  return parsed;
}

function readPositiveNumber(flags: Map<string, string>, key: string, fallback: number): number {
  const value = flags.get(key);
  if (value === undefined) {
    return fallback;
  }
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new MemoryBenchmarkUsageError(`Flag --${key} must be > 0`);
  }
  return parsed;
}

function presetConfig(presetName: PresetName): ReturnType<typeof resolveConfig> {
  const preset: ModelPreset = presetName === "gpt-small" ? GPT_SMALL : GPT_TINY;
  return resolveConfig(preset, 65);
}

function currentMemory(): MemorySnapshot {
  const stats = getMemoryStats();
  return {
    activeBytes: stats.activeBytes,
    cacheBytes: stats.cacheBytes,
    peakBytes: stats.peakBytes,
    limitBytes: stats.limitBytes,
  };
}

function bytesToMb(bytes: number): number {
  return bytes / (1024 * 1024);
}

function prepareBenchmark(): void {
  clearMemoryCache();
  resetPeakMemory();
  synchronize();
}

function finalizeIteration(): MemorySnapshot {
  synchronize();
  return currentMemory();
}

function createReshapeTransposeRunner(sequenceLength: number): ScenarioRunner {
  const base = random.normal([1, sequenceLength, 384], "float32");
  return {
    runOnce() {
      using reshaped = reshape(base, [1, sequenceLength, 6, 64]);
      using transposed = transpose(reshaped, [0, 2, 1, 3]);
      mxEval(transposed);
      return finalizeIteration();
    },
    dispose() {
      base.free();
    },
  };
}

function createAttentionRunner(presetName: PresetName, sequenceLength: number): ScenarioRunner {
  random.seed(42);
  const config = presetConfig(presetName);
  const attention = new CausalSelfAttention(config);
  attention.eval();
  const input = random.normal([1, sequenceLength, config.nEmbd], "float32");
  return {
    runOnce() {
      using output = attention.forward(input);
      mxEval(output);
      return finalizeIteration();
    },
    dispose() {
      input.free();
      attention[Symbol.dispose]();
    },
  };
}

function tokenWindow(sequenceLength: number, vocabSize: number) {
  const values = Array.from({ length: sequenceLength + 1 }, (_, index) => index % vocabSize);
  const flatInput = array(values.slice(0, sequenceLength), "int32");
  const flatTarget = array(values.slice(1), "int32");
  return {
    input: reshape(flatInput, [1, sequenceLength]),
    target: reshape(flatTarget, [1, sequenceLength]),
    flatInput,
    flatTarget,
  };
}

function createGptLossRunner(presetName: PresetName, sequenceLength: number): ScenarioRunner {
  random.seed(42);
  const config = presetConfig(presetName);
  const model = new GPT(config);
  initializeGPT(model, config);
  model.eval();

  const { input, target, flatInput, flatTarget } = tokenWindow(sequenceLength, config.vocabSize);
  return {
    runOnce() {
      using logits = model.forward(input);
      const [batch, time, vocab] = logits.shape;
      if (batch === undefined || time === undefined || vocab === undefined) {
        throw new Error("memory bench: unexpected logits shape");
      }
      using flatLogits = reshape(logits, [batch * time, vocab]);
      using flatTargets = reshape(target, [batch * time]);
      using loss = crossEntropy(flatLogits, flatTargets);
      mxEval(loss);
      return finalizeIteration();
    },
    dispose() {
      input.free();
      target.free();
      flatInput.free();
      flatTarget.free();
      model[Symbol.dispose]();
    },
  };
}

function createScenarioRunner(
  scenario: Scenario,
  presetName: PresetName,
  sequenceLength: number,
): ScenarioRunner {
  if (scenario === "reshape-transpose") {
    return createReshapeTransposeRunner(sequenceLength);
  }
  if (scenario === "attention") {
    return createAttentionRunner(presetName, sequenceLength);
  }
  return createGptLossRunner(presetName, sequenceLength);
}

function average(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function quoteScalar(value: string): string {
  return JSON.stringify(value);
}

function formatError(message: string, code: "usage" | "runtime"): string {
  return [
    "error:",
    `  code: ${quoteScalar(code)}`,
    `  message: ${quoteScalar(message)}`,
    "help[1]:",
    '  "Run `bun run bench:memory --help` for memory benchmark options"',
  ].join("\n");
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function readOptions(flags: Map<string, string>): MemoryBenchmarkOptions {
  return {
    maxEndGrowthMb: readPositiveNumber(flags, "max-end-growth-mb", 64),
    measuredIterations: readPositiveInteger(flags, "iterations", 20),
    presetName: readPreset(flags),
    scenario: readScenario(flags),
    sequenceLength: readPositiveInteger(flags, "sequence-length", 64),
    useJson: flags.has("json"),
    warmupIterations: readPositiveInteger(flags, "warmup", 5),
  };
}

function runBenchmark(options: MemoryBenchmarkOptions): void {
  prepareBenchmark();
  const runner = createScenarioRunner(options.scenario, options.presetName, options.sequenceLength);

  try {
    const setupMemory = currentMemory();
    for (let index = 0; index < options.warmupIterations; index++) {
      runner.runOnce();
    }

    clearMemoryCache();
    resetPeakMemory();
    synchronize();
    const baseline = currentMemory();

    const activeSamples: number[] = [];
    const durationsMs: number[] = [];
    for (let index = 0; index < options.measuredIterations; index++) {
      const start = performance.now();
      const snapshot = runner.runOnce();
      durationsMs.push(performance.now() - start);
      activeSamples.push(snapshot.activeBytes);
    }

    clearMemoryCache();
    synchronize();
    const finalMemory = currentMemory();
    const maxActiveBytes = Math.max(...activeSamples);
    const result = {
      scenario: options.scenario,
      preset: options.scenario === "reshape-transpose" ? undefined : options.presetName,
      sequenceLength: options.sequenceLength,
      warmupIterations: options.warmupIterations,
      measuredIterations: options.measuredIterations,
      setupActiveMb: bytesToMb(setupMemory.activeBytes),
      baselineActiveMb: bytesToMb(baseline.activeBytes),
      finalActiveMb: bytesToMb(finalMemory.activeBytes),
      peakActiveMb: bytesToMb(maxActiveBytes),
      endGrowthMb: bytesToMb(finalMemory.activeBytes - baseline.activeBytes),
      averageIterationMs: average(durationsMs),
    };

    if (options.useJson) {
      process.stdout.write(`${JSON.stringify(result)}\n`);
    } else {
      process.stdout.write(`scenario=${result.scenario}\n`);
      if (result.preset !== undefined) {
        process.stdout.write(`preset=${result.preset}\n`);
      }
      process.stdout.write(`sequenceLength=${result.sequenceLength}\n`);
      process.stdout.write(
        `activeMemoryMb=${result.baselineActiveMb.toFixed(2)} -> ${result.finalActiveMb.toFixed(2)} (peak ${result.peakActiveMb.toFixed(2)})\n`,
      );
      process.stdout.write(
        `endGrowthMb=${result.endGrowthMb.toFixed(2)} averageIterationMs=${result.averageIterationMs.toFixed(2)}\n`,
      );
    }

    if (result.endGrowthMb > options.maxEndGrowthMb) {
      throw new Error(
        `Memory benchmark failed: end growth ${result.endGrowthMb.toFixed(2)} MB exceeded threshold ${options.maxEndGrowthMb.toFixed(2)} MB`,
      );
    }
  } finally {
    runner.dispose();
  }
}

export function main(argv = process.argv): number {
  try {
    const { flags, valuedFlags } = parseArgs(argv);
    validateFlags(flags, valuedFlags);
    if (flags.has("help")) {
      process.stdout.write(`${usage()}\n`);
      return 0;
    }
    const options = readOptions(flags);

    {
      using _runtimeLock = acquireRuntimeCommandLock("bench:memory");
      runBenchmark(options);
    }
    return 0;
  } catch (error) {
    const code = error instanceof MemoryBenchmarkUsageError ? "usage" : "runtime";
    process.stdout.write(`${formatError(errorMessage(error), code)}\n`);
    if (code === "runtime" && error instanceof Error && error.stack !== undefined) {
      process.stderr.write(`${error.stack}\n`);
    }
    return code === "usage" ? 2 : 1;
  }
}

process.exit(main());
