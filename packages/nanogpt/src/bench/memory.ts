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

import { GPT_SMALL, GPT_TINY, type ModelPreset, resolveConfig } from "../config";
import { CausalSelfAttention } from "../model/causal-self-attention";
import { GPT } from "../model/gpt";
import { initializeGPT } from "../model/init";

type Scenario = "reshape-transpose" | "attention" | "gpt-loss";
type PresetName = "gpt-tiny" | "gpt-small";

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

function usage(): string {
  return `nanogpt memory benchmark

Usage:
  bun run packages/nanogpt/src/bench/memory.ts [options]

Options:
  --scenario <reshape-transpose|attention|gpt-loss>   Scenario to run (default: attention)
  --preset <gpt-tiny|gpt-small>                       Model preset for attention/gpt-loss (default: gpt-small)
  --sequence-length <n>                               Sequence length (default: 64)
  --iterations <n>                                    Measured iterations (default: 20)
  --warmup <n>                                        Warmup iterations (default: 5)
  --max-end-growth-mb <n>                             Fail if end growth exceeds this (default: 64)
  --json                                              Emit JSON
`;
}

function parseArgs(argv: string[]): Map<string, string> {
  const flags = new Map<string, string>();
  for (let index = 2; index < argv.length; index++) {
    const argument = argv[index];
    if (argument === undefined || !argument.startsWith("--")) {
      continue;
    }
    const key = argument.slice(2);
    const value = argv[index + 1];
    if (value !== undefined && !value.startsWith("--")) {
      flags.set(key, value);
      index += 1;
      continue;
    }
    flags.set(key, "true");
  }
  return flags;
}

function readScenario(flags: Map<string, string>): Scenario {
  const value = flags.get("scenario") ?? "attention";
  if (value === "reshape-transpose" || value === "attention" || value === "gpt-loss") {
    return value;
  }
  throw new Error(`Unknown scenario "${value}"`);
}

function readPreset(flags: Map<string, string>): PresetName {
  const value = flags.get("preset") ?? "gpt-small";
  if (value === "gpt-tiny" || value === "gpt-small") {
    return value;
  }
  throw new Error(`Unknown preset "${value}"`);
}

function readPositiveInteger(flags: Map<string, string>, key: string, fallback: number): number {
  const value = flags.get(key);
  if (value === undefined) {
    return fallback;
  }
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`Flag --${key} must be a positive integer`);
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
    throw new Error(`Flag --${key} must be > 0`);
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

function main(): void {
  const flags = parseArgs(process.argv);
  if (flags.has("help")) {
    process.stdout.write(`${usage()}\n`);
    return;
  }

  const useJson = flags.has("json");
  const scenario = readScenario(flags);
  const presetName = readPreset(flags);
  const sequenceLength = readPositiveInteger(flags, "sequence-length", 64);
  const warmupIterations = readPositiveInteger(flags, "warmup", 5);
  const measuredIterations = readPositiveInteger(flags, "iterations", 20);
  const maxEndGrowthMb = readPositiveNumber(flags, "max-end-growth-mb", 64);

  prepareBenchmark();
  const runner = createScenarioRunner(scenario, presetName, sequenceLength);

  try {
    const setupMemory = currentMemory();
    for (let index = 0; index < warmupIterations; index++) {
      runner.runOnce();
    }

    clearMemoryCache();
    resetPeakMemory();
    synchronize();
    const baseline = currentMemory();

    const activeSamples: number[] = [];
    const durationsMs: number[] = [];
    for (let index = 0; index < measuredIterations; index++) {
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
      scenario,
      preset: scenario === "reshape-transpose" ? undefined : presetName,
      sequenceLength,
      warmupIterations,
      measuredIterations,
      setupActiveMb: bytesToMb(setupMemory.activeBytes),
      baselineActiveMb: bytesToMb(baseline.activeBytes),
      finalActiveMb: bytesToMb(finalMemory.activeBytes),
      peakActiveMb: bytesToMb(maxActiveBytes),
      endGrowthMb: bytesToMb(finalMemory.activeBytes - baseline.activeBytes),
      averageIterationMs: average(durationsMs),
    };

    if (useJson) {
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

    if (result.endGrowthMb > maxEndGrowthMb) {
      throw new Error(
        `Memory benchmark failed: end growth ${result.endGrowthMb.toFixed(2)} MB exceeded threshold ${maxEndGrowthMb.toFixed(2)} MB`,
      );
    }
  } finally {
    runner.dispose();
  }
}

main();
