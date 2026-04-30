#!/usr/bin/env bun

import {
  clearMemoryCache,
  getPeakMemoryBytes,
  type MxArray,
  mxAsyncEval,
  resetPeakMemory,
} from "@mlxts/core";
import { acquireRuntimeCommandLock } from "../../../scripts/runtime-command-lock";
import {
  isCoreRuntimeProfilingEnabled,
  resetCoreRuntimeProfile,
  snapshotCoreRuntimeProfile,
} from "../../core/src/runtime-profile";
import { materializeCacheState } from "../src/infrastructure/generation/helpers";
import {
  isTransformerRuntimeProfilingEnabled,
  resetTransformerRuntimeProfile,
  snapshotTransformerRuntimeProfile,
} from "../src/infrastructure/runtime-profile";
import { loadCausalLM } from "../src/load";
import {
  type BenchmarkCommandReport,
  type BenchmarkOptions,
  type BenchmarkProgress,
  type BenchmarkTarget,
  BenchmarkUsageError,
  compareAgainstBaseline,
  createDecodeMemoryTracker,
  createPromptTokenIds,
  formatBenchmarkError,
  formatBenchmarkSuccess,
  formatBenchmarkUsage,
  loadBaselines,
  mean,
  type ParsedBenchmarkArgs,
  parseBenchmarkCommand,
  printTrial,
  resolveCachedSnapshotPath,
  selectTargets,
  type TrialMetrics,
  withBenchmarkRuntimeScope,
} from "./benchmark-common";
import { type BenchmarkModel, predictGreedyToken, prefillBenchmarkCache } from "./benchmark-model";

const PERIODIC_CACHE_CLEAR_INTERVAL = 256;

type RuntimeLock = {
  [Symbol.dispose](): void;
};

type GenerationBenchmarkRuntime = {
  stdout?: (text: string) => void;
  stderr?: (text: string) => void;
  acquireLock?: () => RuntimeLock;
  runBenchmarks?: (
    parsed: ParsedBenchmarkArgs,
    progress: BenchmarkProgress,
  ) => Promise<BenchmarkCommandReport[]>;
};

function formatNsPerToken(totalNs: number, generatedTokens: number, trials: number): string {
  const tokenCount = Math.max(generatedTokens * trials, 1);
  return (totalNs / tokenCount / 1e6).toFixed(4);
}

function resetRuntimeProfiles(): void {
  resetCoreRuntimeProfile();
  resetTransformerRuntimeProfile();
}

function printRuntimeProfile(
  generationTokens: number,
  trials: number,
  progress: BenchmarkProgress,
): void {
  if (!isCoreRuntimeProfilingEnabled() && !isTransformerRuntimeProfilingEnabled()) {
    return;
  }

  const core = snapshotCoreRuntimeProfile();
  const transformer = snapshotTransformerRuntimeProfile();
  progress("Runtime profile (steady-state decode):");
  if (core.enabled) {
    progress(
      `  core: out_slot_ms_per_token=${formatNsPerToken(core.outSlot.totalNs, generationTokens, trials)} ffi_ms_per_token=${formatNsPerToken(core.ffiInvoke.totalNs, generationTokens, trials)} wrapper_ms_per_token=${formatNsPerToken(core.wrapperConstruct.totalNs, generationTokens, trials)} free_ms_per_token=${formatNsPerToken(core.explicitFree.totalNs, generationTokens, trials)}`,
    );
    const topLabels = Object.entries(core.ffiLabels)
      .sort((left, right) => right[1].totalNs - left[1].totalNs)
      .slice(0, 8)
      .map(
        ([label, metric]) =>
          `${label}:${(metric.count / Math.max(trials, 1)).toFixed(1)}/trial@${formatNsPerToken(metric.totalNs, generationTokens, trials)}ms`,
      );
    progress(`  core labels: ${topLabels.join(", ")}`);
  }
  if (transformer.enabled) {
    const activeCounters = Object.entries(transformer.counters).filter(([, count]) => count > 0);
    const formattedCounters = activeCounters.map(
      ([name, count]) =>
        `${name}:${(count / Math.max(generationTokens * trials, 1)).toFixed(3)}/token`,
    );
    progress(`  transformer: ${formattedCounters.join(", ")}`);
  }
}

function materializeCacheIfRequested(
  cache: Parameters<typeof materializeCacheState>[0],
  options: BenchmarkOptions,
): void {
  if (options.materializeCacheEachToken) {
    materializeCacheState(cache);
  }
}

function recordDecodeStep(
  index: number,
  options: BenchmarkOptions,
  decodeMemory: ReturnType<typeof createDecodeMemoryTracker>,
): boolean {
  const completedTokens = index + 1;
  if (completedTokens % options.memorySampleInterval === 0) {
    decodeMemory.sample();
  }
  if (completedTokens % PERIODIC_CACHE_CLEAR_INTERVAL === 0) {
    clearMemoryCache();
  }
  return completedTokens >= options.generationTokens;
}

function takeScheduledToken(currentToken: MxArray, nextToken: MxArray | null): MxArray {
  currentToken.free();
  if (nextToken === null) {
    throw new Error("benchmark-generation: async decode did not schedule the next token.");
  }
  return nextToken;
}

function runAsyncDecodeTrial(
  model: BenchmarkModel,
  cache: ReturnType<BenchmarkModel["createCache"]>,
  currentToken: MxArray,
  options: BenchmarkOptions,
  decodeMemory: ReturnType<typeof createDecodeMemoryTracker>,
): number {
  let decodeSyncCount = 0;
  let activeToken = currentToken;
  let nextToken: MxArray | null = null;

  try {
    mxAsyncEval(activeToken);
    for (let index = 0; index < options.generationTokens; index += 1) {
      if (index + 1 < options.generationTokens) {
        nextToken = predictGreedyToken(model, activeToken, cache);
        materializeCacheIfRequested(cache, options);
        mxAsyncEval(nextToken);
      }
      decodeSyncCount += 1;
      activeToken.item();
      if (recordDecodeStep(index, options, decodeMemory)) {
        return decodeSyncCount;
      }
      activeToken = takeScheduledToken(activeToken, nextToken);
      nextToken = null;
    }
    return decodeSyncCount;
  } finally {
    activeToken.free();
    nextToken?.free();
  }
}

function runSyncDecodeTrial(
  model: BenchmarkModel,
  cache: ReturnType<BenchmarkModel["createCache"]>,
  currentToken: MxArray,
  options: BenchmarkOptions,
  decodeMemory: ReturnType<typeof createDecodeMemoryTracker>,
): number {
  let decodeSyncCount = 0;
  let activeToken = currentToken;
  let nextToken: MxArray | null = null;

  try {
    for (let index = 0; index < options.generationTokens; index += 1) {
      decodeSyncCount += 1;
      const tokenId = activeToken.item();
      if (recordDecodeStep(index, options, decodeMemory)) {
        return decodeSyncCount;
      }
      nextToken = predictGreedyToken(model, [tokenId], cache);
      materializeCacheIfRequested(cache, options);
      activeToken.free();
      activeToken = nextToken;
      nextToken = null;
    }
    return decodeSyncCount;
  } finally {
    activeToken.free();
    nextToken?.free();
  }
}

function runSyntheticTrial(
  model: BenchmarkModel,
  promptTokenIds: readonly number[],
  options: BenchmarkOptions,
): TrialMetrics {
  resetPeakMemory();
  using cache = model.createCache();
  const promptStarted = performance.now();
  const remainingPrompt = prefillBenchmarkCache(
    model,
    promptTokenIds,
    cache,
    options.prefillStepSize,
  );
  let decodeSyncCount = 0;
  const currentToken = predictGreedyToken(model, remainingPrompt, cache);

  currentToken.item();
  if (options.materializeCacheEachToken) {
    materializeCacheState(cache);
  }
  const promptSeconds = (performance.now() - promptStarted) / 1000;
  resetRuntimeProfiles();
  const decodeMemory = createDecodeMemoryTracker();
  const decodeStarted = performance.now();
  decodeSyncCount =
    options.decodeSchedule === "async"
      ? runAsyncDecodeTrial(model, cache, currentToken, options, decodeMemory)
      : runSyncDecodeTrial(model, cache, currentToken, options, decodeMemory);

  const decodeSeconds = (performance.now() - decodeStarted) / 1000;
  return {
    promptTps: promptTokenIds.length / promptSeconds,
    generationTps: options.generationTokens / decodeSeconds,
    peakMemoryGb: getPeakMemoryBytes() / 1e9,
    ...decodeMemory.finish(options.generationTokens),
    // item() performs one blocking scalar sync per token in the steady-state decode loop.
    explicitEvalCountPerToken: decodeSyncCount / options.generationTokens,
    totalTimeSeconds: promptSeconds + decodeSeconds,
  };
}

function averageTrialMetrics(trials: readonly TrialMetrics[]): TrialMetrics {
  return {
    promptTps: mean(trials.map((trial) => trial.promptTps)),
    generationTps: mean(trials.map((trial) => trial.generationTps)),
    peakMemoryGb: mean(trials.map((trial) => trial.peakMemoryGb)),
    activeMemoryStartGb: mean(trials.map((trial) => trial.activeMemoryStartGb)),
    activeMemoryEndGb: mean(trials.map((trial) => trial.activeMemoryEndGb)),
    activeMemoryDeltaGb: mean(trials.map((trial) => trial.activeMemoryDeltaGb)),
    activeMemoryMaxGb: mean(trials.map((trial) => trial.activeMemoryMaxGb)),
    activeMemorySlopeMbPerToken: mean(trials.map((trial) => trial.activeMemorySlopeMbPerToken)),
    explicitEvalCountPerToken: mean(trials.map((trial) => trial.explicitEvalCountPerToken)),
    totalTimeSeconds: mean(trials.map((trial) => trial.totalTimeSeconds)),
  };
}

function runSyntheticBenchmarks(
  model: BenchmarkModel,
  promptTokenIds: readonly number[],
  target: BenchmarkTarget,
  options: BenchmarkOptions,
  progress: BenchmarkProgress,
): { metrics: TrialMetrics; warnings: string[] } {
  return withBenchmarkRuntimeScope(
    target.name,
    options.metalTrace,
    () => {
      runSyntheticTrial(model, promptTokenIds, options);
      clearMemoryCache();
      resetRuntimeProfiles();

      const trials: TrialMetrics[] = [];
      for (let index = 0; index < options.trials; index += 1) {
        const metrics = runSyntheticTrial(model, promptTokenIds, options);
        trials.push(metrics);
        printTrial(`Trial ${index + 1}:  `, metrics, progress);
        clearMemoryCache();
      }

      const averages = averageTrialMetrics(trials);
      printTrial("Averages: ", averages, progress);
      printRuntimeProfile(options.generationTokens, options.trials, progress);

      const warnings = compareAgainstBaseline(target, averages);
      for (const warning of warnings) {
        progress(`Warning: ${warning}`);
      }
      return { metrics: averages, warnings };
    },
    progress,
  );
}

async function benchmarkTarget(
  target: BenchmarkTarget,
  options: BenchmarkOptions,
  progress: BenchmarkProgress,
): Promise<BenchmarkCommandReport> {
  const resolvedModelSource = await resolveCachedSnapshotPath(target.model);
  const targetOptions: BenchmarkOptions = {
    ...options,
    promptTokens: target.promptTokens,
    generationTokens: target.generationTokens,
    prefillStepSize: target.prefillStepSize ?? options.prefillStepSize,
    memorySampleInterval: options.memorySampleInterval,
    decodeSchedule: options.decodeSchedule,
    materializeCacheEachToken: options.materializeCacheEachToken,
  };

  progress(
    `Benchmarking ${target.name} (${resolvedModelSource}) with prompt_tokens=${targetOptions.promptTokens}, generation_tokens=${targetOptions.generationTokens}, trials=${targetOptions.trials}, decode_schedule=${targetOptions.decodeSchedule}, materialize_cache_each_token=${targetOptions.materializeCacheEachToken}.`,
  );

  using model = await loadCausalLM(resolvedModelSource, { localFilesOnly: true });
  const promptTokenIds = createPromptTokenIds(targetOptions.promptTokens, model.config.vocabSize);
  const summary = runSyntheticBenchmarks(model, promptTokenIds, target, targetOptions, progress);
  return {
    name: target.name,
    model: target.model,
    snapshotPath: resolvedModelSource,
    promptTokens: targetOptions.promptTokens,
    generationTokens: targetOptions.generationTokens,
    prefillStepSize: targetOptions.prefillStepSize,
    trials: targetOptions.trials,
    decodeSchedule: targetOptions.decodeSchedule,
    materializeCacheEachToken: targetOptions.materializeCacheEachToken,
    metrics: summary.metrics,
    warnings: summary.warnings,
  };
}

async function runGenerationBenchmarks(
  parsed: ParsedBenchmarkArgs,
  progress: BenchmarkProgress,
): Promise<BenchmarkCommandReport[]> {
  const baselines = await loadBaselines();
  const targets = selectTargets("synthetic", baselines, parsed);
  const reports: BenchmarkCommandReport[] = [];

  for (const [index, target] of targets.entries()) {
    reports.push(await benchmarkTarget(target, parsed.options, progress));
    if (index + 1 < targets.length) {
      progress("");
    }
  }
  return reports;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function parseGenerationBenchmarkCliCommand(
  argv: readonly string[],
  stdout: (text: string) => void,
): ReturnType<typeof parseBenchmarkCommand> | number {
  try {
    return parseBenchmarkCommand(argv);
  } catch (error) {
    stdout(
      formatBenchmarkError(
        errorMessage(error),
        "bun run bench:generation -- --model <repo-or-path>",
      ),
    );
    return error instanceof BenchmarkUsageError ? 2 : 1;
  }
}

async function runGenerationBenchmarkWithLock(
  parsed: ParsedBenchmarkArgs,
  runtime: GenerationBenchmarkRuntime,
  stdout: (text: string) => void,
  stderr: (text: string) => void,
): Promise<number> {
  const acquireLock = runtime.acquireLock ?? (() => acquireRuntimeCommandLock("bench:generation"));
  const runBenchmarks = runtime.runBenchmarks ?? runGenerationBenchmarks;
  let lock: RuntimeLock | undefined;
  try {
    lock = acquireLock();
    const reports = await runBenchmarks(parsed, stderr);
    stdout(formatBenchmarkSuccess("synthetic", reports));
    return 0;
  } catch (error) {
    stdout(
      formatBenchmarkError(
        errorMessage(error),
        "rerun with --model <repo-or-path> and smaller prompt/generation rungs",
      ),
    );
    return 1;
  } finally {
    lock?.[Symbol.dispose]();
  }
}

export async function runGenerationBenchmarkCommand(
  argv: readonly string[],
  runtime: GenerationBenchmarkRuntime = {},
): Promise<number> {
  const stdout = runtime.stdout ?? console.log;
  const stderr = runtime.stderr ?? console.error;
  const command = parseGenerationBenchmarkCliCommand(argv, stdout);
  if (typeof command === "number") {
    return command;
  }
  if (command.kind === "help") {
    stdout(formatBenchmarkUsage("synthetic"));
    return 0;
  }
  return runGenerationBenchmarkWithLock(command.parsed, runtime, stdout, stderr);
}

if (import.meta.main) {
  const exitCode = await runGenerationBenchmarkCommand(Bun.argv.slice(2));
  process.exit(exitCode);
}
