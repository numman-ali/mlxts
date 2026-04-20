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
import {
  isTransformerRuntimeProfilingEnabled,
  resetTransformerRuntimeProfile,
  snapshotTransformerRuntimeProfile,
} from "../src/infrastructure/runtime-profile";
import { loadCausalLM } from "../src/load";
import {
  type BenchmarkOptions,
  type BenchmarkTarget,
  compareAgainstBaseline,
  createPromptTokenIds,
  loadBaselines,
  mean,
  parseBenchmarkArgs,
  printTrial,
  resolveCachedSnapshotPath,
  selectTargets,
  type TrialMetrics,
  withBenchmarkRuntimeScope,
} from "./benchmark-common";
import { type BenchmarkModel, predictGreedyToken, prefillBenchmarkCache } from "./benchmark-model";

const PERIODIC_CACHE_CLEAR_INTERVAL = 256;

function formatNsPerToken(totalNs: number, generatedTokens: number, trials: number): string {
  const tokenCount = Math.max(generatedTokens * trials, 1);
  return (totalNs / tokenCount / 1e6).toFixed(4);
}

function resetRuntimeProfiles(): void {
  resetCoreRuntimeProfile();
  resetTransformerRuntimeProfile();
}

function printRuntimeProfile(generationTokens: number, trials: number): void {
  if (!isCoreRuntimeProfilingEnabled() && !isTransformerRuntimeProfilingEnabled()) {
    return;
  }

  const core = snapshotCoreRuntimeProfile();
  const transformer = snapshotTransformerRuntimeProfile();
  console.log("Runtime profile (steady-state decode):");
  if (core.enabled) {
    console.log(
      `  core: out_slot_ms_per_token=${formatNsPerToken(core.outSlot.totalNs, generationTokens, trials)} ffi_ms_per_token=${formatNsPerToken(core.ffiInvoke.totalNs, generationTokens, trials)} wrapper_ms_per_token=${formatNsPerToken(core.wrapperConstruct.totalNs, generationTokens, trials)} free_ms_per_token=${formatNsPerToken(core.explicitFree.totalNs, generationTokens, trials)}`,
    );
    const topLabels = Object.entries(core.ffiLabels)
      .sort((left, right) => right[1].totalNs - left[1].totalNs)
      .slice(0, 8)
      .map(
        ([label, metric]) =>
          `${label}:${(metric.count / Math.max(trials, 1)).toFixed(1)}/trial@${formatNsPerToken(metric.totalNs, generationTokens, trials)}ms`,
      );
    console.log(`  core labels: ${topLabels.join(", ")}`);
  }
  if (transformer.enabled) {
    const activeCounters = Object.entries(transformer.counters).filter(([, count]) => count > 0);
    const formattedCounters = activeCounters.map(
      ([name, count]) =>
        `${name}:${(count / Math.max(generationTokens * trials, 1)).toFixed(3)}/token`,
    );
    console.log(`  transformer: ${formattedCounters.join(", ")}`);
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
  let currentToken = predictGreedyToken(model, remainingPrompt, cache);

  currentToken.item();
  const promptSeconds = (performance.now() - promptStarted) / 1000;
  resetRuntimeProfiles();
  const decodeStarted = performance.now();
  let nextToken: MxArray | null = null;

  try {
    mxAsyncEval(currentToken);

    for (let index = 0; index < options.generationTokens; index += 1) {
      if (index + 1 < options.generationTokens) {
        nextToken = predictGreedyToken(model, currentToken, cache);
        mxAsyncEval(nextToken);
      }

      decodeSyncCount += 1;
      currentToken.item();

      if ((index + 1) % PERIODIC_CACHE_CLEAR_INTERVAL === 0) {
        clearMemoryCache();
      }
      if (index + 1 >= options.generationTokens) {
        break;
      }

      currentToken.free();
      const scheduledToken = nextToken;
      nextToken = null;
      if (scheduledToken === null) {
        throw new Error("benchmark-generation: async decode did not schedule the next token.");
      }
      currentToken = scheduledToken;
    }
  } finally {
    currentToken.free();
    nextToken?.free();
  }

  const decodeSeconds = (performance.now() - decodeStarted) / 1000;
  return {
    promptTps: promptTokenIds.length / promptSeconds,
    generationTps: options.generationTokens / decodeSeconds,
    peakMemoryGb: getPeakMemoryBytes() / 1e9,
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
    explicitEvalCountPerToken: mean(trials.map((trial) => trial.explicitEvalCountPerToken)),
    totalTimeSeconds: mean(trials.map((trial) => trial.totalTimeSeconds)),
  };
}

function runSyntheticBenchmarks(
  model: BenchmarkModel,
  promptTokenIds: readonly number[],
  target: BenchmarkTarget,
  options: BenchmarkOptions,
): void {
  withBenchmarkRuntimeScope(target.name, options.metalTrace, () => {
    runSyntheticTrial(model, promptTokenIds, options);
    clearMemoryCache();
    resetRuntimeProfiles();

    const trials: TrialMetrics[] = [];
    for (let index = 0; index < options.trials; index += 1) {
      const metrics = runSyntheticTrial(model, promptTokenIds, options);
      trials.push(metrics);
      printTrial(`Trial ${index + 1}:  `, metrics);
      clearMemoryCache();
    }

    const averages = averageTrialMetrics(trials);
    printTrial("Averages: ", averages);
    printRuntimeProfile(options.generationTokens, options.trials);

    for (const warning of compareAgainstBaseline(target, averages)) {
      console.warn(`Warning: ${warning}`);
    }
  });
}

async function benchmarkTarget(target: BenchmarkTarget, options: BenchmarkOptions): Promise<void> {
  const resolvedModelSource = await resolveCachedSnapshotPath(target.model);
  const targetOptions: BenchmarkOptions = {
    ...options,
    promptTokens: target.promptTokens,
    generationTokens: target.generationTokens,
    prefillStepSize: target.prefillStepSize ?? options.prefillStepSize,
  };

  console.log(
    `Benchmarking ${target.name} (${resolvedModelSource}) with prompt_tokens=${targetOptions.promptTokens}, generation_tokens=${targetOptions.generationTokens}, trials=${targetOptions.trials}.`,
  );

  using model = await loadCausalLM(resolvedModelSource, { localFilesOnly: true });
  const promptTokenIds = createPromptTokenIds(targetOptions.promptTokens, model.config.vocabSize);
  runSyntheticBenchmarks(model, promptTokenIds, target, targetOptions);
}

async function main(): Promise<void> {
  using _runtimeLock = acquireRuntimeCommandLock("bench:generation");
  const parsed = parseBenchmarkArgs(Bun.argv.slice(2));
  const baselines = await loadBaselines();
  const targets = selectTargets("synthetic", baselines, parsed);

  for (const [index, target] of targets.entries()) {
    await benchmarkTarget(target, parsed.options);
    if (index + 1 < targets.length) {
      console.log("");
    }
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
