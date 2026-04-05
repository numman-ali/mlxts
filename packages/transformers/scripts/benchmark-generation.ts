#!/usr/bin/env bun

import {
  clearMemoryCache,
  getPeakMemoryBytes,
  type MxArray,
  mxAsyncEval,
  mxEval,
  resetPeakMemory,
} from "@mlxts/core";
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
  let decodeEvalCount = 0;
  let currentToken = predictGreedyToken(model, remainingPrompt, cache);

  mxEval(currentToken);
  currentToken.item();
  const promptSeconds = (performance.now() - promptStarted) / 1000;

  const decodeStarted = performance.now();
  let nextToken: MxArray | null = null;

  try {
    mxAsyncEval(currentToken);

    for (let index = 0; index < options.generationTokens; index += 1) {
      if (index + 1 < options.generationTokens) {
        nextToken = predictGreedyToken(model, currentToken, cache);
        mxAsyncEval(nextToken);
      }

      mxEval(currentToken);
      decodeEvalCount += 1;
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
    explicitEvalCountPerToken: decodeEvalCount / options.generationTokens,
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

    const trials: TrialMetrics[] = [];
    for (let index = 0; index < options.trials; index += 1) {
      const metrics = runSyntheticTrial(model, promptTokenIds, options);
      trials.push(metrics);
      printTrial(`Trial ${index + 1}:  `, metrics);
      clearMemoryCache();
    }

    const averages = averageTrialMetrics(trials);
    printTrial("Averages: ", averages);

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

  using model = await loadCausalLM(resolvedModelSource);
  const promptTokenIds = createPromptTokenIds(targetOptions.promptTokens, model.config.vocabSize);
  runSyntheticBenchmarks(model, promptTokenIds, target, targetOptions);
}

async function main(): Promise<void> {
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
