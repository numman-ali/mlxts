#!/usr/bin/env bun

import { clearMemoryCache, getPeakMemoryBytes, mxAsyncEval, resetPeakMemory } from "@mlxts/core";
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
  captureMlxLmReference,
  compareAgainstBaseline,
  compareAgainstMlxLmReference,
  createDecodeMemoryTracker,
  createPromptTokenIds,
  enforceMlxLmDecodeBar,
  formatMlxLmReference,
  loadBaselines,
  mean,
  parseBenchmarkArgs,
  printTrial,
  type ReferenceBenchmarkOptions,
  resolveCachedSnapshotPath,
  selectTargets,
  type TrialMetrics,
  withBenchmarkRuntimeScope,
} from "./benchmark-common";
import {
  type BenchmarkModel,
  type GreedyStepResult,
  predictGreedyStep,
  prefillBenchmarkCache,
} from "./benchmark-model";

const PERIODIC_CACHE_CLEAR_INTERVAL = 256;

function formatNsPerToken(totalNs: number, generationTokens: number, trials: number): string {
  const tokenCount = Math.max(generationTokens * trials, 1);
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

function freeGreedyStep(step: GreedyStepResult | null): void {
  step?.token.free();
  step?.logprobs.free();
}

function runParityTrial(
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
  let promptSeconds = 0;
  let decodeStarted = 0;
  let currentStep = predictGreedyStep(model, remainingPrompt, cache);
  const decodeMemory = createDecodeMemoryTracker();

  try {
    mxAsyncEval(currentStep.token, currentStep.logprobs);

    for (let index = 0; index < options.generationTokens; index += 1) {
      let nextStep: GreedyStepResult | null = null;

      try {
        if (index + 1 < options.generationTokens) {
          nextStep = predictGreedyStep(model, currentStep.token, cache);
          mxAsyncEval(nextStep.token, nextStep.logprobs);
        }

        decodeSyncCount += 1;
        if (index === 0) {
          promptSeconds = (performance.now() - promptStarted) / 1000;
          resetRuntimeProfiles();
          decodeStarted = performance.now();
        }

        currentStep.token.item();
        if ((index + 1) % options.memorySampleInterval === 0) {
          decodeMemory.sample();
        }
        if ((index + 1) % PERIODIC_CACHE_CLEAR_INTERVAL === 0) {
          clearMemoryCache();
        }

        if (index + 1 >= options.generationTokens) {
          break;
        }

        freeGreedyStep(currentStep);
        currentStep = nextStep;
        nextStep = null;
        if (currentStep === null) {
          throw new Error("benchmark-generation: parity decode did not schedule the next token.");
        }
      } finally {
        freeGreedyStep(nextStep);
      }
    }
  } finally {
    freeGreedyStep(currentStep);
  }

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

function runParityBenchmarks(
  model: BenchmarkModel,
  promptTokenIds: readonly number[],
  target: BenchmarkTarget,
  options: BenchmarkOptions,
): TrialMetrics {
  return withBenchmarkRuntimeScope(target.name, options.metalTrace, () => {
    runParityTrial(model, promptTokenIds, options);
    clearMemoryCache();
    resetRuntimeProfiles();

    const trials: TrialMetrics[] = [];
    for (let index = 0; index < options.trials; index += 1) {
      const metrics = runParityTrial(model, promptTokenIds, options);
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

    return averages;
  });
}

async function benchmarkTarget(
  target: BenchmarkTarget,
  options: BenchmarkOptions,
  referenceOptions: ReferenceBenchmarkOptions,
): Promise<void> {
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

  console.log(
    `Benchmarking ${target.name} parity (${resolvedModelSource}) with prompt_tokens=${targetOptions.promptTokens}, generation_tokens=${targetOptions.generationTokens}, trials=${targetOptions.trials}.`,
  );
  let mlxLmReference = target.mlxLmReference ?? null;

  using model = await loadCausalLM(resolvedModelSource, { localFilesOnly: true });
  const promptTokenIds = createPromptTokenIds(targetOptions.promptTokens, model.config.vocabSize);

  try {
    const liveReference = await captureMlxLmReference(
      resolvedModelSource,
      promptTokenIds,
      targetOptions.generationTokens,
      referenceOptions,
    );
    if (liveReference !== null) {
      mlxLmReference = liveReference;
    } else if (mlxLmReference === null) {
      console.warn(
        `Warning: MLX-LM reference unavailable for ${target.name}; falling back to no external comparison.`,
      );
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.warn(`Warning: unable to run mlx-lm reference for ${target.name}: ${message}`);
  }

  if (mlxLmReference !== null) {
    console.log(
      formatMlxLmReference({
        ...target,
        mlxLmReference,
      }),
    );
  }

  const averages = runParityBenchmarks(
    model,
    promptTokenIds,
    mlxLmReference === null ? target : { ...target, mlxLmReference },
    targetOptions,
  );

  if (mlxLmReference !== null) {
    const comparisonWarnings = compareAgainstMlxLmReference(averages, mlxLmReference);
    for (const warning of comparisonWarnings) {
      console.warn(`Warning: ${warning}`);
    }
  }

  enforceMlxLmDecodeBar(target.model, averages, mlxLmReference, referenceOptions);
}

async function main(): Promise<void> {
  using _runtimeLock = acquireRuntimeCommandLock("bench:generation:parity");
  const parsed = parseBenchmarkArgs(Bun.argv.slice(2));
  const baselines = await loadBaselines();
  const targets = selectTargets("parity", baselines, parsed);

  for (const [index, target] of targets.entries()) {
    await benchmarkTarget(target, parsed.options, parsed.reference);
    if (index + 1 < targets.length) {
      console.log("");
    }
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
