#!/usr/bin/env bun

import { clearMemoryCache, getPeakMemoryBytes, mxAsyncEval, resetPeakMemory } from "@mlxts/core";
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
  type MlxLmReference,
  mean,
  parseBenchmarkArgs,
  printTrial,
  type ReferenceBenchmarkOptions,
  readBenchmarkVocabSize,
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

type BenchmarkCache = ReturnType<BenchmarkModel["createCache"]>;

type DecodeTiming = {
  promptSeconds: number;
  decodeSeconds: number;
  decodeSyncCount: number;
};

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

function materializeCacheIfRequested(cache: BenchmarkCache, options: BenchmarkOptions): void {
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

function runAsyncParityDecode(
  model: BenchmarkModel,
  cache: BenchmarkCache,
  initialStep: GreedyStepResult,
  promptStarted: number,
  options: BenchmarkOptions,
  decodeMemory: ReturnType<typeof createDecodeMemoryTracker>,
): DecodeTiming {
  let currentStep = initialStep;
  let decodeSyncCount = 0;
  let promptSeconds = 0;
  let decodeStarted = 0;

  try {
    materializeCacheIfRequested(cache, options);
    mxAsyncEval(currentStep.token, currentStep.logprobs);

    for (let index = 0; index < options.generationTokens; index += 1) {
      let nextStep: GreedyStepResult | null = null;

      try {
        if (index + 1 < options.generationTokens) {
          nextStep = predictGreedyStep(model, currentStep.token, cache);
          materializeCacheIfRequested(cache, options);
          mxAsyncEval(nextStep.token, nextStep.logprobs);
        }

        decodeSyncCount += 1;
        if (index === 0) {
          promptSeconds = (performance.now() - promptStarted) / 1000;
          resetRuntimeProfiles();
          decodeStarted = performance.now();
        }

        currentStep.token.item();
        if (recordDecodeStep(index, options, decodeMemory)) {
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

  return {
    promptSeconds,
    decodeSeconds: (performance.now() - decodeStarted) / 1000,
    decodeSyncCount,
  };
}

function runSyncParityDecode(
  model: BenchmarkModel,
  cache: BenchmarkCache,
  initialStep: GreedyStepResult,
  promptStarted: number,
  options: BenchmarkOptions,
  decodeMemory: ReturnType<typeof createDecodeMemoryTracker>,
): DecodeTiming {
  let currentStep = initialStep;
  let decodeSyncCount = 0;
  let promptSeconds = 0;
  let decodeStarted = 0;

  try {
    for (let index = 0; index < options.generationTokens; index += 1) {
      materializeCacheIfRequested(cache, options);
      mxAsyncEval(currentStep.token, currentStep.logprobs);

      decodeSyncCount += 1;
      if (index === 0) {
        promptSeconds = (performance.now() - promptStarted) / 1000;
        resetRuntimeProfiles();
        decodeStarted = performance.now();
      }

      const tokenId = currentStep.token.item();
      if (recordDecodeStep(index, options, decodeMemory)) {
        break;
      }

      const nextStep = predictGreedyStep(model, [tokenId], cache);
      freeGreedyStep(currentStep);
      currentStep = nextStep;
    }
  } finally {
    freeGreedyStep(currentStep);
  }

  return {
    promptSeconds,
    decodeSeconds: (performance.now() - decodeStarted) / 1000,
    decodeSyncCount,
  };
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
  const currentStep = predictGreedyStep(model, remainingPrompt, cache);
  const decodeMemory = createDecodeMemoryTracker();
  const timing =
    options.decodeSchedule === "async"
      ? runAsyncParityDecode(model, cache, currentStep, promptStarted, options, decodeMemory)
      : runSyncParityDecode(model, cache, currentStep, promptStarted, options, decodeMemory);

  return {
    promptTps: promptTokenIds.length / timing.promptSeconds,
    generationTps: options.generationTokens / timing.decodeSeconds,
    peakMemoryGb: getPeakMemoryBytes() / 1e9,
    ...decodeMemory.finish(options.generationTokens),
    // item() performs one blocking scalar sync per token in the steady-state decode loop.
    explicitEvalCountPerToken: timing.decodeSyncCount / options.generationTokens,
    totalTimeSeconds: timing.promptSeconds + timing.decodeSeconds,
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

async function resolveMlxLmReference(
  target: BenchmarkTarget,
  resolvedModelSource: string,
  promptTokenIds: readonly number[],
  targetOptions: BenchmarkOptions,
  referenceOptions: ReferenceBenchmarkOptions,
): Promise<MlxLmReference | null> {
  const liveReference = await captureMlxLmReference(
    resolvedModelSource,
    promptTokenIds,
    referenceOptions,
    {
      generationTokens: targetOptions.generationTokens,
      prefillStepSize: targetOptions.prefillStepSize,
      trials: targetOptions.trials,
    },
  );
  if (liveReference !== null) {
    return liveReference;
  }
  if (target.mlxLmReference !== undefined) {
    return target.mlxLmReference;
  }
  if (referenceOptions.requireMlxLmReference) {
    throw new Error(
      `benchmark-generation: MLX-LM reference is required for ${target.name} but unavailable.`,
    );
  }
  console.warn(
    `Warning: MLX-LM reference unavailable for ${target.name}; falling back to no external comparison.`,
  );
  return null;
}

async function captureTargetReference(
  target: BenchmarkTarget,
  resolvedModelSource: string,
  promptTokenIds: readonly number[],
  targetOptions: BenchmarkOptions,
  referenceOptions: ReferenceBenchmarkOptions,
): Promise<MlxLmReference | null> {
  try {
    return await resolveMlxLmReference(
      target,
      resolvedModelSource,
      promptTokenIds,
      targetOptions,
      referenceOptions,
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (referenceOptions.requireMlxLmReference) {
      throw new Error(`benchmark-generation: required mlx-lm reference failed: ${message}`);
    }
    console.warn(`Warning: unable to run mlx-lm reference for ${target.name}: ${message}`);
    return target.mlxLmReference ?? null;
  }
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
    `Benchmarking ${target.name} parity (${resolvedModelSource}) with prompt_tokens=${targetOptions.promptTokens}, generation_tokens=${targetOptions.generationTokens}, trials=${targetOptions.trials}, decode_schedule=${targetOptions.decodeSchedule}, materialize_cache_each_token=${targetOptions.materializeCacheEachToken}.`,
  );
  const vocabSize = await readBenchmarkVocabSize(resolvedModelSource);
  const promptTokenIds = createPromptTokenIds(targetOptions.promptTokens, vocabSize);
  const mlxLmReference = await captureTargetReference(
    target,
    resolvedModelSource,
    promptTokenIds,
    targetOptions,
    referenceOptions,
  );

  if (mlxLmReference !== null) {
    console.log(
      formatMlxLmReference({
        ...target,
        mlxLmReference,
      }),
    );
  }

  using model = await loadCausalLM(resolvedModelSource, { localFilesOnly: true });
  if (model.config.vocabSize !== vocabSize) {
    throw new Error(
      `benchmark-generation: loaded model vocab size ${model.config.vocabSize} did not match config prompt vocab size ${vocabSize}.`,
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
