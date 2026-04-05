#!/usr/bin/env bun

import {
  clearMemoryCache,
  getPeakMemoryBytes,
  mxAsyncEval,
  mxEval,
  resetPeakMemory,
} from "@mlxts/core";
import type { Tokenizer } from "@mlxts/tokenizers";
import { loadCausalLM, loadPretrainedTokenizer } from "../src/load";
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
import {
  type BenchmarkModel,
  type GreedyStepResult,
  predictGreedyStep,
  prefillBenchmarkCache,
} from "./benchmark-model";

const PERIODIC_CACHE_CLEAR_INTERVAL = 256;

class BenchmarkDecodeSink {
  #tokenizer: Tokenizer;
  #decodedLength = 0;

  constructor(tokenizer: Tokenizer) {
    this.#tokenizer = tokenizer;
  }

  addToken(tokenId: number): void {
    this.#decodedLength += this.#tokenizer.decode([tokenId], { skipSpecialTokens: false }).length;
  }

  finalize(): number {
    return this.#decodedLength;
  }
}

function freeGreedyStep(step: GreedyStepResult | null): void {
  step?.token.free();
  step?.logprobs.free();
}

function runParityTrial(
  model: BenchmarkModel,
  tokenizer: Tokenizer,
  promptTokenIds: readonly number[],
  options: BenchmarkOptions,
): TrialMetrics {
  resetPeakMemory();
  using cache = model.createCache();
  const decodeSink = new BenchmarkDecodeSink(tokenizer);
  const promptStarted = performance.now();
  const remainingPrompt = prefillBenchmarkCache(
    model,
    promptTokenIds,
    cache,
    options.prefillStepSize,
  );
  let explicitEvalCount = 0;
  let promptSeconds = 0;
  let decodeStarted = 0;
  let currentStep = predictGreedyStep(model, remainingPrompt, cache);

  try {
    mxAsyncEval(currentStep.token, currentStep.logprobs);

    for (let index = 0; index < options.generationTokens; index += 1) {
      let nextStep: GreedyStepResult | null = null;

      try {
        if (index + 1 < options.generationTokens) {
          nextStep = predictGreedyStep(model, currentStep.token, cache);
          mxAsyncEval(nextStep.token, nextStep.logprobs);
        }

        mxEval(currentStep.token);
        explicitEvalCount += 1;
        if (index === 0) {
          promptSeconds = (performance.now() - promptStarted) / 1000;
          decodeStarted = performance.now();
        }

        decodeSink.addToken(currentStep.token.item());
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

  decodeSink.finalize();
  const decodeSeconds = (performance.now() - decodeStarted) / 1000;
  return {
    promptTps: promptTokenIds.length / promptSeconds,
    generationTps: options.generationTokens / decodeSeconds,
    peakMemoryGb: getPeakMemoryBytes() / 1e9,
    explicitEvalCountPerToken: explicitEvalCount / options.generationTokens,
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

function runParityBenchmarks(
  model: BenchmarkModel,
  tokenizer: Tokenizer,
  promptTokenIds: readonly number[],
  target: BenchmarkTarget,
  options: BenchmarkOptions,
): void {
  withBenchmarkRuntimeScope(target.name, options.metalTrace, () => {
    runParityTrial(model, tokenizer, promptTokenIds, options);
    clearMemoryCache();

    const trials: TrialMetrics[] = [];
    for (let index = 0; index < options.trials; index += 1) {
      const metrics = runParityTrial(model, tokenizer, promptTokenIds, options);
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
    `Benchmarking ${target.name} parity (${resolvedModelSource}) with prompt_tokens=${targetOptions.promptTokens}, generation_tokens=${targetOptions.generationTokens}, trials=${targetOptions.trials}.`,
  );

  using model = await loadCausalLM(resolvedModelSource);
  const tokenizer = await loadPretrainedTokenizer(resolvedModelSource);
  const promptTokenIds = createPromptTokenIds(targetOptions.promptTokens, model.config.vocabSize);
  runParityBenchmarks(model, tokenizer, promptTokenIds, target, targetOptions);
}

async function main(): Promise<void> {
  const parsed = parseBenchmarkArgs(Bun.argv.slice(2));
  const baselines = await loadBaselines();
  const targets = selectTargets("parity", baselines, parsed);

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
