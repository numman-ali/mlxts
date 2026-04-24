#!/usr/bin/env bun

import { clearMemoryCache, getMemoryStats, getPeakMemoryBytes, resetPeakMemory } from "@mlxts/core";
import { acquireRuntimeCommandLock } from "../../../scripts/runtime-command-lock";
import {
  loadCausalLM,
  loadInteractionProfile,
  loadPretrainedTokenizer,
} from "../../transformers/src";
import { serveLoadedModel } from "../src/model-server";
import type { ServeEvent } from "../src/types";
import { runCompletionRequest } from "./benchmark-serve-completions";
import {
  buildServeBenchmarkRungs,
  parseServeBenchmarkArgs,
  type ServeBenchmarkOptions,
  type ServeBenchmarkRung,
} from "./benchmark-serve-options";

type TrialMetrics = {
  wallMs: number;
  requestTps: number;
  completionTps: number;
  meanTtftMs: number | null;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  meanRequestMs: number;
  peakMemoryGb: number;
  activeMemoryGb: number;
  cacheMemoryGb: number;
  activeDeltaGb: number;
  admissionBatches: number;
  staticBatches: number;
  streamChunks: number;
  streamBytes: number;
  finishReasons: string[];
};

function mean(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0) / Math.max(values.length, 1);
}

function isPresentNumber(value: number | null): value is number {
  return value !== null;
}

function meanPresent(values: ReadonlyArray<number | null>): number | null {
  const present = values.filter(isPresentNumber);
  return present.length === 0 ? null : mean(present);
}

function directoryExists(path: string): boolean {
  const result = Bun.spawnSync(["/bin/test", "-d", path]);
  return result.exitCode === 0;
}

async function resolveCachedSnapshotPath(modelSource: string): Promise<string> {
  if (modelSource.startsWith("/") || modelSource.startsWith(".")) {
    return modelSource;
  }

  const [owner, name] = modelSource.split("/");
  if (owner === undefined || name === undefined) {
    return modelSource;
  }

  const homeDir = Bun.env.HOME;
  if (homeDir === undefined) {
    return modelSource;
  }

  const cacheRoot =
    Bun.env.HF_HUB_CACHE ??
    Bun.env.HUGGINGFACE_HUB_CACHE ??
    Bun.env.HF_HOME?.concat("/hub") ??
    `${homeDir}/.cache/huggingface/hub`;
  const repoCacheDir = `${cacheRoot}/models--${owner}--${name}`;
  const mainRefPath = `${repoCacheDir}/refs/main`;
  const snapshotsDir = `${repoCacheDir}/snapshots`;
  const mainRef = Bun.file(mainRefPath);
  if (await mainRef.exists()) {
    const revision = (await mainRef.text()).trim();
    const snapshotPath = `${snapshotsDir}/${revision}`;
    if (directoryExists(snapshotPath)) {
      return snapshotPath;
    }
  }

  throw new Error(
    `benchmark-serve: no cached snapshot for ${modelSource}. Use --allow-download if this run may download from the Hub.`,
  );
}

function createPromptTokenIds(length: number, vocabSize: number): number[] {
  const tokenIds: number[] = [];
  let state = 0x12345678;
  const usableVocab = Math.max(2, vocabSize - 1);

  for (let index = 0; index < length; index += 1) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    tokenIds.push((state % usableVocab) + 1);
  }

  return tokenIds;
}

function countEvents(events: readonly ServeEvent[], type: ServeEvent["type"]): number {
  return events.filter((event) => event.type === type).length;
}

async function runTrial(
  endpoint: string,
  modelId: string,
  promptTokenIds: readonly number[],
  rung: ServeBenchmarkRung,
  options: ServeBenchmarkOptions,
  serveEvents: readonly ServeEvent[],
): Promise<TrialMetrics> {
  resetPeakMemory();
  clearMemoryCache();
  const memoryBefore = getMemoryStats();
  const eventStart = serveEvents.length;
  const started = performance.now();
  const requests = Array.from({ length: rung.concurrency }, () =>
    runCompletionRequest(endpoint, modelId, promptTokenIds, rung, options),
  );
  const results = await Promise.all(requests);
  const wallMs = performance.now() - started;
  const memoryAfter = getMemoryStats();
  const events = serveEvents.slice(eventStart);
  const completionTokens = results.reduce((total, result) => total + result.completionTokens, 0);
  const promptTokens = results.reduce((total, result) => total + result.promptTokens, 0);
  const totalTokens = results.reduce((total, result) => total + result.totalTokens, 0);
  const streamChunks = results.reduce((total, result) => total + result.streamChunks, 0);
  const streamBytes = results.reduce((total, result) => total + result.streamBytes, 0);
  const wallSeconds = Math.max(wallMs / 1000, 1e-9);

  return {
    wallMs,
    requestTps: rung.concurrency / wallSeconds,
    completionTps: completionTokens / wallSeconds,
    meanTtftMs: meanPresent(results.map((result) => result.ttftMs)),
    promptTokens,
    completionTokens,
    totalTokens,
    meanRequestMs: mean(results.map((result) => result.durationMs)),
    peakMemoryGb: getPeakMemoryBytes() / 1e9,
    activeMemoryGb: memoryAfter.activeBytes / 1e9,
    cacheMemoryGb: memoryAfter.cacheBytes / 1e9,
    activeDeltaGb: (memoryAfter.activeBytes - memoryBefore.activeBytes) / 1e9,
    admissionBatches: countEvents(events, "generation_admission_batch"),
    staticBatches: countEvents(events, "generation_batch_start"),
    streamChunks,
    streamBytes,
    finishReasons: results.map((result) => result.finishReason),
  };
}

function averageTrialMetrics(trials: readonly TrialMetrics[]): TrialMetrics {
  return {
    wallMs: mean(trials.map((trial) => trial.wallMs)),
    requestTps: mean(trials.map((trial) => trial.requestTps)),
    completionTps: mean(trials.map((trial) => trial.completionTps)),
    meanTtftMs: meanPresent(trials.map((trial) => trial.meanTtftMs)),
    promptTokens: mean(trials.map((trial) => trial.promptTokens)),
    completionTokens: mean(trials.map((trial) => trial.completionTokens)),
    totalTokens: mean(trials.map((trial) => trial.totalTokens)),
    meanRequestMs: mean(trials.map((trial) => trial.meanRequestMs)),
    peakMemoryGb: mean(trials.map((trial) => trial.peakMemoryGb)),
    activeMemoryGb: mean(trials.map((trial) => trial.activeMemoryGb)),
    cacheMemoryGb: mean(trials.map((trial) => trial.cacheMemoryGb)),
    activeDeltaGb: mean(trials.map((trial) => trial.activeDeltaGb)),
    admissionBatches: mean(trials.map((trial) => trial.admissionBatches)),
    staticBatches: mean(trials.map((trial) => trial.staticBatches)),
    streamChunks: mean(trials.map((trial) => trial.streamChunks)),
    streamBytes: mean(trials.map((trial) => trial.streamBytes)),
    finishReasons: trials.flatMap((trial) => trial.finishReasons),
  };
}

function formatNullableMs(value: number | null): string {
  return value === null ? "n/a" : value.toFixed(1);
}

function printMetrics(prefix: string, metrics: TrialMetrics): void {
  console.log(
    [
      `${prefix}wall_ms=${metrics.wallMs.toFixed(1)}`,
      `request_tps=${metrics.requestTps.toFixed(3)}`,
      `completion_tps=${metrics.completionTps.toFixed(3)}`,
      `mean_ttft_ms=${formatNullableMs(metrics.meanTtftMs)}`,
      `mean_request_ms=${metrics.meanRequestMs.toFixed(1)}`,
      `prompt_tokens=${metrics.promptTokens.toFixed(0)}`,
      `completion_tokens=${metrics.completionTokens.toFixed(0)}`,
      `total_tokens=${metrics.totalTokens.toFixed(0)}`,
      `peak_memory=${metrics.peakMemoryGb.toFixed(3)}`,
      `active_memory=${metrics.activeMemoryGb.toFixed(3)}`,
      `cache_memory=${metrics.cacheMemoryGb.toFixed(3)}`,
      `active_delta=${metrics.activeDeltaGb.toFixed(3)}`,
      `admission_batches=${metrics.admissionBatches.toFixed(0)}`,
      `static_batches=${metrics.staticBatches.toFixed(0)}`,
      `stream_chunks=${metrics.streamChunks.toFixed(0)}`,
      `stream_bytes=${metrics.streamBytes.toFixed(0)}`,
      `finish_reasons=${[...new Set(metrics.finishReasons)].join("|") || "none"}`,
    ].join(" "),
  );
}

async function benchmarkRung(
  endpoint: string,
  modelId: string,
  modelVocabSize: number,
  rung: ServeBenchmarkRung,
  options: ServeBenchmarkOptions,
  serveEvents: readonly ServeEvent[],
): Promise<void> {
  const promptTokenIds = createPromptTokenIds(rung.promptTokens, modelVocabSize);
  console.log(
    [
      `rung prompt_tokens=${rung.promptTokens}`,
      `generation_tokens=${rung.generationTokens}`,
      `concurrency=${rung.concurrency}`,
      `sampling=${options.samplingMode}`,
      `transport=${options.transportMode}`,
    ].join(" "),
  );

  if (options.warmup) {
    await runCompletionRequest(endpoint, modelId, promptTokenIds, rung, options);
    clearMemoryCache();
  }

  const trials: TrialMetrics[] = [];
  for (let index = 0; index < options.trials; index += 1) {
    const metrics = await runTrial(endpoint, modelId, promptTokenIds, rung, options, serveEvents);
    trials.push(metrics);
    printMetrics(`Trial ${index + 1}:  `, metrics);
    clearMemoryCache();
  }

  printMetrics("Averages: ", averageTrialMetrics(trials));
  console.log("");
}

function maximum(values: readonly number[]): number {
  return Math.max(...values);
}

async function main(): Promise<void> {
  using _runtimeLock = acquireRuntimeCommandLock("bench:serve");
  const options = parseServeBenchmarkArgs(Bun.argv.slice(2));
  const snapshotPath = options.localFilesOnly
    ? await resolveCachedSnapshotPath(options.model)
    : options.model;
  console.log(`Benchmarking serve completions for ${snapshotPath}`);
  console.log(
    `prompt_tokens=${options.promptTokens.join(",")} generation_tokens=${options.generationTokens.join(",")} concurrency=${options.concurrency.join(",")} matrix=${options.matrix} transport=${options.transportMode}`,
  );

  const [model, tokenizer, interactionProfile] = await Promise.all([
    loadCausalLM(snapshotPath, { localFilesOnly: options.localFilesOnly }),
    loadPretrainedTokenizer(snapshotPath, { localFilesOnly: options.localFilesOnly }),
    loadInteractionProfile(snapshotPath, { localFilesOnly: options.localFilesOnly }),
  ]);
  using loadedModel = model;

  const serveEvents: ServeEvent[] = [];
  const server = serveLoadedModel({
    model: loadedModel,
    tokenizer,
    interactionProfile,
    modelId: options.modelId,
    port: options.port,
    maxGeneratedTokens: maximum(options.generationTokens),
    maxPromptTokens: options.maxPromptTokens ?? maximum(options.promptTokens),
    maxTotalTokens:
      options.maxTotalTokens ?? maximum(options.promptTokens) + maximum(options.generationTokens),
    maxBatchSize: options.maxBatchSize,
    batchWindowMs: options.batchWindowMs,
    maxConcurrentRequests: options.maxConcurrentRequests,
    gpuMemoryUtilization: options.gpuMemoryUtilization,
    onEvent(event) {
      serveEvents.push(event);
    },
  });

  try {
    console.log(`endpoint=${server.endpoint} model_id=${options.modelId}`);
    for (const rung of buildServeBenchmarkRungs(options)) {
      await benchmarkRung(
        server.endpoint,
        options.modelId,
        loadedModel.config.vocabSize,
        rung,
        options,
        serveEvents,
      );
    }
  } finally {
    server.stop(true);
  }
}

if (import.meta.main) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  });
}
