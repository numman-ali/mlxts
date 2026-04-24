#!/usr/bin/env bun

import { clearMemoryCache, getMemoryStats, getPeakMemoryBytes, resetPeakMemory } from "@mlxts/core";
import { mkdirSync } from "fs";
import { dirname } from "path";
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
  requestLaunchDelayMs,
  type ServeBenchmarkOptions,
  type ServeBenchmarkRung,
} from "./benchmark-serve-options";
import { createBenchmarkPrompt } from "./benchmark-serve-prompts";

type TrialMetrics = {
  wallMs: number;
  requestTps: number;
  completionTps: number;
  totalTps: number;
  meanTtftMs: number | null;
  meanPromptToFirstTokenTps: number | null;
  meanPostTtftCompletionTps: number | null;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  meanRequestMs: number;
  p95RequestMs: number;
  maxRequestMs: number;
  peakMemoryGb: number;
  activeMemoryGb: number;
  cacheMemoryGb: number;
  activeDeltaGb: number;
  admissionBatches: number;
  admissionRows: number;
  maxAdmissionBatchSize: number;
  staticBatches: number;
  staticBatchRows: number;
  continuousAdmissions: number;
  continuousAdmissionRows: number;
  maxGenerationBatchSize: number;
  streamChunks: number;
  streamBytes: number;
  finishReasons: string[];
};

type RungReport = {
  rung: ServeBenchmarkRung;
  arrivalSpanMs: number;
  trials: TrialMetrics[];
  averages: TrialMetrics;
};

type BenchmarkReport = {
  createdAt: string;
  model: string;
  modelId: string;
  snapshotPath: string;
  samplingMode: ServeBenchmarkOptions["samplingMode"];
  transportMode: ServeBenchmarkOptions["transportMode"];
  protocolMode: ServeBenchmarkOptions["protocolMode"];
  ignoreEos: boolean;
  maxBatchSize: number;
  batchWindowMs: number;
  requestStaggerMs: number;
  maxConcurrentRequests: number;
  gpuMemoryUtilization: number;
  rungs: RungReport[];
};

function mean(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0) / Math.max(values.length, 1);
}

function percentile(values: readonly number[], quantile: number): number {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.ceil(sorted.length * quantile) - 1;
  return sorted[Math.max(0, Math.min(index, sorted.length - 1))] ?? 0;
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

function countEvents(
  events: readonly ServeEvent[],
  type: ServeEvent["type"],
  mode?: "static" | "continuous",
): number {
  return events.filter((event) => {
    if (event.type !== type) {
      return false;
    }
    return mode === undefined || ("mode" in event && event.mode === mode);
  }).length;
}

function batchSizeEvents(events: readonly ServeEvent[], mode?: "static" | "continuous"): number[] {
  return events
    .filter((event) => event.type === "generation_batch_start")
    .filter((event) => mode === undefined || event.mode === mode)
    .map((event) => event.batchSize);
}

function admissionBatchSizes(events: readonly ServeEvent[]): number[] {
  return events
    .filter((event) => event.type === "generation_admission_batch")
    .map((event) => event.batchSize);
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function arrivalSpanMs(rung: ServeBenchmarkRung, options: ServeBenchmarkOptions): number {
  return requestLaunchDelayMs(rung.concurrency - 1, options.requestStaggerMs);
}

async function runTrial(
  endpoint: string,
  modelId: string,
  prompt: { tokenIds: readonly number[]; text: string },
  rung: ServeBenchmarkRung,
  options: ServeBenchmarkOptions,
  serveEvents: readonly ServeEvent[],
): Promise<TrialMetrics> {
  resetPeakMemory();
  clearMemoryCache();
  const memoryBefore = getMemoryStats();
  const eventStart = serveEvents.length;
  const started = performance.now();
  const requests = Array.from({ length: rung.concurrency }, async (_, requestIndex) => {
    const delayMs = requestLaunchDelayMs(requestIndex, options.requestStaggerMs);
    if (delayMs > 0) {
      await Bun.sleep(delayMs);
    }
    return runCompletionRequest(endpoint, modelId, prompt, rung, options);
  });
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
  const requestDurations = results.map((result) => result.durationMs);
  const admissionSizes = admissionBatchSizes(events);
  const staticBatchSizes = batchSizeEvents(events, "static");
  const continuousAdmissionSizes = batchSizeEvents(events, "continuous");
  const generationBatchSizes = batchSizeEvents(events);

  return {
    wallMs,
    requestTps: rung.concurrency / wallSeconds,
    completionTps: completionTokens / wallSeconds,
    totalTps: totalTokens / wallSeconds,
    meanTtftMs: meanPresent(results.map((result) => result.ttftMs)),
    meanPromptToFirstTokenTps: meanPresent(results.map((result) => result.promptToFirstTokenTps)),
    meanPostTtftCompletionTps: meanPresent(results.map((result) => result.postTtftCompletionTps)),
    promptTokens,
    completionTokens,
    totalTokens,
    meanRequestMs: mean(requestDurations),
    p95RequestMs: percentile(requestDurations, 0.95),
    maxRequestMs: Math.max(...requestDurations),
    peakMemoryGb: getPeakMemoryBytes() / 1e9,
    activeMemoryGb: memoryAfter.activeBytes / 1e9,
    cacheMemoryGb: memoryAfter.cacheBytes / 1e9,
    activeDeltaGb: (memoryAfter.activeBytes - memoryBefore.activeBytes) / 1e9,
    admissionBatches: countEvents(events, "generation_admission_batch"),
    admissionRows: sum(admissionSizes),
    maxAdmissionBatchSize: Math.max(0, ...admissionSizes),
    staticBatches: countEvents(events, "generation_batch_start", "static"),
    staticBatchRows: sum(staticBatchSizes),
    continuousAdmissions: countEvents(events, "generation_batch_start", "continuous"),
    continuousAdmissionRows: sum(continuousAdmissionSizes),
    maxGenerationBatchSize: Math.max(0, ...generationBatchSizes),
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
    totalTps: mean(trials.map((trial) => trial.totalTps)),
    meanTtftMs: meanPresent(trials.map((trial) => trial.meanTtftMs)),
    meanPromptToFirstTokenTps: meanPresent(trials.map((trial) => trial.meanPromptToFirstTokenTps)),
    meanPostTtftCompletionTps: meanPresent(trials.map((trial) => trial.meanPostTtftCompletionTps)),
    promptTokens: mean(trials.map((trial) => trial.promptTokens)),
    completionTokens: mean(trials.map((trial) => trial.completionTokens)),
    totalTokens: mean(trials.map((trial) => trial.totalTokens)),
    meanRequestMs: mean(trials.map((trial) => trial.meanRequestMs)),
    p95RequestMs: mean(trials.map((trial) => trial.p95RequestMs)),
    maxRequestMs: mean(trials.map((trial) => trial.maxRequestMs)),
    peakMemoryGb: mean(trials.map((trial) => trial.peakMemoryGb)),
    activeMemoryGb: mean(trials.map((trial) => trial.activeMemoryGb)),
    cacheMemoryGb: mean(trials.map((trial) => trial.cacheMemoryGb)),
    activeDeltaGb: mean(trials.map((trial) => trial.activeDeltaGb)),
    admissionBatches: mean(trials.map((trial) => trial.admissionBatches)),
    admissionRows: mean(trials.map((trial) => trial.admissionRows)),
    maxAdmissionBatchSize: mean(trials.map((trial) => trial.maxAdmissionBatchSize)),
    staticBatches: mean(trials.map((trial) => trial.staticBatches)),
    staticBatchRows: mean(trials.map((trial) => trial.staticBatchRows)),
    continuousAdmissions: mean(trials.map((trial) => trial.continuousAdmissions)),
    continuousAdmissionRows: mean(trials.map((trial) => trial.continuousAdmissionRows)),
    maxGenerationBatchSize: mean(trials.map((trial) => trial.maxGenerationBatchSize)),
    streamChunks: mean(trials.map((trial) => trial.streamChunks)),
    streamBytes: mean(trials.map((trial) => trial.streamBytes)),
    finishReasons: trials.flatMap((trial) => trial.finishReasons),
  };
}

function formatNullableMs(value: number | null): string {
  return value === null ? "n/a" : value.toFixed(1);
}

function formatNullableTps(value: number | null): string {
  return value === null ? "n/a" : value.toFixed(3);
}

function printMetrics(prefix: string, metrics: TrialMetrics): void {
  console.log(
    [
      `${prefix}wall_ms=${metrics.wallMs.toFixed(1)}`,
      `request_tps=${metrics.requestTps.toFixed(3)}`,
      `completion_tps=${metrics.completionTps.toFixed(3)}`,
      `total_tps=${metrics.totalTps.toFixed(3)}`,
      `mean_ttft_ms=${formatNullableMs(metrics.meanTtftMs)}`,
      `mean_prompt_to_first_token_tps=${formatNullableTps(metrics.meanPromptToFirstTokenTps)}`,
      `mean_post_ttft_completion_tps=${formatNullableTps(metrics.meanPostTtftCompletionTps)}`,
      `mean_request_ms=${metrics.meanRequestMs.toFixed(1)}`,
      `p95_request_ms=${metrics.p95RequestMs.toFixed(1)}`,
      `max_request_ms=${metrics.maxRequestMs.toFixed(1)}`,
      `prompt_tokens=${metrics.promptTokens.toFixed(0)}`,
      `completion_tokens=${metrics.completionTokens.toFixed(0)}`,
      `total_tokens=${metrics.totalTokens.toFixed(0)}`,
      `peak_memory=${metrics.peakMemoryGb.toFixed(3)}`,
      `active_memory=${metrics.activeMemoryGb.toFixed(3)}`,
      `cache_memory=${metrics.cacheMemoryGb.toFixed(3)}`,
      `active_delta=${metrics.activeDeltaGb.toFixed(3)}`,
      `admission_batches=${metrics.admissionBatches.toFixed(0)}`,
      `admission_rows=${metrics.admissionRows.toFixed(0)}`,
      `max_admission_batch=${metrics.maxAdmissionBatchSize.toFixed(0)}`,
      `static_batches=${metrics.staticBatches.toFixed(0)}`,
      `static_batch_rows=${metrics.staticBatchRows.toFixed(0)}`,
      `continuous_admissions=${metrics.continuousAdmissions.toFixed(0)}`,
      `continuous_admission_rows=${metrics.continuousAdmissionRows.toFixed(0)}`,
      `max_generation_batch=${metrics.maxGenerationBatchSize.toFixed(0)}`,
      `stream_chunks=${metrics.streamChunks.toFixed(0)}`,
      `stream_bytes=${metrics.streamBytes.toFixed(0)}`,
      `finish_reasons=${[...new Set(metrics.finishReasons)].join("|") || "none"}`,
    ].join(" "),
  );
}

function formatRung(rung: ServeBenchmarkRung): string {
  return `${rung.promptTokens}x${rung.generationTokens}@${rung.concurrency}`;
}

export async function writeBenchmarkReport(path: string, report: BenchmarkReport): Promise<void> {
  mkdirSync(dirname(path), { recursive: true });
  await Bun.write(path, `${JSON.stringify(report, null, 2)}\n`);
}

async function benchmarkRung(
  endpoint: string,
  modelId: string,
  modelVocabSize: number,
  tokenizer: { encode(text: string): number[] },
  rung: ServeBenchmarkRung,
  options: ServeBenchmarkOptions,
  serveEvents: readonly ServeEvent[],
): Promise<RungReport> {
  const prompt = createBenchmarkPrompt(
    rung.promptTokens,
    modelVocabSize,
    tokenizer,
    options.protocolMode,
  );
  console.log(
    [
      `rung prompt_tokens=${rung.promptTokens}`,
      `generation_tokens=${rung.generationTokens}`,
      `concurrency=${rung.concurrency}`,
      `sampling=${options.samplingMode}`,
      `transport=${options.transportMode}`,
      `protocol=${options.protocolMode}`,
      `request_stagger_ms=${options.requestStaggerMs}`,
      `arrival_span_ms=${arrivalSpanMs(rung, options)}`,
    ].join(" "),
  );

  if (options.warmup) {
    await runCompletionRequest(endpoint, modelId, prompt, rung, options);
    clearMemoryCache();
  }

  const trials: TrialMetrics[] = [];
  for (let index = 0; index < options.trials; index += 1) {
    const metrics = await runTrial(endpoint, modelId, prompt, rung, options, serveEvents);
    trials.push(metrics);
    printMetrics(`Trial ${index + 1}:  `, metrics);
    clearMemoryCache();
  }

  const averages = averageTrialMetrics(trials);
  printMetrics("Averages: ", averages);
  console.log("");
  return { rung, arrivalSpanMs: arrivalSpanMs(rung, options), trials, averages };
}

function maximum(values: readonly number[]): number {
  return Math.max(...values);
}

async function main(): Promise<void> {
  using _runtimeLock = acquireRuntimeCommandLock("bench:serve");
  const options = parseServeBenchmarkArgs(Bun.argv.slice(2));
  const rungs = buildServeBenchmarkRungs(options);
  const snapshotPath = options.localFilesOnly
    ? await resolveCachedSnapshotPath(options.model)
    : options.model;
  console.log(`Benchmarking serve completions for ${snapshotPath}`);
  console.log(
    [
      `rungs=${rungs.map(formatRung).join(",")}`,
      `matrix=${options.matrix}`,
      `transport=${options.transportMode}`,
      `protocol=${options.protocolMode}`,
      `sampling=${options.samplingMode}`,
      `ignore_eos=${options.ignoreEos}`,
      `max_batch_size=${options.maxBatchSize}`,
      `batch_window_ms=${options.batchWindowMs}`,
      `request_stagger_ms=${options.requestStaggerMs}`,
      `max_concurrent_requests=${options.maxConcurrentRequests}`,
      `gpu_memory_utilization=${options.gpuMemoryUtilization}`,
    ].join(" "),
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
    maxGeneratedTokens: maximum(rungs.map((rung) => rung.generationTokens)),
    maxPromptTokens: options.maxPromptTokens ?? maximum(rungs.map((rung) => rung.promptTokens)),
    maxTotalTokens:
      options.maxTotalTokens ??
      maximum(rungs.map((rung) => rung.promptTokens + rung.generationTokens)),
    maxBatchSize: options.maxBatchSize,
    batchWindowMs: options.batchWindowMs,
    maxConcurrentRequests: options.maxConcurrentRequests,
    gpuMemoryUtilization: options.gpuMemoryUtilization,
    onEvent(event) {
      serveEvents.push(event);
    },
  });

  try {
    const reports: RungReport[] = [];
    console.log(`endpoint=${server.endpoint} model_id=${options.modelId}`);
    for (const rung of rungs) {
      const report = await benchmarkRung(
        server.endpoint,
        options.modelId,
        loadedModel.config.vocabSize,
        tokenizer,
        rung,
        options,
        serveEvents,
      );
      reports.push(report);
    }
    if (options.reportJson !== undefined) {
      await writeBenchmarkReport(options.reportJson, {
        createdAt: new Date().toISOString(),
        model: options.model,
        modelId: options.modelId,
        snapshotPath,
        samplingMode: options.samplingMode,
        transportMode: options.transportMode,
        protocolMode: options.protocolMode,
        ignoreEos: options.ignoreEos,
        maxBatchSize: options.maxBatchSize,
        batchWindowMs: options.batchWindowMs,
        requestStaggerMs: options.requestStaggerMs,
        maxConcurrentRequests: options.maxConcurrentRequests,
        gpuMemoryUtilization: options.gpuMemoryUtilization,
        rungs: reports,
      });
      console.log(`report_json=${options.reportJson}`);
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
