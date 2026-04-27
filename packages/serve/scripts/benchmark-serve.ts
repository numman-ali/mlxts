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
import { type RequestMetrics, runCompletionRequest } from "./benchmark-serve-completions";
import {
  buildServeBenchmarkRungs,
  formatServeBenchmarkRung,
  maxGenerationTokensForRung,
  maxPromptTokensForRung,
  maxTotalTokensForRung,
  parseServeBenchmarkArgs,
  requestLaunchDelayMs,
  requestShapesForRung,
  rungConcurrency,
  type ServeBenchmarkOptions,
  type ServeBenchmarkRequestShape,
  type ServeBenchmarkRung,
} from "./benchmark-serve-options";
import { createBenchmarkPrompt } from "./benchmark-serve-prompts";

export type RouteDecisionReport = {
  id: string;
  model: string;
  protocol: string;
  route: string;
  eligible: boolean;
  reason: string;
  modelType: string;
  maxBatchSize: number;
  schedulerMode: string;
  cacheBackend: string;
  attentionBackend: string;
  decodingBackend: string;
  stream: boolean;
};

export type RouteDecisionSummary = {
  key: string;
  route: string;
  reason: string;
  count: number;
};

export type ServerRequestTimingReport = {
  id: string;
  model: string;
  protocol: string;
  inputKind?: string;
  route?: string;
  routeReason?: string;
  routeDecisionMs: number | null;
  modelLaneWaitMs: number | null;
  modelLaneQueuedAhead: number | null;
  modelLaneInFlightAtQueue: number | null;
  schedulerQueuedMs: number | null;
  schedulerPrefillStartMs: number | null;
  schedulerAdmittedMs: number | null;
  schedulerFirstTokenMs: number | null;
  schedulerFinishedMs: number | null;
  schedulerPhaseEvents: number;
  schedulerAdmittedBatchSize: number | null;
  schedulerWaitingTotalTokens: number | null;
  schedulerPrefillingTotalTokens: number | null;
  schedulerActiveTotalTokens: number | null;
  schedulerScheduledTotalTokens: number | null;
  schedulerMaxScheduledTotalTokens: number | null;
  firstPrefillProgressMs: number | null;
  lastPrefillProgressMs: number | null;
  prefillObservedMs: number | null;
  firstCompletionProgressMs: number | null;
  completeObservedMs: number | null;
  durationMs: number | null;
  maxSilentEventGapMs: number | null;
  prefillEvents: number;
  progressEvents: number;
  maxCompletionTokens: number;
  serverStreamChunkEvents: number;
  serverStreamEndEvents: number;
  serverStreamFirstChunkMs: number | null;
  serverStreamLastChunkMs: number | null;
  serverStreamChunks: number | null;
  serverStreamBytes: number | null;
  serverStreamOutputChunks: number | null;
  serverStreamOutputBytes: number | null;
  serverStreamTtftMs: number | null;
  serverStreamDurationMs: number | null;
  serverStreamResult?: "completed" | "cancelled" | "error";
  serverStreamFinishReason?: string;
  finishReason?: string;
  errorCode?: string;
};

export type TrialMetrics = {
  wallMs: number;
  requestTps: number;
  completionTps: number;
  totalTps: number;
  meanTtftMs: number | null;
  meanPromptToFirstTokenTps: number | null;
  meanPostTtftCompletionTps: number | null;
  meanStreamChunkGapMs: number | null;
  maxStreamChunkGapMs: number | null;
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
  continuousSchedulerPhases: number;
  maxContinuousBatchSize: number;
  maxGenerationBatchSize: number;
  streamChunks: number;
  streamBytes: number;
  finishReasons: string[];
  routeDecisions: RouteDecisionReport[];
  routeSummary: RouteDecisionSummary[];
  requests: RequestMetricsReport[];
  serverRequests: ServerRequestTimingReport[];
};

export type RequestMetricsReport = RequestMetrics & {
  index: number;
  launchDelayMs: number;
};

export type RungReport = {
  rung: ServeBenchmarkRung;
  arrivalSpanMs: number;
  trials: TrialMetrics[];
  averages: TrialMetrics;
};

export type BenchmarkReport = {
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
  streamDecodeInterval: number;
  requestStaggerMs: number;
  maxConcurrentRequests: number;
  gpuMemoryUtilization: number;
  rungs: RungReport[];
};

export type RecordedServeEvent = ServeEvent & {
  observedAtMs: number;
};

type PreparedBenchmarkRequest = {
  shape: ServeBenchmarkRequestShape;
  prompt: { tokenIds: readonly number[]; text: string };
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

function maxPresent(values: ReadonlyArray<number | null>): number | null {
  const present = values.filter(isPresentNumber);
  return present.length === 0 ? null : Math.max(...present);
}

function eventRequestIds(event: ServeEvent): readonly string[] {
  if ("ids" in event) {
    return event.ids;
  }
  return "id" in event ? [event.id] : [];
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
  events: readonly RecordedServeEvent[],
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

function batchSizeEvents(
  events: readonly RecordedServeEvent[],
  mode?: "static" | "continuous",
): number[] {
  return events
    .filter((event) => event.type === "generation_batch_start")
    .filter((event) => mode === undefined || event.mode === mode)
    .map((event) => event.batchSize);
}

function continuousSchedulerBatchSizeEvents(
  events: readonly RecordedServeEvent[],
  phase: "admitted",
): number[] {
  return events
    .filter((event) => event.type === "generation_scheduler_phase" && event.phase === phase)
    .map((event) => event.batchSize);
}

function admissionBatchSizes(events: readonly RecordedServeEvent[]): number[] {
  return events
    .filter((event) => event.type === "generation_admission_batch")
    .map((event) => event.batchSize);
}

function routeDecisionReports(events: readonly RecordedServeEvent[]): RouteDecisionReport[] {
  return events
    .filter((event) => event.type === "generation_route_decision")
    .map((event) => ({
      id: event.id,
      model: event.model,
      protocol: event.protocol,
      route: event.route,
      eligible: event.eligible,
      reason: event.reason,
      modelType: event.modelType,
      maxBatchSize: event.maxBatchSize,
      schedulerMode: event.schedulerMode,
      cacheBackend: event.cacheBackend,
      attentionBackend: event.attentionBackend,
      decodingBackend: event.decodingBackend,
      stream: event.stream,
    }));
}

function relativeMs(event: RecordedServeEvent | undefined, startedAt: number): number | null {
  return event === undefined ? null : event.observedAtMs - startedAt;
}

function maxObservedEventGapMs(events: readonly RecordedServeEvent[]): number | null {
  if (events.length < 2) {
    return null;
  }
  let maxGapMs = 0;
  for (let index = 1; index < events.length; index += 1) {
    const previous = events[index - 1];
    const current = events[index];
    if (previous !== undefined && current !== undefined) {
      maxGapMs = Math.max(maxGapMs, current.observedAtMs - previous.observedAtMs);
    }
  }
  return maxGapMs;
}

function groupEventsByRequestId(
  events: readonly RecordedServeEvent[],
): Map<string, RecordedServeEvent[]> {
  const groups = new Map<string, RecordedServeEvent[]>();
  for (const event of events) {
    for (const id of eventRequestIds(event)) {
      const group = groups.get(id);
      if (group === undefined) {
        groups.set(id, [event]);
        continue;
      }
      group.push(event);
    }
  }
  return groups;
}

function maxCompletionTokens(events: readonly RecordedServeEvent[]): number {
  return Math.max(
    0,
    ...events
      .filter((event) => event.type === "generation_progress")
      .map((event) => event.completionTokens),
  );
}

function schedulerAdmittedQueuedMs(
  id: string,
  event:
    | Extract<RecordedServeEvent, { type: "generation_scheduler_phase"; phase: "admitted" }>
    | undefined,
): number | null {
  if (event === undefined) {
    return null;
  }
  const index = event.ids.indexOf(id);
  return index < 0 ? null : (event.queuedMsByRequest[index] ?? null);
}

type RequestEventSlice = {
  sorted: RecordedServeEvent[];
  start: Extract<RecordedServeEvent, { type: "generation_start" }> | undefined;
  route: Extract<RecordedServeEvent, { type: "generation_route_decision" }> | undefined;
  modelLaneWait: Extract<RecordedServeEvent, { type: "generation_model_lane_wait" }> | undefined;
  complete: Extract<RecordedServeEvent, { type: "generation_complete" }> | undefined;
  error: Extract<RecordedServeEvent, { type: "generation_error" }> | undefined;
  schedulerEvents: Extract<RecordedServeEvent, { type: "generation_scheduler_phase" }>[];
  prefillEvents: Extract<RecordedServeEvent, { type: "generation_prefill_progress" }>[];
  progressEvents: Extract<RecordedServeEvent, { type: "generation_progress" }>[];
  streamChunkEvents: Extract<RecordedServeEvent, { type: "generation_stream_chunk" }>[];
  streamEndEvents: Extract<RecordedServeEvent, { type: "generation_stream_end" }>[];
};

function requestEventSlice(group: readonly RecordedServeEvent[]): RequestEventSlice {
  const sorted = [...group].sort((left, right) => left.observedAtMs - right.observedAtMs);
  return {
    sorted,
    start: sorted.find((event) => event.type === "generation_start"),
    route: sorted.find((event) => event.type === "generation_route_decision"),
    modelLaneWait: sorted.find((event) => event.type === "generation_model_lane_wait"),
    complete: sorted.find((event) => event.type === "generation_complete"),
    error: sorted.find((event) => event.type === "generation_error"),
    schedulerEvents: sorted.filter((event) => event.type === "generation_scheduler_phase"),
    prefillEvents: sorted.filter((event) => event.type === "generation_prefill_progress"),
    progressEvents: sorted.filter((event) => event.type === "generation_progress"),
    streamChunkEvents: sorted.filter((event) => event.type === "generation_stream_chunk"),
    streamEndEvents: sorted.filter((event) => event.type === "generation_stream_end"),
  };
}

function sliceStartMs(slice: RequestEventSlice): number {
  return slice.start?.observedAtMs ?? slice.sorted[0]?.observedAtMs ?? 0;
}

function prefillObservedMs(firstMs: number | null, lastMs: number | null): number | null {
  return firstMs === null || lastMs === null ? null : lastMs - firstMs;
}

function requestDurationMs(slice: RequestEventSlice): number | null {
  if (slice.complete !== undefined) {
    return slice.complete.durationMs;
  }
  return slice.error?.durationMs ?? null;
}

function requestModel(slice: RequestEventSlice): string {
  return (
    slice.start?.model ??
    slice.route?.model ??
    slice.complete?.model ??
    slice.error?.model ??
    slice.streamEndEvents[slice.streamEndEvents.length - 1]?.model ??
    "unknown"
  );
}

function requestProtocol(slice: RequestEventSlice): string {
  return (
    slice.start?.protocol ??
    slice.route?.protocol ??
    slice.complete?.protocol ??
    slice.error?.protocol ??
    slice.streamEndEvents[slice.streamEndEvents.length - 1]?.protocol ??
    "unknown"
  );
}

function inputKindField(slice: RequestEventSlice): { inputKind?: string } {
  return slice.start === undefined ? {} : { inputKind: slice.start.inputKind };
}

function routeFields(slice: RequestEventSlice): { route?: string; routeReason?: string } {
  return slice.route === undefined
    ? {}
    : { route: slice.route.route, routeReason: slice.route.reason };
}

function terminalFields(slice: RequestEventSlice): { finishReason?: string; errorCode?: string } {
  return {
    ...(slice.complete === undefined ? {} : { finishReason: slice.complete.finishReason }),
    ...(slice.error === undefined ? {} : { errorCode: slice.error.code }),
  };
}

function serverStreamFields(slice: RequestEventSlice): Pick<
  ServerRequestTimingReport,
  | "serverStreamChunkEvents"
  | "serverStreamEndEvents"
  | "serverStreamFirstChunkMs"
  | "serverStreamLastChunkMs"
  | "serverStreamChunks"
  | "serverStreamBytes"
  | "serverStreamOutputChunks"
  | "serverStreamOutputBytes"
  | "serverStreamTtftMs"
  | "serverStreamDurationMs"
> & {
  serverStreamResult?: "completed" | "cancelled" | "error";
  serverStreamFinishReason?: string;
} {
  const streamEnd = slice.streamEndEvents[slice.streamEndEvents.length - 1];
  const firstStreamChunk = slice.streamChunkEvents[0];
  const lastStreamChunk = slice.streamChunkEvents[slice.streamChunkEvents.length - 1];
  return {
    serverStreamChunkEvents: slice.streamChunkEvents.length,
    serverStreamEndEvents: slice.streamEndEvents.length,
    serverStreamFirstChunkMs: firstStreamChunk?.elapsedMs ?? null,
    serverStreamLastChunkMs: lastStreamChunk?.elapsedMs ?? null,
    serverStreamChunks: streamEnd?.chunks ?? null,
    serverStreamBytes: streamEnd?.bytes ?? null,
    serverStreamOutputChunks: streamEnd?.outputChunks ?? null,
    serverStreamOutputBytes: streamEnd?.outputBytes ?? null,
    serverStreamTtftMs: streamEnd?.ttftMs ?? null,
    serverStreamDurationMs: streamEnd?.durationMs ?? null,
    ...(streamEnd === undefined ? {} : { serverStreamResult: streamEnd.result }),
    ...(streamEnd === undefined ? {} : { serverStreamFinishReason: streamEnd.finishReason }),
  };
}

function schedulerTokenFields(
  slice: RequestEventSlice,
): Pick<
  ServerRequestTimingReport,
  | "schedulerWaitingTotalTokens"
  | "schedulerPrefillingTotalTokens"
  | "schedulerActiveTotalTokens"
  | "schedulerScheduledTotalTokens"
  | "schedulerMaxScheduledTotalTokens"
> {
  const event = slice.schedulerEvents[slice.schedulerEvents.length - 1];
  return {
    schedulerWaitingTotalTokens: event?.waitingTotalTokens ?? null,
    schedulerPrefillingTotalTokens: event?.prefillingTotalTokens ?? null,
    schedulerActiveTotalTokens: event?.activeTotalTokens ?? null,
    schedulerScheduledTotalTokens: event?.scheduledTotalTokens ?? null,
    schedulerMaxScheduledTotalTokens: event?.maxScheduledTotalTokens ?? null,
  };
}

function requestTimingReport(id: string, group: readonly RecordedServeEvent[]) {
  const slice = requestEventSlice(group);
  const startedAt = sliceStartMs(slice);
  const firstPrefillMs = relativeMs(slice.prefillEvents[0], startedAt);
  const lastPrefillMs = relativeMs(slice.prefillEvents[slice.prefillEvents.length - 1], startedAt);
  const firstCompletion = slice.progressEvents.find((event) => event.completionTokens > 0);
  const prefillStart = slice.schedulerEvents.find((event) => event.phase === "prefill_start");
  const admitted = slice.schedulerEvents.find((event) => event.phase === "admitted");
  const firstToken = slice.schedulerEvents.find((event) => event.phase === "first_token");
  const finished = slice.schedulerEvents.find((event) => event.phase === "finished");

  return {
    id,
    model: requestModel(slice),
    protocol: requestProtocol(slice),
    ...inputKindField(slice),
    ...routeFields(slice),
    routeDecisionMs: relativeMs(slice.route, startedAt),
    modelLaneWaitMs: slice.modelLaneWait?.waitMs ?? null,
    modelLaneQueuedAhead: slice.modelLaneWait?.queuedAhead ?? null,
    modelLaneInFlightAtQueue: slice.modelLaneWait?.inFlightAtQueue ?? null,
    schedulerQueuedMs:
      schedulerAdmittedQueuedMs(id, admitted) ?? firstToken?.queuedMs ?? finished?.queuedMs ?? null,
    schedulerPrefillStartMs: prefillStart?.schedulerMs ?? null,
    schedulerAdmittedMs: admitted?.schedulerMs ?? null,
    schedulerFirstTokenMs: firstToken?.schedulerMs ?? null,
    schedulerFinishedMs: finished?.schedulerMs ?? null,
    schedulerPhaseEvents: slice.schedulerEvents.length,
    schedulerAdmittedBatchSize: admitted?.batchSize ?? null,
    ...schedulerTokenFields(slice),
    firstPrefillProgressMs: firstPrefillMs,
    lastPrefillProgressMs: lastPrefillMs,
    prefillObservedMs: prefillObservedMs(firstPrefillMs, lastPrefillMs),
    firstCompletionProgressMs: relativeMs(firstCompletion, startedAt),
    completeObservedMs: relativeMs(slice.complete ?? slice.error, startedAt),
    durationMs: requestDurationMs(slice),
    maxSilentEventGapMs: maxObservedEventGapMs(slice.sorted),
    prefillEvents: slice.prefillEvents.length,
    progressEvents: slice.progressEvents.length,
    maxCompletionTokens: maxCompletionTokens(slice.sorted),
    ...serverStreamFields(slice),
    ...terminalFields(slice),
  };
}

export function serverRequestTimingReports(
  events: readonly RecordedServeEvent[],
): ServerRequestTimingReport[] {
  return [...groupEventsByRequestId(events)]
    .map(([id, group]) => requestTimingReport(id, group))
    .sort((left, right) => left.id.localeCompare(right.id));
}

function summarizeRouteDecisions(
  decisions: readonly RouteDecisionReport[],
): RouteDecisionSummary[] {
  const counts = new Map<string, RouteDecisionSummary>();
  for (const decision of decisions) {
    const key = `${decision.route}:${decision.reason}`;
    const existing = counts.get(key);
    if (existing === undefined) {
      counts.set(key, {
        key,
        route: decision.route,
        reason: decision.reason,
        count: 1,
      });
      continue;
    }
    existing.count += 1;
  }
  return [...counts.values()].sort((left, right) => left.key.localeCompare(right.key));
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function arrivalSpanMs(rung: ServeBenchmarkRung, options: ServeBenchmarkOptions): number {
  return requestLaunchDelayMs(rungConcurrency(rung) - 1, options.requestStaggerMs);
}

function requestRung(shape: ServeBenchmarkRequestShape): ServeBenchmarkRung {
  return { ...shape, concurrency: 1 };
}

function uniquePreparedRequests(
  requests: readonly PreparedBenchmarkRequest[],
): PreparedBenchmarkRequest[] {
  const seen = new Set<string>();
  const unique: PreparedBenchmarkRequest[] = [];
  for (const request of requests) {
    const key = `${request.shape.promptTokens}x${request.shape.generationTokens}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    unique.push(request);
  }
  return unique;
}

async function runTrial(
  endpoint: string,
  modelId: string,
  preparedRequests: readonly PreparedBenchmarkRequest[],
  options: ServeBenchmarkOptions,
  serveEvents: readonly RecordedServeEvent[],
): Promise<TrialMetrics> {
  resetPeakMemory();
  clearMemoryCache();
  const memoryBefore = getMemoryStats();
  const eventStart = serveEvents.length;
  const started = performance.now();
  const requests = preparedRequests.map(async (prepared, requestIndex) => {
    const delayMs = requestLaunchDelayMs(requestIndex, options.requestStaggerMs);
    if (delayMs > 0) {
      await Bun.sleep(delayMs);
    }
    const metrics = await runCompletionRequest(
      endpoint,
      modelId,
      prepared.prompt,
      requestRung(prepared.shape),
      options,
    );
    return {
      index: requestIndex,
      launchDelayMs: delayMs,
      ...metrics,
    };
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
  const continuousAdmissionSizes = continuousSchedulerBatchSizeEvents(events, "admitted");
  const generationBatchSizes = [...staticBatchSizes, ...continuousAdmissionSizes];
  const routeDecisions = routeDecisionReports(events);
  const serverRequests = serverRequestTimingReports(events);

  return {
    wallMs,
    requestTps: preparedRequests.length / wallSeconds,
    completionTps: completionTokens / wallSeconds,
    totalTps: totalTokens / wallSeconds,
    meanTtftMs: meanPresent(results.map((result) => result.ttftMs)),
    meanPromptToFirstTokenTps: meanPresent(results.map((result) => result.promptToFirstTokenTps)),
    meanPostTtftCompletionTps: meanPresent(results.map((result) => result.postTtftCompletionTps)),
    meanStreamChunkGapMs: meanPresent(results.map((result) => result.meanStreamChunkGapMs)),
    maxStreamChunkGapMs: maxPresent(results.map((result) => result.maxStreamChunkGapMs)),
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
    continuousAdmissions: continuousAdmissionSizes.length,
    continuousAdmissionRows: sum(continuousAdmissionSizes),
    continuousSchedulerPhases: countEvents(events, "generation_scheduler_phase"),
    maxContinuousBatchSize: Math.max(0, ...continuousAdmissionSizes),
    maxGenerationBatchSize: Math.max(0, ...generationBatchSizes),
    streamChunks,
    streamBytes,
    finishReasons: results.map((result) => result.finishReason),
    routeDecisions,
    routeSummary: summarizeRouteDecisions(routeDecisions),
    requests: results,
    serverRequests,
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
    meanStreamChunkGapMs: meanPresent(trials.map((trial) => trial.meanStreamChunkGapMs)),
    maxStreamChunkGapMs: maxPresent(trials.map((trial) => trial.maxStreamChunkGapMs)),
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
    continuousSchedulerPhases: mean(trials.map((trial) => trial.continuousSchedulerPhases)),
    maxContinuousBatchSize: mean(trials.map((trial) => trial.maxContinuousBatchSize)),
    maxGenerationBatchSize: mean(trials.map((trial) => trial.maxGenerationBatchSize)),
    streamChunks: mean(trials.map((trial) => trial.streamChunks)),
    streamBytes: mean(trials.map((trial) => trial.streamBytes)),
    finishReasons: trials.flatMap((trial) => trial.finishReasons),
    routeDecisions: trials.flatMap((trial) => trial.routeDecisions),
    routeSummary: summarizeRouteDecisions(trials.flatMap((trial) => trial.routeDecisions)),
    requests: trials.flatMap((trial) => trial.requests),
    serverRequests: trials.flatMap((trial) => trial.serverRequests),
  };
}

function formatNullableMs(value: number | null): string {
  return value === null ? "n/a" : value.toFixed(1);
}

function formatNullableTps(value: number | null): string {
  return value === null ? "n/a" : value.toFixed(3);
}

function formatRouteSummary(summary: readonly RouteDecisionSummary[]): string {
  return summary.map((entry) => `${entry.key}=${entry.count}`).join("|") || "none";
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
      `mean_stream_chunk_gap_ms=${formatNullableMs(metrics.meanStreamChunkGapMs)}`,
      `max_stream_chunk_gap_ms=${formatNullableMs(metrics.maxStreamChunkGapMs)}`,
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
      `continuous_scheduler_phases=${metrics.continuousSchedulerPhases.toFixed(0)}`,
      `max_continuous_batch=${metrics.maxContinuousBatchSize.toFixed(0)}`,
      `max_generation_batch=${metrics.maxGenerationBatchSize.toFixed(0)}`,
      `stream_chunks=${metrics.streamChunks.toFixed(0)}`,
      `stream_bytes=${metrics.streamBytes.toFixed(0)}`,
      `finish_reasons=${[...new Set(metrics.finishReasons)].join("|") || "none"}`,
      `routes=${formatRouteSummary(metrics.routeSummary)}`,
    ].join(" "),
  );
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
  serveEvents: readonly RecordedServeEvent[],
): Promise<RungReport> {
  const preparedRequests = requestShapesForRung(rung).map((shape) => ({
    shape,
    prompt: createBenchmarkPrompt(
      shape.promptTokens,
      modelVocabSize,
      tokenizer,
      options.protocolMode,
    ),
  }));
  console.log(
    [
      `rung=${formatServeBenchmarkRung(rung)}`,
      `concurrency=${rungConcurrency(rung)}`,
      `sampling=${options.samplingMode}`,
      `transport=${options.transportMode}`,
      `protocol=${options.protocolMode}`,
      `request_stagger_ms=${options.requestStaggerMs}`,
      `arrival_span_ms=${arrivalSpanMs(rung, options)}`,
    ].join(" "),
  );

  if (options.warmup) {
    for (const prepared of uniquePreparedRequests(preparedRequests)) {
      await runCompletionRequest(
        endpoint,
        modelId,
        prepared.prompt,
        requestRung(prepared.shape),
        options,
      );
    }
    clearMemoryCache();
  }

  const trials: TrialMetrics[] = [];
  for (let index = 0; index < options.trials; index += 1) {
    const metrics = await runTrial(endpoint, modelId, preparedRequests, options, serveEvents);
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
      `rungs=${rungs.map(formatServeBenchmarkRung).join(",")}`,
      `matrix=${options.matrix}`,
      `transport=${options.transportMode}`,
      `protocol=${options.protocolMode}`,
      `sampling=${options.samplingMode}`,
      `ignore_eos=${options.ignoreEos}`,
      `max_batch_size=${options.maxBatchSize}`,
      `batch_window_ms=${options.batchWindowMs}`,
      `stream_decode_interval=${options.streamDecodeInterval}`,
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

  const serveEvents: RecordedServeEvent[] = [];
  const server = serveLoadedModel({
    model: loadedModel,
    tokenizer,
    interactionProfile,
    modelId: options.modelId,
    port: options.port,
    maxGeneratedTokens: maximum(rungs.map(maxGenerationTokensForRung)),
    maxPromptTokens: options.maxPromptTokens ?? maximum(rungs.map(maxPromptTokensForRung)),
    maxTotalTokens: options.maxTotalTokens ?? maximum(rungs.map(maxTotalTokensForRung)),
    maxBatchSize: options.maxBatchSize,
    batchWindowMs: options.batchWindowMs,
    streamDecodeInterval: options.streamDecodeInterval,
    maxConcurrentRequests: options.maxConcurrentRequests,
    gpuMemoryUtilization: options.gpuMemoryUtilization,
    onEvent(event) {
      serveEvents.push({ ...event, observedAtMs: performance.now() });
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
        streamDecodeInterval: options.streamDecodeInterval,
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
