#!/usr/bin/env bun

import { clearMemoryCache } from "@mlxts/core";
import { loadCausalLM, loadInteractionProfile, loadPretrainedTokenizer } from "@mlxts/transformers";
import { mkdirSync } from "fs";
import { dirname } from "path";
import { acquireRuntimeCommandLock } from "../../../scripts/runtime-command-lock";
import { serveLoadedModels } from "../src/model-loading/server";
import {
  DEFAULT_MODEL_SERVER_MAX_CONCURRENT_REQUESTS,
  DEFAULT_MODEL_SERVER_PROMPT_PREFIX_CACHE_MAX_ENTRIES,
  type LoadedModelServerEntry,
} from "../src/model-loading/server-options";
import type { ServeEvent } from "../src/types";
import {
  type BenchmarkPrompt,
  type RequestMetrics,
  runCompletionRequest,
} from "./benchmark-serve-completions";
import type { ServeBenchmarkOptions } from "./benchmark-serve-options";

type ScenarioName =
  | "qwen-dense"
  | "gemma-dense"
  | "multi-dense"
  | "qwen-moe"
  | "gemma-moe"
  | "multi-moe";

export type AgentCacheModelTarget = {
  id: string;
  source: string;
};

export type AgentCacheRegressionOptions = {
  scenarios: readonly ScenarioName[];
  qwenModel: string;
  gemmaModel: string;
  qwenMoeModel: string;
  gemmaMoeModel: string;
  reportJson: string;
  promptTokens: number;
  maxTokens: number;
  maxConcurrentRequests: number;
  promptPrefixCacheMaxEntries: number;
  requestTimeoutMs: number;
  gpuMemoryUtilization: number;
  allowDownload: boolean;
};

type AgentCacheRegressionCommand =
  | { kind: "help" }
  | { kind: "run"; options: AgentCacheRegressionOptions };

type RuntimeLock = {
  [Symbol.dispose](): void;
};

type AgentCacheRegressionRuntime = {
  stdout?: (text: string) => void;
  stderr?: (text: string) => void;
  acquireLock?: () => RuntimeLock;
  runRegression?: (
    options: AgentCacheRegressionOptions,
    progress: (text: string) => void,
  ) => Promise<AgentCacheRegressionReport>;
  writeReport?: (path: string, report: AgentCacheRegressionReport) => Promise<void>;
};

type PromptCacheStats = {
  hits: number;
  misses: number;
  writes: number;
  readTokens: number;
  writeTokens: number;
};

type PromptCacheEvent = Extract<ServeEvent, { type: "generation_prompt_cache" }>;
type RouteDecisionEvent = Extract<ServeEvent, { type: "generation_route_decision" }>;
type PromptPrepareEvent = Extract<ServeEvent, { type: "generation_prompt_prepare" }>;
type PrefillProgressEvent = Extract<ServeEvent, { type: "generation_prefill_progress" }>;
type StreamChunkEvent = Extract<ServeEvent, { type: "generation_stream_chunk" }>;
type StreamEndEvent = Extract<ServeEvent, { type: "generation_stream_end" }>;

export type AgentCacheProbePhase = "cold-a" | "cold-b" | "warm-a" | "warm-b" | "exact-a";

export type AgentCacheServerRouteReport = {
  id: string;
  route: string;
  eligible: boolean;
  reason: string;
  modelType: string;
  stream: boolean;
};

export type AgentCacheServerPromptPrepareReport = {
  id: string;
  inputKind: string;
  promptTokens?: number;
  durationMs?: number;
};

export type AgentCacheServerPromptCacheEventReport = {
  id: string;
  result: "hit" | "miss" | "write";
  promptTokens: number;
  readTokens: number;
  writeTokens: number;
  matchType?: string;
  sourceTokenLength?: number;
  sourceLayerKinds?: readonly string[];
  sourceTrimmable?: boolean;
  retainedSnapshots?: number;
  retainedSnapshotBytes?: number;
  tokenBlocks?: number;
  tokenBlockReferences?: number;
};

export type AgentCacheServerPrefillReport = {
  id: string;
  events: number;
  processedPrefillTokens: number;
  totalPrefillTokens: number;
  maxChunkTokens: number;
};

export type AgentCacheServerStreamReport = {
  id: string;
  chunks: number;
  bytes: number;
  outputChunks: number;
  outputBytes: number;
  ttftMs: number | null;
  durationMs: number | null;
  finishReason: string | null;
  result: string | null;
};

export type AgentCacheServerEvidenceReport = {
  routes: readonly AgentCacheServerRouteReport[];
  promptPrepare: readonly AgentCacheServerPromptPrepareReport[];
  promptCache: readonly AgentCacheServerPromptCacheEventReport[];
  prefill: readonly AgentCacheServerPrefillReport[];
  streams: readonly AgentCacheServerStreamReport[];
};

export type AgentCacheRequestEvidenceReport = {
  phase: AgentCacheProbePhase;
  session: "A" | "B";
  responseId?: string;
  durationMs: number;
  ttftMs: number | null;
  finishReason: string;
  streamChunks: number;
  streamBytes: number;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  cacheReadTokens: number;
  cacheWriteTokens: number;
  server: AgentCacheServerEvidenceReport;
};

export type AgentCacheClientUsageReport = {
  session: "A" | "B";
  readTokens: number;
  writeTokens: number;
};

export type AgentCacheProbeReport = {
  id: string;
  modelId: string;
  cold: PromptCacheStats;
  warm: PromptCacheStats;
  exactReplay: PromptCacheStats;
  requests: readonly AgentCacheRequestEvidenceReport[];
  coldClientUsage: readonly AgentCacheClientUsageReport[];
  warmClientUsage: readonly AgentCacheClientUsageReport[];
  warmClientReadTokens: number;
  warmClientWriteTokens: number;
  exactReplayClientReadTokens: number;
  exactReplayClientWriteTokens: number;
};

export type AgentCacheScenarioReport = {
  id: ScenarioName;
  models: readonly string[];
  targets: readonly AgentCacheModelTarget[];
  probes: readonly AgentCacheProbeReport[];
  status: "passed";
};

export type AgentCacheRegressionReport = {
  createdAt: string;
  promptTokens: number;
  maxTokens: number;
  maxConcurrentRequests: number;
  promptPrefixCacheMaxEntries: number;
  scenarios: readonly AgentCacheScenarioReport[];
};

class AgentCacheRegressionUsageError extends Error {}

const DEFAULT_SCENARIOS: ScenarioName[] = ["qwen-dense", "gemma-dense", "multi-dense"];

export function formatAgentCacheRegressionUsage(): string {
  return [
    "description: Run the @mlxts/serve Pi-style prompt-prefix cache regression",
    "usage[3]:",
    "  bun run regression:agent-cache",
    "  bun run regression:agent-cache -- --scenarios qwen-dense,gemma-dense,multi-dense",
    "  bun run regression:agent-cache -- --include-moe --include-moe-multi",
    "options[16]{flag,description}:",
    '  "--scenarios <list>","Comma-separated qwen-dense,gemma-dense,multi-dense,qwen-moe,gemma-moe,multi-moe"',
    '  "--include-moe","Add qwen-moe and gemma-moe to the default scenario list"',
    '  "--include-moe-multi","Add multi-moe to the scenario list"',
    '  "--qwen-model <id>","Dense Qwen model id/path"',
    '  "--gemma-model <id>","Dense Gemma model id/path"',
    '  "--qwen-moe-model <id>","Qwen MoE model id/path"',
    '  "--gemma-moe-model <id>","Gemma MoE model id/path"',
    '  "--prompt-tokens <n>","Approximate chat prompt length per session; default 128"',
    '  "--max-tokens <n>","Max generated tokens per probe; default 16"',
    `  "--max-concurrent-requests <n>","Active model jobs allowed during probes; default ${DEFAULT_MODEL_SERVER_MAX_CONCURRENT_REQUESTS}"`,
    `  "--prompt-prefix-cache-max-entries <n>","Retained prompt-boundary snapshots per served model; default ${DEFAULT_MODEL_SERVER_PROMPT_PREFIX_CACHE_MAX_ENTRIES}"`,
    '  "--request-timeout-ms <n>","Client timeout per request; default 3600000"',
    '  "--gpu-memory-utilization <f>","Serving memory preflight budget in (0,1]; default 0.85"',
    '  "--report-json <path>","Report path; default .tmp/agent-cache-regression/report.json"',
    '  "--allow-download","Allow Hub downloads; default is cached/local only"',
    '  "--help","Show this help"',
    "exit_codes[3]{code,meaning}:",
    '  0,"regression passed"',
    '  1,"runtime or regression failure"',
    '  2,"usage error"',
  ].join("\n");
}

function readValue(args: readonly string[], index: number, flag: string): string {
  const value = args[index + 1];
  if (value === undefined || value.trim() === "" || value.startsWith("-")) {
    throw new AgentCacheRegressionUsageError(`${flag} requires a value.`);
  }
  return value;
}

function readPositiveInteger(args: readonly string[], index: number, flag: string): number {
  const rawValue = args[index + 1]?.trim();
  if (rawValue === undefined || rawValue === "" || rawValue.startsWith("--")) {
    throw new AgentCacheRegressionUsageError(`${flag} requires a value.`);
  }
  const parsed = /^\d+$/.test(rawValue) ? Number(rawValue) : Number.NaN;
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new AgentCacheRegressionUsageError(`${flag} must be a positive integer.`);
  }
  return parsed;
}

function readPositiveFraction(args: readonly string[], index: number, flag: string): number {
  const rawValue = args[index + 1]?.trim();
  if (rawValue === undefined || rawValue === "" || rawValue.startsWith("--")) {
    throw new AgentCacheRegressionUsageError(`${flag} requires a value.`);
  }
  const parsed = Number(rawValue);
  if (!Number.isFinite(parsed) || parsed <= 0 || parsed > 1) {
    throw new AgentCacheRegressionUsageError(`${flag} must be between 0 and 1.`);
  }
  return parsed;
}

function parseScenario(value: string): ScenarioName {
  switch (value) {
    case "qwen-dense":
    case "gemma-dense":
    case "multi-dense":
    case "qwen-moe":
    case "gemma-moe":
    case "multi-moe":
      return value;
    default:
      throw new AgentCacheRegressionUsageError(`unknown scenario "${value}".`);
  }
}

function parseScenarios(value: string): ScenarioName[] {
  if (value.trim() === "") {
    throw new AgentCacheRegressionUsageError("--scenarios requires at least one scenario.");
  }
  return [...new Set(value.split(",").map((entry) => parseScenario(entry.trim())))];
}

function addScenario(scenarios: ScenarioName[], scenario: ScenarioName): void {
  if (!scenarios.includes(scenario)) {
    scenarios.push(scenario);
  }
}

export function parseAgentCacheRegressionArgs(
  argv: readonly string[],
): AgentCacheRegressionCommand {
  if (argv.length === 1 && (argv[0] === "--help" || argv[0] === "-h")) {
    return { kind: "help" };
  }
  const options: AgentCacheRegressionOptions = {
    scenarios: DEFAULT_SCENARIOS,
    qwenModel: "mlx-community/Qwen3.6-27B-4bit",
    gemmaModel: "google/gemma-4-E2B-it",
    qwenMoeModel: "unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit",
    gemmaMoeModel: "mlx-community/gemma-4-26b-a4b-it-4bit",
    reportJson: ".tmp/agent-cache-regression/report.json",
    promptTokens: 128,
    maxTokens: 16,
    maxConcurrentRequests: DEFAULT_MODEL_SERVER_MAX_CONCURRENT_REQUESTS,
    promptPrefixCacheMaxEntries: DEFAULT_MODEL_SERVER_PROMPT_PREFIX_CACHE_MAX_ENTRIES,
    requestTimeoutMs: 3_600_000,
    gpuMemoryUtilization: 0.85,
    allowDownload: false,
  };
  let scenarios = [...options.scenarios];

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === undefined) {
      throw new AgentCacheRegressionUsageError("argument parsing reached an empty slot.");
    }
    switch (arg) {
      case "--help":
      case "-h":
        return { kind: "help" };
      case "--scenarios":
        scenarios = parseScenarios(readValue(argv, index, arg));
        index += 1;
        break;
      case "--include-moe":
        addScenario(scenarios, "qwen-moe");
        addScenario(scenarios, "gemma-moe");
        break;
      case "--include-moe-multi":
        addScenario(scenarios, "multi-moe");
        break;
      case "--qwen-model":
        options.qwenModel = readValue(argv, index, arg);
        index += 1;
        break;
      case "--gemma-model":
        options.gemmaModel = readValue(argv, index, arg);
        index += 1;
        break;
      case "--qwen-moe-model":
        options.qwenMoeModel = readValue(argv, index, arg);
        index += 1;
        break;
      case "--gemma-moe-model":
        options.gemmaMoeModel = readValue(argv, index, arg);
        index += 1;
        break;
      case "--prompt-tokens":
        options.promptTokens = readPositiveInteger(argv, index, arg);
        index += 1;
        break;
      case "--max-tokens":
        options.maxTokens = readPositiveInteger(argv, index, arg);
        index += 1;
        break;
      case "--max-concurrent-requests":
        options.maxConcurrentRequests = readPositiveInteger(argv, index, arg);
        index += 1;
        break;
      case "--prompt-prefix-cache-max-entries":
        options.promptPrefixCacheMaxEntries = readPositiveInteger(argv, index, arg);
        index += 1;
        break;
      case "--request-timeout-ms":
        options.requestTimeoutMs = readPositiveInteger(argv, index, arg);
        index += 1;
        break;
      case "--gpu-memory-utilization":
        options.gpuMemoryUtilization = readPositiveFraction(argv, index, arg);
        index += 1;
        break;
      case "--report-json":
        options.reportJson = readValue(argv, index, arg);
        index += 1;
        break;
      case "--allow-download":
        options.allowDownload = true;
        break;
      default:
        throw new AgentCacheRegressionUsageError(
          arg.startsWith("-") ? `unknown option "${arg}".` : `unexpected argument "${arg}".`,
        );
    }
  }

  return { kind: "run", options: { ...options, scenarios } };
}

function benchmarkOptions(options: AgentCacheRegressionOptions): ServeBenchmarkOptions {
  return {
    model: "agent-cache-regression",
    modelId: "agent-cache-regression",
    promptTokens: [options.promptTokens],
    generationTokens: [options.maxTokens],
    concurrency: [2],
    requestStaggerMs: 0,
    trials: 1,
    warmup: false,
    matrix: "cartesian",
    samplingMode: "greedy",
    transportMode: "streaming",
    protocolMode: "chat",
    ignoreEos: false,
    localFilesOnly: !options.allowDownload,
    port: 0,
    maxBatchSize: 8,
    batchWindowMs: 2,
    prefillStepSize: 512,
    activePrefillStepSize: 128,
    activeDecodeStepsPerPrefillChunk: 16,
    streamDecodeInterval: 1,
    maxConcurrentRequests: options.maxConcurrentRequests,
    requestTimeoutMs: options.requestTimeoutMs,
    gpuMemoryUtilization: options.gpuMemoryUtilization,
    maxPromptTokens: Math.max(options.promptTokens + 256, 512),
    maxTotalTokens: Math.max(options.promptTokens + options.maxTokens + 256, 512),
  };
}

function isPromptCacheEventForModel(event: ServeEvent, modelId: string): event is PromptCacheEvent {
  return event.type === "generation_prompt_cache" && event.model === modelId;
}

function promptCacheStats(events: readonly ServeEvent[], modelId: string): PromptCacheStats {
  const cacheEvents = events.filter((event) => isPromptCacheEventForModel(event, modelId));
  return {
    hits: cacheEvents.filter((event) => event.result === "hit").length,
    misses: cacheEvents.filter((event) => event.result === "miss").length,
    writes: cacheEvents.filter((event) => event.result === "write").length,
    readTokens: cacheEvents.reduce((total, event) => total + event.cacheReadTokens, 0),
    writeTokens: cacheEvents.reduce((total, event) => total + event.cacheWriteTokens, 0),
  };
}

function eventsForRequest(
  events: readonly ServeEvent[],
  modelId: string,
  metrics: RequestMetrics,
): readonly ServeEvent[] {
  const modelEvents = events.filter((event) => "model" in event && event.model === modelId);
  if (metrics.id === undefined) {
    return modelEvents;
  }
  const requestEvents = modelEvents.filter((event) => "id" in event && event.id === metrics.id);
  return requestEvents.length === 0 ? modelEvents : requestEvents;
}

function routeReport(event: RouteDecisionEvent): AgentCacheServerRouteReport {
  return {
    id: event.id,
    route: event.route,
    eligible: event.eligible,
    reason: event.reason,
    modelType: event.modelType,
    stream: event.stream,
  };
}

function promptPrepareReports(
  events: readonly PromptPrepareEvent[],
): AgentCacheServerPromptPrepareReport[] {
  const reports = new Map<string, AgentCacheServerPromptPrepareReport>();
  for (const event of events) {
    const existing = reports.get(event.id);
    if (event.phase === "start") {
      reports.set(event.id, {
        id: event.id,
        inputKind: event.inputKind,
        ...(existing?.promptTokens === undefined ? {} : { promptTokens: existing.promptTokens }),
        ...(existing?.durationMs === undefined ? {} : { durationMs: existing.durationMs }),
      });
      continue;
    }
    reports.set(event.id, {
      id: event.id,
      inputKind: existing?.inputKind ?? event.inputKind,
      promptTokens: event.promptTokens,
      durationMs: Math.round(event.durationMs * 100) / 100,
    });
  }
  return [...reports.values()];
}

function promptCacheEventReport(event: PromptCacheEvent): AgentCacheServerPromptCacheEventReport {
  return {
    id: event.id,
    result: event.result,
    promptTokens: event.promptTokens,
    readTokens: event.cacheReadTokens,
    writeTokens: event.cacheWriteTokens,
    ...(event.cacheMatchType === undefined ? {} : { matchType: event.cacheMatchType }),
    ...(event.cacheSourceTokenLength === undefined
      ? {}
      : { sourceTokenLength: event.cacheSourceTokenLength }),
    ...(event.cacheSourceLayerKinds === undefined
      ? {}
      : { sourceLayerKinds: event.cacheSourceLayerKinds }),
    ...(event.cacheSourceTrimmable === undefined
      ? {}
      : { sourceTrimmable: event.cacheSourceTrimmable }),
    ...(event.retainedSnapshots === undefined
      ? {}
      : { retainedSnapshots: event.retainedSnapshots }),
    ...(event.retainedSnapshotBytes === undefined
      ? {}
      : { retainedSnapshotBytes: event.retainedSnapshotBytes }),
    ...(event.tokenBlockCount === undefined ? {} : { tokenBlocks: event.tokenBlockCount }),
    ...(event.tokenBlockReferences === undefined
      ? {}
      : { tokenBlockReferences: event.tokenBlockReferences }),
  };
}

function prefillReports(events: readonly PrefillProgressEvent[]): AgentCacheServerPrefillReport[] {
  const summaries = new Map<string, AgentCacheServerPrefillReport>();
  for (const event of events) {
    const existing = summaries.get(event.id);
    summaries.set(event.id, {
      id: event.id,
      events: (existing?.events ?? 0) + 1,
      processedPrefillTokens: Math.max(
        existing?.processedPrefillTokens ?? 0,
        event.processedPrefillTokens,
      ),
      totalPrefillTokens: Math.max(existing?.totalPrefillTokens ?? 0, event.totalPrefillTokens),
      maxChunkTokens: Math.max(existing?.maxChunkTokens ?? 0, event.chunkTokens),
    });
  }
  return [...summaries.values()];
}

function streamReports(
  chunkEvents: readonly StreamChunkEvent[],
  endEvents: readonly StreamEndEvent[],
): AgentCacheServerStreamReport[] {
  const ids = new Set([
    ...chunkEvents.map((event) => event.id),
    ...endEvents.map((event) => event.id),
  ]);
  return [...ids].map((id) => {
    const chunks = chunkEvents.filter((event) => event.id === id);
    const end = endEvents.find((event) => event.id === id);
    return {
      id,
      chunks: end?.chunks ?? chunks.length,
      bytes: end?.bytes ?? chunks.reduce((total, event) => total + event.bytes, 0),
      outputChunks: end?.outputChunks ?? 0,
      outputBytes: end?.outputBytes ?? 0,
      ttftMs: end?.ttftMs ?? null,
      durationMs: end?.durationMs ?? null,
      finishReason: end?.finishReason ?? null,
      result: end?.result ?? null,
    };
  });
}

function serverEvidenceReport(
  events: readonly ServeEvent[],
  modelId: string,
  metrics: RequestMetrics,
): AgentCacheServerEvidenceReport {
  const requestEvents = eventsForRequest(events, modelId, metrics);
  return {
    routes: requestEvents
      .filter((event): event is RouteDecisionEvent => event.type === "generation_route_decision")
      .map(routeReport),
    promptPrepare: promptPrepareReports(
      requestEvents.filter(
        (event): event is PromptPrepareEvent => event.type === "generation_prompt_prepare",
      ),
    ),
    promptCache: requestEvents
      .filter((event): event is PromptCacheEvent => event.type === "generation_prompt_cache")
      .map(promptCacheEventReport),
    prefill: prefillReports(
      requestEvents.filter(
        (event): event is PrefillProgressEvent => event.type === "generation_prefill_progress",
      ),
    ),
    streams: streamReports(
      requestEvents.filter(
        (event): event is StreamChunkEvent => event.type === "generation_stream_chunk",
      ),
      requestEvents.filter(
        (event): event is StreamEndEvent => event.type === "generation_stream_end",
      ),
    ),
  };
}

function requestEvidenceReport(
  phase: AgentCacheProbePhase,
  session: "A" | "B",
  metrics: RequestMetrics,
  events: readonly ServeEvent[],
  modelId: string,
): AgentCacheRequestEvidenceReport {
  return {
    phase,
    session,
    ...(metrics.id === undefined ? {} : { responseId: metrics.id }),
    durationMs: Math.round(metrics.durationMs * 100) / 100,
    ttftMs: metrics.ttftMs === null ? null : Math.round(metrics.ttftMs * 100) / 100,
    finishReason: metrics.finishReason,
    streamChunks: metrics.streamChunks,
    streamBytes: metrics.streamBytes,
    promptTokens: metrics.promptTokens,
    completionTokens: metrics.completionTokens,
    totalTokens: metrics.totalTokens,
    cacheReadTokens: metrics.cacheReadTokens,
    cacheWriteTokens: metrics.cacheWriteTokens,
    server: serverEvidenceReport(events, modelId, metrics),
  };
}

function chatPrompt(
  tokenizer: { encode(text: string): number[] },
  targetTokens: number,
  session: "A" | "B",
): BenchmarkPrompt {
  const shared =
    "You are running inside the mlxts repository. Follow AGENTS.md, preserve cache evidence, and answer with concise engineering judgment.";
  const divergent =
    session === "A"
      ? "Session A is editing a serve cache regression and asks for a short status."
      : "Session B is reviewing a Qwen and Gemma cache proof and asks for a short status.";
  let text = `${shared}\n${divergent}`;
  while (tokenizer.encode(text).length < targetTokens) {
    text = `${text}\n${shared}`;
  }
  return { tokenIds: [], text };
}

async function loadEntry(
  target: AgentCacheModelTarget,
  options: AgentCacheRegressionOptions,
  progress: (text: string) => void,
): Promise<LoadedModelServerEntry> {
  progress(`agent-cache-regression: loading ${target.id} from ${target.source}`);
  const loadOptions = { localFilesOnly: !options.allowDownload };
  const [tokenizer, interactionProfile] = await Promise.all([
    loadPretrainedTokenizer(target.source, loadOptions),
    loadInteractionProfile(target.source, loadOptions),
  ]);
  const model = await loadCausalLM(target.source, loadOptions);
  return { modelId: target.id, model, tokenizer, interactionProfile };
}

function disposeLoadedEntries(entries: readonly LoadedModelServerEntry[]): void {
  for (const entry of entries) {
    entry.model[Symbol.dispose]();
  }
}

async function loadEntries(
  targets: readonly AgentCacheModelTarget[],
  options: AgentCacheRegressionOptions,
  progress: (text: string) => void,
): Promise<LoadedModelServerEntry[]> {
  const entries: LoadedModelServerEntry[] = [];
  try {
    for (const target of targets) {
      entries.push(await loadEntry(target, options, progress));
    }
    return entries;
  } catch (error) {
    disposeLoadedEntries(entries);
    throw error;
  }
}

async function runProbe(
  endpoint: string,
  entry: LoadedModelServerEntry,
  options: AgentCacheRegressionOptions,
  events: readonly ServeEvent[],
): Promise<AgentCacheProbeReport> {
  const requestOptions = benchmarkOptions(options);
  const leftPrompt = chatPrompt(entry.tokenizer, options.promptTokens, "A");
  const rightPrompt = chatPrompt(entry.tokenizer, options.promptTokens, "B");
  const rung = {
    promptTokens: options.promptTokens,
    generationTokens: options.maxTokens,
    concurrency: 1,
  };

  const coldStart = events.length;
  const [coldAResult, coldBResult] = await Promise.all([
    runCompletionRequest(endpoint, entry.modelId, leftPrompt, rung, requestOptions),
    runCompletionRequest(endpoint, entry.modelId, rightPrompt, rung, requestOptions),
  ]);
  const coldEvents = events.slice(coldStart);
  const cold = promptCacheStats(coldEvents, entry.modelId);

  const warmStart = events.length;
  const [warmAResult, warmBResult] = await Promise.all([
    runCompletionRequest(endpoint, entry.modelId, leftPrompt, rung, requestOptions),
    runCompletionRequest(endpoint, entry.modelId, rightPrompt, rung, requestOptions),
  ]);
  const warmEvents = events.slice(warmStart);
  const warm = promptCacheStats(warmEvents, entry.modelId);

  const exactReplayStart = events.length;
  const exactReplayResult = await runCompletionRequest(
    endpoint,
    entry.modelId,
    leftPrompt,
    rung,
    requestOptions,
  );
  const exactReplayEvents = events.slice(exactReplayStart);
  const exactReplay = promptCacheStats(exactReplayEvents, entry.modelId);

  const warmClientUsage = [
    {
      session: "A",
      readTokens: warmAResult.cacheReadTokens,
      writeTokens: warmAResult.cacheWriteTokens,
    },
    {
      session: "B",
      readTokens: warmBResult.cacheReadTokens,
      writeTokens: warmBResult.cacheWriteTokens,
    },
  ] as const;
  const coldClientUsage = [
    {
      session: "A",
      readTokens: coldAResult.cacheReadTokens,
      writeTokens: coldAResult.cacheWriteTokens,
    },
    {
      session: "B",
      readTokens: coldBResult.cacheReadTokens,
      writeTokens: coldBResult.cacheWriteTokens,
    },
  ] as const;

  const report: AgentCacheProbeReport = {
    id: `${entry.modelId}-ab-warm-replay`,
    modelId: entry.modelId,
    cold,
    warm,
    exactReplay,
    requests: [
      requestEvidenceReport("cold-a", "A", coldAResult, coldEvents, entry.modelId),
      requestEvidenceReport("cold-b", "B", coldBResult, coldEvents, entry.modelId),
      requestEvidenceReport("warm-a", "A", warmAResult, warmEvents, entry.modelId),
      requestEvidenceReport("warm-b", "B", warmBResult, warmEvents, entry.modelId),
      requestEvidenceReport("exact-a", "A", exactReplayResult, exactReplayEvents, entry.modelId),
    ],
    coldClientUsage,
    warmClientUsage,
    warmClientReadTokens: warmClientUsage.reduce((total, result) => total + result.readTokens, 0),
    warmClientWriteTokens: warmClientUsage.reduce((total, result) => total + result.writeTokens, 0),
    exactReplayClientReadTokens: exactReplayResult.cacheReadTokens,
    exactReplayClientWriteTokens: exactReplayResult.cacheWriteTokens,
  };
  const failures = agentCacheProbeFailures(report);
  if (failures.length > 0) {
    throw new Error(`${report.id} failed: ${failures.join("; ")}`);
  }
  return report;
}

function clientBoundaryTokens(
  usage: readonly AgentCacheClientUsageReport[],
  session: "A" | "B",
): number | undefined {
  const entry = usage.find((candidate) => candidate.session === session);
  if (entry === undefined) {
    return undefined;
  }
  return entry.readTokens + entry.writeTokens;
}

function coldClientBoundaryFailure(report: AgentCacheProbeReport): string | undefined {
  const coldSessionBoundaries = [
    clientBoundaryTokens(report.coldClientUsage, "A") ?? 0,
    clientBoundaryTokens(report.coldClientUsage, "B") ?? 0,
  ];
  if (coldSessionBoundaries.some((tokens) => tokens <= 0)) {
    return "each cold divergent session must report retained prompt boundary tokens";
  }
  return undefined;
}

function warmClientBoundaryFailures(report: AgentCacheProbeReport): string[] {
  const failures: string[] = [];
  for (const usage of report.warmClientUsage) {
    const boundaryTokens = clientBoundaryTokens(report.coldClientUsage, usage.session);
    if (boundaryTokens !== undefined && usage.readTokens < boundaryTokens) {
      failures.push(
        `warm replay session ${usage.session} cached ${usage.readTokens} tokens below retained boundary ${boundaryTokens}`,
      );
    }
  }
  return failures;
}

export function agentCacheProbeFailures(report: AgentCacheProbeReport): string[] {
  const failures: string[] = [];
  if (report.cold.writes < 2 || report.cold.writeTokens <= 0) {
    failures.push("cold divergent sessions did not write two retained prompt boundaries");
  }
  const coldFailure = coldClientBoundaryFailure(report);
  if (coldFailure !== undefined) {
    failures.push(coldFailure);
  }
  if (report.warm.hits < 2 || report.warm.readTokens <= 0) {
    failures.push("warm divergent session replay did not hit both retained prompt boundaries");
  }
  failures.push(...warmClientBoundaryFailures(report));
  if (report.warmClientReadTokens <= 0) {
    failures.push("OpenAI-compatible chat usage did not report cached prompt tokens");
  }
  const exactReplayBoundaryTokens = clientBoundaryTokens(report.coldClientUsage, "A") ?? 0;
  if (report.exactReplay.hits < 1 || report.exactReplay.readTokens < exactReplayBoundaryTokens) {
    failures.push("exact A replay after divergent A/B did not hit the retained prompt boundary");
  }
  if (report.exactReplayClientReadTokens < exactReplayBoundaryTokens) {
    failures.push(
      `exact A replay cached ${report.exactReplayClientReadTokens} tokens below retained boundary ${exactReplayBoundaryTokens}`,
    );
  }
  return failures;
}

function scenarioTargets(
  scenario: ScenarioName,
  options: AgentCacheRegressionOptions,
): AgentCacheModelTarget[] {
  switch (scenario) {
    case "qwen-dense":
      return [{ id: "qwen-dense-local", source: options.qwenModel }];
    case "gemma-dense":
      return [{ id: "gemma-dense-local", source: options.gemmaModel }];
    case "multi-dense":
      return [
        { id: "qwen-dense-local", source: options.qwenModel },
        { id: "gemma-dense-local", source: options.gemmaModel },
      ];
    case "qwen-moe":
      return [{ id: "qwen-moe-local", source: options.qwenMoeModel }];
    case "gemma-moe":
      return [{ id: "gemma-moe-local", source: options.gemmaMoeModel }];
    case "multi-moe":
      return [
        { id: "qwen-moe-local", source: options.qwenMoeModel },
        { id: "gemma-moe-local", source: options.gemmaMoeModel },
      ];
  }
}

async function runScenario(
  scenario: ScenarioName,
  options: AgentCacheRegressionOptions,
  progress: (text: string) => void,
): Promise<AgentCacheScenarioReport> {
  const targets = scenarioTargets(scenario, options);
  const entries = await loadEntries(targets, options, progress);
  const events: ServeEvent[] = [];
  let server: ReturnType<typeof serveLoadedModels> | undefined;

  try {
    server = serveLoadedModels({
      models: entries,
      disposeModelsOnStop: true,
      port: 0,
      maxGeneratedTokens: options.maxTokens,
      maxPromptTokens: Math.max(options.promptTokens + 256, 512),
      maxTotalTokens: Math.max(options.promptTokens + options.maxTokens + 256, 512),
      maxBatchSize: 8,
      batchWindowMs: 2,
      streamDecodeInterval: 1,
      maxConcurrentRequests: options.maxConcurrentRequests,
      promptPrefixCacheMaxEntries: options.promptPrefixCacheMaxEntries,
      gpuMemoryUtilization: options.gpuMemoryUtilization,
      onEvent(event) {
        events.push(event);
      },
    });
    progress(`agent-cache-regression: probing ${scenario} at ${server.endpoint}`);
    const probes: AgentCacheProbeReport[] = [];
    for (const entry of entries) {
      probes.push(await runProbe(server.endpoint, entry, options, events));
    }
    return {
      id: scenario,
      models: entries.map((entry) => entry.modelId),
      targets,
      probes,
      status: "passed",
    };
  } finally {
    if (server === undefined) {
      disposeLoadedEntries(entries);
    } else {
      server.stop(true);
    }
    clearMemoryCache();
  }
}

async function writeReport(path: string, report: AgentCacheRegressionReport): Promise<void> {
  mkdirSync(dirname(path), { recursive: true });
  await Bun.write(path, `${JSON.stringify(report, null, 2)}\n`);
}

function toon(value: string | number | boolean | null): string {
  return typeof value === "string" ? JSON.stringify(value) : String(value);
}

function formatMultilineField(name: string, value: string): string[] {
  const lines = value.split(/\r?\n/);
  if (lines.length === 1) {
    return [`  ${name}: ${toon(value)}`];
  }
  return [`  ${name}: |`, ...lines.map((line) => `    ${line}`)];
}

export function formatAgentCacheRegressionSuccess(
  report: AgentCacheRegressionReport,
  reportJson: string,
): string {
  const rows = report.scenarios.map((scenario) => {
    const warmHits = scenario.probes.reduce((total, probe) => total + probe.warm.hits, 0);
    const readTokens = scenario.probes.reduce((total, probe) => total + probe.warm.readTokens, 0);
    const clientReadTokens = scenario.probes.reduce(
      (total, probe) => total + probe.warmClientReadTokens,
      0,
    );
    const exactReplayHits = scenario.probes.reduce(
      (total, probe) => total + probe.exactReplay.hits,
      0,
    );
    const exactReplayClientReadTokens = scenario.probes.reduce(
      (total, probe) => total + probe.exactReplayClientReadTokens,
      0,
    );
    return [
      toon(scenario.id),
      toon(scenario.status),
      toon(scenario.models.join("|")),
      String(warmHits),
      String(readTokens),
      String(clientReadTokens),
      String(exactReplayHits),
      String(exactReplayClientReadTokens),
    ].join(",");
  });
  return [
    "agent_cache_regression:",
    "  status: passed",
    `  scenarios: ${report.scenarios.length}`,
    `  max_concurrent_requests: ${report.maxConcurrentRequests}`,
    `  prompt_prefix_cache_max_entries: ${report.promptPrefixCacheMaxEntries}`,
    `  report_json: ${toon(reportJson)}`,
    `scenarios[${rows.length}]{id,status,models,warm_hits,warm_read_tokens,warm_client_read_tokens,exact_replay_hits,exact_replay_client_read_tokens}:`,
    ...rows.map((row) => `  ${row}`),
  ].join("\n");
}

export async function runAgentCacheRegression(
  options: AgentCacheRegressionOptions,
  progress: (text: string) => void = console.error,
): Promise<AgentCacheRegressionReport> {
  const scenarios: AgentCacheScenarioReport[] = [];
  for (const scenario of options.scenarios) {
    scenarios.push(await runScenario(scenario, options, progress));
  }
  return {
    createdAt: new Date().toISOString(),
    promptTokens: options.promptTokens,
    maxTokens: options.maxTokens,
    maxConcurrentRequests: options.maxConcurrentRequests,
    promptPrefixCacheMaxEntries: options.promptPrefixCacheMaxEntries,
    scenarios,
  };
}

export function formatAgentCacheRegressionError(message: string, help: string): string {
  return ["error:", ...formatMultilineField("message", message), `help: ${toon(help)}`].join("\n");
}

export async function runAgentCacheRegressionCommand(
  argv: readonly string[],
  runtime: AgentCacheRegressionRuntime = {},
): Promise<number> {
  const stdout = runtime.stdout ?? console.log;
  const stderr = runtime.stderr ?? console.error;
  const acquireLock =
    runtime.acquireLock ?? (() => acquireRuntimeCommandLock("regression:agent-cache"));
  const runRegression = runtime.runRegression ?? runAgentCacheRegression;
  const writeRegressionReport = runtime.writeReport ?? writeReport;
  let command: AgentCacheRegressionCommand;

  try {
    command = parseAgentCacheRegressionArgs(argv);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    stdout(
      formatAgentCacheRegressionError(
        message,
        "bun run regression:agent-cache -- --scenarios qwen-dense",
      ),
    );
    return error instanceof AgentCacheRegressionUsageError ? 2 : 1;
  }

  if (command.kind === "help") {
    stdout(formatAgentCacheRegressionUsage());
    return 0;
  }

  let lock: RuntimeLock | undefined;
  try {
    lock = acquireLock();
    const report = await runRegression(command.options, stderr);
    await writeRegressionReport(command.options.reportJson, report);
    stdout(formatAgentCacheRegressionSuccess(report, command.options.reportJson));
    return 0;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    stdout(
      formatAgentCacheRegressionError(
        message,
        "inspect the report path or rerun with a narrower --scenarios list",
      ),
    );
    return 1;
  } finally {
    lock?.[Symbol.dispose]();
  }
}

if (import.meta.main) {
  const exitCode = await runAgentCacheRegressionCommand(Bun.argv.slice(2));
  process.exit(exitCode);
}
