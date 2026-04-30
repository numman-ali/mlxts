#!/usr/bin/env bun

import { clearMemoryCache } from "@mlxts/core";
import { loadCausalLM, loadInteractionProfile, loadPretrainedTokenizer } from "@mlxts/transformers";
import { mkdirSync } from "fs";
import { dirname } from "path";
import { acquireRuntimeCommandLock } from "../../../scripts/runtime-command-lock";
import { serveLoadedModels } from "../src/model-loading/server";
import type { LoadedModelServerEntry } from "../src/model-loading/server-options";
import type { ServeEvent } from "../src/types";
import { type BenchmarkPrompt, runCompletionRequest } from "./benchmark-serve-completions";
import type { ServeBenchmarkOptions } from "./benchmark-serve-options";

type ScenarioName =
  | "qwen-dense"
  | "gemma-dense"
  | "multi-dense"
  | "qwen-moe"
  | "gemma-moe"
  | "multi-moe";

type ModelTarget = {
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
  requestTimeoutMs: number;
  gpuMemoryUtilization: number;
  allowDownload: boolean;
};

type PromptCacheStats = {
  hits: number;
  misses: number;
  writes: number;
  readTokens: number;
  writeTokens: number;
};

type PromptCacheEvent = Extract<ServeEvent, { type: "generation_prompt_cache" }>;

export type AgentCacheProbeReport = {
  id: string;
  modelId: string;
  cold: PromptCacheStats;
  warm: PromptCacheStats;
  warmClientReadTokens: number;
  warmClientWriteTokens: number;
};

export type AgentCacheScenarioReport = {
  id: ScenarioName;
  models: readonly string[];
  probes: readonly AgentCacheProbeReport[];
  status: "passed";
};

export type AgentCacheRegressionReport = {
  createdAt: string;
  promptTokens: number;
  maxTokens: number;
  scenarios: readonly AgentCacheScenarioReport[];
};

class UsageError extends Error {}

const DEFAULT_SCENARIOS: ScenarioName[] = ["qwen-dense", "gemma-dense", "multi-dense"];

function usage(): string {
  return [
    "Usage: bun run regression:agent-cache -- [options]",
    "",
    "Options:",
    "  --scenarios <list>              Comma-separated scenarios.",
    "                                  qwen-dense,gemma-dense,multi-dense,qwen-moe,gemma-moe,multi-moe",
    "  --include-moe                  Add qwen-moe and gemma-moe to the default scenario list.",
    "  --include-moe-multi            Add multi-moe to the scenario list.",
    "  --qwen-model <id>              Dense Qwen model id/path.",
    "  --gemma-model <id>             Dense Gemma model id/path.",
    "  --qwen-moe-model <id>          Qwen MoE model id/path.",
    "  --gemma-moe-model <id>         Gemma MoE model id/path.",
    "  --prompt-tokens <n>            Approximate chat prompt length per session, default 128.",
    "  --max-tokens <n>               Max generated tokens per probe, default 16.",
    "  --request-timeout-ms <n>       Client timeout per request, default 3600000.",
    "  --gpu-memory-utilization <f>   Serving memory preflight budget, default 0.85.",
    "  --report-json <path>           Report path, default .tmp/agent-cache-regression/report.json.",
    "  --allow-download               Allow Hub downloads; default is cached/local only.",
  ].join("\n");
}

function readValue(args: readonly string[], index: number, flag: string): string {
  const value = args[index + 1];
  if (value === undefined || value.startsWith("--")) {
    throw new UsageError(`${flag} requires a value.`);
  }
  return value;
}

function readPositiveInteger(args: readonly string[], index: number, flag: string): number {
  const parsed = Number.parseInt(readValue(args, index, flag), 10);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new UsageError(`${flag} must be a positive integer.`);
  }
  return parsed;
}

function readPositiveFraction(args: readonly string[], index: number, flag: string): number {
  const parsed = Number.parseFloat(readValue(args, index, flag));
  if (!Number.isFinite(parsed) || parsed <= 0 || parsed > 1) {
    throw new UsageError(`${flag} must be between 0 and 1.`);
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
      throw new UsageError(`unknown scenario "${value}".`);
  }
}

function parseScenarios(value: string): ScenarioName[] {
  if (value.trim() === "") {
    throw new UsageError("--scenarios requires at least one scenario.");
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
): AgentCacheRegressionOptions {
  const options: AgentCacheRegressionOptions = {
    scenarios: DEFAULT_SCENARIOS,
    qwenModel: "mlx-community/Qwen3.6-27B-4bit",
    gemmaModel: "google/gemma-4-E2B-it",
    qwenMoeModel: "unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit",
    gemmaMoeModel: "mlx-community/gemma-4-26b-a4b-it-4bit",
    reportJson: ".tmp/agent-cache-regression/report.json",
    promptTokens: 128,
    maxTokens: 16,
    requestTimeoutMs: 3_600_000,
    gpuMemoryUtilization: 0.85,
    allowDownload: false,
  };
  let scenarios = [...options.scenarios];

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
      case "--help":
      case "-h":
        console.log(usage());
        process.exit(0);
        return options;
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
        throw new UsageError(`unknown option "${arg}".`);
    }
  }

  return { ...options, scenarios };
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
    maxConcurrentRequests: 1,
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
  target: ModelTarget,
  options: AgentCacheRegressionOptions,
): Promise<LoadedModelServerEntry> {
  console.error(`agent-cache-regression: loading ${target.id} from ${target.source}`);
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
  targets: readonly ModelTarget[],
  options: AgentCacheRegressionOptions,
): Promise<LoadedModelServerEntry[]> {
  const entries: LoadedModelServerEntry[] = [];
  try {
    for (const target of targets) {
      entries.push(await loadEntry(target, options));
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
  await Promise.all([
    runCompletionRequest(endpoint, entry.modelId, leftPrompt, rung, requestOptions),
    runCompletionRequest(endpoint, entry.modelId, rightPrompt, rung, requestOptions),
  ]);
  const cold = promptCacheStats(events.slice(coldStart), entry.modelId);

  const warmStart = events.length;
  const warmResults = await Promise.all([
    runCompletionRequest(endpoint, entry.modelId, leftPrompt, rung, requestOptions),
    runCompletionRequest(endpoint, entry.modelId, rightPrompt, rung, requestOptions),
  ]);
  const warm = promptCacheStats(events.slice(warmStart), entry.modelId);

  const report: AgentCacheProbeReport = {
    id: `${entry.modelId}-ab-warm-replay`,
    modelId: entry.modelId,
    cold,
    warm,
    warmClientReadTokens: warmResults.reduce((total, result) => total + result.cacheReadTokens, 0),
    warmClientWriteTokens: warmResults.reduce(
      (total, result) => total + result.cacheWriteTokens,
      0,
    ),
  };
  const failures = agentCacheProbeFailures(report);
  if (failures.length > 0) {
    throw new Error(`${report.id} failed: ${failures.join("; ")}`);
  }
  return report;
}

export function agentCacheProbeFailures(report: AgentCacheProbeReport): string[] {
  const failures: string[] = [];
  if (report.cold.writes < 2 || report.cold.writeTokens <= 0) {
    failures.push("cold divergent sessions did not write two retained prompt boundaries");
  }
  if (report.warm.hits < 2 || report.warm.readTokens <= 0) {
    failures.push("warm divergent session replay did not hit both retained prompt boundaries");
  }
  if (report.warmClientReadTokens <= 0) {
    failures.push("OpenAI-compatible chat usage did not report cached prompt tokens");
  }
  return failures;
}

function scenarioTargets(
  scenario: ScenarioName,
  options: AgentCacheRegressionOptions,
): ModelTarget[] {
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
): Promise<AgentCacheScenarioReport> {
  const targets = scenarioTargets(scenario, options);
  const entries = await loadEntries(targets, options);
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
      maxConcurrentRequests: 1,
      gpuMemoryUtilization: options.gpuMemoryUtilization,
      onEvent(event) {
        events.push(event);
      },
    });
    console.error(`agent-cache-regression: probing ${scenario} at ${server.endpoint}`);
    const probes: AgentCacheProbeReport[] = [];
    for (const entry of entries) {
      probes.push(await runProbe(server.endpoint, entry, options, events));
    }
    return {
      id: scenario,
      models: entries.map((entry) => entry.modelId),
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

function formatSummary(report: AgentCacheRegressionReport, reportJson: string): string {
  const rows = report.scenarios.map((scenario) => {
    const warmHits = scenario.probes.reduce((total, probe) => total + probe.warm.hits, 0);
    const readTokens = scenario.probes.reduce((total, probe) => total + probe.warm.readTokens, 0);
    const clientReadTokens = scenario.probes.reduce(
      (total, probe) => total + probe.warmClientReadTokens,
      0,
    );
    return `  ${scenario.id},${scenario.status},${scenario.models.join("|")},${warmHits},${readTokens},${clientReadTokens}`;
  });
  return [
    "agent_cache_regression:",
    "  status: passed",
    `  scenarios: ${report.scenarios.length}`,
    `  report_json: ${reportJson}`,
    `scenarios[${rows.length}]{id,status,models,warm_hits,warm_read_tokens,warm_client_read_tokens}:`,
    ...rows,
  ].join("\n");
}

export async function runAgentCacheRegression(
  options: AgentCacheRegressionOptions,
): Promise<AgentCacheRegressionReport> {
  const scenarios: AgentCacheScenarioReport[] = [];
  for (const scenario of options.scenarios) {
    scenarios.push(await runScenario(scenario, options));
  }
  return {
    createdAt: new Date().toISOString(),
    promptTokens: options.promptTokens,
    maxTokens: options.maxTokens,
    scenarios,
  };
}

function formatError(message: string, help: string): string {
  return ["error:", `  message: ${message}`, `help: ${help}`].join("\n");
}

async function main(): Promise<void> {
  let options: AgentCacheRegressionOptions;
  try {
    options = parseAgentCacheRegressionArgs(Bun.argv.slice(2));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.log(formatError(message, "bun run regression:agent-cache -- --scenarios qwen-dense"));
    process.exit(error instanceof UsageError ? 2 : 1);
    return;
  }

  try {
    using _runtimeLock = acquireRuntimeCommandLock("regression:agent-cache");
    const report = await runAgentCacheRegression(options);
    await writeReport(options.reportJson, report);
    console.log(formatSummary(report, options.reportJson));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.log(
      formatError(message, "inspect the report path or rerun with a narrower --scenarios list"),
    );
    process.exit(1);
  }
}

if (import.meta.main) {
  await main();
}
