#!/usr/bin/env bun

import { clearMemoryCache, getMemoryStats, mxAsyncEval, resetPeakMemory } from "@mlxts/core";
import type { Tokenizer } from "@mlxts/tokenizers";
import { mkdirSync } from "fs";
import { dirname } from "path";
import { acquireRuntimeCommandLock } from "../../../scripts/runtime-command-lock";
import {
  type InteractionProfile,
  loadCausalLM,
  loadInteractionProfile,
  loadPretrainedTokenizer,
} from "../src";
import {
  type BenchmarkProgress,
  BenchmarkUsageError,
  formatBenchmarkError,
  resolveCachedSnapshotPath,
  sanitizePathSegment,
  withBenchmarkRuntimeScope,
} from "./benchmark-common";
import { predictGreedyToken, prefillBenchmarkCache } from "./benchmark-model";

export type LongContextOptions = {
  model: string;
  rungs: number[] | null;
  generationTokens: number;
  prefillStepSize: number;
  metalTrace: boolean;
  needlePositions: NeedlePosition[];
  reportJson: string | null;
  failOnMismatch: boolean;
  maxActiveSlopeMbPerToken: number | null;
};

export type NeedlePosition = "early" | "middle" | "late";

type LongContextPrompt = {
  promptText: string;
  promptTokenIds: number[];
  secret: string;
  needlePosition: NeedlePosition;
  needleTokenOffset: number | null;
  needleTokenStart: number | null;
  needleTokenEnd: number | null;
  needleTokenCenter: number | null;
  needleTokenFraction: number | null;
  needleTokenCenterFraction: number | null;
};

type CompiledLongContextPrompt = {
  promptText: string;
  promptTokenIds: number[];
};

export type LongContextResult = {
  needlePosition: NeedlePosition;
  rungTokens: number;
  promptTokens: number;
  expectedMarker: string;
  needleTokenOffset: number | null;
  needleTokenStart: number | null;
  needleTokenEnd: number | null;
  needleTokenCenter: number | null;
  needleTokenFraction: number | null;
  needleTokenCenterFraction: number | null;
  prefillSeconds: number;
  prefillTps: number;
  firstTokenSeconds: number;
  decodeTokens: number;
  decodeTps: number;
  prefillPeakMemoryGb: number;
  activeMemoryAfterPrefillGb: number;
  cacheMemoryAfterPrefillGb: number;
  activeMemoryAfterFirstTokenGb: number;
  cacheMemoryAfterFirstTokenGb: number;
  activeMemoryAfterDecodeGb: number;
  cacheMemoryAfterDecodeGb: number;
  activeMemoryDecodeDeltaGb: number;
  activeMemoryDecodeSlopeMbPerToken: number;
  peakMemoryAfterDecodeGb: number;
  exactMatch: boolean;
  containsSecret: boolean;
  responseText: string;
};

export type LongContextReport = {
  createdAt: string;
  model: string;
  resolvedModelSource: string;
  maxContextTokens: number;
  rungTargets: number[];
  generationTokens: number;
  prefillStepSize: number;
  needlePositions: NeedlePosition[];
  results: LongContextResult[];
};

type LongContextCommand = { kind: "help" } | { kind: "run"; options: LongContextOptions };

type RuntimeLock = {
  [Symbol.dispose](): void;
};

type LongContextBenchmarkRuntime = {
  stdout?: (text: string) => void;
  stderr?: (text: string) => void;
  acquireLock?: () => RuntimeLock;
  runBenchmark?: (
    options: LongContextOptions,
    progress: BenchmarkProgress,
  ) => Promise<LongContextReport>;
};

const CONTEXT_LADDER = [32768, 65536, 131072, 262144] as const;
const FILLER_BLOCK =
  "Context block: this is filler text for long-context benchmarking. Keep reading carefully.\n";
const RETRIEVAL_SYSTEM_PROMPT =
  "You are a retrieval checker. Reply with the exact benchmark marker only, with no extra words or punctuation.";

function toon(value: string | number | boolean | null): string {
  if (typeof value === "number") {
    return Number.isFinite(value) ? value.toString() : "null";
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  if (value === null) {
    return "null";
  }
  return JSON.stringify(value);
}

function readInteger(flag: string, value: string | undefined): number {
  const raw = readRequiredValue(flag, value);
  const parsed = /^\d+$/.test(raw) ? Number(raw) : Number.NaN;
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new BenchmarkUsageError(`benchmark-long-context: ${flag} expects a positive integer.`);
  }
  return parsed;
}

function readNonNegativeNumber(flag: string, value: string | undefined): number {
  const raw = readRequiredValue(flag, value);
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new BenchmarkUsageError(`benchmark-long-context: ${flag} expects a non-negative number.`);
  }
  return parsed;
}

function readRequiredValue(flag: string, value: string | undefined): string {
  if (value === undefined || value.trim() === "" || value.startsWith("--")) {
    throw new BenchmarkUsageError(`benchmark-long-context: ${flag} expects a value.`);
  }
  return value;
}

function readPositiveIntegerEntry(flag: string, value: string): number {
  const parsed = /^\d+$/.test(value) ? Number(value) : Number.NaN;
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new BenchmarkUsageError(`benchmark-long-context: ${flag} expects positive integers.`);
  }
  return parsed;
}

function parseRungs(value: string | undefined): number[] | null {
  if (value === undefined || value.trim() === "") {
    return null;
  }

  const rungs = value.split(",").map((entry) => readPositiveIntegerEntry("--rungs", entry.trim()));
  if (rungs.length === 0) {
    throw new BenchmarkUsageError(
      "benchmark-long-context: --rungs must include at least one positive integer.",
    );
  }
  return [...new Set(rungs)].sort((left, right) => left - right);
}

function parseNeedlePosition(value: string): NeedlePosition {
  switch (value) {
    case "early":
    case "middle":
    case "late":
      return value;
    default:
      throw new BenchmarkUsageError(
        'benchmark-long-context: --needle-placements entries must be "early", "middle", "late", or "all".',
      );
  }
}

export function parseNeedlePositions(value: string | undefined): NeedlePosition[] {
  if (value === undefined || value.trim() === "") {
    return ["late"];
  }
  if (value.trim() === "all") {
    return ["early", "middle", "late"];
  }

  const positions: NeedlePosition[] = [];
  for (const rawEntry of value.split(",")) {
    const entry = rawEntry.trim();
    if (entry === "") {
      continue;
    }
    const position = parseNeedlePosition(entry);
    if (!positions.includes(position)) {
      positions.push(position);
    }
  }
  if (positions.length === 0) {
    throw new BenchmarkUsageError(
      "benchmark-long-context: --needle-placements must include at least one entry.",
    );
  }
  return positions;
}

export function defaultContextTargets(maxContextTokens: number): number[] {
  return CONTEXT_LADDER.filter((target) => target <= maxContextTokens);
}

export function inferMaxContextTokens(config: Record<string, unknown>): number {
  const keys = [
    "max_position_embeddings",
    "max_sequence_length",
    "model_max_length",
    "n_positions",
    "max_seq_len",
  ] as const;
  for (const key of keys) {
    const value = config[key];
    if (typeof value === "number" && Number.isInteger(value) && value > 0) {
      return value;
    }
  }
  const textConfig = config.text_config;
  if (typeof textConfig === "object" && textConfig !== null && !Array.isArray(textConfig)) {
    return inferMaxContextTokens(textConfig as Record<string, unknown>);
  }
  throw new Error(
    "benchmark-long-context: checkpoint config did not expose a known max context field.",
  );
}

export function parseLongContextArgs(argv: readonly string[]): LongContextOptions {
  let model: string | undefined;
  let rungs: number[] | null = null;
  let generationTokens = 24;
  let prefillStepSize = 2048;
  let metalTrace = false;
  let needlePositions = parseNeedlePositions(undefined);
  let reportJson: string | null = null;
  let failOnMismatch = false;
  let maxActiveSlopeMbPerToken: number | null = null;

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === undefined) {
      continue;
    }

    switch (arg) {
      case "--model":
        model = readRequiredValue(arg, argv[index + 1]);
        index += 1;
        break;
      case "--rungs":
        rungs = parseRungs(readRequiredValue(arg, argv[index + 1]));
        index += 1;
        break;
      case "--generation-tokens":
        generationTokens = readInteger(arg, argv[index + 1]);
        index += 1;
        break;
      case "--prefill-step-size":
        prefillStepSize = readInteger(arg, argv[index + 1]);
        index += 1;
        break;
      case "--needle-placements":
      case "--needle-positions":
        needlePositions = parseNeedlePositions(readRequiredValue(arg, argv[index + 1]));
        index += 1;
        break;
      case "--report-json":
        reportJson = readRequiredValue(arg, argv[index + 1]);
        index += 1;
        break;
      case "--fail-on-mismatch":
        failOnMismatch = true;
        break;
      case "--max-active-slope-mb-per-token":
        maxActiveSlopeMbPerToken = readNonNegativeNumber(arg, argv[index + 1]);
        index += 1;
        break;
      case "--metal-trace":
        metalTrace = true;
        break;
      default:
        if (!arg.startsWith("--") && model === undefined) {
          model = arg;
          break;
        }
        throw new BenchmarkUsageError(`benchmark-long-context: unknown argument "${arg}".`);
    }
  }

  if (model === undefined || model.trim() === "") {
    throw new BenchmarkUsageError("benchmark-long-context: --model is required.");
  }

  return {
    model,
    rungs,
    generationTokens,
    prefillStepSize,
    metalTrace,
    needlePositions,
    reportJson,
    failOnMismatch,
    maxActiveSlopeMbPerToken,
  };
}

export function parseLongContextCommand(argv: readonly string[]): LongContextCommand {
  if (argv.some((arg) => arg === "--help" || arg === "-h")) {
    return { kind: "help" };
  }
  return { kind: "run", options: parseLongContextArgs(argv) };
}

export function formatLongContextUsage(): string {
  return [
    "description: Benchmark long-context retrieval and decode memory for @mlxts/transformers",
    "usage[3]:",
    "  bun run bench:generation:context -- --model <repo-or-path>",
    "  bun run bench:generation:context -- --model <repo-or-path> --rungs 32768 --needle-placements all",
    "  bun run bench:generation:context -- <repo-or-path> --generation-tokens 24 --report-json .tmp/context.json",
    "options[10]{flag,description}:",
    '  "--model <repo-or-path>","Model id/path; may also be the first positional argument"',
    '  "--rungs <list>","Comma-separated context token targets; default uses the advertised ladder"',
    '  "--needle-placements <list|all>","early, middle, late, or all; default late"',
    '  "--generation-tokens <n>","Decode token count; default 24"',
    '  "--prefill-step-size <n>","Prompt prefill chunk size; default 2048"',
    '  "--report-json <path>","Write incremental JSON evidence after each rung"',
    '  "--fail-on-mismatch","Fail when the generated response is not the exact marker"',
    '  "--max-active-slope-mb-per-token <n>","Fail when decode active memory slope exceeds this limit"',
    '  "--metal-trace","Capture a Metal trace under benchmarks/traces"',
    '  "--help","Show this help"',
    "exit_codes[3]{code,meaning}:",
    '  0,"benchmark completed"',
    '  1,"runtime or benchmark failure"',
    '  2,"usage error"',
  ].join("\n");
}

function nullableMetric(value: number | null, fractionDigits: number): string {
  return value === null ? "null" : value.toFixed(fractionDigits);
}

function responsePreview(text: string): string {
  if (text.length <= 120) {
    return text;
  }
  return `${text.slice(0, 120)}...`;
}

export function formatLongContextSuccess(
  report: LongContextReport,
  reportJson: string | null,
): string {
  const rows = report.results.map((result) =>
    [
      toon(result.needlePosition),
      result.rungTokens.toString(),
      result.promptTokens.toString(),
      toon(result.expectedMarker),
      nullableMetric(result.needleTokenCenterFraction, 3),
      result.prefillTps.toFixed(3),
      result.firstTokenSeconds.toFixed(3),
      result.decodeTps.toFixed(3),
      result.activeMemoryDecodeSlopeMbPerToken.toFixed(2),
      result.peakMemoryAfterDecodeGb.toFixed(3),
      toon(result.exactMatch),
      toon(result.containsSecret),
      toon(responsePreview(result.responseText)),
    ].join(","),
  );
  return [
    "long_context_benchmark:",
    "  status: passed",
    `  model: ${toon(report.model)}`,
    `  resolved_model_source: ${toon(report.resolvedModelSource)}`,
    `  max_context_tokens: ${report.maxContextTokens}`,
    `  generation_tokens: ${report.generationTokens}`,
    `  prefill_step_size: ${report.prefillStepSize}`,
    `  rungs: ${toon(report.rungTargets.join(","))}`,
    `  needle_positions: ${toon(report.needlePositions.join(","))}`,
    ...(reportJson === null ? [] : [`  report_json: ${toon(reportJson)}`]),
    `results[${rows.length}]{needle_position,rung_tokens,prompt_tokens,expected_marker,needle_center_fraction,prefill_tps,first_token_seconds,decode_tps,active_slope_mb_per_token,peak_memory_gb,exact_match,contains_secret,response_preview}:`,
    ...rows.map((row) => `  ${row}`),
  ].join("\n");
}

function shortModelTag(model: string): string {
  let hash = 0;
  for (const character of model) {
    hash = (Math.imul(hash, 33) + character.charCodeAt(0)) >>> 0;
  }
  return hash.toString(36).toUpperCase().slice(0, 6);
}

function sanitizeSecret(model: string, rungTokens: number): string {
  return `MKR-${shortModelTag(model)}-${rungTokens}`;
}

function positionSecret(model: string, rungTokens: number, needlePosition: NeedlePosition): string {
  return `${sanitizeSecret(model, rungTokens)}-${needlePosition.toUpperCase()}`;
}

export function buildNeedlePromptTokenIds(
  tokenizer: Tokenizer,
  tokenBudget: number,
  secret: string,
): number[] {
  const tail = `\nBenchmark marker: ${secret}\nInstruction: Output only the benchmark marker.\nAnswer:`;
  const prefix = "Long-context retrieval benchmark.\n";
  const prefixIds = tokenizer.encode(prefix, { addSpecialTokens: true });
  const fillerIds = tokenizer.encode(FILLER_BLOCK, { addSpecialTokens: false });
  const tailIds = tokenizer.encode(tail, { addSpecialTokens: false });
  if (prefixIds.length + tailIds.length > tokenBudget) {
    throw new Error("benchmark-long-context: token budget is too small for the retrieval tail.");
  }

  const promptTokenIds = [...prefixIds];
  while (promptTokenIds.length + fillerIds.length + tailIds.length <= tokenBudget) {
    promptTokenIds.push(...fillerIds);
  }
  while (promptTokenIds.length + tailIds.length < tokenBudget) {
    const remaining = tokenBudget - tailIds.length - promptTokenIds.length;
    promptTokenIds.push(...fillerIds.slice(0, remaining));
  }
  promptTokenIds.push(...tailIds);
  return promptTokenIds.slice(0, tokenBudget);
}

export function splitNeedleFillerRepetitions(
  totalRepetitions: number,
  needlePosition: NeedlePosition,
): { before: number; after: number } {
  if (!Number.isInteger(totalRepetitions) || totalRepetitions < 0) {
    throw new Error("benchmark-long-context: filler repetitions must be a non-negative integer.");
  }

  const ratio = needlePosition === "early" ? 0.1 : needlePosition === "middle" ? 0.5 : 1;
  const before = Math.round(totalRepetitions * ratio);
  return {
    before,
    after: totalRepetitions - before,
  };
}

function promptBody(
  secret: string,
  fillerBeforeRepetitions: number,
  fillerAfterRepetitions: number,
): string {
  return [
    FILLER_BLOCK.repeat(fillerBeforeRepetitions),
    `Benchmark marker: ${secret}\n`,
    FILLER_BLOCK.repeat(fillerAfterRepetitions),
    "Instruction: Output only the benchmark marker. Do not explain.\n",
    "Answer:",
  ].join("");
}

function compilePrompt(
  profile: InteractionProfile,
  tokenizer: Tokenizer,
  body: string,
): CompiledLongContextPrompt {
  if (profile.kind === "chat") {
    const compiled = profile.compileMessages(
      tokenizer,
      [
        { role: "system", content: RETRIEVAL_SYSTEM_PROMPT },
        { role: "user", content: body },
      ],
      {
        addGenerationPrompt: true,
        enableThinking: false,
      },
    );
    return {
      promptText: compiled.text,
      promptTokenIds: compiled.tokenIds,
      secret: "",
    };
  }

  const compiled = profile.compileTextPrompt(tokenizer, body, { addSpecialTokens: true });
  return {
    promptText: compiled.text,
    promptTokenIds: compiled.tokenIds,
    secret: "",
  };
}

export function findNeedleTokenOffset(
  tokenizer: Tokenizer,
  promptText: string,
  promptTokenIds: readonly number[],
  secret: string,
): number | null {
  return findNeedleTokenSpan(tokenizer, promptText, promptTokenIds, secret)?.start ?? null;
}

export function findNeedleTokenSpan(
  tokenizer: Tokenizer,
  promptText: string,
  promptTokenIds: readonly number[],
  secret: string,
): { start: number; end: number; center: number; centerFraction: number } | null {
  const markerStart = promptText.indexOf(secret);
  if (markerStart === -1) {
    return null;
  }
  const markerEnd = markerStart + secret.length;

  try {
    const encoding = tokenizer.encodeWithOffsets(promptText, {
      addSpecialTokens: true,
      returnOffsets: true,
    });
    if (encoding.offsets === undefined || encoding.ids.length !== promptTokenIds.length) {
      return null;
    }
    const overlapping = encoding.offsets
      .map((offset, index) => ({ offset, index }))
      .filter(({ offset }) => offset.end > markerStart && offset.start < markerEnd);
    if (overlapping.length === 0) {
      return null;
    }
    const start = overlapping[0]?.index;
    const last = overlapping[overlapping.length - 1]?.index;
    if (start === undefined || last === undefined) {
      return null;
    }
    const end = last + 1;
    const center = (start + end) / 2;
    return {
      start,
      end,
      center,
      centerFraction: center / Math.max(promptTokenIds.length, 1),
    };
  } catch {
    return null;
  }
}

function buildPromptForRung(
  modelSource: string,
  profile: InteractionProfile,
  tokenizer: Tokenizer,
  rungTokens: number,
  needlePosition: NeedlePosition,
): LongContextPrompt {
  const secret = positionSecret(modelSource, rungTokens, needlePosition);
  const baseCompiled = compilePrompt(profile, tokenizer, promptBody(secret, 0, 0));
  const fillerTokenCount = Math.max(
    tokenizer.encode(FILLER_BLOCK, { addSpecialTokens: false }).length,
    1,
  );
  const roughRepetitions = Math.max(
    0,
    Math.floor((rungTokens - baseCompiled.promptTokenIds.length) / fillerTokenCount),
  );

  let repetitions = roughRepetitions;
  let split = splitNeedleFillerRepetitions(repetitions, needlePosition);
  let compiled = compilePrompt(profile, tokenizer, promptBody(secret, split.before, split.after));

  while (compiled.promptTokenIds.length < rungTokens) {
    repetitions += 1;
    split = splitNeedleFillerRepetitions(repetitions, needlePosition);
    compiled = compilePrompt(profile, tokenizer, promptBody(secret, split.before, split.after));
  }

  while (repetitions > 0) {
    const tighterSplit = splitNeedleFillerRepetitions(repetitions - 1, needlePosition);
    const tighter = compilePrompt(
      profile,
      tokenizer,
      promptBody(secret, tighterSplit.before, tighterSplit.after),
    );
    if (tighter.promptTokenIds.length < rungTokens) {
      break;
    }
    repetitions -= 1;
    compiled = tighter;
  }

  const needleTokenSpan = findNeedleTokenSpan(
    tokenizer,
    compiled.promptText,
    compiled.promptTokenIds,
    secret,
  );
  return {
    promptText: compiled.promptText,
    promptTokenIds: compiled.promptTokenIds,
    secret,
    needlePosition,
    needleTokenOffset: needleTokenSpan?.start ?? null,
    needleTokenStart: needleTokenSpan?.start ?? null,
    needleTokenEnd: needleTokenSpan?.end ?? null,
    needleTokenCenter: needleTokenSpan?.center ?? null,
    needleTokenFraction:
      needleTokenSpan === null
        ? null
        : needleTokenSpan.start / Math.max(compiled.promptTokenIds.length, 1),
    needleTokenCenterFraction: needleTokenSpan?.centerFraction ?? null,
  };
}

function decodeOutput(tokenizer: Tokenizer, generated: readonly number[]): string {
  return tokenizer.decode(generated, { skipSpecialTokens: true }).trim();
}

export function normalizeExactResponse(text: string): string {
  const firstLine = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .find((line) => line !== "");
  return (firstLine ?? text.trim()).replace(/^[`"'“”]+|[`"'“”.,!?]+$/g, "");
}

function runLongContextRung(
  model: Awaited<ReturnType<typeof loadCausalLM>>,
  tokenizer: Tokenizer,
  rungTokens: number,
  prompt: LongContextPrompt,
  options: LongContextOptions,
  progress: BenchmarkProgress,
): LongContextResult {
  return withBenchmarkRuntimeScope(
    `${sanitizePathSegment(options.model)}-${rungTokens}-${prompt.needlePosition}`,
    options.metalTrace,
    () => {
      resetPeakMemory();
      clearMemoryCache();

      using cache = model.createCache();
      const prefillStarted = performance.now();
      const remainingPrompt = prefillBenchmarkCache(
        model,
        prompt.promptTokenIds,
        cache,
        options.prefillStepSize,
      );
      const prefillSeconds = (performance.now() - prefillStarted) / 1000;
      const afterPrefill = getMemoryStats();

      const generated: number[] = [];
      let currentToken = predictGreedyToken(model, remainingPrompt, cache);
      let nextToken: ReturnType<typeof predictGreedyToken> | null = null;
      const firstTokenStarted = performance.now();

      try {
        mxAsyncEval(currentToken);
        const firstTokenId = currentToken.item();
        generated.push(firstTokenId);
        const firstTokenSeconds = (performance.now() - firstTokenStarted) / 1000;
        const afterFirstToken = getMemoryStats();

        const decodeStarted = performance.now();
        for (let index = 1; index < options.generationTokens; index += 1) {
          nextToken = predictGreedyToken(model, currentToken, cache);
          mxAsyncEval(nextToken);
          currentToken.free();
          currentToken = nextToken;
          nextToken = null;
          generated.push(currentToken.item());
        }
        const decodeSeconds = Math.max((performance.now() - decodeStarted) / 1000, 1e-9);
        const afterDecode = getMemoryStats();
        const activeDecodeDeltaBytes = afterDecode.activeBytes - afterFirstToken.activeBytes;
        const responseText = decodeOutput(tokenizer, generated);
        const normalized = normalizeExactResponse(responseText);

        return {
          needlePosition: prompt.needlePosition,
          rungTokens,
          promptTokens: prompt.promptTokenIds.length,
          expectedMarker: prompt.secret,
          needleTokenOffset: prompt.needleTokenOffset,
          needleTokenStart: prompt.needleTokenStart,
          needleTokenEnd: prompt.needleTokenEnd,
          needleTokenCenter: prompt.needleTokenCenter,
          needleTokenFraction: prompt.needleTokenFraction,
          needleTokenCenterFraction: prompt.needleTokenCenterFraction,
          prefillSeconds,
          prefillTps: prompt.promptTokenIds.length / Math.max(prefillSeconds, 1e-9),
          firstTokenSeconds,
          decodeTokens: options.generationTokens,
          decodeTps: options.generationTokens / decodeSeconds,
          prefillPeakMemoryGb: afterPrefill.peakBytes / 1e9,
          activeMemoryAfterPrefillGb: afterPrefill.activeBytes / 1e9,
          cacheMemoryAfterPrefillGb: afterPrefill.cacheBytes / 1e9,
          activeMemoryAfterFirstTokenGb: afterFirstToken.activeBytes / 1e9,
          cacheMemoryAfterFirstTokenGb: afterFirstToken.cacheBytes / 1e9,
          activeMemoryAfterDecodeGb: afterDecode.activeBytes / 1e9,
          cacheMemoryAfterDecodeGb: afterDecode.cacheBytes / 1e9,
          activeMemoryDecodeDeltaGb: activeDecodeDeltaBytes / 1e9,
          activeMemoryDecodeSlopeMbPerToken:
            activeDecodeDeltaBytes / 1e6 / Math.max(options.generationTokens, 1),
          peakMemoryAfterDecodeGb: afterDecode.peakBytes / 1e9,
          exactMatch: normalized === prompt.secret,
          containsSecret: responseText.includes(prompt.secret),
          responseText,
        };
      } finally {
        currentToken.free();
        nextToken?.free();
      }
    },
    progress,
  );
}

function printResult(result: LongContextResult, progress: BenchmarkProgress): void {
  progress(
    [
      `needle_position=${result.needlePosition}`,
      `rung=${result.rungTokens}`,
      `prompt_tokens=${result.promptTokens}`,
      `expected_marker=${result.expectedMarker}`,
      `needle_token_offset=${result.needleTokenOffset ?? "unknown"}`,
      `needle_token_span=${result.needleTokenStart ?? "unknown"}:${result.needleTokenEnd ?? "unknown"}`,
      `needle_token_fraction=${result.needleTokenFraction?.toFixed(3) ?? "unknown"}`,
      `needle_center_fraction=${result.needleTokenCenterFraction?.toFixed(3) ?? "unknown"}`,
      `prefill_seconds=${result.prefillSeconds.toFixed(3)}`,
      `prefill_tps=${result.prefillTps.toFixed(3)}`,
      `first_token_seconds=${result.firstTokenSeconds.toFixed(3)}`,
      `decode_tps=${result.decodeTps.toFixed(3)}`,
      `prefill_peak_memory=${result.prefillPeakMemoryGb.toFixed(3)}`,
      `active_after_prefill=${result.activeMemoryAfterPrefillGb.toFixed(3)}`,
      `cache_after_prefill=${result.cacheMemoryAfterPrefillGb.toFixed(3)}`,
      `active_after_first_token=${result.activeMemoryAfterFirstTokenGb.toFixed(3)}`,
      `cache_after_first_token=${result.cacheMemoryAfterFirstTokenGb.toFixed(3)}`,
      `active_after_decode=${result.activeMemoryAfterDecodeGb.toFixed(3)}`,
      `cache_after_decode=${result.cacheMemoryAfterDecodeGb.toFixed(3)}`,
      `active_decode_delta=${result.activeMemoryDecodeDeltaGb.toFixed(3)}`,
      `active_decode_slope_mb_per_token=${result.activeMemoryDecodeSlopeMbPerToken.toFixed(2)}`,
      `peak_after_decode=${result.peakMemoryAfterDecodeGb.toFixed(3)}`,
      `exact_match=${result.exactMatch}`,
      `contains_secret=${result.containsSecret}`,
    ].join(" "),
  );
  progress(`response=${JSON.stringify(result.responseText)}`);
}

export async function writeLongContextReport(
  path: string,
  report: LongContextReport,
): Promise<void> {
  mkdirSync(dirname(path), { recursive: true });
  await Bun.write(path, `${JSON.stringify(report, null, 2)}\n`);
}

function assertPromptWithinContext(prompt: LongContextPrompt, maxContextTokens: number): void {
  if (prompt.promptTokenIds.length <= maxContextTokens) {
    return;
  }
  throw new Error(
    `benchmark-long-context: ${prompt.needlePosition} needle prompt has ${prompt.promptTokenIds.length} tokens, exceeding checkpoint max context ${maxContextTokens}.`,
  );
}

function assertRungTargetsWithinContext(rungs: readonly number[], maxContextTokens: number): void {
  const overContextRung = rungs.find((rung) => rung > maxContextTokens);
  if (overContextRung === undefined) {
    return;
  }
  throw new Error(
    `benchmark-long-context: rung target ${overContextRung} exceeds checkpoint max context ${maxContextTokens}.`,
  );
}

export function assertLongContextResult(
  result: LongContextResult,
  options: LongContextOptions,
): void {
  const failures: string[] = [];
  if (options.failOnMismatch && !result.exactMatch) {
    failures.push(
      `expected exact marker ${result.expectedMarker}, got ${JSON.stringify(result.responseText)}`,
    );
  }
  if (
    options.maxActiveSlopeMbPerToken !== null &&
    result.activeMemoryDecodeSlopeMbPerToken > options.maxActiveSlopeMbPerToken
  ) {
    failures.push(
      `active_decode_slope ${result.activeMemoryDecodeSlopeMbPerToken.toFixed(
        2,
      )} MB/token > ${options.maxActiveSlopeMbPerToken.toFixed(2)} MB/token`,
    );
  }

  if (failures.length > 0) {
    throw new Error(
      `benchmark-long-context: ${result.rungTokens}/${result.needlePosition} failed: ${failures.join(
        "; ",
      )}.`,
    );
  }
}

async function runLongContextBenchmark(
  options: LongContextOptions,
  progress: BenchmarkProgress,
): Promise<LongContextReport> {
  const resolvedModelSource = await resolveCachedSnapshotPath(options.model);
  progress(`Benchmarking long-context ladder for ${resolvedModelSource}`);
  const configRecord = (await Bun.file(`${resolvedModelSource}/config.json`).json()) as Record<
    string,
    unknown
  >;
  const maxContextTokens = inferMaxContextTokens(configRecord);
  const rungs = options.rungs ?? defaultContextTargets(maxContextTokens);
  if (rungs.length === 0) {
    throw new Error(
      `benchmark-long-context: ${options.model} advertises max context ${maxContextTokens}, which is below the default ladder.`,
    );
  }
  assertRungTargetsWithinContext(rungs, maxContextTokens);

  progress(
    `Max context=${maxContextTokens} rung_targets=${rungs.join(",")} needle_positions=${options.needlePositions.join(",")}`,
  );

  const [model, tokenizer, profile] = await Promise.all([
    loadCausalLM(resolvedModelSource, { localFilesOnly: true }),
    loadPretrainedTokenizer(resolvedModelSource, { localFilesOnly: true }),
    loadInteractionProfile(resolvedModelSource, { localFilesOnly: true }),
  ]);

  using loadedModel = model;
  const report: LongContextReport = {
    createdAt: new Date().toISOString(),
    model: options.model,
    resolvedModelSource,
    maxContextTokens,
    rungTargets: rungs,
    generationTokens: options.generationTokens,
    prefillStepSize: options.prefillStepSize,
    needlePositions: options.needlePositions,
    results: [],
  };
  for (const rungTokens of rungs) {
    for (const needlePosition of options.needlePositions) {
      const prompt = buildPromptForRung(
        options.model,
        profile,
        tokenizer,
        rungTokens,
        needlePosition,
      );
      assertPromptWithinContext(prompt, maxContextTokens);
      const result = runLongContextRung(
        loadedModel,
        tokenizer,
        rungTokens,
        prompt,
        options,
        progress,
      );
      report.results.push(result);
      printResult(result, progress);
      progress("");
      if (options.reportJson !== null) {
        await writeLongContextReport(options.reportJson, report);
      }
      assertLongContextResult(result, options);
    }
  }

  if (options.reportJson !== null) {
    progress(`report_json=${options.reportJson}`);
  }
  return report;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function parseLongContextCliCommand(
  argv: readonly string[],
  stdout: (text: string) => void,
): LongContextCommand | number {
  try {
    return parseLongContextCommand(argv);
  } catch (error) {
    stdout(
      formatBenchmarkError(
        errorMessage(error),
        "bun run bench:generation:context -- --model <repo-or-path>",
      ),
    );
    return error instanceof BenchmarkUsageError ? 2 : 1;
  }
}

async function runLongContextBenchmarkWithLock(
  options: LongContextOptions,
  runtime: LongContextBenchmarkRuntime,
  stdout: (text: string) => void,
  stderr: (text: string) => void,
): Promise<number> {
  const acquireLock =
    runtime.acquireLock ?? (() => acquireRuntimeCommandLock("bench:generation:context"));
  const runBenchmark = runtime.runBenchmark ?? runLongContextBenchmark;
  let lock: RuntimeLock | undefined;
  try {
    lock = acquireLock();
    const report = await runBenchmark(options, stderr);
    stdout(formatLongContextSuccess(report, options.reportJson));
    return 0;
  } catch (error) {
    stdout(
      formatBenchmarkError(
        errorMessage(error),
        "rerun with --rungs <smaller-context> or disable strict failure flags",
      ),
    );
    return 1;
  } finally {
    lock?.[Symbol.dispose]();
  }
}

export async function runLongContextBenchmarkCommand(
  argv: readonly string[],
  runtime: LongContextBenchmarkRuntime = {},
): Promise<number> {
  const stdout = runtime.stdout ?? console.log;
  const stderr = runtime.stderr ?? console.error;
  const command = parseLongContextCliCommand(argv, stdout);
  if (typeof command === "number") {
    return command;
  }
  if (command.kind === "help") {
    stdout(formatLongContextUsage());
    return 0;
  }
  return runLongContextBenchmarkWithLock(command.options, runtime, stdout, stderr);
}

if (import.meta.main) {
  const exitCode = await runLongContextBenchmarkCommand(Bun.argv.slice(2));
  process.exit(exitCode);
}
