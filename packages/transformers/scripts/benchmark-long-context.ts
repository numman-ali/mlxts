#!/usr/bin/env bun

import { clearMemoryCache, getMemoryStats, mxAsyncEval, resetPeakMemory } from "@mlxts/core";
import type { Tokenizer } from "@mlxts/tokenizers";
import { acquireRuntimeCommandLock } from "../../../scripts/runtime-command-lock";
import {
  type InteractionProfile,
  loadCausalLM,
  loadInteractionProfile,
  loadPretrainedTokenizer,
} from "../src";
import {
  resolveCachedSnapshotPath,
  sanitizePathSegment,
  withBenchmarkRuntimeScope,
} from "./benchmark-common";
import { predictGreedyToken, prefillBenchmarkCache } from "./benchmark-model";

type LongContextOptions = {
  model: string;
  rungs: number[] | null;
  generationTokens: number;
  prefillStepSize: number;
  metalTrace: boolean;
};

type LongContextPrompt = {
  promptText: string;
  promptTokenIds: number[];
  secret: string;
};

type LongContextResult = {
  rungTokens: number;
  promptTokens: number;
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

const CONTEXT_LADDER = [32768, 65536, 131072, 262144] as const;
const FILLER_BLOCK =
  "Context block: this is filler text for long-context benchmarking. Keep reading carefully.\n";
const RETRIEVAL_SYSTEM_PROMPT =
  "You are a retrieval checker. Reply with the exact benchmark marker only, with no extra words or punctuation.";

function usage(): never {
  console.error(
    "Usage: bun run packages/transformers/scripts/benchmark-long-context.ts --model <repo-or-path> [--rungs 32768,65536,131072] [--generation-tokens <n>] [--prefill-step-size <n>] [--metal-trace]",
  );
  process.exit(1);
}

function readInteger(flag: string, value: string | undefined): number {
  const parsed = Number.parseInt(value ?? "", 10);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`benchmark-long-context: ${flag} expects a positive integer.`);
  }
  return parsed;
}

function parseRungs(value: string | undefined): number[] | null {
  if (value === undefined || value.trim() === "") {
    return null;
  }

  const rungs = value
    .split(",")
    .map((entry) => Number.parseInt(entry.trim(), 10))
    .filter((entry) => Number.isInteger(entry) && entry > 0);
  if (rungs.length === 0) {
    throw new Error("benchmark-long-context: --rungs must include at least one positive integer.");
  }
  return [...new Set(rungs)].sort((left, right) => left - right);
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

function parseArgs(argv: readonly string[]): LongContextOptions {
  let model: string | undefined;
  let rungs: number[] | null = null;
  let generationTokens = 24;
  let prefillStepSize = 2048;
  let metalTrace = false;

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === undefined) {
      continue;
    }

    switch (arg) {
      case "--model":
        model = argv[index + 1];
        index += 1;
        break;
      case "--rungs":
        rungs = parseRungs(argv[index + 1]);
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
      case "--metal-trace":
        metalTrace = true;
        break;
      default:
        if (!arg.startsWith("--") && model === undefined) {
          model = arg;
          break;
        }
        throw new Error(`benchmark-long-context: unknown argument "${arg}".`);
    }
  }

  if (model === undefined || model.trim() === "") {
    usage();
  }

  return {
    model,
    rungs,
    generationTokens,
    prefillStepSize,
    metalTrace,
  };
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

function promptBody(secret: string, fillerRepetitions: number): string {
  return [
    FILLER_BLOCK.repeat(fillerRepetitions),
    `Benchmark marker: ${secret}\n`,
    "Instruction: Output only the benchmark marker. Do not explain.\n",
    "Answer:",
  ].join("");
}

function compilePrompt(
  profile: InteractionProfile,
  tokenizer: Tokenizer,
  body: string,
): LongContextPrompt {
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

function buildPromptForRung(
  modelSource: string,
  profile: InteractionProfile,
  tokenizer: Tokenizer,
  rungTokens: number,
): LongContextPrompt {
  const secret = sanitizeSecret(modelSource, rungTokens);
  const baseCompiled = compilePrompt(profile, tokenizer, promptBody(secret, 0));
  const fillerTokenCount = Math.max(
    tokenizer.encode(FILLER_BLOCK, { addSpecialTokens: false }).length,
    1,
  );
  const roughRepetitions = Math.max(
    0,
    Math.floor((rungTokens - baseCompiled.promptTokenIds.length) / fillerTokenCount),
  );

  let compiled = compilePrompt(profile, tokenizer, promptBody(secret, roughRepetitions));
  let repetitions = roughRepetitions;

  while (compiled.promptTokenIds.length < rungTokens) {
    repetitions += 1;
    compiled = compilePrompt(profile, tokenizer, promptBody(secret, repetitions));
  }

  while (repetitions > 0) {
    const tighter = compilePrompt(profile, tokenizer, promptBody(secret, repetitions - 1));
    if (tighter.promptTokenIds.length < rungTokens) {
      break;
    }
    repetitions -= 1;
    compiled = tighter;
  }

  return {
    promptText: compiled.promptText,
    promptTokenIds: compiled.promptTokenIds,
    secret,
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
  promptTokenIds: readonly number[],
  secret: string,
  options: LongContextOptions,
): LongContextResult {
  return withBenchmarkRuntimeScope(
    `${sanitizePathSegment(options.model)}-${rungTokens}`,
    options.metalTrace,
    () => {
      resetPeakMemory();
      clearMemoryCache();

      using cache = model.createCache();
      const prefillStarted = performance.now();
      const remainingPrompt = prefillBenchmarkCache(
        model,
        promptTokenIds,
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
          rungTokens,
          promptTokens: promptTokenIds.length,
          prefillSeconds,
          prefillTps: promptTokenIds.length / Math.max(prefillSeconds, 1e-9),
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
          exactMatch: normalized === secret,
          containsSecret: responseText.includes(secret),
          responseText,
        };
      } finally {
        currentToken.free();
        nextToken?.free();
      }
    },
  );
}

function printResult(result: LongContextResult): void {
  console.log(
    [
      `rung=${result.rungTokens}`,
      `prompt_tokens=${result.promptTokens}`,
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
  console.log(`response=${JSON.stringify(result.responseText)}`);
}

async function main(): Promise<void> {
  using _runtimeLock = acquireRuntimeCommandLock("bench:generation:context");
  const options = parseArgs(Bun.argv.slice(2));
  const resolvedModelSource = await resolveCachedSnapshotPath(options.model);
  console.log(`Benchmarking long-context ladder for ${resolvedModelSource}`);
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

  console.log(`Max context=${maxContextTokens} rung_targets=${rungs.join(",")}`);

  const [model, tokenizer, profile] = await Promise.all([
    loadCausalLM(resolvedModelSource, { localFilesOnly: true }),
    loadPretrainedTokenizer(resolvedModelSource, { localFilesOnly: true }),
    loadInteractionProfile(resolvedModelSource, { localFilesOnly: true }),
  ]);

  using loadedModel = model;
  for (const rungTokens of rungs) {
    const prompt = buildPromptForRung(options.model, profile, tokenizer, rungTokens);
    const result = runLongContextRung(
      loadedModel,
      tokenizer,
      rungTokens,
      prompt.promptTokenIds,
      prompt.secret,
      options,
    );
    printResult(result);
    console.log("");
  }
}

if (import.meta.main) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exit(1);
  });
}
