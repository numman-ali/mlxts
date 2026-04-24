/**
 * Shared request helpers for the transformer-backed serving engine.
 * @module
 */

import type { Tokenizer } from "@mlxts/tokenizers";
import type { GenerationOptions, PrefillProgressEvent } from "@mlxts/transformers";
import { ServeError } from "./errors";
import { readGenerationMemoryUsage } from "./memory-telemetry";
import {
  effectiveTotalTokenLimit,
  estimateGenerationMemory,
  modelContextWindow,
} from "./model-context";
import type { TransformersGenerationEngineOptions } from "./transformers-engine";
import type {
  GenerationMemoryUsage,
  NormalizedFinishReason,
  NormalizedGenerationRequest,
} from "./types";

export type CompiledPrompt = {
  text: string;
  tokenIds: number[];
};

export const THINK_OPEN = "<think>";
const THINK_CLOSE = "</think>";
const PROGRESS_TOKEN_INTERVAL = 64;
const DEFAULT_PREFILL_STEP_SIZE = 2048;

/** Convert a normalized serving request into transformer generation options. */
export function generationOptions(
  request: NormalizedGenerationRequest,
  onPrefillProgress?: (event: PrefillProgressEvent) => void,
): GenerationOptions {
  return {
    maxTokens: request.sampling.maxTokens,
    ...(request.sampling.temperature === undefined
      ? {}
      : { temperature: request.sampling.temperature }),
    ...(request.sampling.topP === undefined ? {} : { topP: request.sampling.topP }),
    ...(request.sampling.topK === undefined ? {} : { topK: request.sampling.topK }),
    ...(request.sampling.seed === undefined ? {} : { seed: request.sampling.seed }),
    ...(onPrefillProgress === undefined ? {} : { onPrefillProgress }),
  };
}

function totalTokenLimit(options: TransformersGenerationEngineOptions): number | undefined {
  return effectiveTotalTokenLimit({
    maxTotalTokens: options.maxTotalTokens,
    contextWindow: modelContextWindow(options.model),
  });
}

/** Reject requests whose prompt alone exceeds the server prompt admission budget. */
export function enforcePromptTokenLimit(
  options: TransformersGenerationEngineOptions,
  request: NormalizedGenerationRequest,
  promptTokens: number,
): void {
  const limit = options.maxPromptTokens;
  if (limit === undefined || promptTokens <= limit) {
    return;
  }

  const effectiveTotal = totalTokenLimit(options);
  const totalSuffix =
    effectiveTotal === undefined ? "" : ` Effective total token limit is ${effectiveTotal}.`;
  throw new ServeError(
    `Requested prompt_tokens ${promptTokens} exceeds this server's prompt token limit of ${limit}. Requested max_tokens is ${request.sampling.maxTokens}.${totalSuffix}`,
    {
      code: "prompt_tokens_exceeded",
      param: "prompt",
    },
  );
}

/** Reject requests that exceed either server or checkpoint context limits. */
export function enforceTotalTokenLimit(
  options: TransformersGenerationEngineOptions,
  request: NormalizedGenerationRequest,
  promptTokens: number,
): void {
  const limit = totalTokenLimit(options);
  if (limit === undefined) {
    return;
  }

  const requestedTotal = promptTokens + request.sampling.maxTokens;
  if (requestedTotal <= limit) {
    return;
  }

  throw new ServeError(
    `Requested prompt_tokens ${promptTokens} plus max_tokens ${request.sampling.maxTokens} exceeds this server's total token limit of ${limit}.`,
    {
      code: "context_length_exceeded",
      param: "max_tokens",
    },
  );
}

function formatBytes(bytes: number): string {
  if (bytes >= 1e9) {
    return `${(bytes / 1e9).toFixed(1)} GB`;
  }
  if (bytes >= 1e6) {
    return `${(bytes / 1e6).toFixed(1)} MB`;
  }
  if (bytes >= 1e3) {
    return `${(bytes / 1e3).toFixed(1)} KB`;
  }
  return `${bytes} B`;
}

/** Reject requests whose estimated cache/prefill memory exceeds the configured MLX budget. */
export function enforceGenerationMemoryBudgetForTokens(
  options: TransformersGenerationEngineOptions,
  promptTokens: number,
  maxTokens: number,
  memory: GenerationMemoryUsage | undefined = readGenerationMemoryUsage(),
  batchSize = 1,
): void {
  if (options.gpuMemoryUtilization === undefined || memory === undefined) {
    return;
  }

  const estimate = estimateGenerationMemory(options.model, {
    promptTokens,
    totalTokens: promptTokens + maxTokens,
    prefillStepSize: DEFAULT_PREFILL_STEP_SIZE,
    batchSize,
  });
  if (estimate === undefined) {
    return;
  }

  const budgetBytes = Math.floor(memory.limitBytes * options.gpuMemoryUtilization);
  const projectedBytes = memory.activeBytes + estimate.totalBytes;
  if (projectedBytes <= budgetBytes) {
    return;
  }

  throw new ServeError(
    `Estimated request memory ${formatBytes(estimate.totalBytes)} plus active MLX memory ${formatBytes(memory.activeBytes)} exceeds the configured GPU memory budget ${formatBytes(budgetBytes)} (${Math.round(options.gpuMemoryUtilization * 100)}%). prompt_tokens=${promptTokens}, max_tokens=${maxTokens}, batch_size=${batchSize}.`,
    {
      code: "memory_budget_exceeded",
      param: "prompt",
    },
  );
}

/** Reject a single request whose estimated cache/prefill memory exceeds the configured budget. */
export function enforceGenerationMemoryBudget(
  options: TransformersGenerationEngineOptions,
  request: NormalizedGenerationRequest,
  promptTokens: number,
  memory?: GenerationMemoryUsage | undefined,
): void {
  enforceGenerationMemoryBudgetForTokens(options, promptTokens, request.sampling.maxTokens, memory);
}

/** Emit a generation progress event with current MLX memory telemetry when available. */
export function emitGenerationProgress(
  options: TransformersGenerationEngineOptions,
  request: NormalizedGenerationRequest,
  promptTokens: number,
  completionTokens: number,
): void {
  if (options.onEvent === undefined) {
    return;
  }
  const memory = readGenerationMemoryUsage();
  options.onEvent({
    type: "generation_progress",
    id: request.id,
    protocol: request.protocol,
    model: request.model,
    promptTokens,
    completionTokens,
    maxTokens: request.sampling.maxTokens,
    ...(memory === undefined ? {} : { memory }),
  });
}

/** Emit prompt-prefill progress for long-context requests before first-token decode. */
export function emitGenerationPrefillProgress(
  options: TransformersGenerationEngineOptions,
  request: NormalizedGenerationRequest,
  promptTokens: number,
  progress: PrefillProgressEvent,
): void {
  if (options.onEvent === undefined) {
    return;
  }
  const memory = readGenerationMemoryUsage();
  options.onEvent({
    type: "generation_prefill_progress",
    id: request.id,
    protocol: request.protocol,
    model: request.model,
    promptTokens,
    processedPrefillTokens: progress.processedTokens,
    totalPrefillTokens: progress.totalTokens,
    chunkTokens: progress.chunkTokens,
    maxTokens: request.sampling.maxTokens,
    ...(memory === undefined ? {} : { memory }),
  });
}

/** Create the chunk-level progress callback used during cached prompt prefill. */
export function createPrefillProgressReporter(
  options: TransformersGenerationEngineOptions,
  request: NormalizedGenerationRequest,
  promptTokens: number,
): (event: PrefillProgressEvent) => void {
  return (event) => {
    emitGenerationPrefillProgress(options, request, promptTokens, event);
  };
}

/** Create the token-level progress callback used by synchronous generation helpers. */
export function createProgressReporter(
  options: TransformersGenerationEngineOptions,
  request: NormalizedGenerationRequest,
  promptTokens: number,
): (tokenId: number, generatedTokenIds: readonly number[]) => void {
  return (_tokenId, generatedTokenIds) => {
    const completionTokens = generatedTokenIds.length;
    if (
      completionTokens % PROGRESS_TOKEN_INTERVAL === 0 ||
      completionTokens === request.sampling.maxTokens
    ) {
      emitGenerationProgress(options, request, promptTokens, completionTokens);
    }
  };
}

/** Apply OpenAI-style stop strings to decoded generated text. */
export function applyStopSequences(
  text: string,
  stop: readonly string[] | undefined,
): { text: string; stopped: boolean } {
  if (stop === undefined || stop.length === 0) {
    return { text, stopped: false };
  }

  const matches = stop
    .map((sequence) => text.indexOf(sequence))
    .filter((index) => index >= 0)
    .sort((left, right) => left - right);

  const firstMatch = matches[0];
  return firstMatch === undefined
    ? { text, stopped: false }
    : { text: text.slice(0, firstMatch), stopped: true };
}

/** Convert transformer finish reasons into the serving finish reason vocabulary. */
export function finishReason(
  generatedReason: "length" | "eos",
  stopped: boolean,
): NormalizedFinishReason {
  if (stopped) {
    return "stop";
  }
  return generatedReason === "eos" ? "eos" : "length";
}

/** Whether the compiled chat prompt opens a Qwen-style reasoning section. */
export function promptHasOpenThinking(prompt: { text: string } | null): boolean {
  if (prompt === null) {
    return false;
  }
  const lastOpen = prompt.text.lastIndexOf(THINK_OPEN);
  const lastClose = prompt.text.lastIndexOf(THINK_CLOSE);
  return lastOpen >= 0 && lastOpen > lastClose && prompt.text.slice(lastOpen).trim() === THINK_OPEN;
}

function cleanReasoningContent(text: string): string {
  return text.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, "").trim();
}

function reasoningResult(text: string, reasoning: string) {
  const reasoningContent = cleanReasoningContent(reasoning);
  return reasoningContent === "" ? { text } : { text, reasoningContent };
}

/** Split prompt-open Qwen reasoning out of visible assistant text. */
export function splitPromptOpenReasoning(prompt: { text: string } | null, text: string) {
  if (!promptHasOpenThinking(prompt)) {
    return { text };
  }

  const generated = text.startsWith(THINK_OPEN) ? text.slice(THINK_OPEN.length) : text;
  const closeIndex = generated.indexOf(THINK_CLOSE);
  if (closeIndex < 0) {
    return reasoningResult("", generated);
  }

  return reasoningResult(
    generated.slice(closeIndex + THINK_CLOSE.length).trimStart(),
    generated.slice(0, closeIndex),
  );
}

/** Compile message input with the model's interaction profile, if needed. */
export function compileMessagePrompt(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
): CompiledPrompt | null {
  if (request.input.kind !== "messages") {
    return null;
  }
  if (options.interactionProfile === undefined) {
    throw new ServeError(
      "The transformers generation engine requires an interaction profile for message input.",
      { code: "unsupported_input" },
    );
  }
  return options.interactionProfile.compileMessages(options.tokenizer, request.input.messages, {
    addGenerationPrompt: true,
    ...(request.input.tools === undefined ? {} : { tools: request.input.tools }),
    ...(request.input.chatTemplate?.enableThinking === undefined
      ? {}
      : { enableThinking: request.input.chatTemplate.enableThinking }),
    ...(request.input.chatTemplate?.preserveThinking === undefined
      ? {}
      : { preserveThinking: request.input.chatTemplate.preserveThinking }),
  });
}

/** Return token ids for already-prepared token/message requests. */
export function promptTokenIds(
  request: NormalizedGenerationRequest,
  prompt: { tokenIds: readonly number[] } | null,
): readonly number[] {
  if (request.input.kind === "tokens") {
    return request.input.tokenIds;
  }
  if (prompt !== null) {
    return prompt.tokenIds;
  }
  throw new ServeError("The transformers generation engine could not compile message input.", {
    code: "unsupported_input",
  });
}

/** Count the prompt tokens using the same tokenization mode used for generation. */
export function promptTokenCount(
  request: NormalizedGenerationRequest,
  options: { tokenizer: Tokenizer },
  prompt: { tokenIds: readonly number[] } | null,
): number {
  if (request.input.kind === "text") {
    return options.tokenizer.encode(request.input.text, { addSpecialTokens: true }).length;
  }
  if (request.input.kind === "tokens") {
    return request.input.tokenIds.length;
  }
  return prompt?.tokenIds.length ?? 0;
}
