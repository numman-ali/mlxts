/**
 * Non-streaming generation helpers for the transformer-backed serving engine.
 * @module
 */

import {
  type BatchGenerationOptions,
  disposePreparedPrompt,
  GenerationAbortError,
  type GenerationResult,
  generatePreparedTokenEvents,
  generateTokenEvents,
  type PreparedPrompt,
  slicePreparedPrompt,
} from "@mlxts/transformers";
import { ServeError } from "../errors";
import type { NormalizedGenerationRequest, NormalizedGenerationResult } from "../types";
import type { TransformersGenerationEngineOptions } from "./index";
import {
  createPromptPrefixCacheSession,
  type PromptPrefixCache,
  type PromptPrefixCacheIdentity,
  type PromptPrefixCacheSession,
  type PromptPrefixCacheUsage,
} from "./prefix-cache";
import {
  applyStopSequences,
  type CompiledPrompt,
  compileMessagePrompt,
  createPrefillProgressReporter,
  createProgressReporter,
  decodeGeneratedTokenIds,
  emitGenerationProgress,
  enforceGenerationMemoryBudget,
  enforcePromptTokenLimit,
  enforceTotalTokenLimit,
  finishReason,
  generationOptions,
  promptTokenCount,
  promptTokenIds,
  splitPromptOpenReasoning,
} from "./shared";
import {
  createStreamingDecodeState,
  handleStreamingDoneEvent,
  handleStreamingTokenEvent,
  streamDecodeInterval,
} from "./streaming";

export type PreparedGenerationRequest = {
  request: NormalizedGenerationRequest;
  prompt: CompiledPrompt | null;
  promptTokens: number;
  tokenIds: readonly number[];
  preparedPrompt?: PreparedPrompt;
  promptCacheIdentity?: PromptPrefixCacheIdentity;
  batchOptions: BatchGenerationOptions;
};

type PreparedPromptWindow = {
  prompt: PreparedPrompt;
  dispose(): void;
};

function throwIfRequestAborted(request: NormalizedGenerationRequest, context: string): void {
  if (request.abortSignal?.aborted === true) {
    throw new GenerationAbortError(`${context}: generation was cancelled.`);
  }
}

function rejectMediaInput(): never {
  throw new ServeError(
    "The transformers generation engine cannot prepare media content until a model-family media adapter is configured.",
    { code: "unsupported_input", param: "messages" },
  );
}

function emitPromptPrepareStart(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
): void {
  options.onEvent?.({
    type: "generation_prompt_prepare",
    phase: "start",
    id: request.id,
    protocol: request.protocol,
    model: request.model,
    inputKind: request.input.kind,
  });
}

function emitPromptPrepareComplete(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
  promptTokens: number,
  durationMs: number,
): void {
  options.onEvent?.({
    type: "generation_prompt_prepare",
    phase: "complete",
    id: request.id,
    protocol: request.protocol,
    model: request.model,
    inputKind: request.input.kind,
    promptTokens,
    durationMs,
  });
}

export function batchGenerationOptions(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
): BatchGenerationOptions {
  const eosTokenIds =
    request.sampling.ignoreEos !== true &&
    request.input.kind === "text" &&
    options.tokenizer.eosTokenIds.length > 0
      ? [...options.tokenizer.eosTokenIds]
      : undefined;
  return {
    ...generationOptions(options, request),
    ...(eosTokenIds === undefined ? {} : { eosTokenIds }),
    padTokenId: options.tokenizer.padTokenId ?? options.tokenizer.eosTokenIds[0] ?? 0,
  };
}

export function batchOptionsKey(options: BatchGenerationOptions): string {
  return JSON.stringify({
    temperature: options.temperature,
    topP: options.topP,
    topK: options.topK,
    minP: options.minP,
    repetitionPenalty: options.repetitionPenalty,
    seed: options.seed,
    padTokenId: options.padTokenId,
    eosTokenIds: options.eosTokenIds ?? [],
    prefillStepSize: options.prefillStepSize,
  });
}

export function prepareGenerationRequest(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
): PreparedGenerationRequest {
  throwIfRequestAborted(request, "prepareGenerationRequest");
  if (request.input.kind === "content") {
    rejectMediaInput();
  }
  const startedAt = performance.now();
  emitPromptPrepareStart(request, options);
  const prompt = compileMessagePrompt(request, options);
  const tokenIds =
    request.input.kind === "text"
      ? options.tokenizer.encode(request.input.text, { addSpecialTokens: true })
      : promptTokenIds(request, prompt);
  const promptTokens = promptTokenCount(request, options, prompt);
  throwIfRequestAborted(request, "prepareGenerationRequest");
  enforcePromptTokenLimit(options, request, promptTokens);
  enforceTotalTokenLimit(options, request, promptTokens);
  enforceGenerationMemoryBudget(options, request, promptTokens);
  emitPromptPrepareComplete(request, options, promptTokens, performance.now() - startedAt);
  return {
    request,
    prompt,
    promptTokens,
    tokenIds,
    batchOptions: batchGenerationOptions(request, options),
  };
}

export function generatedResultToServeResult(
  prepared: PreparedGenerationRequest,
  options: TransformersGenerationEngineOptions,
  result: GenerationResult,
  cacheUsage: PromptPrefixCacheUsage = { readTokens: 0, writeTokens: 0 },
): NormalizedGenerationResult {
  const text = decodeGeneratedTokenIds(options, prepared.request, result.tokenIds);
  const reasoning = splitPromptOpenReasoning(prepared.prompt, text);
  const stopped = applyStopSequences(reasoning.text, prepared.request.sampling.stop);

  return {
    text: stopped.text,
    ...(reasoning.reasoningContent === undefined
      ? {}
      : { reasoningContent: reasoning.reasoningContent }),
    finishReason: finishReason(result.finishReason, stopped.stopped),
    tokenIds: result.tokenIds,
    usage: {
      promptTokens: prepared.promptTokens,
      completionTokens: result.tokenIds.length,
      totalTokens: prepared.promptTokens + result.tokenIds.length,
      ...(cacheUsage.readTokens === 0 && cacheUsage.writeTokens === 0
        ? {}
        : {
            cacheReadTokens: cacheUsage.readTokens,
            cacheWriteTokens: cacheUsage.writeTokens,
          }),
    },
  };
}

function emitPromptCacheEvent(
  prepared: PreparedGenerationRequest,
  options: TransformersGenerationEngineOptions,
  result: "hit" | "miss" | "write",
  cacheUsage: PromptPrefixCacheUsage,
): void {
  options.onEvent?.({
    type: "generation_prompt_cache",
    id: prepared.request.id,
    protocol: prepared.request.protocol,
    model: prepared.request.model,
    result,
    promptTokens: prepared.promptTokens,
    cacheReadTokens: cacheUsage.readTokens,
    cacheWriteTokens: cacheUsage.writeTokens,
  });
}

function createPromptCacheSession(
  prepared: PreparedGenerationRequest,
  options: TransformersGenerationEngineOptions,
  promptCache: PromptPrefixCache | undefined,
): PromptPrefixCacheSession {
  return createPromptPrefixCacheSession({
    tokenIds: prepared.tokenIds,
    enabled:
      prepared.request.input.kind === "messages" || prepared.promptCacheIdentity !== undefined,
    ...(prepared.promptCacheIdentity === undefined
      ? {}
      : { identity: prepared.promptCacheIdentity }),
    ...(promptCache === undefined ? {} : { promptCache }),
    onEvent(result, cacheUsage) {
      emitPromptCacheEvent(prepared, options, result, cacheUsage);
    },
  });
}

function generationOptionsForPrepared(
  prepared: PreparedGenerationRequest,
  options: TransformersGenerationEngineOptions,
  onPrefillProgress: ReturnType<typeof createPrefillProgressReporter>,
  cacheSession: PromptPrefixCacheSession,
) {
  return {
    ...generationOptions(options, prepared.request, onPrefillProgress),
    ...cacheSession.generationOptions(),
  };
}

function preparedPromptWindow(
  prompt: PreparedPrompt | undefined,
  cacheSession: PromptPrefixCacheSession,
): PreparedPromptWindow | null {
  if (prompt === undefined) {
    return null;
  }

  const cachedPrefixLength = cacheSession.cachedPrefixLength();
  if (cachedPrefixLength === 0) {
    return {
      prompt,
      dispose() {},
    };
  }

  const tokenIds = cacheSession.tokenIdsForGeneration();
  const window = slicePreparedPrompt(
    prompt,
    cachedPrefixLength,
    tokenIds.length,
    "transformers prepared prompt cache suffix",
  );
  return {
    prompt: window,
    dispose() {
      disposePreparedPrompt(window);
    },
  };
}

function tokenEventsForPreparedRequest(
  prepared: PreparedGenerationRequest,
  options: TransformersGenerationEngineOptions,
  onPrefillProgress: ReturnType<typeof createPrefillProgressReporter>,
  cacheSession: PromptPrefixCacheSession,
  promptWindow: PreparedPrompt | null,
) {
  const generation = generationOptionsForPrepared(
    prepared,
    options,
    onPrefillProgress,
    cacheSession,
  );
  if (promptWindow !== null) {
    return generatePreparedTokenEvents(options.model, promptWindow, generation);
  }

  const tokenIds = cacheSession.tokenIdsForGeneration();
  if (prepared.request.input.kind === "text") {
    return generateTokenEvents(options.model, tokenIds, {
      ...generation,
      ...(prepared.request.sampling.ignoreEos === true || options.tokenizer.eosTokenIds.length === 0
        ? {}
        : { eosTokenIds: [...options.tokenizer.eosTokenIds] }),
    });
  }

  return generatePreparedTokenEvents(options.model, { tokenIds }, generation);
}

function disposeGenerationPreparedPrompt(prepared: PreparedGenerationRequest): void {
  if (prepared.preparedPrompt !== undefined) {
    disposePreparedPrompt(prepared.preparedPrompt);
  }
}

export async function generateSinglePreparedRequest(
  prepared: PreparedGenerationRequest,
  options: TransformersGenerationEngineOptions,
  promptCache?: PromptPrefixCache,
): Promise<NormalizedGenerationResult> {
  const onToken = createProgressReporter(options, prepared.request, prepared.promptTokens);
  const onPrefillProgress = createPrefillProgressReporter(
    options,
    prepared.request,
    prepared.promptTokens,
  );
  emitGenerationProgress(options, prepared.request, prepared.promptTokens, 0);
  const cacheSession = createPromptCacheSession(prepared, options, promptCache);
  const tokenIds: number[] = [];
  let finishReason: GenerationResult["finishReason"] = "length";
  let cacheUsage: PromptPrefixCacheUsage = { readTokens: 0, writeTokens: 0 };
  const promptWindow = preparedPromptWindow(prepared.preparedPrompt, cacheSession);

  try {
    for await (const event of tokenEventsForPreparedRequest(
      prepared,
      options,
      onPrefillProgress,
      cacheSession,
      promptWindow?.prompt ?? null,
    )) {
      if (event.type === "token") {
        tokenIds.push(event.tokenId);
        onToken(event.tokenId, tokenIds);
        continue;
      }
      tokenIds.length = 0;
      tokenIds.push(...event.tokenIds);
      finishReason = event.finishReason;
    }
  } finally {
    promptWindow?.dispose();
    cacheUsage = cacheSession.usage();
    cacheSession[Symbol.dispose]();
    disposeGenerationPreparedPrompt(prepared);
  }

  const result: GenerationResult = { tokenIds, finishReason };
  return generatedResultToServeResult(prepared, options, result, cacheUsage);
}

export async function* streamSinglePreparedRequest(
  prepared: PreparedGenerationRequest,
  options: TransformersGenerationEngineOptions,
  promptCache?: PromptPrefixCache,
) {
  emitGenerationProgress(options, prepared.request, prepared.promptTokens, 0);
  const onPrefillProgress = createPrefillProgressReporter(
    options,
    prepared.request,
    prepared.promptTokens,
  );
  const decodeInterval = streamDecodeInterval(options, prepared.request.sampling.stop);
  const state = createStreamingDecodeState(prepared.prompt);
  const cacheSession = createPromptCacheSession(prepared, options, promptCache);
  const promptWindow = preparedPromptWindow(prepared.preparedPrompt, cacheSession);

  try {
    const tokenEvents = tokenEventsForPreparedRequest(
      prepared,
      options,
      onPrefillProgress,
      cacheSession,
      promptWindow?.prompt ?? null,
    );
    for await (const event of tokenEvents) {
      if (event.type === "token") {
        const text = handleStreamingTokenEvent(
          prepared.request,
          options,
          prepared.promptTokens,
          decodeInterval,
          state,
          event,
        );
        if (text !== undefined) {
          yield { type: "text", text } as const;
        }
        continue;
      }

      const finished = handleStreamingDoneEvent(
        prepared.request,
        options,
        prepared.promptTokens,
        state,
        event,
        cacheSession.usage(),
      );
      if (finished.text !== undefined) {
        yield { type: "text", text: finished.text } as const;
      }
      yield finished.done;
      return;
    }
  } finally {
    promptWindow?.dispose();
    cacheSession[Symbol.dispose]();
    disposeGenerationPreparedPrompt(prepared);
  }
}

/** Generate one non-streaming request through the transformer-backed engine. */
export function generateTransformersRequest(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
): Promise<NormalizedGenerationResult> {
  return generateSinglePreparedRequest(prepareGenerationRequest(request, options), options);
}
