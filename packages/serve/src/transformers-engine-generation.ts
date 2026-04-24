/**
 * Non-streaming generation helpers for the transformer-backed serving engine.
 * @module
 */

import {
  type BatchGenerationOptions,
  type CausalLM,
  type GenerationResult,
  generateBatchTokens,
  generateTextStream,
  generateTokens,
} from "@mlxts/transformers";
import { ServeError } from "./errors";
import type { TransformersGenerationEngineOptions } from "./transformers-engine";
import {
  applyStopSequences,
  type CompiledPrompt,
  compileMessagePrompt,
  createPrefillProgressReporter,
  createProgressReporter,
  emitGenerationProgress,
  enforcePromptTokenLimit,
  enforceTotalTokenLimit,
  finishReason,
  generationOptions,
  promptTokenCount,
  promptTokenIds,
  splitPromptOpenReasoning,
} from "./transformers-engine-shared";
import type { NormalizedGenerationRequest, NormalizedGenerationResult } from "./types";

const STATIC_BATCH_MODEL_TYPES = new Set(["gemma", "llama", "mistral", "mistral3", "phi3"]);

type PreparedGenerationRequest = {
  request: NormalizedGenerationRequest;
  prompt: CompiledPrompt | null;
  promptTokens: number;
  tokenIds: readonly number[];
  batchOptions: BatchGenerationOptions;
};

function configHasSlidingWindow(model: CausalLM): boolean {
  const config = model.config;
  if (!("slidingWindow" in config)) {
    return false;
  }
  const value = config.slidingWindow;
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

function supportsStaticBatchGeneration(model: CausalLM): boolean {
  return STATIC_BATCH_MODEL_TYPES.has(model.config.modelType) && !configHasSlidingWindow(model);
}

function effectiveTemperature(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
): number {
  return (
    request.sampling.temperature ?? options.model.config.generationDefaults?.temperature ?? 1.0
  );
}

function effectiveRepetitionPenalty(options: TransformersGenerationEngineOptions): number {
  return options.model.config.generationDefaults?.repetitionPenalty ?? 1.0;
}

function canUseStaticBatchGeneration(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
): boolean {
  return (
    !request.stream &&
    supportsStaticBatchGeneration(options.model) &&
    effectiveTemperature(request, options) === 0 &&
    effectiveRepetitionPenalty(options) === 1.0
  );
}

function batchGenerationOptions(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
): BatchGenerationOptions {
  const eosTokenIds =
    request.input.kind === "text" && options.tokenizer.eosTokenIds.length > 0
      ? [...options.tokenizer.eosTokenIds]
      : undefined;
  return {
    ...generationOptions(request),
    ...(eosTokenIds === undefined ? {} : { eosTokenIds }),
    padTokenId: options.tokenizer.padTokenId ?? options.tokenizer.eosTokenIds[0] ?? 0,
  };
}

function batchOptionsKey(options: BatchGenerationOptions): string {
  return JSON.stringify({
    temperature: options.temperature,
    topP: options.topP,
    topK: options.topK,
    seed: options.seed,
    padTokenId: options.padTokenId,
    eosTokenIds: options.eosTokenIds ?? [],
  });
}

function prepareGenerationRequest(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
): PreparedGenerationRequest {
  const prompt = compileMessagePrompt(request, options);
  const tokenIds =
    request.input.kind === "text"
      ? options.tokenizer.encode(request.input.text, { addSpecialTokens: true })
      : promptTokenIds(request, prompt);
  const promptTokens = promptTokenCount(request, options, prompt);
  enforcePromptTokenLimit(options, request, promptTokens);
  enforceTotalTokenLimit(options, request, promptTokens);
  return {
    request,
    prompt,
    promptTokens,
    tokenIds,
    batchOptions: batchGenerationOptions(request, options),
  };
}

function generateTokenPrompt(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
  prompt: { tokenIds: readonly number[] } | null,
  onPrefillProgress: ReturnType<typeof createPrefillProgressReporter>,
  onToken?: (tokenId: number, generatedTokenIds: readonly number[]) => void,
) {
  const generated = generateTokens(
    options.model,
    promptTokenIds(request, prompt),
    generationOptions(request, onPrefillProgress),
    onToken,
  );
  return {
    ...generated,
    text: options.tokenizer.decode(generated.tokenIds, { skipSpecialTokens: true }),
  };
}

function generatedResultToServeResult(
  prepared: PreparedGenerationRequest,
  options: TransformersGenerationEngineOptions,
  result: GenerationResult,
): NormalizedGenerationResult {
  const text = options.tokenizer.decode(result.tokenIds, { skipSpecialTokens: true });
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
    },
  };
}

function generateSinglePreparedRequest(
  prepared: PreparedGenerationRequest,
  options: TransformersGenerationEngineOptions,
): NormalizedGenerationResult {
  const onToken = createProgressReporter(options, prepared.request, prepared.promptTokens);
  const onPrefillProgress = createPrefillProgressReporter(
    options,
    prepared.request,
    prepared.promptTokens,
  );
  emitGenerationProgress(options, prepared.request, prepared.promptTokens, 0);
  const result =
    prepared.request.input.kind === "text"
      ? generateTextStream(
          options.model,
          options.tokenizer,
          prepared.request.input.text,
          generationOptions(prepared.request, onPrefillProgress),
          () => undefined,
          onToken,
        )
      : generateTokenPrompt(prepared.request, options, prepared.prompt, onPrefillProgress, onToken);
  return generatedResultToServeResult(prepared, options, result);
}

function invalidBatchResult(): ServeError {
  return new ServeError("The transformers generation engine produced an incomplete batch.", {
    code: "invalid_engine_result",
    status: 500,
  });
}

function preparedAt(
  preparedRequests: readonly PreparedGenerationRequest[],
  index: number,
): PreparedGenerationRequest {
  const prepared = preparedRequests[index];
  if (prepared === undefined) {
    throw invalidBatchResult();
  }
  return prepared;
}

function addBatchGroup(groups: Map<string, number[]>, key: string, index: number): void {
  const group = groups.get(key);
  if (group === undefined) {
    groups.set(key, [index]);
    return;
  }
  group.push(index);
}

function routePreparedRequests(
  preparedRequests: readonly PreparedGenerationRequest[],
  options: TransformersGenerationEngineOptions,
  results: (NormalizedGenerationResult | undefined)[],
): Map<string, number[]> {
  const groups = new Map<string, number[]>();
  for (let index = 0; index < preparedRequests.length; index += 1) {
    const prepared = preparedAt(preparedRequests, index);
    if (!canUseStaticBatchGeneration(prepared.request, options)) {
      results[index] = generateSinglePreparedRequest(prepared, options);
      continue;
    }
    addBatchGroup(groups, batchOptionsKey(prepared.batchOptions), index);
  }
  return groups;
}

function batchGroupPrompts(
  preparedRequests: readonly PreparedGenerationRequest[],
  group: readonly number[],
): readonly (readonly number[])[] {
  return group.map((index) => preparedAt(preparedRequests, index).tokenIds);
}

function batchGroupMaxTokens(
  preparedRequests: readonly PreparedGenerationRequest[],
  group: readonly number[],
): readonly number[] {
  return group.map((index) => preparedAt(preparedRequests, index).request.sampling.maxTokens);
}

function batchGroupOptions(
  preparedRequests: readonly PreparedGenerationRequest[],
  group: readonly number[],
): BatchGenerationOptions {
  const first = preparedAt(preparedRequests, firstGroupIndex(group));
  return {
    ...first.batchOptions,
    maxTokens: batchGroupMaxTokens(preparedRequests, group),
  };
}

function emitBatchStart(
  preparedRequests: readonly PreparedGenerationRequest[],
  group: readonly number[],
  options: TransformersGenerationEngineOptions,
): void {
  const first = preparedAt(preparedRequests, firstGroupIndex(group));
  const maxTokensByRequest = batchGroupMaxTokens(preparedRequests, group);
  options.onEvent?.({
    type: "generation_batch_start",
    mode: "static",
    model: first.request.model,
    ids: group.map((index) => preparedAt(preparedRequests, index).request.id),
    batchSize: group.length,
    maxTokens: Math.max(...maxTokensByRequest),
    maxTokensByRequest,
  });
  for (const index of group) {
    const prepared = preparedAt(preparedRequests, index);
    emitGenerationProgress(options, prepared.request, prepared.promptTokens, 0);
  }
}

function firstGroupIndex(group: readonly number[]): number {
  const firstIndex = group[0];
  if (firstIndex === undefined) {
    throw invalidBatchResult();
  }
  return firstIndex;
}

function assignBatchResults(
  preparedRequests: readonly PreparedGenerationRequest[],
  group: readonly number[],
  generated: readonly GenerationResult[],
  options: TransformersGenerationEngineOptions,
  results: (NormalizedGenerationResult | undefined)[],
): void {
  for (let position = 0; position < group.length; position += 1) {
    const index = group[position];
    const result = generated[position];
    if (index === undefined || result === undefined) {
      throw invalidBatchResult();
    }
    const prepared = preparedAt(preparedRequests, index);
    const serveResult = generatedResultToServeResult(prepared, options, result);
    emitGenerationProgress(
      options,
      prepared.request,
      prepared.promptTokens,
      serveResult.usage?.completionTokens ?? 0,
    );
    results[index] = serveResult;
  }
}

function runBatchGroup(
  preparedRequests: readonly PreparedGenerationRequest[],
  group: readonly number[],
  options: TransformersGenerationEngineOptions,
  results: (NormalizedGenerationResult | undefined)[],
): void {
  emitBatchStart(preparedRequests, group, options);
  assignBatchResults(
    preparedRequests,
    group,
    generateBatchTokens(
      options.model,
      batchGroupPrompts(preparedRequests, group),
      batchGroupOptions(preparedRequests, group),
    ),
    options,
    results,
  );
}

function runBatchGroups(
  preparedRequests: readonly PreparedGenerationRequest[],
  groups: Iterable<readonly number[]>,
  options: TransformersGenerationEngineOptions,
  results: (NormalizedGenerationResult | undefined)[],
): void {
  for (const group of groups) {
    if (group.length === 1) {
      const index = group[0];
      if (index === undefined) {
        throw invalidBatchResult();
      }
      results[index] = generateSinglePreparedRequest(preparedAt(preparedRequests, index), options);
      continue;
    }
    runBatchGroup(preparedRequests, group, options, results);
  }
}

function completedBatchResults(
  results: readonly (NormalizedGenerationResult | undefined)[],
): NormalizedGenerationResult[] {
  const completed: NormalizedGenerationResult[] = [];
  for (const result of results) {
    if (result === undefined) {
      throw invalidBatchResult();
    }
    completed.push(result);
  }
  return completed;
}

function generateBatchForPreparedRequests(
  preparedRequests: readonly PreparedGenerationRequest[],
  options: TransformersGenerationEngineOptions,
): NormalizedGenerationResult[] {
  const results: (NormalizedGenerationResult | undefined)[] = Array.from({
    length: preparedRequests.length,
  });
  const groups = routePreparedRequests(preparedRequests, options, results);
  runBatchGroups(preparedRequests, groups.values(), options, results);
  return completedBatchResults(results);
}

/** Generate one non-streaming request through the transformer-backed engine. */
export function generateTransformersRequest(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
): NormalizedGenerationResult {
  return generateSinglePreparedRequest(prepareGenerationRequest(request, options), options);
}

/** Generate a non-streaming request batch through static batching when eligible. */
export function generateTransformersBatch(
  requests: readonly NormalizedGenerationRequest[],
  options: TransformersGenerationEngineOptions,
): NormalizedGenerationResult[] {
  return generateBatchForPreparedRequests(
    requests.map((request) => prepareGenerationRequest(request, options)),
    options,
  );
}
