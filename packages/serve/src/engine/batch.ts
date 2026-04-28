/**
 * Static batch assembly for transformer-backed generation.
 * @module
 */

import {
  type BatchGenerationOptions,
  type GenerationResult,
  generateBatchTokens,
} from "@mlxts/transformers";
import { ServeError } from "../errors";
import { type LinkedAbortSignal, linkAbortSignals } from "../http/abort";
import type { NormalizedGenerationRequest, NormalizedGenerationResult } from "../types";
import {
  batchOptionsKey,
  generatedResultToServeResult,
  generateSinglePreparedRequest,
  type PreparedGenerationRequest,
  prepareGenerationRequest,
} from "./generation";
import type { TransformersGenerationEngineOptions } from "./index";
import { emitGenerationRouteDecision, staticBatchIneligibilityReason } from "./routing";
import { emitGenerationProgress, enforceGenerationMemoryBudgetForTokens } from "./shared";

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

async function routePreparedRequests(
  preparedRequests: readonly PreparedGenerationRequest[],
  options: TransformersGenerationEngineOptions,
  results: (NormalizedGenerationResult | undefined)[],
): Promise<Map<string, number[]>> {
  const groups = new Map<string, number[]>();
  for (let index = 0; index < preparedRequests.length; index += 1) {
    const prepared = preparedAt(preparedRequests, index);
    const reason = staticBatchIneligibilityReason(prepared.request, options);
    if (reason !== "eligible") {
      emitGenerationRouteDecision(options, prepared.request, "single", false, reason);
      results[index] = await generateSinglePreparedRequest(prepared, options);
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

function batchGroupAbortSignalScope(
  preparedRequests: readonly PreparedGenerationRequest[],
  group: readonly number[],
): LinkedAbortSignal | undefined {
  const signals: AbortSignal[] = [];
  for (const index of group) {
    const signal = preparedAt(preparedRequests, index).request.abortSignal;
    if (signal === undefined) {
      continue;
    }
    signals.push(signal);
  }
  return signals.length === 0 ? undefined : linkAbortSignals(...signals);
}

function firstGroupIndex(group: readonly number[]): number {
  const firstIndex = group[0];
  if (firstIndex === undefined) {
    throw invalidBatchResult();
  }
  return firstIndex;
}

function batchGroupOptions(
  preparedRequests: readonly PreparedGenerationRequest[],
  group: readonly number[],
  abortSignal: AbortSignal | undefined,
): BatchGenerationOptions {
  const first = preparedAt(preparedRequests, firstGroupIndex(group));
  return {
    ...first.batchOptions,
    maxTokens: batchGroupMaxTokens(preparedRequests, group),
    ...(abortSignal === undefined ? {} : { abortSignal }),
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
  const maxPromptTokens = Math.max(
    ...group.map((index) => preparedAt(preparedRequests, index).promptTokens),
  );
  const maxTotalTokens = Math.max(
    ...group.map((index) => {
      const prepared = preparedAt(preparedRequests, index);
      return prepared.promptTokens + prepared.request.sampling.maxTokens;
    }),
  );
  enforceGenerationMemoryBudgetForTokens(
    options,
    maxPromptTokens,
    Math.max(0, maxTotalTokens - maxPromptTokens),
    undefined,
    group.length,
  );
  emitBatchStart(preparedRequests, group, options);
  const abortScope = batchGroupAbortSignalScope(preparedRequests, group);
  try {
    assignBatchResults(
      preparedRequests,
      group,
      generateBatchTokens(
        options.model,
        batchGroupPrompts(preparedRequests, group),
        batchGroupOptions(preparedRequests, group, abortScope?.signal),
      ),
      options,
      results,
    );
  } finally {
    abortScope?.dispose();
  }
}

async function runBatchGroups(
  preparedRequests: readonly PreparedGenerationRequest[],
  groups: Iterable<readonly number[]>,
  options: TransformersGenerationEngineOptions,
  results: (NormalizedGenerationResult | undefined)[],
): Promise<void> {
  for (const group of groups) {
    if (group.length === 1) {
      const index = group[0];
      if (index === undefined) {
        throw invalidBatchResult();
      }
      const prepared = preparedAt(preparedRequests, index);
      emitGenerationRouteDecision(
        options,
        prepared.request,
        "single",
        false,
        "single_request_group",
      );
      results[index] = await generateSinglePreparedRequest(prepared, options);
      continue;
    }
    for (const index of group) {
      const prepared = preparedAt(preparedRequests, index);
      emitGenerationRouteDecision(options, prepared.request, "static", true, "eligible");
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

async function generateBatchForPreparedRequests(
  preparedRequests: readonly PreparedGenerationRequest[],
  options: TransformersGenerationEngineOptions,
): Promise<NormalizedGenerationResult[]> {
  const results: (NormalizedGenerationResult | undefined)[] = Array.from({
    length: preparedRequests.length,
  });
  const groups = await routePreparedRequests(preparedRequests, options, results);
  await runBatchGroups(preparedRequests, groups.values(), options, results);
  return completedBatchResults(results);
}

/** Generate a non-streaming request batch through static batching when eligible. */
export function generateTransformersBatch(
  requests: readonly NormalizedGenerationRequest[],
  options: TransformersGenerationEngineOptions,
): Promise<NormalizedGenerationResult[]> {
  return generateBatchForPreparedRequests(
    requests.map((request) => prepareGenerationRequest(request, options)),
    options,
  );
}
