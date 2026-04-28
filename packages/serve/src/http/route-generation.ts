/**
 * Shared generation dispatch helpers for serving routes.
 * @module
 */

import { ServeError } from "../errors";
import type { NormalizedCompletionBatch } from "../protocols/openai-completions";
import type { NormalizedGenerationResult } from "../types";
import { emitGenerationComplete, emitGenerationError, emitGenerationStart } from "./events";
import type { ServeAppOptions } from "./server";

type CompletionRequests = NormalizedCompletionBatch["requests"];

function assertBatchResultCount(
  results: readonly NormalizedGenerationResult[],
  requests: CompletionRequests,
): void {
  if (results.length !== requests.length) {
    throw new ServeError("Generation engine returned the wrong number of batch results.", {
      code: "invalid_engine_result",
      status: 500,
    });
  }
}

function emitBatchGenerationErrors(
  options: ServeAppOptions,
  requests: CompletionRequests,
  error: unknown,
  startedAt: number,
): void {
  const durationMs = performance.now() - startedAt;
  for (const normalized of requests) {
    emitGenerationError(options, normalized, error, durationMs);
  }
}

function emitBatchGenerationComplete(
  options: ServeAppOptions,
  requests: CompletionRequests,
  results: readonly NormalizedGenerationResult[],
  durationMs: number,
): void {
  for (let index = 0; index < requests.length; index += 1) {
    const normalized = requests[index];
    const result = results[index];
    if (normalized !== undefined && result !== undefined) {
      emitGenerationComplete(options, normalized, result, durationMs);
    }
  }
}

async function generateWithBatchEngine(
  options: ServeAppOptions,
  requests: CompletionRequests,
): Promise<NormalizedGenerationResult[]> {
  const generateBatch = options.engine.generateBatch;
  if (generateBatch === undefined) {
    throw new Error("generateWithBatchEngine requires an engine batch function.");
  }

  const startedAt = performance.now();
  for (const normalized of requests) {
    emitGenerationStart(options, normalized);
  }
  try {
    const results = await generateBatch(requests);
    assertBatchResultCount(results, requests);
    emitBatchGenerationComplete(options, requests, results, performance.now() - startedAt);
    return [...results];
  } catch (error) {
    emitBatchGenerationErrors(options, requests, error, startedAt);
    throw error;
  }
}

async function generateSequentially(
  options: ServeAppOptions,
  requests: CompletionRequests,
): Promise<NormalizedGenerationResult[]> {
  const results: NormalizedGenerationResult[] = [];
  for (const normalized of requests) {
    const startedAt = performance.now();
    emitGenerationStart(options, normalized);
    try {
      const result = await options.engine.generate(normalized);
      emitGenerationComplete(options, normalized, result, performance.now() - startedAt);
      results.push(result);
    } catch (error) {
      emitGenerationError(options, normalized, error, performance.now() - startedAt);
      throw error;
    }
  }
  return results;
}

/** Generate one or more normalized completion requests with matching lifecycle events. */
export async function generateCompletionBatch(
  options: ServeAppOptions,
  requests: CompletionRequests,
): Promise<NormalizedGenerationResult[]> {
  if (requests.length > 1 && options.engine.generateBatch !== undefined) {
    return await generateWithBatchEngine(options, requests);
  }
  return await generateSequentially(options, requests);
}
