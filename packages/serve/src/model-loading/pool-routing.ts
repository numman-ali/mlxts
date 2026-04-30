/**
 * Lazy model-pool request grouping and engine delegation.
 * @module
 */

import { ServeError } from "../errors";
import type {
  GenerationEngine,
  NormalizedGenerationRequest,
  NormalizedGenerationResult,
} from "../types";
import type { SourceModelPoolEntry } from "./pool-types";

/** Validate and index lazy model-pool entries by served model id. */
export function requireEntries(
  entries: readonly SourceModelPoolEntry[],
): Map<string, SourceModelPoolEntry> {
  if (entries.length === 0) {
    throw new ServeError("Source model pool requires at least one model source.", {
      code: "no_models_loaded",
      status: 500,
    });
  }
  const byModelId = new Map<string, SourceModelPoolEntry>();
  for (const entry of entries) {
    if (byModelId.has(entry.modelId)) {
      throw new ServeError(`Source model pool received duplicate model id "${entry.modelId}".`, {
        code: "duplicate_model_id",
        status: 500,
      });
    }
    byModelId.set(entry.modelId, entry);
  }
  return byModelId;
}

/** Group a batch by target model while preserving original result indexes. */
export function groupByModel(
  requests: readonly NormalizedGenerationRequest[],
): Map<string, { index: number; request: NormalizedGenerationRequest }[]> {
  const groups = new Map<string, { index: number; request: NormalizedGenerationRequest }[]>();
  requests.forEach((request, index) => {
    const group = groups.get(request.model) ?? [];
    group.push({ index, request });
    groups.set(request.model, group);
  });
  return groups;
}

/** Delegate grouped work to batch-capable engines when available. */
export async function generateForEngine(
  engine: GenerationEngine,
  requests: readonly NormalizedGenerationRequest[],
): Promise<NormalizedGenerationResult[]> {
  if (requests.length > 1 && engine.generateBatch !== undefined) {
    const results = await engine.generateBatch(requests);
    if (results.length !== requests.length) {
      throw new ServeError("Generation engine returned the wrong number of batch results.", {
        code: "invalid_engine_result",
        status: 500,
      });
    }
    return [...results];
  }

  const results: NormalizedGenerationResult[] = [];
  for (const request of requests) {
    results.push(await engine.generate(request));
  }
  return results;
}
