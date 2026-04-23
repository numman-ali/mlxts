/**
 * Request admission limits for generation engines.
 * @module
 */

import { ServeError } from "./errors";
import type {
  GenerationEngine,
  NormalizedGenerationRequest,
  NormalizedGenerationResult,
} from "./types";

export type RequestLimitGenerationEngineOptions = {
  engine: GenerationEngine;
  maxGeneratedTokens: number;
};

function positiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value <= 0) {
    throw new ServeError(`${name} must be a positive integer.`, {
      code: "invalid_limit_options",
      param: name,
    });
  }
  return value;
}

function enforceGeneratedTokenLimit(
  request: NormalizedGenerationRequest,
  maxGeneratedTokens: number,
): void {
  if (request.sampling.maxTokens <= maxGeneratedTokens) {
    return;
  }

  throw new ServeError(
    `Requested max_tokens ${request.sampling.maxTokens} exceeds this server's limit of ${maxGeneratedTokens}.`,
    {
      code: "max_tokens_exceeded",
      param: "max_tokens",
    },
  );
}

function enforceBatchGeneratedTokenLimit(
  requests: readonly NormalizedGenerationRequest[],
  maxGeneratedTokens: number,
): void {
  for (const request of requests) {
    enforceGeneratedTokenLimit(request, maxGeneratedTokens);
  }
}

async function generateBatch(
  engine: GenerationEngine,
  requests: readonly NormalizedGenerationRequest[],
): Promise<NormalizedGenerationResult[]> {
  if (engine.generateBatch !== undefined) {
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

/** Reject generation requests that exceed configured server admission limits. */
export function createRequestLimitGenerationEngine(
  options: RequestLimitGenerationEngineOptions,
): GenerationEngine {
  const maxGeneratedTokens = positiveInteger(options.maxGeneratedTokens, "maxGeneratedTokens");

  const limited: GenerationEngine = {
    generate(request) {
      enforceGeneratedTokenLimit(request, maxGeneratedTokens);
      return options.engine.generate(request);
    },
    generateBatch(requests) {
      enforceBatchGeneratedTokenLimit(requests, maxGeneratedTokens);
      return generateBatch(options.engine, requests);
    },
  };

  const stream = options.engine.stream;
  if (stream !== undefined) {
    limited.stream = (request) => {
      enforceGeneratedTokenLimit(request, maxGeneratedTokens);
      return stream(request);
    };
  }

  return limited;
}
