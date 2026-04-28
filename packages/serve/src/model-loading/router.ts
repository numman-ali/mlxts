/**
 * Model-id routing for serving multiple loaded engines behind one endpoint.
 * @module
 */

import { ServeError } from "../errors";
import type {
  GenerationEngine,
  NormalizedGenerationRequest,
  NormalizedGenerationResult,
} from "../types";

export type ModelRouterGenerationEngineOptions = {
  engines: ReadonlyMap<string, GenerationEngine> | Record<string, GenerationEngine>;
};

function entriesFromEngines(
  engines: ModelRouterGenerationEngineOptions["engines"],
): [string, GenerationEngine][] {
  return engines instanceof Map ? [...engines.entries()] : Object.entries(engines);
}

function groupByModel(
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

async function generateForEngine(
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

/** Route generation requests to engines by their OpenAI `model` id. */
export function createModelRouterGenerationEngine(
  options: ModelRouterGenerationEngineOptions,
): GenerationEngine {
  const engines = new Map(entriesFromEngines(options.engines));
  if (engines.size === 0) {
    throw new ServeError("Model router requires at least one engine.", {
      code: "no_models_loaded",
      status: 500,
    });
  }

  function engineFor(model: string): GenerationEngine {
    const engine = engines.get(model);
    if (engine === undefined) {
      throw new ServeError(`Model "${model}" is not served by this endpoint.`, {
        code: "model_not_found",
        param: "model",
        status: 404,
      });
    }
    return engine;
  }

  return {
    generate(request) {
      return engineFor(request.model).generate(request);
    },
    async generateBatch(requests) {
      const results: (NormalizedGenerationResult | undefined)[] = [];
      for (const [model, group] of groupByModel(requests)) {
        const groupedResults = await generateForEngine(
          engineFor(model),
          group.map((entry) => entry.request),
        );
        for (const [groupIndex, result] of groupedResults.entries()) {
          const original = group[groupIndex];
          if (original === undefined) {
            throw new ServeError("Model router received an unexpected batch result.", {
              code: "invalid_engine_result",
              status: 500,
            });
          }
          results[original.index] = result;
        }
      }

      const completed: NormalizedGenerationResult[] = [];
      for (let index = 0; index < requests.length; index += 1) {
        const result = results[index];
        if (result === undefined) {
          throw new ServeError("Model router did not receive a result for every request.", {
            code: "invalid_engine_result",
            status: 500,
          });
        }
        completed.push(result);
      }
      return completed;
    },
    stream(request) {
      const engine = engineFor(request.model);
      const stream = engine.stream;
      if (stream === undefined) {
        throw new ServeError(`Model "${request.model}" does not support streaming.`, {
          code: "stream_not_supported",
          param: "stream",
        });
      }
      return stream(request);
    },
    [Symbol.dispose]() {
      for (const engine of engines.values()) {
        engine[Symbol.dispose]?.();
      }
    },
  };
}
