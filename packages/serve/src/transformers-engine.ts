/**
 * Adapter from @mlxts/transformers generation into the serving engine contract.
 * @module
 */

import type { Tokenizer } from "@mlxts/tokenizers";
import type { CausalLM, InteractionProfile } from "@mlxts/transformers";
import { ModelExecutionLane, type ModelExecutionLaneStats } from "./model-execution-lane";
import { transformersRuntimeStrategy } from "./serve-runtime-strategy";
import { createContinuousTransformersGeneration } from "./transformers-engine-continuous";
import {
  generateSinglePreparedRequest,
  prepareGenerationRequest,
  streamSinglePreparedRequest,
} from "./transformers-engine-generation";
import { PromptPrefixCache } from "./transformers-engine-prefix-cache";
import {
  continuousBatchIneligibilityReason,
  emitGenerationRouteDecision,
} from "./transformers-engine-routing";
import {
  createStaticTransformersGeneration,
  runStaticBatchOnModelLane,
} from "./transformers-engine-static";
import type {
  GenerationEngine,
  NormalizedGenerationRequest,
  NormalizedGenerationResult,
  ServeEvent,
} from "./types";

export type TransformersGenerationEngineOptions = {
  model: CausalLM;
  tokenizer: Tokenizer;
  interactionProfile?: InteractionProfile;
  maxGeneratedTokens?: number;
  maxPromptTokens?: number;
  maxTotalTokens?: number;
  maxBatchSize?: number;
  batchWindowMs?: number;
  prefillStepSize?: number;
  activePrefillStepSize?: number;
  activeDecodeStepsPerPrefillChunk?: number;
  streamDecodeInterval?: number;
  maxConcurrentRequests?: number;
  gpuMemoryUtilization?: number;
  onEvent?: (event: ServeEvent) => void;
};

function emitGenerationModelLaneWait(
  options: TransformersGenerationEngineOptions,
  request: NormalizedGenerationRequest,
  queuedStats: ModelExecutionLaneStats,
  dispatchStats: ModelExecutionLaneStats,
  waitMs: number,
): void {
  options.onEvent?.({
    type: "generation_model_lane_wait",
    id: request.id,
    protocol: request.protocol,
    model: request.model,
    lane: "model",
    waitMs,
    inFlightAtQueue: queuedStats.inFlight,
    queuedAhead: queuedStats.queued,
    inFlightAtDispatch: dispatchStats.inFlight,
    queuedAtDispatch: dispatchStats.queued,
    maxConcurrentJobs: queuedStats.maxConcurrentJobs,
  });
}

async function acquireModelLane(
  lane: ModelExecutionLane,
  options: TransformersGenerationEngineOptions,
  request: NormalizedGenerationRequest,
): Promise<() => void> {
  const queuedStats = lane.stats();
  const queuedAt = performance.now();
  const release = await lane.acquire(request.abortSignal);
  emitGenerationModelLaneWait(
    options,
    request,
    queuedStats,
    lane.stats(),
    performance.now() - queuedAt,
  );
  return release;
}

async function runOnModelLane<T>(
  lane: ModelExecutionLane,
  options: TransformersGenerationEngineOptions,
  request: NormalizedGenerationRequest,
  work: () => Promise<T>,
): Promise<T> {
  const release = await acquireModelLane(lane, options, request);
  try {
    return await work();
  } finally {
    release();
  }
}

function maxBatchSize(options: TransformersGenerationEngineOptions): number {
  return transformersRuntimeStrategy(options).scheduler.maxBatchSize;
}

/** Create a text-generation engine from an already loaded CausalLM and tokenizer. */
export function createTransformersGenerationEngine(
  options: TransformersGenerationEngineOptions,
): GenerationEngine {
  const strategy = transformersRuntimeStrategy(options);
  const lane = new ModelExecutionLane(strategy.scheduler.maxConcurrentRequests);
  const promptPrefixCache = new PromptPrefixCache();
  const staticGeneration = createStaticTransformersGeneration(options, lane);
  const continuous = createContinuousTransformersGeneration(options, lane);

  function generate(
    request: NormalizedGenerationRequest,
  ): NormalizedGenerationResult | Promise<NormalizedGenerationResult> {
    if (request.input.kind === "messages") {
      emitGenerationRouteDecision(options, request, "single", false, "prompt_prefix_cache");
      const prepared = prepareGenerationRequest(request, options);
      return runOnModelLane(lane, options, request, () =>
        generateSinglePreparedRequest(prepared, options, promptPrefixCache),
      );
    }
    const staticallyBatched = staticGeneration.generate(request);
    if (staticallyBatched !== null) {
      return staticallyBatched;
    }
    const scheduled = continuous.generate(request);
    if (scheduled !== null) {
      return scheduled;
    }
    const prepared = prepareGenerationRequest(request, options);
    return runOnModelLane(lane, options, request, () =>
      generateSinglePreparedRequest(prepared, options),
    );
  }

  return {
    generate,
    generateBatch(requests) {
      if (
        maxBatchSize(options) > 1 &&
        requests.every(
          (request) => continuousBatchIneligibilityReason(request, options) === "eligible",
        )
      ) {
        return Promise.all(requests.map((request) => generate(request)));
      }
      return runStaticBatchOnModelLane(lane, options, requests);
    },
    async *stream(request) {
      if (request.input.kind === "messages") {
        emitGenerationRouteDecision(options, request, "single", false, "prompt_prefix_cache");
        const prepared = prepareGenerationRequest(request, options);
        const release = await acquireModelLane(lane, options, request);
        try {
          yield* streamSinglePreparedRequest(prepared, options, promptPrefixCache);
        } finally {
          release();
        }
        return;
      }
      const scheduled = continuous.stream(request);
      if (scheduled !== null) {
        yield* scheduled;
        return;
      }
      const prepared = prepareGenerationRequest(request, options);
      const release = await acquireModelLane(lane, options, request);
      try {
        yield* streamSinglePreparedRequest(prepared, options);
      } finally {
        release();
      }
    },
    [Symbol.dispose]() {
      promptPrefixCache[Symbol.dispose]();
    },
  };
}
