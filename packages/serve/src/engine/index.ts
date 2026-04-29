/**
 * Adapter from @mlxts/transformers generation into the serving engine contract.
 * @module
 */

import type { Tokenizer } from "@mlxts/tokenizers";
import type { CausalLM, InteractionProfile } from "@mlxts/transformers";
import { ServeError } from "../errors";
import { transformersRuntimeStrategy } from "../runtime/strategy";
import type {
  GenerationEngine,
  GenerationStreamEvent,
  NormalizedGenerationRequest,
  NormalizedGenerationResult,
  ServeEvent,
} from "../types";
import {
  loadContentGenerationRequest,
  prepareLoadedContentGenerationRequest,
  type TransformersContentAdapter,
} from "./content";
import {
  type ContinuousTransformersGeneration,
  createContinuousTransformersGeneration,
} from "./continuous";
import { ModelExecutionLane, type ModelExecutionLaneStats } from "./execution-lane";
import {
  generateSinglePreparedRequest,
  prepareGenerationRequest,
  streamSinglePreparedRequest,
} from "./generation";
import { PromptPrefixCache } from "./prefix-cache";
import { continuousBatchIneligibilityReason, emitGenerationRouteDecision } from "./routing";
import { createStaticTransformersGeneration, runStaticBatchOnModelLane } from "./static";

type StaticTransformersGeneration = ReturnType<typeof createStaticTransformersGeneration>;

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
  promptPrefixCacheMaxEntries?: number;
  gpuMemoryUtilization?: number;
  contentAdapter?: TransformersContentAdapter;
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

function rejectMediaInput(): never {
  throw new ServeError(
    "The transformers generation engine accepted media-shaped input, but this loaded model does not prepare media tensors.",
    { code: "unsupported_input", param: "messages" },
  );
}

function rejectMediaBatchInputs(
  options: TransformersGenerationEngineOptions,
  requests: readonly NormalizedGenerationRequest[],
): void {
  let hasMediaInput = false;
  for (const request of requests) {
    if (request.input.kind !== "content") {
      continue;
    }
    hasMediaInput = true;
    emitGenerationRouteDecision(options, request, "single", false, "media_input");
  }
  if (hasMediaInput) {
    rejectMediaInput();
  }
}

function generateContentRequest(
  lane: ModelExecutionLane,
  options: TransformersGenerationEngineOptions,
  promptPrefixCache: PromptPrefixCache,
  request: NormalizedGenerationRequest,
): Promise<NormalizedGenerationResult> {
  emitGenerationRouteDecision(options, request, "single", false, "media_input");
  if (options.contentAdapter === undefined) {
    return rejectMediaInput();
  }
  return (async () => {
    const loaded = await loadContentGenerationRequest(request, options);
    return await runOnModelLane(lane, options, request, async () => {
      const prepared = await prepareLoadedContentGenerationRequest(loaded, options);
      return await generateSinglePreparedRequest(prepared, options, promptPrefixCache);
    });
  })();
}

function generateMessageRequest(
  lane: ModelExecutionLane,
  options: TransformersGenerationEngineOptions,
  promptPrefixCache: PromptPrefixCache,
  continuous: ContinuousTransformersGeneration,
  request: NormalizedGenerationRequest,
): NormalizedGenerationResult | Promise<NormalizedGenerationResult> {
  if (maxBatchSize(options) > 1) {
    const scheduled = continuous.generate(request);
    if (scheduled !== null) {
      return scheduled;
    }
  } else {
    emitGenerationRouteDecision(options, request, "single", false, "prompt_prefix_cache");
  }
  const prepared = prepareGenerationRequest(request, options);
  return runOnModelLane(lane, options, request, () =>
    generateSinglePreparedRequest(prepared, options, promptPrefixCache),
  );
}

function generateTextRequest(
  lane: ModelExecutionLane,
  options: TransformersGenerationEngineOptions,
  staticGeneration: StaticTransformersGeneration,
  continuous: ContinuousTransformersGeneration,
  request: NormalizedGenerationRequest,
): NormalizedGenerationResult | Promise<NormalizedGenerationResult> {
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

async function* streamContentRequest(
  lane: ModelExecutionLane,
  options: TransformersGenerationEngineOptions,
  promptPrefixCache: PromptPrefixCache,
  request: NormalizedGenerationRequest,
): AsyncIterable<GenerationStreamEvent> {
  emitGenerationRouteDecision(options, request, "single", false, "media_input");
  if (options.contentAdapter === undefined) {
    rejectMediaInput();
  }
  const loaded = await loadContentGenerationRequest(request, options);
  const release = await acquireModelLane(lane, options, request);
  try {
    const prepared = await prepareLoadedContentGenerationRequest(loaded, options);
    yield* streamSinglePreparedRequest(prepared, options, promptPrefixCache);
  } finally {
    release();
  }
}

async function* streamMessageRequest(
  lane: ModelExecutionLane,
  options: TransformersGenerationEngineOptions,
  promptPrefixCache: PromptPrefixCache,
  continuous: ContinuousTransformersGeneration,
  request: NormalizedGenerationRequest,
): AsyncIterable<GenerationStreamEvent> {
  if (maxBatchSize(options) > 1) {
    const scheduled = continuous.stream(request);
    if (scheduled !== null) {
      yield* scheduled;
      return;
    }
  } else {
    emitGenerationRouteDecision(options, request, "single", false, "prompt_prefix_cache");
  }
  const prepared = prepareGenerationRequest(request, options);
  const release = await acquireModelLane(lane, options, request);
  try {
    yield* streamSinglePreparedRequest(prepared, options, promptPrefixCache);
  } finally {
    release();
  }
}

async function* streamTextRequest(
  lane: ModelExecutionLane,
  options: TransformersGenerationEngineOptions,
  continuous: ContinuousTransformersGeneration,
  request: NormalizedGenerationRequest,
): AsyncIterable<GenerationStreamEvent> {
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
}

/** Create a text-generation engine from an already loaded CausalLM and tokenizer. */
export function createTransformersGenerationEngine(
  options: TransformersGenerationEngineOptions,
): GenerationEngine {
  const strategy = transformersRuntimeStrategy(options);
  const lane = new ModelExecutionLane(strategy.scheduler.maxConcurrentRequests);
  const promptPrefixCache = new PromptPrefixCache({
    maxEntries: strategy.cache.promptPrefixMaxEntries,
  });
  const staticGeneration = createStaticTransformersGeneration(options, lane);
  const continuous = createContinuousTransformersGeneration(options, lane, promptPrefixCache);

  function generate(
    request: NormalizedGenerationRequest,
  ): NormalizedGenerationResult | Promise<NormalizedGenerationResult> {
    if (request.input.kind === "content") {
      return generateContentRequest(lane, options, promptPrefixCache, request);
    }
    if (request.input.kind === "messages") {
      return generateMessageRequest(lane, options, promptPrefixCache, continuous, request);
    }
    return generateTextRequest(lane, options, staticGeneration, continuous, request);
  }

  return {
    generate,
    async generateBatch(requests) {
      rejectMediaBatchInputs(options, requests);
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
      if (request.input.kind === "content") {
        yield* streamContentRequest(lane, options, promptPrefixCache, request);
        return;
      }
      if (request.input.kind === "messages") {
        yield* streamMessageRequest(lane, options, promptPrefixCache, continuous, request);
      } else {
        yield* streamTextRequest(lane, options, continuous, request);
      }
    },
    [Symbol.dispose]() {
      promptPrefixCache[Symbol.dispose]();
    },
  };
}
