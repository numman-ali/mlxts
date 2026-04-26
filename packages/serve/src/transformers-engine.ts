/**
 * Adapter from @mlxts/transformers generation into the serving engine contract.
 * @module
 */

import type { Tokenizer } from "@mlxts/tokenizers";
import {
  type CausalLM,
  generatePreparedTokenEvents,
  generateTokenEvents,
  type InteractionProfile,
} from "@mlxts/transformers";
import { ModelExecutionLane, type ModelExecutionLaneStats } from "./model-execution-lane";
import { createContinuousTransformersGeneration } from "./transformers-engine-continuous";
import {
  generateSinglePreparedRequest,
  generateTransformersBatch,
  prepareGenerationRequest,
} from "./transformers-engine-generation";
import {
  compileMessagePrompt,
  createPrefillProgressReporter,
  emitGenerationProgress,
  enforceGenerationMemoryBudget,
  enforcePromptTokenLimit,
  enforceTotalTokenLimit,
  generationOptions,
  promptTokenCount,
  promptTokenIds,
} from "./transformers-engine-shared";
import {
  createStreamingDecodeState,
  handleStreamingDoneEvent,
  handleStreamingTokenEvent,
  streamDecodeInterval,
} from "./transformers-engine-streaming";
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
  maxPromptTokens?: number;
  maxTotalTokens?: number;
  maxBatchSize?: number;
  batchWindowMs?: number;
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

function streamTokenEventsForRequest(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
  prompt: { tokenIds: readonly number[] } | null,
  onPrefillProgress: Parameters<typeof generationOptions>[1],
) {
  if (request.input.kind === "text") {
    return generateTokenEvents(
      options.model,
      options.tokenizer.encode(request.input.text, { addSpecialTokens: true }),
      {
        ...generationOptions(request, onPrefillProgress),
        ...(request.sampling.ignoreEos === true || options.tokenizer.eosTokenIds.length === 0
          ? {}
          : { eosTokenIds: [...options.tokenizer.eosTokenIds] }),
      },
    );
  }

  return generatePreparedTokenEvents(
    options.model,
    { tokenIds: promptTokenIds(request, prompt) },
    generationOptions(request, onPrefillProgress),
  );
}

/** Create a text-generation engine from an already loaded CausalLM and tokenizer. */
export function createTransformersGenerationEngine(
  options: TransformersGenerationEngineOptions,
): GenerationEngine {
  const lane = new ModelExecutionLane(options.maxConcurrentRequests ?? 1);
  const continuous = createContinuousTransformersGeneration(options, lane);

  function generate(
    request: NormalizedGenerationRequest,
  ): NormalizedGenerationResult | Promise<NormalizedGenerationResult> {
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
      if ((options.maxBatchSize ?? 1) > 1) {
        return Promise.all(requests.map((request) => generate(request)));
      }
      return generateTransformersBatch(requests, options);
    },
    async *stream(request) {
      const scheduled = continuous.stream(request);
      if (scheduled !== null) {
        yield* scheduled;
        return;
      }
      const release = await acquireModelLane(lane, options, request);
      try {
        const prompt = compileMessagePrompt(request, options);
        const promptTokens = promptTokenCount(request, options, prompt);
        enforcePromptTokenLimit(options, request, promptTokens);
        enforceTotalTokenLimit(options, request, promptTokens);
        enforceGenerationMemoryBudget(options, request, promptTokens);
        emitGenerationProgress(options, request, promptTokens, 0);
        const onPrefillProgress = createPrefillProgressReporter(options, request, promptTokens);
        const decodeInterval = streamDecodeInterval(request.sampling.stop);
        const tokenEvents = streamTokenEventsForRequest(
          request,
          options,
          prompt,
          onPrefillProgress,
        );
        const state = createStreamingDecodeState(prompt);

        for await (const event of tokenEvents) {
          if (event.type === "token") {
            const text = handleStreamingTokenEvent(
              request,
              options,
              promptTokens,
              decodeInterval,
              state,
              event,
            );
            if (text !== undefined) {
              yield { type: "text", text };
            }
            continue;
          }

          const finished = handleStreamingDoneEvent(options, promptTokens, state, event);
          if (finished.text !== undefined) {
            yield { type: "text", text: finished.text };
          }
          yield finished.done;
          return;
        }
      } finally {
        release();
      }
    },
  };
}
