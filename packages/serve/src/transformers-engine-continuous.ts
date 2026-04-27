/**
 * Continuous batching adapter for transformer-backed serving.
 * @module
 */

import {
  type ContinuousBatchAdmissionController,
  type ContinuousBatchSchedulerEvent,
  type ContinuousBatchTokenSchedulerOptions,
  createContinuousBatchTokenScheduler,
} from "@mlxts/transformers";
import { createContinuousSchedulerTokenBudget } from "./continuous-scheduler-budget";
import type { ModelExecutionLane } from "./model-execution-lane";
import { transformersRuntimeStrategy } from "./serve-runtime-strategy";
import { linkAbortSignals } from "./server-abort";
import type { TransformersGenerationEngineOptions } from "./transformers-engine";
import {
  batchOptionsKey,
  generatedResultToServeResult,
  type PreparedGenerationRequest,
  prepareGenerationRequest,
} from "./transformers-engine-generation";
import {
  continuousBatchIneligibilityReason,
  emitGenerationRouteDecision,
  routeDecisionForRequest,
} from "./transformers-engine-routing";
import {
  createPrefillProgressReporter,
  createProgressReporter,
  emitGenerationProgress,
} from "./transformers-engine-shared";
import {
  createStreamingDecodeState,
  handleStreamingDone,
  handleStreamingTokenDelta,
  streamDecodeInterval,
} from "./transformers-engine-streaming";
import type {
  GenerationStreamEvent,
  NormalizedGenerationRequest,
  NormalizedGenerationResult,
} from "./types";

type SchedulerEntry = {
  scheduler: ReturnType<typeof createContinuousBatchTokenScheduler>;
  modelsByRequestId: Map<string, string>;
};

export type ContinuousTransformersGeneration = {
  generate(request: NormalizedGenerationRequest): Promise<NormalizedGenerationResult> | null;
  stream(request: NormalizedGenerationRequest): AsyncIterable<GenerationStreamEvent> | null;
};

function maxBatchSize(options: TransformersGenerationEngineOptions): number {
  return transformersRuntimeStrategy(options).scheduler.maxBatchSize;
}

function canUseContinuousBatchGeneration(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
): boolean {
  return (
    maxBatchSize(options) > 1 && continuousBatchIneligibilityReason(request, options) === "eligible"
  );
}

function schedulerOptions(
  prepared: PreparedGenerationRequest,
  options: TransformersGenerationEngineOptions,
  lane: ModelExecutionLane,
  modelsByRequestId: Map<string, string>,
  admissionController: ContinuousBatchAdmissionController | undefined,
): ContinuousBatchTokenSchedulerOptions {
  const batchOptions = prepared.batchOptions;
  const strategy = transformersRuntimeStrategy(options);
  return {
    ...(batchOptions.temperature === undefined ? {} : { temperature: batchOptions.temperature }),
    ...(batchOptions.topK === undefined ? {} : { topK: batchOptions.topK }),
    ...(batchOptions.topP === undefined ? {} : { topP: batchOptions.topP }),
    ...(batchOptions.minP === undefined ? {} : { minP: batchOptions.minP }),
    ...(batchOptions.repetitionPenalty === undefined
      ? {}
      : { repetitionPenalty: batchOptions.repetitionPenalty }),
    ...(batchOptions.seed === undefined ? {} : { seed: batchOptions.seed }),
    ...(batchOptions.eosTokenIds === undefined ? {} : { eosTokenIds: batchOptions.eosTokenIds }),
    ...(batchOptions.useCache === undefined ? {} : { useCache: batchOptions.useCache }),
    ...(batchOptions.prefillStepSize === undefined
      ? {}
      : { prefillStepSize: batchOptions.prefillStepSize }),
    ...(batchOptions.padTokenId === undefined ? {} : { padTokenId: batchOptions.padTokenId }),
    maxBatchSize: strategy.scheduler.maxBatchSize,
    batchWindowMs: strategy.scheduler.batchWindowMs,
    activePrefillStepSize: strategy.scheduler.activePrefillStepSize,
    activeDecodeStepsPerPrefillChunk: strategy.scheduler.activeDecodeStepsPerPrefillChunk,
    runExclusive: (work) => lane.run(work),
    ...(admissionController === undefined ? {} : { admissionController }),
    onSchedulerEvent(event) {
      emitContinuousSchedulerPhase(options, prepared.request.model, modelsByRequestId, event);
    },
  };
}

function schedulerEventModel(
  fallbackModel: string,
  modelsByRequestId: Map<string, string>,
  event: ContinuousBatchSchedulerEvent,
): string {
  const id = "id" in event ? event.id : event.ids[0];
  return id === undefined ? fallbackModel : (modelsByRequestId.get(id) ?? fallbackModel);
}

function schedulerCounts(event: ContinuousBatchSchedulerEvent) {
  return {
    waiting: event.waiting,
    prefilling: event.prefilling,
    active: event.active,
    maxBatchSize: event.maxBatchSize,
    waitingTotalTokens: event.waitingTotalTokens,
    prefillingTotalTokens: event.prefillingTotalTokens,
    activeTotalTokens: event.activeTotalTokens,
    scheduledPromptTokens: event.scheduledPromptTokens,
    maxScheduledPromptTokens: event.maxScheduledPromptTokens,
    scheduledCompletionTokens: event.scheduledCompletionTokens,
    maxScheduledCompletionTokens: event.maxScheduledCompletionTokens,
    scheduledTotalTokens: event.scheduledTotalTokens,
    maxScheduledTotalTokens: event.maxScheduledTotalTokens,
  };
}

function emitContinuousSchedulerPhase(
  options: TransformersGenerationEngineOptions,
  fallbackModel: string,
  modelsByRequestId: Map<string, string>,
  event: ContinuousBatchSchedulerEvent,
): void {
  const model = schedulerEventModel(fallbackModel, modelsByRequestId, event);
  switch (event.type) {
    case "queued":
      options.onEvent?.({
        type: "generation_scheduler_phase",
        mode: "continuous",
        phase: event.type,
        model,
        id: event.id,
        ids: [event.id],
        queuedAhead: event.queuedAhead,
        promptTokens: event.promptTokens,
        maxTokens: event.maxTokens,
        schedulerMs: event.schedulerMs,
        ...schedulerCounts(event),
      });
      return;
    case "deferred":
      options.onEvent?.({
        type: "generation_scheduler_phase",
        mode: "continuous",
        phase: event.type,
        model,
        id: event.id,
        ids: [event.id],
        reason: event.reason,
        promptTokens: event.promptTokens,
        maxTokens: event.maxTokens,
        queuedMs: event.queuedMs,
        schedulerMs: event.schedulerMs,
        ...schedulerCounts(event),
      });
      return;
    case "prefill_start":
      options.onEvent?.({
        type: "generation_scheduler_phase",
        mode: "continuous",
        phase: event.type,
        model,
        id: event.id,
        ids: [event.id],
        promptTokens: event.promptTokens,
        maxTokens: event.maxTokens,
        queuedMs: event.queuedMs,
        schedulerMs: event.schedulerMs,
        ...schedulerCounts(event),
      });
      return;
    case "admitted":
      options.onEvent?.({
        type: "generation_scheduler_phase",
        mode: "continuous",
        phase: event.type,
        model,
        ids: event.ids,
        batchSize: event.batchSize,
        maxTokens: event.maxTokens,
        maxTokensByRequest: event.maxTokensByRequest,
        queuedMsByRequest: event.queuedMsByRequest,
        schedulerMs: event.schedulerMs,
        ...schedulerCounts(event),
      });
      return;
    case "first_token":
      options.onEvent?.({
        type: "generation_scheduler_phase",
        mode: "continuous",
        phase: event.type,
        model,
        id: event.id,
        ids: [event.id],
        completionTokens: event.completionTokens,
        queuedMs: event.queuedMs,
        schedulerMs: event.schedulerMs,
        ...schedulerCounts(event),
      });
      return;
    case "finished":
      options.onEvent?.({
        type: "generation_scheduler_phase",
        mode: "continuous",
        phase: event.type,
        model,
        id: event.id,
        ids: [event.id],
        completionTokens: event.completionTokens,
        finishReason: event.finishReason,
        queuedMs: event.queuedMs,
        schedulerMs: event.schedulerMs,
        ...schedulerCounts(event),
      });
      return;
    case "cancelled":
      options.onEvent?.({
        type: "generation_scheduler_phase",
        mode: "continuous",
        phase: event.type,
        model,
        id: event.id,
        ids: [event.id],
        completionTokens: event.completionTokens,
        queuedMs: event.queuedMs,
        schedulerMs: event.schedulerMs,
        ...schedulerCounts(event),
      });
  }
}

function schedulerKey(prepared: PreparedGenerationRequest): string {
  return batchOptionsKey(prepared.batchOptions);
}

type StreamQueueState = {
  events: GenerationStreamEvent[];
  notify: (() => void) | null;
  done: boolean;
  error: unknown;
};

function enqueueStreamEvent(state: StreamQueueState, event: GenerationStreamEvent): void {
  state.events.push(event);
  state.notify?.();
  state.notify = null;
}

function finishStreamQueue(state: StreamQueueState, error?: unknown): void {
  state.done = true;
  if (error !== undefined) {
    state.error = error;
  }
  state.notify?.();
  state.notify = null;
}

function waitForStreamQueue(state: StreamQueueState): Promise<void> {
  return new Promise((resolve) => {
    state.notify = resolve;
  });
}

async function* readStreamQueue(state: StreamQueueState): AsyncIterable<GenerationStreamEvent> {
  while (true) {
    const event = state.events.shift();
    if (event !== undefined) {
      yield event;
      continue;
    }
    if (state.error !== undefined) {
      throw state.error;
    }
    if (state.done) {
      return;
    }
    await waitForStreamQueue(state);
  }
}

/** Create the optional continuous-batching path for eligible transformer requests. */
export function createContinuousTransformersGeneration(
  options: TransformersGenerationEngineOptions,
  lane: ModelExecutionLane,
): ContinuousTransformersGeneration {
  const schedulers = new Map<string, SchedulerEntry>();
  const strategy = transformersRuntimeStrategy(options);
  const admissionController = createContinuousSchedulerTokenBudget({
    maxBatchSize: strategy.scheduler.maxBatchSize,
    ...(options.maxGeneratedTokens === undefined
      ? {}
      : { maxGeneratedTokens: options.maxGeneratedTokens }),
    ...(options.maxPromptTokens === undefined ? {} : { maxPromptTokens: options.maxPromptTokens }),
    ...(options.maxTotalTokens === undefined ? {} : { maxTotalTokens: options.maxTotalTokens }),
  });

  function schedulerFor(prepared: PreparedGenerationRequest): SchedulerEntry {
    const key = schedulerKey(prepared);
    const existing = schedulers.get(key);
    if (existing !== undefined) {
      return existing;
    }

    const modelsByRequestId = new Map<string, string>();
    const scheduler = createContinuousBatchTokenScheduler(
      options.model,
      schedulerOptions(prepared, options, lane, modelsByRequestId, admissionController),
    );
    const entry = { scheduler, modelsByRequestId };
    schedulers.set(key, entry);
    return entry;
  }

  async function generateContinuous(
    prepared: PreparedGenerationRequest,
  ): Promise<NormalizedGenerationResult> {
    const entry = schedulerFor(prepared);
    entry.modelsByRequestId.set(prepared.request.id, prepared.request.model);
    emitGenerationProgress(options, prepared.request, prepared.promptTokens, 0);
    const onPrefillProgress = createPrefillProgressReporter(
      options,
      prepared.request,
      prepared.promptTokens,
    );
    const onToken = createProgressReporter(options, prepared.request, prepared.promptTokens);
    try {
      const result = await entry.scheduler.enqueue({
        id: prepared.request.id,
        promptTokenIds: prepared.tokenIds,
        maxTokens: prepared.request.sampling.maxTokens,
        ...(prepared.request.abortSignal === undefined
          ? {}
          : { abortSignal: prepared.request.abortSignal }),
        onPrefillProgress,
        onToken,
      });
      return generatedResultToServeResult(prepared, options, result);
    } finally {
      entry.modelsByRequestId.delete(prepared.request.id);
    }
  }

  async function* streamContinuous(
    prepared: PreparedGenerationRequest,
  ): AsyncIterable<GenerationStreamEvent> {
    const entry = schedulerFor(prepared);
    const abortScope = linkAbortSignals(prepared.request.abortSignal);
    const state: StreamQueueState = {
      events: [],
      notify: null,
      done: false,
      error: undefined,
    };
    const decodeState = createStreamingDecodeState(prepared.prompt);
    const decodeInterval = streamDecodeInterval(options, prepared.request.sampling.stop);
    entry.modelsByRequestId.set(prepared.request.id, prepared.request.model);
    emitGenerationProgress(options, prepared.request, prepared.promptTokens, 0);
    const onPrefillProgress = createPrefillProgressReporter(
      options,
      prepared.request,
      prepared.promptTokens,
    );
    const scheduled = entry.scheduler
      .enqueue({
        id: prepared.request.id,
        promptTokenIds: prepared.tokenIds,
        maxTokens: prepared.request.sampling.maxTokens,
        abortSignal: abortScope.signal,
        onPrefillProgress,
        onToken(tokenId, generatedTokenIds) {
          const text = handleStreamingTokenDelta(
            prepared.request,
            options,
            prepared.promptTokens,
            decodeInterval,
            decodeState,
            tokenId,
            generatedTokenIds.length,
          );
          if (text !== undefined) {
            enqueueStreamEvent(state, { type: "text", text });
          }
        },
      })
      .then((result) => {
        const finished = handleStreamingDone(
          options,
          prepared.promptTokens,
          decodeState,
          result.tokenIds,
          result.finishReason,
        );
        if (finished.text !== undefined) {
          enqueueStreamEvent(state, { type: "text", text: finished.text });
        }
        enqueueStreamEvent(state, finished.done);
        finishStreamQueue(state);
      })
      .catch((error) => finishStreamQueue(state, error));

    try {
      yield* readStreamQueue(state);
    } finally {
      if (!state.done) {
        abortScope.abort();
      }
      await scheduled.catch(() => undefined);
      abortScope.dispose();
      entry.modelsByRequestId.delete(prepared.request.id);
    }
  }

  return {
    generate(request) {
      const decision = routeDecisionForRequest(request, options);
      if (!canUseContinuousBatchGeneration(request, options)) {
        emitGenerationRouteDecision(
          options,
          request,
          decision.route,
          decision.eligible,
          decision.reason,
        );
        return null;
      }
      emitGenerationRouteDecision(options, request, "continuous", true, "eligible");
      return generateContinuous(prepareGenerationRequest(request, options));
    },
    stream(request) {
      const reason = continuousBatchIneligibilityReason(request, options);
      if (reason !== "eligible" || maxBatchSize(options) <= 1) {
        emitGenerationRouteDecision(
          options,
          request,
          "single",
          false,
          reason === "eligible" ? "max_batch_size" : reason,
        );
        return null;
      }
      emitGenerationRouteDecision(options, request, "continuous", true, "eligible");
      return streamContinuous(prepareGenerationRequest(request, options));
    },
  };
}
