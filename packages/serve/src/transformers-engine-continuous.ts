/**
 * Continuous batching adapter for transformer-backed serving.
 * @module
 */

import {
  type ContinuousBatchSchedulerEvent,
  type ContinuousBatchTokenSchedulerOptions,
  createContinuousBatchTokenScheduler,
} from "@mlxts/transformers";
import type { ModelExecutionLane } from "./model-execution-lane";
import type { TransformersGenerationEngineOptions } from "./transformers-engine";
import {
  batchOptionsKey,
  generatedResultToServeResult,
  type PreparedGenerationRequest,
  prepareGenerationRequest,
} from "./transformers-engine-generation";
import {
  canUseStaticBatchGeneration,
  emitGenerationRouteDecision,
  routeDecisionForRequest,
} from "./transformers-engine-routing";
import {
  createPrefillProgressReporter,
  createProgressReporter,
  emitGenerationProgress,
} from "./transformers-engine-shared";
import type { NormalizedGenerationRequest, NormalizedGenerationResult } from "./types";

type SchedulerEntry = {
  scheduler: ReturnType<typeof createContinuousBatchTokenScheduler>;
  modelsByRequestId: Map<string, string>;
};

export type ContinuousTransformersGeneration = {
  generate(request: NormalizedGenerationRequest): Promise<NormalizedGenerationResult> | null;
};

function maxBatchSize(options: TransformersGenerationEngineOptions): number {
  return options.maxBatchSize ?? 1;
}

function canUseContinuousBatchGeneration(
  request: NormalizedGenerationRequest,
  options: TransformersGenerationEngineOptions,
): boolean {
  return maxBatchSize(options) > 1 && canUseStaticBatchGeneration(request, options);
}

function schedulerOptions(
  prepared: PreparedGenerationRequest,
  options: TransformersGenerationEngineOptions,
  lane: ModelExecutionLane,
  modelsByRequestId: Map<string, string>,
): ContinuousBatchTokenSchedulerOptions {
  const batchOptions = prepared.batchOptions;
  return {
    ...(batchOptions.temperature === undefined ? {} : { temperature: batchOptions.temperature }),
    ...(batchOptions.topK === undefined ? {} : { topK: batchOptions.topK }),
    ...(batchOptions.topP === undefined ? {} : { topP: batchOptions.topP }),
    ...(batchOptions.minP === undefined ? {} : { minP: batchOptions.minP }),
    ...(batchOptions.seed === undefined ? {} : { seed: batchOptions.seed }),
    ...(batchOptions.eosTokenIds === undefined ? {} : { eosTokenIds: batchOptions.eosTokenIds }),
    ...(batchOptions.useCache === undefined ? {} : { useCache: batchOptions.useCache }),
    ...(batchOptions.prefillStepSize === undefined
      ? {}
      : { prefillStepSize: batchOptions.prefillStepSize }),
    ...(batchOptions.padTokenId === undefined ? {} : { padTokenId: batchOptions.padTokenId }),
    maxBatchSize: maxBatchSize(options),
    batchWindowMs: options.batchWindowMs ?? 0,
    runExclusive: (work) => lane.run(work),
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
        waiting: event.waiting,
        prefilling: event.prefilling,
        active: event.active,
        maxBatchSize: event.maxBatchSize,
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
        waiting: event.waiting,
        prefilling: event.prefilling,
        active: event.active,
        maxBatchSize: event.maxBatchSize,
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
        waiting: event.waiting,
        prefilling: event.prefilling,
        active: event.active,
        maxBatchSize: event.maxBatchSize,
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
        waiting: event.waiting,
        prefilling: event.prefilling,
        active: event.active,
        maxBatchSize: event.maxBatchSize,
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
        waiting: event.waiting,
        prefilling: event.prefilling,
        active: event.active,
        maxBatchSize: event.maxBatchSize,
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
        waiting: event.waiting,
        prefilling: event.prefilling,
        active: event.active,
        maxBatchSize: event.maxBatchSize,
      });
  }
}

function schedulerKey(prepared: PreparedGenerationRequest): string {
  return batchOptionsKey(prepared.batchOptions);
}

/** Create the optional continuous-batching path for eligible transformer requests. */
export function createContinuousTransformersGeneration(
  options: TransformersGenerationEngineOptions,
  lane: ModelExecutionLane,
): ContinuousTransformersGeneration {
  const schedulers = new Map<string, SchedulerEntry>();

  function schedulerFor(prepared: PreparedGenerationRequest): SchedulerEntry {
    const key = schedulerKey(prepared);
    const existing = schedulers.get(key);
    if (existing !== undefined) {
      return existing;
    }

    const modelsByRequestId = new Map<string, string>();
    const scheduler = createContinuousBatchTokenScheduler(
      options.model,
      schedulerOptions(prepared, options, lane, modelsByRequestId),
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
  };
}
