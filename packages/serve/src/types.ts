/**
 * Protocol-neutral generation contracts for serving adapters.
 * @module
 */

import type { ChatMessage, ChatTool } from "@mlxts/transformers";

export type GenerationInput =
  | {
      kind: "text";
      text: string;
    }
  | {
      kind: "tokens";
      tokenIds: readonly number[];
    }
  | {
      kind: "messages";
      messages: readonly ChatMessage[];
      tools?: readonly ChatTool[];
      chatTemplate?: {
        enableThinking?: boolean;
        preserveThinking?: boolean;
      };
    };

export type GenerationSamplingOptions = {
  maxTokens: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  seed?: number;
  stop?: readonly string[];
  ignoreEos?: boolean;
};

export type GenerationProtocol =
  | "openai.completions"
  | "openai.chat_completions"
  | "openai.responses"
  | "anthropic.messages";

export type NormalizedGenerationRequest = {
  id: string;
  model: string;
  input: GenerationInput;
  sampling: GenerationSamplingOptions;
  stream: boolean;
  protocol: GenerationProtocol;
  abortSignal?: AbortSignal;
  metadata?: Record<string, unknown>;
};

export type GenerationUsage = {
  promptTokens?: number;
  completionTokens?: number;
  totalTokens?: number;
  cacheReadTokens?: number;
  cacheWriteTokens?: number;
};

export type GenerationMemoryUsage = {
  activeBytes: number;
  cacheBytes: number;
  peakBytes: number;
  limitBytes: number;
};

export type NormalizedFinishReason = "stop" | "length" | "eos" | "cancelled" | "error";

export type NormalizedGenerationResult = {
  text: string;
  reasoningContent?: string;
  finishReason: NormalizedFinishReason;
  tokenIds?: readonly number[];
  usage?: GenerationUsage;
};

export type GenerationRoute = "single" | "static" | "continuous";

export type GenerationRouteDecisionReason =
  | "eligible"
  | "streaming"
  | "max_batch_size"
  | "single_request_group"
  | "unsupported_model_type"
  | "sliding_window_cache"
  | "sampled_generation"
  | "repetition_penalty";

type ContinuousSchedulerCounts = {
  waiting: number;
  prefilling: number;
  active: number;
  maxBatchSize: number;
  waitingTotalTokens: number;
  prefillingTotalTokens: number;
  activeTotalTokens: number;
  scheduledPromptTokens: number;
  maxScheduledPromptTokens: number | null;
  scheduledCompletionTokens: number;
  maxScheduledCompletionTokens: number | null;
  scheduledTotalTokens: number;
  maxScheduledTotalTokens: number | null;
};

type ContinuousSchedulerTiming = {
  schedulerMs: number;
};

type ContinuousSchedulerRequestTiming = ContinuousSchedulerTiming & {
  queuedMs: number;
};

type ContinuousSchedulerServeEvent =
  | (ContinuousSchedulerCounts &
      ContinuousSchedulerTiming & {
        type: "generation_scheduler_phase";
        mode: "continuous";
        phase: "queued";
        model: string;
        id: string;
        ids: readonly string[];
        queuedAhead: number;
        promptTokens: number;
        maxTokens: number;
      })
  | (ContinuousSchedulerCounts &
      ContinuousSchedulerRequestTiming & {
        type: "generation_scheduler_phase";
        mode: "continuous";
        phase: "deferred";
        model: string;
        id: string;
        ids: readonly string[];
        reason:
          | "scheduled_prompt_budget"
          | "scheduled_completion_budget"
          | "scheduled_token_budget";
        promptTokens: number;
        maxTokens: number;
      })
  | (ContinuousSchedulerCounts &
      ContinuousSchedulerRequestTiming & {
        type: "generation_scheduler_phase";
        mode: "continuous";
        phase: "prefill_start";
        model: string;
        id: string;
        ids: readonly string[];
        promptTokens: number;
        maxTokens: number;
      })
  | (ContinuousSchedulerCounts &
      ContinuousSchedulerTiming & {
        type: "generation_scheduler_phase";
        mode: "continuous";
        phase: "admitted";
        model: string;
        ids: readonly string[];
        batchSize: number;
        maxTokens: number;
        maxTokensByRequest: readonly number[];
        queuedMsByRequest: readonly number[];
      })
  | (ContinuousSchedulerCounts &
      ContinuousSchedulerRequestTiming & {
        type: "generation_scheduler_phase";
        mode: "continuous";
        phase: "first_token";
        model: string;
        id: string;
        ids: readonly string[];
        completionTokens: number;
      })
  | (ContinuousSchedulerCounts &
      ContinuousSchedulerRequestTiming & {
        type: "generation_scheduler_phase";
        mode: "continuous";
        phase: "finished";
        model: string;
        id: string;
        ids: readonly string[];
        completionTokens: number;
        finishReason: Extract<NormalizedFinishReason, "length" | "eos">;
      })
  | (ContinuousSchedulerCounts &
      ContinuousSchedulerRequestTiming & {
        type: "generation_scheduler_phase";
        mode: "continuous";
        phase: "cancelled";
        model: string;
        id: string;
        ids: readonly string[];
        completionTokens: number;
      });

export type GenerationStreamEvent =
  | {
      type: "text";
      text: string;
    }
  | {
      type: "done";
      finishReason: NormalizedFinishReason;
      usage?: GenerationUsage;
    };

export type ServeEvent =
  | {
      type: "request_start";
      method: string;
      path: string;
    }
  | {
      type: "request_complete";
      method: string;
      path: string;
      status: number;
      durationMs: number;
    }
  | {
      type: "request_error";
      method: string;
      path: string;
      message: string;
      code: string;
      status: number;
      durationMs: number;
    }
  | {
      type: "generation_start";
      id: string;
      protocol: GenerationProtocol;
      model: string;
      inputKind: GenerationInput["kind"];
      maxTokens: number;
      stream: boolean;
    }
  | {
      type: "generation_route_decision";
      id: string;
      protocol: GenerationProtocol;
      model: string;
      route: GenerationRoute;
      eligible: boolean;
      reason: GenerationRouteDecisionReason;
      modelType: string;
      maxBatchSize: number;
      schedulerMode: "auto";
      cacheBackend: "managed";
      attentionBackend: "auto";
      decodingBackend: "model";
      stream: boolean;
    }
  | {
      type: "generation_model_lane_wait";
      id: string;
      protocol: GenerationProtocol;
      model: string;
      lane: "model";
      waitMs: number;
      inFlightAtQueue: number;
      queuedAhead: number;
      inFlightAtDispatch: number;
      queuedAtDispatch: number;
      maxConcurrentJobs: number;
    }
  | {
      type: "generation_progress";
      id: string;
      protocol: GenerationProtocol;
      model: string;
      promptTokens: number;
      completionTokens: number;
      maxTokens: number;
      memory?: GenerationMemoryUsage;
    }
  | {
      type: "generation_prefill_progress";
      id: string;
      protocol: GenerationProtocol;
      model: string;
      promptTokens: number;
      processedPrefillTokens: number;
      totalPrefillTokens: number;
      chunkTokens: number;
      maxTokens: number;
      memory?: GenerationMemoryUsage;
    }
  | {
      type: "generation_stream_chunk";
      id: string;
      protocol: GenerationProtocol;
      model: string;
      chunkIndex: number;
      elapsedMs: number;
      bytes: number;
    }
  | {
      type: "generation_stream_end";
      id: string;
      protocol: GenerationProtocol;
      model: string;
      result: "completed" | "cancelled" | "error";
      finishReason: NormalizedFinishReason;
      chunks: number;
      bytes: number;
      outputChunks: number;
      outputBytes: number;
      ttftMs?: number;
      durationMs: number;
    }
  | {
      type: "generation_batch_start";
      mode: "static" | "continuous";
      model: string;
      ids: readonly string[];
      batchSize: number;
      maxTokens: number;
      maxTokensByRequest: readonly number[];
    }
  | ContinuousSchedulerServeEvent
  | {
      type: "generation_admission_batch";
      mode: "micro";
      engineMode: "batch" | "sequential";
      model: string;
      ids: readonly string[];
      batchSize: number;
      maxTokens: number;
      maxTokensByRequest: readonly number[];
    }
  | {
      type: "generation_complete";
      id: string;
      protocol: GenerationProtocol;
      model: string;
      finishReason: NormalizedFinishReason;
      promptTokens?: number;
      completionTokens?: number;
      totalTokens?: number;
      durationMs: number;
      memory?: GenerationMemoryUsage;
    }
  | {
      type: "generation_error";
      id: string;
      protocol: GenerationProtocol;
      model: string;
      code: string;
      message: string;
      durationMs: number;
    };

export type GenerationEngine = {
  generate(
    request: NormalizedGenerationRequest,
  ): NormalizedGenerationResult | Promise<NormalizedGenerationResult>;
  generateBatch?(
    requests: readonly NormalizedGenerationRequest[],
  ): readonly NormalizedGenerationResult[] | Promise<readonly NormalizedGenerationResult[]>;
  stream?(
    request: NormalizedGenerationRequest,
  ): AsyncIterable<GenerationStreamEvent> | Promise<AsyncIterable<GenerationStreamEvent>>;
};
