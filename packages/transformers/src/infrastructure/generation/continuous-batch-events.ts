import type { GenerationResult } from "../../types";

export type ContinuousBatchQueueSnapshot = {
  waiting: number;
  prefilling: number;
  active: number;
  maxBatchSize: number;
};

export type ContinuousBatchEvent = ContinuousBatchQueueSnapshot & {
  ids: readonly string[];
  batchSize: number;
  maxTokens: number;
  maxTokensByRequest: readonly number[];
};

type ContinuousBatchSchedulerTiming = {
  schedulerMs: number;
};

type ContinuousBatchRequestTiming = ContinuousBatchSchedulerTiming & {
  queuedMs: number;
};

export type ContinuousBatchSchedulerEvent =
  | (ContinuousBatchQueueSnapshot &
      ContinuousBatchSchedulerTiming & {
        type: "queued";
        id: string;
        queuedAhead: number;
        promptTokens: number;
        maxTokens: number;
      })
  | (ContinuousBatchQueueSnapshot &
      ContinuousBatchRequestTiming & {
        type: "prefill_start";
        id: string;
        promptTokens: number;
        maxTokens: number;
      })
  | (ContinuousBatchQueueSnapshot &
      ContinuousBatchSchedulerTiming & {
        type: "admitted";
        ids: readonly string[];
        batchSize: number;
        maxTokens: number;
        maxTokensByRequest: readonly number[];
        queuedMsByRequest: readonly number[];
      })
  | (ContinuousBatchQueueSnapshot &
      ContinuousBatchRequestTiming & {
        type: "first_token";
        id: string;
        completionTokens: number;
      })
  | (ContinuousBatchQueueSnapshot &
      ContinuousBatchRequestTiming & {
        type: "finished";
        id: string;
        completionTokens: number;
        finishReason: GenerationResult["finishReason"];
      })
  | (ContinuousBatchQueueSnapshot &
      ContinuousBatchRequestTiming & {
        type: "cancelled";
        id: string;
        completionTokens: number;
      });
