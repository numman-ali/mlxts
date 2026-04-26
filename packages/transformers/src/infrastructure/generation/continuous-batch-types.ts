import type { BatchGenerationOptions, GenerationResult, PrefillProgressEvent } from "../../types";
import type { BatchKVCache } from "../cache";
import type {
  ContinuousBatchEvent,
  ContinuousBatchSchedulerEvent,
} from "./continuous-batch-events";

export type RunExclusive = <T>(work: () => Promise<T>) => Promise<T>;

export const defaultRunExclusive: RunExclusive = (work) => work();

export type ScheduledRequest = {
  id: string;
  promptTokenIds: readonly number[];
  maxTokens: number;
  enqueuedAtMs: number;
  generated: number[];
  firstTokenEmitted: boolean;
  finishReason: GenerationResult["finishReason"];
  abortSignal?: AbortSignal;
  onAbort?: () => void;
  onPrefillProgress?: (event: PrefillProgressEvent) => void;
  onToken?: (tokenId: number, generatedTokenIds: readonly number[]) => void;
  resolve(result: GenerationResult): void;
  reject(error: unknown): void;
};

export type PrefillingRequest = {
  request: ScheduledRequest;
  cache: BatchKVCache;
  cursor: number;
};

export type ContinuousBatchTokenSchedulerOptions = Omit<
  BatchGenerationOptions,
  "maxTokens" | "abortSignal"
> & {
  /** Maximum active rows admitted into one decode step. */
  maxBatchSize?: number;
  /** Delay before starting a fresh scheduler loop so nearby requests can join. */
  batchWindowMs?: number;
  /** Optional model-lane guard shared with fallback generation. */
  runExclusive?: RunExclusive;
  /** Emitted whenever the scheduler admits rows into the active decode batch. */
  onBatch?: (event: ContinuousBatchEvent) => void;
  /** Emitted for scheduler-owned queue, prefill, admission, decode, and finish phases. */
  onSchedulerEvent?: (event: ContinuousBatchSchedulerEvent) => void;
};

export type ContinuousBatchTokenRequest = {
  id?: string;
  promptTokenIds: readonly number[];
  maxTokens: number;
  abortSignal?: AbortSignal;
  onPrefillProgress?: (event: PrefillProgressEvent) => void;
  onToken?: (tokenId: number, generatedTokenIds: readonly number[]) => void;
};
