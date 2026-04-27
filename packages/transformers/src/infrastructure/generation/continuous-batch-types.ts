import type {
  BatchGenerationOptions,
  GenerationResult,
  PrefillProgressEvent,
  SamplerOptions,
  TransformerBatchCache,
} from "../../types";
import type { SamplerState } from "../sampling";
import type {
  ContinuousBatchEvent,
  ContinuousBatchSchedulerEvent,
} from "./continuous-batch-events";

export type RunExclusive = <T>(work: () => Promise<T>) => Promise<T>;

export const defaultRunExclusive: RunExclusive = (work) => work();

export type ContinuousBatchAdmissionRequest = {
  id: string;
  promptTokens: number;
  maxTokens: number;
  totalTokens: number;
};

export type ContinuousBatchAdmissionBudgetSnapshot = {
  scheduledPromptTokens: number;
  maxScheduledPromptTokens: number | null;
  scheduledCompletionTokens: number;
  maxScheduledCompletionTokens: number | null;
  scheduledTotalTokens: number;
  maxScheduledTotalTokens: number | null;
};

export type ContinuousBatchAdmissionReservation = Disposable;

export type ContinuousBatchAdmissionDecision =
  | {
      type: "reserved";
      reservation: ContinuousBatchAdmissionReservation;
    }
  | (ContinuousBatchAdmissionBudgetSnapshot & {
      type: "deferred";
      reason: "scheduled_prompt_budget" | "scheduled_completion_budget" | "scheduled_token_budget";
    })
  | {
      type: "rejected";
      message: string;
    };

export type ContinuousBatchAdmissionController = {
  tryReserve(request: ContinuousBatchAdmissionRequest): ContinuousBatchAdmissionDecision;
  snapshot(): ContinuousBatchAdmissionBudgetSnapshot;
  onRelease(listener: () => void): () => void;
};

export type ScheduledRequest = {
  id: string;
  promptTokenIds: readonly number[];
  maxTokens: number;
  samplerOptions: SamplerOptions;
  samplerState: SamplerState;
  enqueuedAtMs: number;
  generated: number[];
  firstTokenEmitted: boolean;
  finishReason: GenerationResult["finishReason"];
  admissionDeferred: boolean;
  admissionReservation?: ContinuousBatchAdmissionReservation;
  abortSignal?: AbortSignal;
  onAbort?: () => void;
  onPrefillProgress?: (event: PrefillProgressEvent) => void;
  onToken?: (tokenId: number, generatedTokenIds: readonly number[]) => void;
  resolve(result: GenerationResult): void;
  reject(error: unknown): void;
};

export type PrefillingRequest = {
  request: ScheduledRequest;
  cache: TransformerBatchCache;
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
  /** Maximum prompt-prefill chunk size while another row is actively decoding. */
  activePrefillStepSize?: number;
  /** Active decode steps allowed between long prompt-prefill chunks. */
  activeDecodeStepsPerPrefillChunk?: number;
  /** Optional model-lane guard shared with fallback generation. */
  runExclusive?: RunExclusive;
  /** Optional shared admission budget for model-wide continuous scheduling. */
  admissionController?: ContinuousBatchAdmissionController;
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
