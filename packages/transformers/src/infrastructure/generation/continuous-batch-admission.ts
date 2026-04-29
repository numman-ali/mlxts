import type { ContinuousBatchQueueSnapshot } from "./continuous-batch-events";
import type {
  ContinuousBatchAdmissionController,
  ContinuousBatchAdmissionDecision,
  ContinuousBatchAdmissionRequest,
  PrefillingRequest,
  ScheduledRequest,
} from "./continuous-batch-types";

type SchedulableRequest = {
  promptTokenIds: readonly number[];
  maxTokens: number;
};

function requestTotalTokens(request: SchedulableRequest): number {
  return request.promptTokenIds.length + request.maxTokens;
}

function requestPromptTokens(request: SchedulableRequest): number {
  return request.promptTokenIds.length;
}

function requestCompletionTokens(request: SchedulableRequest): number {
  return request.maxTokens;
}

function sumTotalTokens(requests: readonly SchedulableRequest[]): number {
  return requests.reduce((total, request) => total + requestTotalTokens(request), 0);
}

function sumPromptTokens(requests: readonly SchedulableRequest[]): number {
  return requests.reduce((total, request) => total + requestPromptTokens(request), 0);
}

function sumCompletionTokens(requests: readonly SchedulableRequest[]): number {
  return requests.reduce((total, request) => total + requestCompletionTokens(request), 0);
}

function prefillingTotalTokens(prefilling: readonly PrefillingRequest[]): number {
  return prefilling.reduce((total, row) => total + requestTotalTokens(row.request), 0);
}

function prefillingPromptTokens(prefilling: readonly PrefillingRequest[]): number {
  return prefilling.reduce((total, row) => total + requestPromptTokens(row.request), 0);
}

function prefillingCompletionTokens(prefilling: readonly PrefillingRequest[]): number {
  return prefilling.reduce((total, row) => total + requestCompletionTokens(row.request), 0);
}

export function admissionRequestFor(request: ScheduledRequest): ContinuousBatchAdmissionRequest {
  const promptTokens = request.promptTokenIds.length;
  return {
    id: request.id,
    promptTokens,
    maxTokens: request.maxTokens,
    totalTokens: promptTokens + request.maxTokens,
  };
}

type DeferredTelemetry = {
  deferred(
    request: ScheduledRequest,
    reason: Extract<ContinuousBatchAdmissionDecision, { type: "deferred" }>["reason"],
    snapshot: ContinuousBatchQueueSnapshot,
  ): void;
};

type AdmissionCleanup = (request: ScheduledRequest) => void;

function tryReserveAdmission(
  request: ScheduledRequest,
  admissionController: ContinuousBatchAdmissionController | undefined,
  snapshot: () => ContinuousBatchQueueSnapshot,
  telemetry: DeferredTelemetry,
  cleanup: AdmissionCleanup,
): "reserved" | "deferred" | "rejected" {
  if (request.admissionReservation !== undefined || admissionController === undefined) {
    request.admissionDeferred = false;
    return "reserved";
  }
  const decision = admissionController.tryReserve(admissionRequestFor(request));
  if (decision.type === "reserved") {
    request.admissionReservation = decision.reservation;
    request.admissionDeferred = false;
    return "reserved";
  }
  if (decision.type === "rejected") {
    cleanup(request);
    request.reject(new Error(decision.message));
    return "rejected";
  }
  if (!request.admissionDeferred) {
    request.admissionDeferred = true;
    telemetry.deferred(request, decision.reason, snapshot());
  }
  return "deferred";
}

export function takeAdmittableWaitingRows(
  waiting: ScheduledRequest[],
  limit: number,
  admissionController: ContinuousBatchAdmissionController | undefined,
  snapshot: () => ContinuousBatchQueueSnapshot,
  telemetry: DeferredTelemetry,
  cleanup: AdmissionCleanup,
): ScheduledRequest[] {
  const admitted: ScheduledRequest[] = [];
  for (let index = 0; index < waiting.length && admitted.length < limit; ) {
    const request = waiting[index];
    if (request === undefined) {
      break;
    }
    const decision = tryReserveAdmission(
      request,
      admissionController,
      snapshot,
      telemetry,
      cleanup,
    );
    if (decision === "reserved") {
      waiting.splice(index, 1);
      admitted.push(request);
      continue;
    }
    if (decision === "rejected") {
      waiting.splice(index, 1);
      continue;
    }
    index += 1;
  }
  return admitted;
}

export function continuousBatchQueueSnapshot(
  waiting: readonly ScheduledRequest[],
  prefilling: readonly PrefillingRequest[],
  active: readonly ScheduledRequest[],
  maxBatchSize: number,
  admissionController: ContinuousBatchAdmissionController | undefined,
): ContinuousBatchQueueSnapshot {
  const waitingTotalTokens = sumTotalTokens(waiting);
  const prefillingTokens = prefillingTotalTokens(prefilling);
  const activeTotalTokens = sumTotalTokens(active);
  const prefillingPrompt = prefillingPromptTokens(prefilling);
  const activePromptTokens = sumPromptTokens(active);
  const prefillingCompletion = prefillingCompletionTokens(prefilling);
  const activeCompletionTokens = sumCompletionTokens(active);
  const admission = admissionController?.snapshot();
  const scheduledPromptTokens =
    admission?.scheduledPromptTokens ?? prefillingPrompt + activePromptTokens;
  const scheduledCompletionTokens =
    admission?.scheduledCompletionTokens ?? prefillingCompletion + activeCompletionTokens;

  return {
    waiting: waiting.length,
    prefilling: prefilling.length,
    active: active.length,
    maxBatchSize,
    waitingTotalTokens,
    prefillingTotalTokens: prefillingTokens,
    activeTotalTokens,
    scheduledPromptTokens,
    maxScheduledPromptTokens: admission?.maxScheduledPromptTokens ?? null,
    scheduledCompletionTokens,
    maxScheduledCompletionTokens: admission?.maxScheduledCompletionTokens ?? null,
    scheduledTotalTokens: admission?.scheduledTotalTokens ?? prefillingTokens + activeTotalTokens,
    maxScheduledTotalTokens: admission?.maxScheduledTotalTokens ?? null,
    scheduledMemoryBytes: admission?.scheduledMemoryBytes ?? 0,
    maxScheduledMemoryBytes: admission?.maxScheduledMemoryBytes ?? null,
  };
}
