import type { MxArray } from "@mlxts/core";

import type { TransformerBatchCache } from "../../types";
import { GenerationAbortError } from "./cancellation";
import type { ContinuousBatchQueueSnapshot } from "./continuous-batch-events";
import type { PrefillingRequest, ScheduledRequest } from "./continuous-batch-types";

type CancellationTelemetry = {
  cancelled(request: ScheduledRequest, snapshot: ContinuousBatchQueueSnapshot): void;
};

type AbortState = {
  waiting: ScheduledRequest[];
  prefilling: PrefillingRequest[];
  telemetry: CancellationTelemetry;
  snapshot(): ContinuousBatchQueueSnapshot;
  cleanup(request: ScheduledRequest): void;
};

function cancellationError(): GenerationAbortError {
  return new GenerationAbortError("ContinuousBatchTokenScheduler: generation was cancelled.");
}

export function cleanupScheduledRequest(entry: ScheduledRequest): void {
  if (entry.onAbort !== undefined) {
    entry.abortSignal?.removeEventListener("abort", entry.onAbort);
    delete entry.onAbort;
  }
  entry.admissionReservation?.[Symbol.dispose]();
  delete entry.admissionReservation;
  entry.samplerState[Symbol.dispose]();
}

export function attachScheduledRequestAbort(entry: ScheduledRequest, state: AbortState): void {
  if (entry.abortSignal === undefined) {
    return;
  }
  entry.onAbort = () => {
    const index = state.waiting.indexOf(entry);
    if (index >= 0) {
      state.waiting.splice(index, 1);
      state.telemetry.cancelled(entry, state.snapshot());
      state.cleanup(entry);
      entry.reject(cancellationError());
    }
    const prefillIndex = state.prefilling.findIndex((row) => row.request === entry);
    if (prefillIndex >= 0) {
      const [prefilling] = state.prefilling.splice(prefillIndex, 1);
      prefilling?.cache[Symbol.dispose]();
      state.telemetry.cancelled(entry, state.snapshot());
      state.cleanup(entry);
      entry.reject(cancellationError());
    }
  };
  entry.abortSignal.addEventListener("abort", entry.onAbort, { once: true });
}

export function failScheduledRequests(
  error: unknown,
  state: {
    cancelAdmissionWakeup(): void;
    waiting: ScheduledRequest[];
    prefilling: PrefillingRequest[];
    active: ScheduledRequest[];
    currentToken: MxArray | null;
    cache: TransformerBatchCache | null;
  },
): {
  active: ScheduledRequest[];
  currentToken: MxArray | null;
  cache: TransformerBatchCache | null;
} {
  state.cancelAdmissionWakeup();
  for (const request of [
    ...state.waiting,
    ...state.prefilling.map((prefilling) => prefilling.request),
    ...state.active,
  ]) {
    cleanupScheduledRequest(request);
    request.reject(error);
  }
  state.waiting.length = 0;
  for (const row of state.prefilling.splice(0)) {
    row.cache[Symbol.dispose]();
  }
  state.currentToken?.free();
  state.cache?.[Symbol.dispose]();
  return { active: [], currentToken: null, cache: null };
}
