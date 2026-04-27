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
