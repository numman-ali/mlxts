/**
 * Abort-signal helpers for request-scoped serving work.
 * @module
 */

import type { NormalizedGenerationRequest } from "./types";

export type LinkedAbortSignal = {
  signal: AbortSignal;
  abort(): void;
  dispose(): void;
};

/** Link several parent signals into one request-local signal with explicit cleanup. */
export function linkAbortSignals(
  ...parents: readonly (AbortSignal | undefined)[]
): LinkedAbortSignal {
  const controller = new AbortController();
  const cleanups: Array<() => void> = [];
  const abort = (parent: AbortSignal) => {
    controller.abort(parent.reason);
  };

  for (const parent of parents) {
    if (parent === undefined) {
      continue;
    }
    if (parent.aborted) {
      abort(parent);
      continue;
    }
    const onAbort = () => abort(parent);
    parent.addEventListener("abort", onAbort, { once: true });
    cleanups.push(() => parent.removeEventListener("abort", onAbort));
  }

  return {
    signal: controller.signal,
    abort() {
      controller.abort();
    },
    dispose() {
      for (const cleanup of cleanups) {
        cleanup();
      }
    },
  };
}

/** Attach a request-local abort signal to a normalized generation request. */
export function withAbortSignal(
  request: NormalizedGenerationRequest,
  signal: AbortSignal,
): NormalizedGenerationRequest {
  return { ...request, abortSignal: signal };
}
