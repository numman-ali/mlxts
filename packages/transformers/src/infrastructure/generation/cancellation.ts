/**
 * Cooperative generation cancellation helpers.
 * @module
 */

/** Raised when generation is cancelled through an AbortSignal. */
export class GenerationAbortError extends Error {
  constructor(message = "generation was cancelled") {
    super(message);
    this.name = "GenerationAbortError";
  }
}

/** Throw a typed cancellation error when the caller's abort signal is set. */
export function throwIfGenerationAborted(signal: AbortSignal | undefined, context: string): void {
  if (signal?.aborted) {
    throw new GenerationAbortError(`${context}: generation was cancelled.`);
  }
}
