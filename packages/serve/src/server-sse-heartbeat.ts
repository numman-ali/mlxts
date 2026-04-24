/**
 * SSE heartbeat helpers for long-running model streams.
 * @module
 */

const HEARTBEAT_INTERVAL_MS = 5_000;

function encodeSse(payload: string): Uint8Array {
  return new TextEncoder().encode(payload);
}

/** Enqueue an SSE comment frame that protocol clients safely ignore. */
export function enqueueSseComment(
  controller: ReadableStreamDefaultController<Uint8Array>,
  text: string,
): void {
  controller.enqueue(encodeSse(`: ${text}\n\n`));
}

/** Keep an SSE stream active while waiting for the model iterator to produce output. */
export async function withSseHeartbeat<T>(
  controller: ReadableStreamDefaultController<Uint8Array>,
  work: () => Promise<T>,
): Promise<T> {
  const heartbeat = setInterval(() => {
    try {
      enqueueSseComment(controller, "mlxts-serve keep-alive");
    } catch {
      clearInterval(heartbeat);
    }
  }, HEARTBEAT_INTERVAL_MS);
  try {
    return await work();
  } finally {
    clearInterval(heartbeat);
  }
}
