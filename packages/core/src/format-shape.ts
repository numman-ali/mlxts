/**
 * Format a tensor shape for human-readable errors.
 *
 * @module
 */

/** Convert a shape into the repo's canonical `[a, b, c]` display format. */
export function formatShape(shape: readonly number[]): string {
  return shape.length === 0 ? "[]" : `[${shape.join(", ")}]`;
}
