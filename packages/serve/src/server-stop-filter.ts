/**
 * Incremental stop-sequence filtering for text streams.
 * @module
 */

/** Create a streaming filter that withholds enough tail text to match stop sequences safely. */
export function createStopSequenceFilter(stop: readonly string[] | undefined): {
  push(text: string): { text: string; stopped: boolean };
  finish(): { text: string; stopped: boolean };
} {
  const sequences = (stop ?? []).filter((sequence) => sequence !== "");
  if (sequences.length === 0) {
    return {
      push(text) {
        return { text, stopped: false };
      },
      finish() {
        return { text: "", stopped: false };
      },
    };
  }

  const maxSequenceLength = Math.max(...sequences.map((sequence) => sequence.length));
  let buffer = "";

  return {
    push(text) {
      buffer += text;
      const matchIndexes = sequences
        .map((sequence) => buffer.indexOf(sequence))
        .filter((index) => index >= 0)
        .sort((left, right) => left - right);
      const firstMatch = matchIndexes[0];
      if (firstMatch !== undefined) {
        const emitted = buffer.slice(0, firstMatch);
        buffer = "";
        return { text: emitted, stopped: true };
      }

      const safeLength = Math.max(0, buffer.length - (maxSequenceLength - 1));
      if (safeLength === 0) {
        return { text: "", stopped: false };
      }
      const emitted = buffer.slice(0, safeLength);
      buffer = buffer.slice(safeLength);
      return { text: emitted, stopped: false };
    },
    finish() {
      const matchIndexes = sequences
        .map((sequence) => buffer.indexOf(sequence))
        .filter((index) => index >= 0)
        .sort((left, right) => left - right);
      const firstMatch = matchIndexes[0];
      const emitted = firstMatch === undefined ? buffer : buffer.slice(0, firstMatch);
      buffer = "";
      return { text: emitted, stopped: firstMatch !== undefined };
    },
  };
}
