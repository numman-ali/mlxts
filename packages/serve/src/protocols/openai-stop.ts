/**
 * Shared OpenAI stop-sequence parsing helpers.
 * @module
 */

import { ServeError } from "../errors";

export function parseOpenAIStopSequences(
  record: Record<string, unknown>,
  protocolName: string,
): readonly string[] | undefined {
  const value = record.stop;
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value === "string") {
    return [value];
  }
  if (Array.isArray(value) && value.every((entry) => typeof entry === "string")) {
    if (value.length > 4) {
      throw new ServeError(`OpenAI ${protocolName}: "stop" can contain at most 4 sequences.`, {
        param: "stop",
      });
    }
    return value;
  }
  throw new ServeError(`OpenAI ${protocolName}: "stop" must be a string, string array, or null.`, {
    param: "stop",
  });
}
