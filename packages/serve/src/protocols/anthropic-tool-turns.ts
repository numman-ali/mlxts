/**
 * Anthropic tool-turn validation.
 * @module
 */

import { ServeError } from "../errors";

export type AnthropicToolTurnMessage = {
  toolUseIds: readonly string[];
  toolResultIds: readonly string[];
};

function assertUniqueIds(ids: readonly string[], message: string): void {
  if (new Set(ids).size !== ids.length) {
    throw new ServeError(message, { param: "messages" });
  }
}

function assertToolResultsMatch(
  toolUseIds: readonly string[],
  toolResultIds: readonly string[],
): void {
  assertUniqueIds(toolUseIds, "Anthropic messages: assistant tool_use ids must be unique.");
  assertUniqueIds(
    toolResultIds,
    "Anthropic messages: tool_result blocks must not repeat tool_use_id values.",
  );

  const expected = new Set(toolUseIds);
  if (expected.size !== toolResultIds.length || !toolResultIds.every((id) => expected.has(id))) {
    throw new ServeError(
      "Anthropic messages: tool_result blocks must match the immediately preceding assistant tool_use ids.",
      { param: "messages" },
    );
  }
}

/** Validate Anthropic's immediate assistant tool_use to user tool_result transcript shape. */
export function validateAnthropicToolTurns(messages: readonly AnthropicToolTurnMessage[]): void {
  for (let index = 0; index < messages.length; index += 1) {
    const message = messages[index];
    if (message === undefined) {
      continue;
    }
    if (message.toolUseIds.length > 0) {
      const next = messages[index + 1];
      if (next === undefined || next.toolResultIds.length === 0) {
        throw new ServeError(
          "Anthropic messages: tool_result blocks must immediately follow the assistant tool_use message.",
          { param: "messages" },
        );
      }
      assertToolResultsMatch(message.toolUseIds, next.toolResultIds);
    }
    if (message.toolResultIds.length > 0) {
      const previous = messages[index - 1];
      if (previous === undefined || previous.toolUseIds.length === 0) {
        throw new ServeError(
          "Anthropic messages: tool_result blocks must immediately follow the assistant tool_use message.",
          { param: "messages" },
        );
      }
      assertToolResultsMatch(previous.toolUseIds, message.toolResultIds);
    }
  }
}
