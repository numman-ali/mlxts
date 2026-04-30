/**
 * OpenResponses turn-item parsing for tool and reasoning history.
 * @module
 */

import type { ChatMessage, ChatToolCall } from "@mlxts/transformers";
import { isRecord, ServeError } from "../errors";
import type { GenerationContentMessage } from "../types";

export type OpenAIResponseParsedInputItem = {
  chat: readonly ChatMessage[];
  content: readonly GenerationContentMessage[];
  hasMedia: boolean;
  toolUseIds: readonly string[];
  toolResultIds: readonly string[];
};

type FunctionOutputTurn = {
  chat: ChatMessage;
  content: GenerationContentMessage;
  id: string;
};

function textContentPart(text: string) {
  return { kind: "text" as const, text };
}

function nonEmptyItemString(value: unknown, field: string, type: string): string {
  if (typeof value !== "string" || value.trim() === "") {
    throw new ServeError(`OpenAI responses: ${type} items require a non-empty "${field}".`, {
      param: "input",
    });
  }
  return value;
}

function responseFunctionCall(value: Record<string, unknown>): ChatToolCall {
  return {
    id: nonEmptyItemString(value.call_id, "call_id", "function_call"),
    type: "function",
    function: {
      name: nonEmptyItemString(value.name, "name", "function_call"),
      arguments: nonEmptyItemString(value.arguments, "arguments", "function_call"),
    },
  };
}

function responseReasoning(value: Record<string, unknown>): string {
  if (!Array.isArray(value.content) || value.content.length === 0) {
    throw new ServeError(
      "OpenAI responses: reasoning items require non-empty reasoning_text content.",
      { param: "input" },
    );
  }
  const text = value.content
    .map((part) => {
      if (!isRecord(part) || part.type !== "reasoning_text" || typeof part.text !== "string") {
        throw new ServeError("OpenAI responses: reasoning items require reasoning_text content.", {
          param: "input",
        });
      }
      return part.text;
    })
    .join("");
  if (text.trim() === "") {
    throw new ServeError(
      "OpenAI responses: reasoning items require non-empty reasoning_text content.",
      { param: "input" },
    );
  }
  return text;
}

function responseFunctionOutput(value: Record<string, unknown>): FunctionOutputTurn {
  const id = nonEmptyItemString(value.call_id, "call_id", "function_call_output");
  if (typeof value.output !== "string") {
    throw new ServeError(
      "OpenAI responses: function_call_output currently supports string output only.",
      { param: "input" },
    );
  }
  return {
    id,
    chat: { role: "tool", content: value.output, tool_call_id: id },
    content: {
      role: "tool",
      content: value.output === "" ? [] : [textContentPart(value.output)],
      tool_call_id: id,
    },
  };
}

function functionCallMessage(
  toolCalls: readonly ChatToolCall[],
  reasoningContent: string | undefined,
): OpenAIResponseParsedInputItem {
  const toolUseIds = toolCalls
    .map((toolCall) => toolCall.id)
    .filter((id): id is string => id !== undefined);
  return {
    chat: [
      {
        role: "assistant",
        content: "",
        ...(reasoningContent === undefined ? {} : { reasoning_content: reasoningContent }),
        tool_calls: toolCalls,
      },
    ],
    content: [
      {
        role: "assistant",
        content: [],
        ...(reasoningContent === undefined ? {} : { reasoning_content: reasoningContent }),
        tool_calls: toolCalls,
      },
    ],
    hasMedia: false,
    toolUseIds,
    toolResultIds: [],
  };
}

function functionOutputMessage(
  outputs: readonly FunctionOutputTurn[],
): OpenAIResponseParsedInputItem {
  return {
    chat: outputs.map((output) => output.chat),
    content: outputs.map((output) => output.content),
    hasMedia: false,
    toolUseIds: [],
    toolResultIds: outputs.map((output) => output.id),
  };
}

function appendReasoningContent(current: string | undefined, next: string): string {
  return current === undefined || current === "" ? next : `${current}\n${next}`;
}

function assistantMessageWithReasoning(
  message: OpenAIResponseParsedInputItem,
  reasoningContent: string,
): OpenAIResponseParsedInputItem {
  const chat = message.chat[0];
  const content = message.content[0];
  if (
    message.chat.length !== 1 ||
    message.content.length !== 1 ||
    chat === undefined ||
    content === undefined ||
    chat.role !== "assistant" ||
    content.role !== "assistant"
  ) {
    throw new ServeError(
      "OpenAI responses: reasoning items must immediately precede assistant output items.",
      { param: "input" },
    );
  }
  return {
    ...message,
    chat: [
      {
        ...chat,
        reasoning_content: appendReasoningContent(chat.reasoning_content, reasoningContent),
      },
    ],
    content: [
      {
        ...content,
        reasoning_content: appendReasoningContent(content.reasoning_content, reasoningContent),
      },
    ],
  };
}

function assertUniqueIds(ids: readonly string[], message: string): void {
  if (new Set(ids).size !== ids.length) {
    throw new ServeError(message, { param: "input" });
  }
}

function assertToolResultsMatch(
  toolUseIds: readonly string[],
  toolResultIds: readonly string[],
): void {
  assertUniqueIds(toolUseIds, "OpenAI responses: function_call call_ids must be unique.");
  assertUniqueIds(
    toolResultIds,
    "OpenAI responses: function_call_output items must not repeat call_id values.",
  );
  const expected = new Set(toolUseIds);
  if (expected.size !== toolResultIds.length || !toolResultIds.every((id) => expected.has(id))) {
    throw new ServeError(
      "OpenAI responses: function_call_output items must match the immediately preceding function_call ids.",
      { param: "input" },
    );
  }
}

function validateToolTurns(messages: readonly OpenAIResponseParsedInputItem[]): void {
  for (let index = 0; index < messages.length; index += 1) {
    const message = messages[index];
    if (message === undefined) {
      continue;
    }
    if (message.toolUseIds.length > 0) {
      const next = messages[index + 1];
      if (next === undefined || next.toolResultIds.length === 0) {
        throw new ServeError(
          "OpenAI responses: function_call_output items must immediately follow function_call items.",
          { param: "input" },
        );
      }
      assertToolResultsMatch(message.toolUseIds, next.toolResultIds);
    }
    if (message.toolResultIds.length > 0) {
      const previous = messages[index - 1];
      if (previous === undefined || previous.toolUseIds.length === 0) {
        throw new ServeError(
          "OpenAI responses: function_call_output items must immediately follow function_call items.",
          { param: "input" },
        );
      }
      assertToolResultsMatch(previous.toolUseIds, message.toolResultIds);
    }
  }
}

function assertReasoningCanStart(pendingCalls: readonly ChatToolCall[]): void {
  if (pendingCalls.length > 0) {
    throw new ServeError(
      "OpenAI responses: reasoning items must immediately precede assistant output items.",
      { param: "input" },
    );
  }
}

/** Parse Responses input array items that carry reasoning and tool-turn state. */
export function parseOpenAIResponseInputItems(
  inputMessage: (value: unknown) => OpenAIResponseParsedInputItem,
  input: readonly unknown[],
): OpenAIResponseParsedInputItem[] {
  const messages: OpenAIResponseParsedInputItem[] = [];
  let pendingCalls: ChatToolCall[] = [];
  let pendingOutputs: FunctionOutputTurn[] = [];
  let pendingReasoning: string | undefined;
  const addPendingReasoning = (reasoningContent: string) => {
    pendingReasoning = appendReasoningContent(pendingReasoning, reasoningContent);
  };
  const takePendingReasoning = () => {
    const reasoningContent = pendingReasoning;
    pendingReasoning = undefined;
    return reasoningContent;
  };
  const assertNoPendingReasoning = () => {
    if (pendingReasoning !== undefined) {
      throw new ServeError(
        "OpenAI responses: reasoning items must immediately precede assistant output items.",
        { param: "input" },
      );
    }
  };
  const flushCalls = () => {
    if (pendingCalls.length > 0) {
      messages.push(functionCallMessage(pendingCalls, takePendingReasoning()));
      pendingCalls = [];
    }
  };
  const flushOutputs = () => {
    if (pendingOutputs.length > 0) {
      assertNoPendingReasoning();
      messages.push(functionOutputMessage(pendingOutputs));
      pendingOutputs = [];
    }
  };
  for (const item of input) {
    if (isRecord(item) && item.type === "reasoning") {
      assertReasoningCanStart(pendingCalls);
      flushOutputs();
      addPendingReasoning(responseReasoning(item));
      continue;
    }
    if (isRecord(item) && item.type === "function_call") {
      flushOutputs();
      pendingCalls.push(responseFunctionCall(item));
      continue;
    }
    if (isRecord(item) && item.type === "function_call_output") {
      flushCalls();
      assertNoPendingReasoning();
      pendingOutputs.push(responseFunctionOutput(item));
      continue;
    }
    flushCalls();
    flushOutputs();
    const parsed = inputMessage(item);
    messages.push(
      pendingReasoning === undefined
        ? parsed
        : assistantMessageWithReasoning(parsed, takePendingReasoning() ?? ""),
    );
  }
  flushCalls();
  flushOutputs();
  assertNoPendingReasoning();
  validateToolTurns(messages);
  return messages;
}
