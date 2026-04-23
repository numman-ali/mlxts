/**
 * OpenAI chat message and tool request parsing.
 * @module
 */

import type { ChatMessage, ChatTool, ChatToolCall } from "@mlxts/transformers";
import { isRecord, ServeError } from "../errors";

function textContentPart(value: unknown): string {
  if (!isRecord(value)) {
    throw new ServeError(
      'OpenAI chat completions: "content" array entries must be content part objects.',
      { param: "messages" },
    );
  }
  if (value.type === "text") {
    if (typeof value.text !== "string") {
      throw new ServeError(
        'OpenAI chat completions: text content parts require a string "text" field.',
        { param: "messages" },
      );
    }
    return value.text;
  }
  if (value.type === "image_url") {
    throw new ServeError(
      "OpenAI chat completions: image content parts are not supported by this endpoint yet.",
      { param: "messages" },
    );
  }
  throw new ServeError(
    'OpenAI chat completions: only text content parts are supported in "messages" today.',
    { param: "messages" },
  );
}

function optionalMessageContent(record: Record<string, unknown>, key: string): string {
  const value = record[key];
  if (value === undefined || value === null) {
    return "";
  }
  if (typeof value !== "string") {
    if (Array.isArray(value)) {
      return value.map(textContentPart).join("");
    }
    throw new ServeError(
      `OpenAI chat completions: "${key}" must be a string, null, or an array of text parts.`,
      { param: key },
    );
  }
  return value;
}

function roleContent(value: Record<string, unknown>, role: "system" | "user"): ChatMessage {
  return {
    role,
    content: optionalMessageContent(value, "content"),
  };
}

function toolCall(value: unknown): ChatToolCall {
  if (!isRecord(value) || value.type !== "function" || !isRecord(value.function)) {
    throw new ServeError(
      'OpenAI chat completions: assistant "tool_calls" must be function calls.',
      { param: "messages" },
    );
  }
  const name = value.function.name;
  const args = value.function.arguments;
  if (typeof name !== "string" || name.trim() === "" || typeof args !== "string") {
    throw new ServeError(
      'OpenAI chat completions: assistant "tool_calls" require function name and arguments.',
      { param: "messages" },
    );
  }
  return {
    ...(typeof value.id === "string" && value.id.trim() !== "" ? { id: value.id } : {}),
    type: "function",
    function: { name, arguments: args },
  };
}

function assistantMessage(value: Record<string, unknown>): ChatMessage {
  const toolCalls =
    value.tool_calls === undefined || value.tool_calls === null
      ? undefined
      : Array.isArray(value.tool_calls)
        ? value.tool_calls.map(toolCall)
        : null;
  if (toolCalls === null) {
    throw new ServeError('OpenAI chat completions: assistant "tool_calls" must be an array.', {
      param: "messages",
    });
  }
  return {
    role: "assistant",
    content: optionalMessageContent(value, "content"),
    ...(typeof value.reasoning_content === "string"
      ? { reasoning_content: value.reasoning_content }
      : {}),
    ...(toolCalls === undefined ? {} : { tool_calls: toolCalls }),
  };
}

function toolMessage(value: Record<string, unknown>): ChatMessage {
  return {
    role: "tool",
    content: optionalMessageContent(value, "content"),
    ...(typeof value.name === "string" ? { name: value.name } : {}),
    ...(typeof value.tool_call_id === "string" ? { tool_call_id: value.tool_call_id } : {}),
  };
}

function chatMessage(value: unknown): ChatMessage {
  if (!isRecord(value)) {
    throw new ServeError('OpenAI chat completions: "messages" entries must be objects.', {
      param: "messages",
    });
  }

  switch (value.role) {
    case "system":
    case "user":
      return roleContent(value, value.role);
    case "developer":
      return roleContent(value, "system");
    case "assistant":
      return assistantMessage(value);
    case "tool":
      return toolMessage(value);
    default:
      throw new ServeError(
        'OpenAI chat completions: message role must be "system", "developer", "user", "assistant", or "tool".',
        { param: "messages" },
      );
  }
}

export function parseOpenAIChatMessages(record: Record<string, unknown>): ChatMessage[] {
  const value = record.messages;
  if (!Array.isArray(value) || value.length === 0) {
    throw new ServeError('OpenAI chat completions: "messages" must be a non-empty array.', {
      param: "messages",
    });
  }
  return value.map(chatMessage);
}

function chatTool(value: unknown): ChatTool {
  if (!isRecord(value) || value.type !== "function" || !isRecord(value.function)) {
    throw new ServeError('OpenAI chat completions: "tools" entries must be function tools.', {
      param: "tools",
    });
  }
  const name = value.function.name;
  if (typeof name !== "string" || name.trim() === "") {
    throw new ServeError("OpenAI chat completions: tool function name must be non-empty.", {
      param: "tools",
    });
  }
  return {
    type: "function",
    function: {
      name,
      ...(typeof value.function.description === "string"
        ? { description: value.function.description }
        : {}),
      ...(isRecord(value.function.parameters) ? { parameters: value.function.parameters } : {}),
    },
  };
}

export function parseOpenAIChatTools(record: Record<string, unknown>): ChatTool[] | undefined {
  const value = record.tools;
  if (value === undefined || value === null) {
    return undefined;
  }
  if (!Array.isArray(value)) {
    throw new ServeError('OpenAI chat completions: "tools" must be an array or null.', {
      param: "tools",
    });
  }
  return value.map(chatTool);
}

export function validateOpenAIChatToolChoice(record: Record<string, unknown>): void {
  const value = record.tool_choice;
  if (value === undefined || value === null || value === "auto" || value === "none") {
    return;
  }
  throw new ServeError(
    'OpenAI chat completions: "tool_choice" currently supports only "auto", "none", or null.',
    { param: "tool_choice" },
  );
}
