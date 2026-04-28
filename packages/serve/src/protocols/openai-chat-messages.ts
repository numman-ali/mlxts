/**
 * OpenAI chat message and tool request parsing.
 * @module
 */

import type { ChatMessage, ChatTool, ChatToolCall } from "@mlxts/transformers";
import { isRecord, ServeError } from "../errors";
import type { GenerationContentMessage, GenerationContentPart, GenerationInput } from "../types";
import { hasMediaContent, openAIImageContentPart, textContentPart } from "./media-content";

type ParsedMessageContent = {
  text: string;
  parts: readonly GenerationContentPart[];
  hasMedia: boolean;
};

type ParsedChatMessage = {
  chat: ChatMessage;
  content: GenerationContentMessage;
  hasMedia: boolean;
};

function parsedTextContent(text: string): ParsedMessageContent {
  return {
    text,
    parts: text === "" ? [] : [textContentPart(text)],
    hasMedia: false,
  };
}

function contentPart(value: unknown): GenerationContentPart {
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
    return textContentPart(value.text);
  }
  if (value.type === "image_url") {
    return openAIImageContentPart(value.image_url, "OpenAI chat completions: image content parts");
  }
  throw new ServeError(
    'OpenAI chat completions: only text and image content parts are supported in "messages" today.',
    { param: "messages" },
  );
}

function joinTextParts(parts: readonly GenerationContentPart[]): string {
  let text = "";
  for (const part of parts) {
    if (part.kind === "text") {
      text += part.text;
    }
  }
  return text;
}

function parsedContentArray(value: readonly unknown[]): ParsedMessageContent {
  const parts = value.map(contentPart);
  return {
    text: joinTextParts(parts),
    parts,
    hasMedia: hasMediaContent(parts),
  };
}

function optionalMessageContent(
  record: Record<string, unknown>,
  key: string,
): ParsedMessageContent {
  const value = record[key];
  if (value === undefined || value === null) {
    return parsedTextContent("");
  }
  if (typeof value === "string") {
    return parsedTextContent(value);
  }
  if (Array.isArray(value)) {
    return parsedContentArray(value);
  }
  throw new ServeError(
    `OpenAI chat completions: "${key}" must be a string, null, or an array of content parts.`,
    { param: key },
  );
}

function roleContent(value: Record<string, unknown>, role: "system" | "user"): ParsedChatMessage {
  const content = optionalMessageContent(value, "content");
  return {
    chat: {
      role,
      content: content.text,
    },
    content: {
      role,
      content: content.parts,
    },
    hasMedia: content.hasMedia,
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

function assistantMessage(value: Record<string, unknown>): ParsedChatMessage {
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
  const content = optionalMessageContent(value, "content");
  return {
    chat: {
      role: "assistant",
      content: content.text,
      ...(typeof value.reasoning_content === "string"
        ? { reasoning_content: value.reasoning_content }
        : {}),
      ...(toolCalls === undefined ? {} : { tool_calls: toolCalls }),
    },
    content: {
      role: "assistant",
      content: content.parts,
      ...(typeof value.reasoning_content === "string"
        ? { reasoning_content: value.reasoning_content }
        : {}),
      ...(toolCalls === undefined ? {} : { tool_calls: toolCalls }),
    },
    hasMedia: content.hasMedia,
  };
}

function toolMessage(value: Record<string, unknown>): ParsedChatMessage {
  const content = optionalMessageContent(value, "content");
  return {
    chat: {
      role: "tool",
      content: content.text,
      ...(typeof value.name === "string" ? { name: value.name } : {}),
      ...(typeof value.tool_call_id === "string" ? { tool_call_id: value.tool_call_id } : {}),
    },
    content: {
      role: "tool",
      content: content.parts,
      ...(typeof value.name === "string" ? { name: value.name } : {}),
      ...(typeof value.tool_call_id === "string" ? { tool_call_id: value.tool_call_id } : {}),
    },
    hasMedia: content.hasMedia,
  };
}

function chatMessage(value: unknown): ParsedChatMessage {
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
  return value.map(chatMessage).map((message) => message.chat);
}

/** Parse chat messages into either text-only messages or ordered media content. */
export function parseOpenAIChatInput(record: Record<string, unknown>): GenerationInput {
  const value = record.messages;
  if (!Array.isArray(value) || value.length === 0) {
    throw new ServeError('OpenAI chat completions: "messages" must be a non-empty array.', {
      param: "messages",
    });
  }
  const messages = value.map(chatMessage);
  if (!messages.some((message) => message.hasMedia)) {
    return {
      kind: "messages",
      messages: messages.map((message) => message.chat),
    };
  }
  return {
    kind: "content",
    messages: messages.map((message) => message.content),
  };
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
