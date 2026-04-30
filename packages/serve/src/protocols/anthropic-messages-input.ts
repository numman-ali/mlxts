/**
 * Anthropic Messages input normalization.
 * @module
 */

import type { ChatMessage, ChatTool, ChatToolCall } from "@mlxts/transformers";
import { isRecord, ServeError } from "../errors";
import type { GenerationContentMessage, GenerationContentPart, GenerationInput } from "../types";
import { validateAnthropicToolTurns } from "./anthropic-tool-turns";
import {
  anthropicImageContentPart,
  textContentPart as normalizedTextContentPart,
} from "./media-content";

export type AnthropicChatTemplateOptions = { enableThinking?: boolean; preserveThinking?: boolean };

type ParsedAssistantContent = {
  content: string;
  reasoningContent?: string;
  toolCall?: ChatToolCall;
  toolCalls?: readonly ChatToolCall[];
};

type ParsedUserContent = { text: string; parts: readonly GenerationContentPart[] };

type ParsedAnthropicMessage = {
  chat: readonly ChatMessage[];
  content: readonly GenerationContentMessage[];
  toolUseIds: readonly string[];
  toolResultIds: readonly string[];
};

function textOnlyContentBlockText(value: unknown, owner: "system" | "message"): string {
  if (!isRecord(value)) {
    throw new ServeError(`Anthropic messages: ${owner} content blocks must be objects.`, {
      param: owner === "system" ? "system" : "messages",
    });
  }
  if (value.type === "text") {
    if (typeof value.text !== "string") {
      throw new ServeError('Anthropic messages: text blocks require a string "text" field.', {
        param: owner === "system" ? "system" : "messages",
      });
    }
    return value.text;
  }
  if (value.type === "image") {
    throw new ServeError(
      "Anthropic messages: image content blocks are only supported in user messages.",
      { param: owner === "system" ? "system" : "messages" },
    );
  }
  if (value.type === "tool_use" || value.type === "tool_result") {
    throw new ServeError(
      "Anthropic messages: tool content blocks are not supported by this endpoint yet.",
      { param: "messages" },
    );
  }
  throw new ServeError("Anthropic messages: only text content blocks are supported today.", {
    param: owner === "system" ? "system" : "messages",
  });
}

export function parseAnthropicSystemContent(record: Record<string, unknown>): string | null {
  const value = record.system;
  if (value === undefined || value === null) {
    return null;
  }
  if (typeof value === "string") {
    return value;
  }
  if (Array.isArray(value)) {
    return value.map((entry) => textOnlyContentBlockText(entry, "system")).join("");
  }
  throw new ServeError('Anthropic messages: "system" must be a string or text content blocks.', {
    param: "system",
  });
}

function nonEmptyString(value: unknown, field: string, context: string): string {
  if (typeof value !== "string" || value.trim() === "") {
    throw new ServeError(`Anthropic messages: ${context} require a non-empty "${field}" field.`, {
      param: "messages",
    });
  }
  return value;
}

function assistantToolUseBlock(value: Record<string, unknown>): ChatToolCall {
  const id = nonEmptyString(value.id, "id", "tool_use blocks");
  const name = nonEmptyString(value.name, "name", "tool_use blocks");
  if (!isRecord(value.input)) {
    throw new ServeError('Anthropic messages: tool_use blocks require an object "input" field.', {
      param: "messages",
    });
  }
  return {
    id,
    type: "function",
    function: {
      name,
      arguments: JSON.stringify(value.input),
    },
  };
}

function assistantContentBlock(value: unknown): ParsedAssistantContent {
  if (!isRecord(value)) {
    throw new ServeError("Anthropic messages: assistant content blocks must be objects.", {
      param: "messages",
    });
  }
  if (value.type === "text") {
    if (typeof value.text !== "string") {
      throw new ServeError(
        'Anthropic messages: assistant text blocks require a string "text" field.',
        { param: "messages" },
      );
    }
    return { content: value.text };
  }
  if (value.type === "thinking") {
    if (typeof value.thinking !== "string") {
      throw new ServeError(
        'Anthropic messages: thinking blocks require a string "thinking" field.',
        { param: "messages" },
      );
    }
    return { content: "", reasoningContent: value.thinking };
  }
  if (value.type === "image") {
    throw new ServeError(
      "Anthropic messages: image content blocks are only supported in user messages.",
      { param: "messages" },
    );
  }
  if (value.type === "tool_use") {
    return {
      content: "",
      toolCall: assistantToolUseBlock(value),
    };
  }
  if (value.type === "tool_result") {
    throw new ServeError(
      "Anthropic messages: tool_result blocks are only supported in user messages.",
      { param: "messages" },
    );
  }
  throw new ServeError(
    "Anthropic messages: assistant content supports text, thinking, and tool_use blocks today.",
    { param: "messages" },
  );
}

function parseAssistantContent(value: unknown): ParsedAssistantContent {
  if (value === undefined || value === null) {
    return { content: "" };
  }
  if (typeof value === "string") {
    return { content: value };
  }
  if (!Array.isArray(value)) {
    throw new ServeError(
      'Anthropic messages: assistant "content" must be a string or content block array.',
      { param: "messages" },
    );
  }
  const parts = value.map(assistantContentBlock);
  const content = parts.map((part) => part.content).join("");
  const reasoningContent = parts
    .map((part) => part.reasoningContent ?? "")
    .join("")
    .trim();
  const toolCalls = parts
    .map((part) => part.toolCall)
    .filter((toolCall): toolCall is ChatToolCall => toolCall !== undefined);
  return {
    content,
    ...(reasoningContent === "" ? {} : { reasoningContent }),
    ...(toolCalls.length === 0 ? {} : { toolCalls }),
  };
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

function parsedTextContent(text: string): ParsedUserContent {
  return {
    text,
    parts: text === "" ? [] : [normalizedTextContentPart(text)],
  };
}

function userContentPart(value: unknown): GenerationContentPart {
  if (!isRecord(value)) {
    throw new ServeError("Anthropic messages: message content blocks must be objects.", {
      param: "messages",
    });
  }
  if (value.type === "text") {
    if (typeof value.text !== "string") {
      throw new ServeError('Anthropic messages: text blocks require a string "text" field.', {
        param: "messages",
      });
    }
    return normalizedTextContentPart(value.text);
  }
  if (value.type === "image") {
    return anthropicImageContentPart(value, "Anthropic messages: image content blocks");
  }
  if (value.type === "tool_use") {
    throw new ServeError(
      "Anthropic messages: tool_use blocks are only supported in assistant messages.",
      { param: "messages" },
    );
  }
  if (value.type === "tool_result") {
    throw new ServeError(
      "Anthropic messages: tool_result blocks must be handled before user text or image blocks.",
      { param: "messages" },
    );
  }
  throw new ServeError(
    "Anthropic messages: only text and image content blocks are supported today.",
    {
      param: "messages",
    },
  );
}

function optionalBooleanField(value: unknown, field: string, context: string): boolean | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "boolean") {
    throw new ServeError(`Anthropic messages: ${context} require a boolean "${field}" field.`, {
      param: "messages",
    });
  }
  return value;
}

function parseToolResultContent(value: unknown): ParsedUserContent {
  if (value === undefined || value === null) {
    return parsedTextContent("");
  }
  if (typeof value === "string") {
    return parsedTextContent(value);
  }
  if (Array.isArray(value)) {
    const parts = value.map(toolResultContentPart);
    return {
      text: joinTextParts(parts),
      parts,
    };
  }
  throw new ServeError(
    'Anthropic messages: tool_result "content" must be a string or content block array.',
    { param: "messages" },
  );
}

function toolResultContentPart(value: unknown): GenerationContentPart {
  if (!isRecord(value)) {
    throw new ServeError("Anthropic messages: tool_result content blocks must be objects.", {
      param: "messages",
    });
  }
  if (value.type === "text") {
    if (typeof value.text !== "string") {
      throw new ServeError(
        'Anthropic messages: tool_result text blocks require a string "text" field.',
        { param: "messages" },
      );
    }
    return normalizedTextContentPart(value.text);
  }
  throw new ServeError(
    "Anthropic messages: tool_result content supports string or text blocks in this local endpoint.",
    { param: "messages" },
  );
}

function toolResultMessage(value: Record<string, unknown>): ParsedAnthropicMessage {
  const toolUseId = nonEmptyString(value.tool_use_id, "tool_use_id", "tool_result blocks");
  if (optionalBooleanField(value.is_error, "is_error", "tool_result blocks") === true) {
    throw new ServeError(
      "Anthropic messages: tool_result is_error=true is not supported by this endpoint yet.",
      { param: "messages" },
    );
  }
  const content = parseToolResultContent(value.content);
  return {
    chat: [
      {
        role: "tool",
        content: content.text,
        tool_call_id: toolUseId,
      },
    ],
    content: [
      {
        role: "tool",
        content: content.parts,
        tool_call_id: toolUseId,
      },
    ],
    toolUseIds: [],
    toolResultIds: [toolUseId],
  };
}

function parseUserContent(value: unknown): ParsedAnthropicMessage {
  if (typeof value === "string") {
    const content = parsedTextContent(value);
    return {
      chat: [{ role: "user", content: content.text }],
      content: [{ role: "user", content: content.parts }],
      toolUseIds: [],
      toolResultIds: [],
    };
  }
  if (!Array.isArray(value)) {
    throw new ServeError(
      'Anthropic messages: user "content" must be a string or content block array.',
      { param: "messages" },
    );
  }

  const chatMessages: ChatMessage[] = [];
  const contentMessages: GenerationContentMessage[] = [];
  const userParts: GenerationContentPart[] = [];
  const toolResultIds: string[] = [];
  let sawNonToolResult = false;

  for (const entry of value) {
    if (!isRecord(entry)) {
      throw new ServeError("Anthropic messages: message content blocks must be objects.", {
        param: "messages",
      });
    }
    if (entry.type === "tool_result") {
      if (sawNonToolResult) {
        throw new ServeError(
          "Anthropic messages: tool_result blocks must precede user text and image blocks.",
          { param: "messages" },
        );
      }
      const parsed = toolResultMessage(entry);
      chatMessages.push(...parsed.chat);
      contentMessages.push(...parsed.content);
      toolResultIds.push(...parsed.toolResultIds);
      continue;
    }
    sawNonToolResult = true;
    const part = userContentPart(entry);
    userParts.push(part);
  }

  if (userParts.length > 0) {
    const text = joinTextParts(userParts);
    chatMessages.push({ role: "user", content: text });
    contentMessages.push({ role: "user", content: userParts });
  }

  return {
    chat: chatMessages,
    content: contentMessages,
    toolUseIds: [],
    toolResultIds,
  };
}

function parseAnthropicMessage(value: unknown): ParsedAnthropicMessage {
  if (!isRecord(value)) {
    throw new ServeError('Anthropic messages: "messages" entries must be objects.', {
      param: "messages",
    });
  }
  if (value.role === "user") {
    return parseUserContent(value.content);
  }
  if (value.role === "assistant") {
    const content = parseAssistantContent(value.content);
    const toolUseIds = (content.toolCalls ?? [])
      .map((toolCall) => toolCall.id)
      .filter((id): id is string => id !== undefined);
    return {
      chat: [
        {
          role: "assistant",
          content: content.content,
          ...(content.reasoningContent === undefined
            ? {}
            : { reasoning_content: content.reasoningContent }),
          ...(content.toolCalls === undefined ? {} : { tool_calls: content.toolCalls }),
        },
      ],
      content: [
        {
          role: "assistant",
          content: content.content === "" ? [] : [normalizedTextContentPart(content.content)],
          ...(content.reasoningContent === undefined
            ? {}
            : { reasoning_content: content.reasoningContent }),
          ...(content.toolCalls === undefined ? {} : { tool_calls: content.toolCalls }),
        },
      ],
      toolUseIds,
      toolResultIds: [],
    };
  }
  if (value.role === "system") {
    throw new ServeError(
      'Anthropic messages: use top-level "system" instead of a system message role.',
      { param: "messages" },
    );
  }
  throw new ServeError('Anthropic messages: message role must be "user" or "assistant".', {
    param: "messages",
  });
}

function parseMessages(record: Record<string, unknown>): ParsedAnthropicMessage[] {
  const value = record.messages;
  if (!Array.isArray(value) || value.length === 0) {
    throw new ServeError('Anthropic messages: "messages" must be a non-empty array.', {
      param: "messages",
    });
  }
  const messages = value.map(parseAnthropicMessage);
  validateAnthropicToolTurns(messages);
  return messages;
}

export function parseAnthropicMessagesInput(
  record: Record<string, unknown>,
  system: string | null,
  templateOptions: AnthropicChatTemplateOptions,
  tools?: readonly ChatTool[],
): GenerationInput {
  const messages = parseMessages(record);
  const hasTemplateOptions = Object.keys(templateOptions).length > 0;
  const chatMessages = messages.flatMap((message) => message.chat);
  const contentMessages = messages.flatMap((message) => message.content);
  const hasMedia = contentMessages.some((message) =>
    message.content.some((part) => part.kind !== "text"),
  );
  if (system !== null) {
    chatMessages.unshift({ role: "system", content: system });
    contentMessages.unshift({
      role: "system",
      content: system === "" ? [] : [normalizedTextContentPart(system)],
    });
  }

  if (!hasMedia) {
    return {
      kind: "messages",
      messages: chatMessages,
      ...(tools === undefined ? {} : { tools }),
      ...(hasTemplateOptions ? { chatTemplate: templateOptions } : {}),
    };
  }
  return {
    kind: "content",
    messages: contentMessages,
    ...(tools === undefined ? {} : { tools }),
    ...(hasTemplateOptions ? { chatTemplate: templateOptions } : {}),
  };
}
