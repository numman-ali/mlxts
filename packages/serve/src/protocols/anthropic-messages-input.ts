/**
 * Anthropic Messages input normalization.
 * @module
 */

import type { ChatMessage } from "@mlxts/transformers";
import { isRecord, ServeError } from "../errors";
import type { GenerationContentMessage, GenerationContentPart, GenerationInput } from "../types";
import {
  anthropicImageContentPart,
  hasMediaContent,
  textContentPart as normalizedTextContentPart,
} from "./media-content";

export type AnthropicChatTemplateOptions = {
  enableThinking?: boolean;
  preserveThinking?: boolean;
};

type ParsedAssistantContent = {
  content: string;
  reasoningContent?: string;
};

type ParsedUserContent = {
  text: string;
  parts: readonly GenerationContentPart[];
  hasMedia: boolean;
};

type ParsedAnthropicMessage = {
  chat: ChatMessage;
  content: GenerationContentMessage;
  hasMedia: boolean;
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
  if (value.type === "tool_use" || value.type === "tool_result") {
    throw new ServeError(
      "Anthropic messages: tool content blocks are not supported by this endpoint yet.",
      { param: "messages" },
    );
  }
  throw new ServeError(
    "Anthropic messages: assistant content supports text and thinking blocks today.",
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
  return {
    content,
    ...(reasoningContent === "" ? {} : { reasoningContent }),
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
    hasMedia: false,
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
  if (value.type === "tool_use" || value.type === "tool_result") {
    throw new ServeError(
      "Anthropic messages: tool content blocks are not supported by this endpoint yet.",
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

function parseUserContent(value: unknown): ParsedUserContent {
  if (typeof value === "string") {
    return parsedTextContent(value);
  }
  if (Array.isArray(value)) {
    const parts = value.map(userContentPart);
    return {
      text: joinTextParts(parts),
      parts,
      hasMedia: hasMediaContent(parts),
    };
  }
  throw new ServeError(
    'Anthropic messages: user "content" must be a string or content block array.',
    { param: "messages" },
  );
}

function parseAnthropicMessage(value: unknown): ParsedAnthropicMessage {
  if (!isRecord(value)) {
    throw new ServeError('Anthropic messages: "messages" entries must be objects.', {
      param: "messages",
    });
  }
  if (value.role === "user") {
    const content = parseUserContent(value.content);
    return {
      chat: { role: "user", content: content.text },
      content: { role: "user", content: content.parts },
      hasMedia: content.hasMedia,
    };
  }
  if (value.role === "assistant") {
    const content = parseAssistantContent(value.content);
    return {
      chat: {
        role: "assistant",
        content: content.content,
        ...(content.reasoningContent === undefined
          ? {}
          : { reasoning_content: content.reasoningContent }),
      },
      content: {
        role: "assistant",
        content: content.content === "" ? [] : [normalizedTextContentPart(content.content)],
        ...(content.reasoningContent === undefined
          ? {}
          : { reasoning_content: content.reasoningContent }),
      },
      hasMedia: false,
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
  return value.map(parseAnthropicMessage);
}

export function parseAnthropicMessagesInput(
  record: Record<string, unknown>,
  system: string | null,
  templateOptions: AnthropicChatTemplateOptions,
): GenerationInput {
  const messages = parseMessages(record);
  const hasTemplateOptions = Object.keys(templateOptions).length > 0;
  const chatMessages = messages.map((message) => message.chat);
  const contentMessages = messages.map((message) => message.content);
  if (system !== null) {
    chatMessages.unshift({ role: "system", content: system });
    contentMessages.unshift({
      role: "system",
      content: system === "" ? [] : [normalizedTextContentPart(system)],
    });
  }

  if (!messages.some((message) => message.hasMedia)) {
    return {
      kind: "messages",
      messages: chatMessages,
      ...(hasTemplateOptions ? { chatTemplate: templateOptions } : {}),
    };
  }
  return {
    kind: "content",
    messages: contentMessages,
    ...(hasTemplateOptions ? { chatTemplate: templateOptions } : {}),
  };
}
