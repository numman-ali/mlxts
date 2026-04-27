/**
 * Anthropic-compatible Messages protocol adapter.
 * @module
 */

import type { ChatMessage } from "@mlxts/transformers";
import { isRecord, ServeError } from "../errors";
import type { NormalizedGenerationRequest } from "../types";

export type AnthropicTextBlock = {
  type: "text";
  text: string;
};

export type AnthropicThinkingBlock = {
  type: "thinking";
  thinking: string;
  signature: string;
};

export type AnthropicContentBlock = AnthropicTextBlock | AnthropicThinkingBlock;

export type AnthropicUsage = {
  input_tokens: number;
  output_tokens: number;
};

export type AnthropicStopReason =
  | "end_turn"
  | "max_tokens"
  | "stop_sequence"
  | "tool_use"
  | "pause_turn"
  | "refusal"
  | null;

export type AnthropicMessageResponse = {
  id: string;
  type: "message";
  role: "assistant";
  content: AnthropicContentBlock[];
  model: string;
  stop_reason: AnthropicStopReason;
  stop_sequence: string | null;
  usage: AnthropicUsage;
};

export type NormalizedAnthropicMessage = {
  model: string;
  stream: boolean;
  maxTokens: number;
  temperature: number | null;
  topP: number | null;
  topK: number | null;
  request: NormalizedGenerationRequest;
};

export {
  anthropicStopReason,
  formatAnthropicMessageResponse,
} from "./anthropic-messages-formatting";

function stringField(record: Record<string, unknown>, key: string): string {
  const value = record[key];
  if (typeof value !== "string" || value.trim() === "") {
    throw new ServeError(`Anthropic messages: "${key}" must be a non-empty string.`, {
      param: key,
    });
  }
  return value;
}

function optionalString(record: Record<string, unknown>, key: string): string | undefined {
  const value = record[key];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "string") {
    throw new ServeError(`Anthropic messages: "${key}" must be a string.`, { param: key });
  }
  return value;
}

function optionalBoolean(record: Record<string, unknown>, key: string): boolean | undefined {
  const value = record[key];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "boolean") {
    throw new ServeError(`Anthropic messages: "${key}" must be a boolean.`, { param: key });
  }
  return value;
}

function optionalNumber(
  record: Record<string, unknown>,
  key: string,
  validate: (value: number) => boolean,
  description: string,
): number | undefined {
  const value = record[key];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "number" || !Number.isFinite(value) || !validate(value)) {
    throw new ServeError(`Anthropic messages: "${key}" must be ${description}.`, {
      param: key,
    });
  }
  return value;
}

function optionalInteger(
  record: Record<string, unknown>,
  key: string,
  validate: (value: number) => boolean,
  description: string,
): number | undefined {
  return optionalNumber(
    record,
    key,
    (value) => Number.isInteger(value) && validate(value),
    description,
  );
}

function requiredMaxTokens(record: Record<string, unknown>): number {
  const maxTokens = optionalInteger(
    record,
    "max_tokens",
    (value) => value > 0,
    "a positive integer",
  );
  if (maxTokens === undefined) {
    throw new ServeError('Anthropic messages: "max_tokens" is required.', {
      param: "max_tokens",
    });
  }
  return maxTokens;
}

function parseStopSequences(record: Record<string, unknown>): readonly string[] | undefined {
  const value = record.stop_sequences;
  if (value === undefined || value === null) {
    return undefined;
  }
  if (Array.isArray(value) && value.every((entry) => typeof entry === "string")) {
    return value;
  }
  throw new ServeError('Anthropic messages: "stop_sequences" must be a string array or null.', {
    param: "stop_sequences",
  });
}

function textContentPart(value: unknown, owner: "system" | "message"): string {
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
      "Anthropic messages: image content blocks are not supported by this endpoint yet.",
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

function systemContent(record: Record<string, unknown>): string | null {
  const value = record.system;
  if (value === undefined || value === null) {
    return null;
  }
  if (typeof value === "string") {
    return value;
  }
  if (Array.isArray(value)) {
    return value.map((entry) => textContentPart(entry, "system")).join("");
  }
  throw new ServeError('Anthropic messages: "system" must be a string or text content blocks.', {
    param: "system",
  });
}

type ParsedAssistantContent = {
  content: string;
  reasoningContent?: string;
};

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

function parseUserContent(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  if (Array.isArray(value)) {
    return value.map((entry) => textContentPart(entry, "message")).join("");
  }
  throw new ServeError(
    'Anthropic messages: user "content" must be a string or text content block array.',
    { param: "messages" },
  );
}

function parseAnthropicMessage(value: unknown): ChatMessage {
  if (!isRecord(value)) {
    throw new ServeError('Anthropic messages: "messages" entries must be objects.', {
      param: "messages",
    });
  }
  if (value.role === "user") {
    return { role: "user", content: parseUserContent(value.content) };
  }
  if (value.role === "assistant") {
    const content = parseAssistantContent(value.content);
    return {
      role: "assistant",
      content: content.content,
      ...(content.reasoningContent === undefined
        ? {}
        : { reasoning_content: content.reasoningContent }),
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

function parseMessages(record: Record<string, unknown>): ChatMessage[] {
  const value = record.messages;
  if (!Array.isArray(value) || value.length === 0) {
    throw new ServeError('Anthropic messages: "messages" must be a non-empty array.', {
      param: "messages",
    });
  }
  return value.map(parseAnthropicMessage);
}

function validateUnsupportedFields(record: Record<string, unknown>): void {
  const tools = record.tools;
  if (tools !== undefined && tools !== null) {
    if (!Array.isArray(tools)) {
      throw new ServeError('Anthropic messages: "tools" must be an array or null.', {
        param: "tools",
      });
    }
    if (tools.length > 0) {
      throw new ServeError("Anthropic messages: tools are not supported yet.", {
        param: "tools",
      });
    }
  }
  if (record.tool_choice !== undefined && record.tool_choice !== null) {
    throw new ServeError("Anthropic messages: tool_choice is not supported yet.", {
      param: "tool_choice",
    });
  }
}

function chatTemplateFlag(
  record: Record<string, unknown>,
  key: "enable_thinking" | "preserve_thinking",
): boolean | undefined {
  return optionalBoolean(record, key);
}

function thinkingFlag(record: Record<string, unknown>): boolean | undefined {
  const thinking = record.thinking;
  if (thinking === undefined || thinking === null) {
    return undefined;
  }
  if (!isRecord(thinking)) {
    throw new ServeError('Anthropic messages: "thinking" must be an object or null.', {
      param: "thinking",
    });
  }
  if (thinking.type === "enabled") {
    return true;
  }
  if (thinking.type === "disabled") {
    return false;
  }
  throw new ServeError(
    'Anthropic messages: "thinking.type" currently supports "enabled" or "disabled".',
    { param: "thinking" },
  );
}

function chatTemplateOptions(record: Record<string, unknown>) {
  const kwargs = record.chat_template_kwargs;
  if (kwargs !== undefined && kwargs !== null && !isRecord(kwargs)) {
    throw new ServeError('Anthropic messages: "chat_template_kwargs" must be an object or null.', {
      param: "chat_template_kwargs",
    });
  }
  const templateRecord = isRecord(kwargs) ? kwargs : {};
  const enableThinking =
    chatTemplateFlag(templateRecord, "enable_thinking") ??
    chatTemplateFlag(record, "enable_thinking") ??
    thinkingFlag(record);
  const preserveThinking =
    chatTemplateFlag(templateRecord, "preserve_thinking") ??
    chatTemplateFlag(record, "preserve_thinking");

  return {
    ...(enableThinking === undefined ? {} : { enableThinking }),
    ...(preserveThinking === undefined ? {} : { preserveThinking }),
  };
}

function parseMetadata(record: Record<string, unknown>): Record<string, unknown> {
  const metadata = record.metadata;
  if (metadata === undefined || metadata === null) {
    return {};
  }
  if (!isRecord(metadata)) {
    throw new ServeError('Anthropic messages: "metadata" must be an object or null.', {
      param: "metadata",
    });
  }
  return { ...metadata };
}

/** Normalize an Anthropic Messages JSON body into one generation request. */
export function normalizeAnthropicMessageRequest(
  body: unknown,
  options: { id: string },
): NormalizedAnthropicMessage {
  if (!isRecord(body)) {
    throw new ServeError("Anthropic messages: request body must be a JSON object.");
  }

  validateUnsupportedFields(body);
  const model = stringField(body, "model");
  const stream = optionalBoolean(body, "stream") ?? false;
  const maxTokens = requiredMaxTokens(body);
  const messages = parseMessages(body);
  const system = systemContent(body);
  const temperature = optionalNumber(
    body,
    "temperature",
    (value) => value >= 0 && value <= 1,
    "a number between 0 and 1",
  );
  const topP = optionalNumber(body, "top_p", (value) => value > 0 && value <= 1, "0 < value <= 1");
  const topK = optionalInteger(body, "top_k", (value) => value >= 0, "a non-negative integer");
  const seed = optionalInteger(body, "seed", (value) => value >= 0, "a non-negative integer");
  const ignoreEos = optionalBoolean(body, "ignore_eos");
  const stop = parseStopSequences(body);
  const templateOptions = chatTemplateOptions(body);
  const metadata = parseMetadata(body);
  const userId = optionalString(metadata, "user_id");
  const normalizedMessages =
    system === null ? messages : [{ role: "system" as const, content: system }, ...messages];

  return {
    model,
    stream,
    maxTokens,
    temperature: temperature ?? null,
    topP: topP ?? null,
    topK: topK ?? null,
    request: {
      id: options.id,
      model,
      input: {
        kind: "messages",
        messages: normalizedMessages,
        ...(Object.keys(templateOptions).length === 0 ? {} : { chatTemplate: templateOptions }),
      },
      sampling: {
        maxTokens,
        ...(temperature === undefined ? {} : { temperature }),
        ...(topP === undefined ? {} : { topP }),
        ...(topK === undefined || topK === 0 ? {} : { topK }),
        ...(seed === undefined ? {} : { seed }),
        ...(ignoreEos === undefined ? {} : { ignoreEos }),
        ...(stop === undefined ? {} : { stop }),
      },
      stream,
      protocol: "anthropic.messages",
      metadata: {
        ...metadata,
        ...(userId === undefined ? {} : { user: userId }),
      },
    },
  };
}
