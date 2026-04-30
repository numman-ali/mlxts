/**
 * Anthropic-compatible Messages protocol adapter.
 * @module
 */

import type { ChatTool } from "@mlxts/transformers";
import { isRecord, ServeError } from "../errors";
import type { NormalizedGenerationRequest } from "../types";
import {
  type AnthropicChatTemplateOptions,
  parseAnthropicMessagesInput,
  parseAnthropicSystemContent,
} from "./anthropic-messages-input";

export type AnthropicTextBlock = {
  type: "text";
  text: string;
};

export type AnthropicThinkingBlock = {
  type: "thinking";
  thinking: string;
  signature: string;
};

export type AnthropicToolUseBlock = {
  type: "tool_use";
  id: string;
  name: string;
  input: Record<string, unknown>;
};

export type AnthropicContentBlock =
  | AnthropicTextBlock
  | AnthropicThinkingBlock
  | AnthropicToolUseBlock;

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

function chatTemplateOptions(record: Record<string, unknown>): AnthropicChatTemplateOptions {
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

function anthropicTool(value: unknown): ChatTool {
  if (!isRecord(value)) {
    throw new ServeError('Anthropic messages: "tools" entries must be objects.', {
      param: "tools",
    });
  }
  const name = value.name;
  if (typeof name !== "string" || name.trim() === "") {
    throw new ServeError('Anthropic messages: tool "name" must be a non-empty string.', {
      param: "tools",
    });
  }
  if (!/^[a-zA-Z0-9_-]{1,64}$/.test(name)) {
    throw new ServeError('Anthropic messages: tool "name" must match /^[a-zA-Z0-9_-]{1,64}$/.', {
      param: "tools",
    });
  }
  if (!isRecord(value.input_schema)) {
    throw new ServeError('Anthropic messages: tool "input_schema" must be an object.', {
      param: "tools",
    });
  }
  if (value.description !== undefined && typeof value.description !== "string") {
    throw new ServeError('Anthropic messages: tool "description" must be a string when present.', {
      param: "tools",
    });
  }
  return {
    type: "function",
    function: {
      name,
      ...(typeof value.description === "string" ? { description: value.description } : {}),
      parameters: value.input_schema,
    },
  };
}

function parseAnthropicTools(record: Record<string, unknown>): ChatTool[] | undefined {
  const value = record.tools;
  if (value === undefined || value === null) {
    return undefined;
  }
  if (!Array.isArray(value)) {
    throw new ServeError('Anthropic messages: "tools" must be an array or null.', {
      param: "tools",
    });
  }
  return value.map(anthropicTool);
}

function selectedAnthropicTools(
  record: Record<string, unknown>,
  tools: readonly ChatTool[] | undefined,
): readonly ChatTool[] | undefined {
  const value = record.tool_choice;
  if (value === undefined || value === null) {
    return tools;
  }
  if (!isRecord(value)) {
    throw new ServeError('Anthropic messages: "tool_choice" must be an object or null.', {
      param: "tool_choice",
    });
  }
  switch (value.type) {
    case "auto":
      return tools;
    case "none":
      return undefined;
    case "any":
    case "tool":
      throw new ServeError(
        'Anthropic messages: tool_choice "any" and "tool" are not supported yet.',
        { param: "tool_choice" },
      );
    default:
      throw new ServeError(
        'Anthropic messages: "tool_choice.type" must be "auto", "none", "any", or "tool".',
        { param: "tool_choice" },
      );
  }
}

/** Normalize an Anthropic Messages JSON body into one generation request. */
export function normalizeAnthropicMessageRequest(
  body: unknown,
  options: { id: string },
): NormalizedAnthropicMessage {
  if (!isRecord(body)) {
    throw new ServeError("Anthropic messages: request body must be a JSON object.");
  }

  const model = stringField(body, "model");
  const stream = optionalBoolean(body, "stream") ?? false;
  const maxTokens = requiredMaxTokens(body);
  const system = parseAnthropicSystemContent(body);
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
  const selectedTools = selectedAnthropicTools(body, parseAnthropicTools(body));
  if (stream && selectedTools !== undefined && selectedTools.length > 0) {
    throw new ServeError(
      "Anthropic messages: streaming tool use is not supported until Anthropic tool SSE blocks are implemented.",
      { param: "stream" },
    );
  }

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
      input: parseAnthropicMessagesInput(body, system, templateOptions, selectedTools),
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
