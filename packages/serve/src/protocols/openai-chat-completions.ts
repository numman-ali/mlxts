/**
 * OpenAI-compatible chat completions protocol adapter.
 * @module
 */

import type { ChatMessage, ChatTool, ChatToolCall } from "@mlxts/transformers";
import { isRecord, ServeError } from "../errors";
import type {
  GenerationUsage,
  NormalizedFinishReason,
  NormalizedGenerationRequest,
  NormalizedGenerationResult,
} from "../types";
import {
  type OpenAIChatCompletionUsage,
  parseOpenAIChatCompletionStreamOptions,
} from "./openai-chat-completion-streaming";
import { extractOpenAIChatToolCalls } from "./openai-chat-tool-calls";
import { parseOpenAIStopSequences } from "./openai-stop";

export type {
  OpenAIChatCompletionChunk,
  OpenAIChatCompletionChunkChoice,
  OpenAIChatCompletionStreamDelta,
  OpenAIChatCompletionUsage,
} from "./openai-chat-completion-streaming";
export {
  createOpenAIChatCompletionReasoningStream,
  formatOpenAIChatCompletionStreamChunk,
  formatOpenAIChatCompletionUsageStreamChunk,
} from "./openai-chat-completion-streaming";

export type OpenAIChatCompletionMessage = {
  role: "assistant";
  content: string | null;
  reasoning_content?: string;
  tool_calls?: ChatToolCall[];
};

export type OpenAIChatCompletionChoice = {
  index: number;
  message: OpenAIChatCompletionMessage;
  finish_reason: "stop" | "length" | "tool_calls" | "content_filter" | null;
};

export type OpenAIChatCompletionResponse = {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: OpenAIChatCompletionChoice[];
  usage?: OpenAIChatCompletionUsage;
};

export type NormalizedChatCompletion = {
  model: string;
  stream: boolean;
  streamOptions: {
    includeUsage: boolean;
  };
  request: NormalizedGenerationRequest;
};

const DEFAULT_CHAT_MAX_TOKENS = 16;
const THINK_OPEN = "<think>";
const THINK_CLOSE = "</think>";

function stringField(record: Record<string, unknown>, key: string): string {
  const value = record[key];
  if (typeof value !== "string" || value.trim() === "") {
    throw new ServeError(`OpenAI chat completions: "${key}" must be a non-empty string.`, {
      param: key,
    });
  }
  return value;
}

function optionalBoolean(record: Record<string, unknown>, key: string): boolean | undefined {
  const value = record[key];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "boolean") {
    throw new ServeError(`OpenAI chat completions: "${key}" must be a boolean.`, { param: key });
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
    throw new ServeError(`OpenAI chat completions: "${key}" must be ${description}.`, {
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

function optionalStringContent(record: Record<string, unknown>, key: string): string {
  const value = record[key];
  if (value === undefined || value === null) {
    return "";
  }
  if (typeof value !== "string") {
    throw new ServeError(`OpenAI chat completions: "${key}" must be a string or null.`, {
      param: key,
    });
  }
  return value;
}

function toolCall(value: unknown): ChatToolCall {
  if (!isRecord(value) || value.type !== "function" || !isRecord(value.function)) {
    throw new ServeError(
      'OpenAI chat completions: assistant "tool_calls" must be function calls.',
      {
        param: "messages",
      },
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
    content: optionalStringContent(value, "content"),
    ...(typeof value.reasoning_content === "string"
      ? { reasoning_content: value.reasoning_content }
      : {}),
    ...(toolCalls === undefined ? {} : { tool_calls: toolCalls }),
  };
}

function toolMessage(value: Record<string, unknown>): ChatMessage {
  return {
    role: "tool",
    content: optionalStringContent(value, "content"),
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
      return { role: value.role, content: optionalStringContent(value, "content") };
    case "assistant":
      return assistantMessage(value);
    case "tool":
      return toolMessage(value);
    default:
      throw new ServeError(
        'OpenAI chat completions: message role must be "system", "user", "assistant", or "tool".',
        { param: "messages" },
      );
  }
}

function messages(record: Record<string, unknown>): ChatMessage[] {
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

function tools(record: Record<string, unknown>): ChatTool[] | undefined {
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

function validateToolChoice(record: Record<string, unknown>): void {
  const value = record.tool_choice;
  if (value === undefined || value === null || value === "auto" || value === "none") {
    return;
  }
  throw new ServeError(
    'OpenAI chat completions: "tool_choice" currently supports only "auto", "none", or null.',
    { param: "tool_choice" },
  );
}

function chatTemplateFlag(
  record: Record<string, unknown>,
  key: "enable_thinking" | "preserve_thinking",
): boolean | undefined {
  return optionalBoolean(record, key);
}

function chatTemplateOptions(record: Record<string, unknown>) {
  const kwargs = record.chat_template_kwargs;
  if (kwargs !== undefined && kwargs !== null && !isRecord(kwargs)) {
    throw new ServeError(
      'OpenAI chat completions: "chat_template_kwargs" must be an object or null.',
      { param: "chat_template_kwargs" },
    );
  }
  const templateRecord = isRecord(kwargs) ? kwargs : {};
  const enableThinking =
    chatTemplateFlag(templateRecord, "enable_thinking") ??
    chatTemplateFlag(record, "enable_thinking");
  const preserveThinking =
    chatTemplateFlag(templateRecord, "preserve_thinking") ??
    chatTemplateFlag(record, "preserve_thinking");

  return {
    ...(enableThinking === undefined ? {} : { enableThinking }),
    ...(preserveThinking === undefined ? {} : { preserveThinking }),
  };
}

function sanitizeReasoningContent(text: string): string {
  return text.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, "").trim();
}

function reasoningSplit(content: string, reasoning: string) {
  const reasoningContent = sanitizeReasoningContent(reasoning);
  return reasoningContent === "" ? { content } : { content, reasoningContent };
}

function splitReasoningContent(text: string): { content: string; reasoningContent?: string } {
  const openIndex = text.indexOf(THINK_OPEN);
  const closeIndex = text.indexOf(THINK_CLOSE);
  if (closeIndex < 0 && openIndex < 0) {
    return { content: text.trim() };
  }

  if (closeIndex >= 0 && (openIndex < 0 || openIndex < closeIndex)) {
    const reasoningStart = openIndex < 0 ? 0 : openIndex + THINK_OPEN.length;
    const contentPrefix = openIndex > 0 ? text.slice(0, openIndex).trimEnd() : "";
    const contentSuffix = text.slice(closeIndex + THINK_CLOSE.length).trimStart();
    const content =
      contentPrefix === "" ? contentSuffix.trim() : `${contentPrefix}\n${contentSuffix}`.trim();
    return reasoningSplit(content, text.slice(reasoningStart, closeIndex));
  }

  return reasoningSplit(
    text.slice(0, openIndex).trimEnd(),
    text.slice(openIndex + THINK_OPEN.length),
  );
}

function finishReason(reason: NormalizedFinishReason): OpenAIChatCompletionChoice["finish_reason"] {
  if (reason === "length") {
    return "length";
  }
  if (reason === "stop" || reason === "eos") {
    return "stop";
  }
  if (reason === "cancelled") {
    return null;
  }
  return "content_filter";
}

function formatUsage(usage: GenerationUsage): OpenAIChatCompletionUsage {
  return {
    ...(usage.promptTokens === undefined ? {} : { prompt_tokens: usage.promptTokens }),
    ...(usage.completionTokens === undefined ? {} : { completion_tokens: usage.completionTokens }),
    ...(usage.totalTokens === undefined ? {} : { total_tokens: usage.totalTokens }),
  };
}

function hasToolOutputEnabled(chat: NormalizedChatCompletion): boolean {
  return (
    chat.request.input.kind === "messages" &&
    chat.request.input.tools !== undefined &&
    chat.request.input.tools.length > 0
  );
}

/** Normalize an OpenAI chat completions JSON body into one generation request. */
export function normalizeOpenAIChatCompletionRequest(
  body: unknown,
  options: { id: string },
): NormalizedChatCompletion {
  if (!isRecord(body)) {
    throw new ServeError("OpenAI chat completions: request body must be a JSON object.");
  }

  validateToolChoice(body);
  const stream = optionalBoolean(body, "stream") ?? false;
  const model = stringField(body, "model");
  const parsedTools = body.tool_choice === "none" ? undefined : tools(body);
  const templateOptions = chatTemplateOptions(body);
  const temperature = optionalNumber(
    body,
    "temperature",
    (value) => value >= 0 && value <= 2,
    "a number between 0 and 2",
  );
  const topP = optionalNumber(body, "top_p", (value) => value > 0 && value <= 1, "0 < value <= 1");
  const topK = optionalInteger(body, "top_k", (value) => value > 0, "a positive integer");
  const stop = parseOpenAIStopSequences(body, "chat completions");
  const parsedStreamOptions = parseOpenAIChatCompletionStreamOptions(body, stream);
  return {
    model,
    stream,
    streamOptions: parsedStreamOptions,
    request: {
      id: options.id,
      model,
      input: {
        kind: "messages",
        messages: messages(body),
        ...(parsedTools === undefined ? {} : { tools: parsedTools }),
        ...(Object.keys(templateOptions).length === 0 ? {} : { chatTemplate: templateOptions }),
      },
      sampling: {
        maxTokens:
          optionalInteger(body, "max_tokens", (value) => value >= 0, "a non-negative integer") ??
          DEFAULT_CHAT_MAX_TOKENS,
        ...(temperature === undefined ? {} : { temperature }),
        ...(topP === undefined ? {} : { topP }),
        ...(topK === undefined ? {} : { topK }),
        ...(stop === undefined ? {} : { stop }),
      },
      stream,
      protocol: "openai.chat_completions",
    },
  };
}

/** Format a generation result as an OpenAI chat completion. */
export function formatOpenAIChatCompletionResponse(
  chat: NormalizedChatCompletion,
  result: NormalizedGenerationResult,
  options: { id: string; created: number },
): OpenAIChatCompletionResponse {
  const reasoning = splitReasoningContent(result.text);
  const reasoningContent = result.reasoningContent ?? reasoning.reasoningContent;
  const extractedToolCalls = hasToolOutputEnabled(chat)
    ? extractOpenAIChatToolCalls(reasoning.content)
    : null;
  const content = extractedToolCalls?.content ?? reasoning.content;
  return {
    id: options.id,
    object: "chat.completion",
    created: options.created,
    model: chat.model,
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: extractedToolCalls !== null && content === "" ? null : content,
          ...(reasoningContent === undefined ? {} : { reasoning_content: reasoningContent }),
          ...(extractedToolCalls === null ? {} : { tool_calls: extractedToolCalls.toolCalls }),
        },
        finish_reason:
          extractedToolCalls === null ? finishReason(result.finishReason) : "tool_calls",
      },
    ],
    ...(result.usage === undefined ? {} : { usage: formatUsage(result.usage) }),
  };
}
