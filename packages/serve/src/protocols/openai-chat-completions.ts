/**
 * OpenAI-compatible chat completions protocol adapter.
 * @module
 */

import type { ChatToolCall } from "@mlxts/transformers";
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
import {
  parseOpenAIChatMessages,
  parseOpenAIChatTools,
  validateOpenAIChatToolChoice,
} from "./openai-chat-messages";
import { extractOpenAIChatToolCalls } from "./openai-chat-tool-calls";
import { parseOpenAIStopSequences } from "./openai-stop";

export type {
  OpenAIChatCompletionChunk,
  OpenAIChatCompletionChunkChoice,
  OpenAIChatCompletionStreamDelta,
  OpenAIChatCompletionStreamToolCall,
  OpenAIChatCompletionUsage,
} from "./openai-chat-completion-streaming";
export {
  createOpenAIChatCompletionReasoningStream,
  formatOpenAIChatCompletionStreamChunk,
  formatOpenAIChatCompletionUsageStreamChunk,
} from "./openai-chat-completion-streaming";
export { createOpenAIChatCompletionToolCallStream } from "./openai-chat-tool-call-stream";

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

function optionalString(record: Record<string, unknown>, key: string): string | undefined {
  const value = record[key];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "string") {
    throw new ServeError(`OpenAI chat completions: "${key}" must be a string.`, { param: key });
  }
  return value;
}

function optionalNoOpInteger(record: Record<string, unknown>, key: string, noOp: number): void {
  const value = optionalInteger(record, key, (entry) => entry > 0, "a positive integer");
  if (value !== undefined && value !== noOp) {
    throw new ServeError(`OpenAI chat completions: "${key}" currently supports only ${noOp}.`, {
      param: key,
    });
  }
}

function rejectNonDefaultBoolean(
  record: Record<string, unknown>,
  key: string,
  supported: boolean,
): void {
  const value = optionalBoolean(record, key);
  if (value !== undefined && value !== supported) {
    throw new ServeError(
      `OpenAI chat completions: "${key}" currently supports only ${supported}.`,
      {
        param: key,
      },
    );
  }
}

function rejectNonZeroPenalty(record: Record<string, unknown>, key: string): void {
  const value = optionalNumber(
    record,
    key,
    (entry) => entry >= -2 && entry <= 2,
    "a number between -2 and 2",
  );
  if (value !== undefined && value !== 0) {
    throw new ServeError(`OpenAI chat completions: non-zero "${key}" is not supported yet.`, {
      param: key,
    });
  }
}

function rejectLogitBias(record: Record<string, unknown>): void {
  const value = record.logit_bias;
  if (value === undefined || value === null) {
    return;
  }
  if (!isRecord(value)) {
    throw new ServeError('OpenAI chat completions: "logit_bias" must be an object or null.', {
      param: "logit_bias",
    });
  }
  if (Object.keys(value).length > 0) {
    throw new ServeError('OpenAI chat completions: non-empty "logit_bias" is not supported yet.', {
      param: "logit_bias",
    });
  }
}

function rejectTopLogprobs(record: Record<string, unknown>): void {
  const value = optionalInteger(
    record,
    "top_logprobs",
    (entry) => entry >= 0 && entry <= 20,
    "an integer between 0 and 20",
  );
  if (value !== undefined) {
    throw new ServeError('OpenAI chat completions: "top_logprobs" is not supported yet.', {
      param: "top_logprobs",
    });
  }
}

function validateResponseFormat(record: Record<string, unknown>): void {
  const value = record.response_format;
  if (value === undefined || value === null) {
    return;
  }
  if (!isRecord(value)) {
    throw new ServeError('OpenAI chat completions: "response_format" must be an object or null.', {
      param: "response_format",
    });
  }
  if (value.type === "text") {
    return;
  }
  if (typeof value.type !== "string" || value.type.trim() === "") {
    throw new ServeError(
      'OpenAI chat completions: "response_format.type" must be a non-empty string.',
      { param: "response_format" },
    );
  }
  throw new ServeError(
    `OpenAI chat completions: response_format "${value.type}" is not supported yet.`,
    { param: "response_format" },
  );
}

function validateNoOpChatFields(record: Record<string, unknown>): void {
  optionalNoOpInteger(record, "n", 1);
  rejectNonDefaultBoolean(record, "parallel_tool_calls", true);
  rejectNonDefaultBoolean(record, "logprobs", false);
  rejectNonZeroPenalty(record, "presence_penalty");
  rejectNonZeroPenalty(record, "frequency_penalty");
  rejectLogitBias(record);
  rejectTopLogprobs(record);
  validateResponseFormat(record);
}

function chatMaxTokens(record: Record<string, unknown>): number {
  const maxTokens = optionalInteger(
    record,
    "max_tokens",
    (value) => value >= 0,
    "a non-negative integer",
  );
  const maxCompletionTokens = optionalInteger(
    record,
    "max_completion_tokens",
    (value) => value >= 0,
    "a non-negative integer",
  );
  if (
    maxTokens !== undefined &&
    maxCompletionTokens !== undefined &&
    maxTokens !== maxCompletionTokens
  ) {
    throw new ServeError(
      'OpenAI chat completions: "max_tokens" and "max_completion_tokens" must match when both are provided.',
      { param: "max_completion_tokens" },
    );
  }
  return maxCompletionTokens ?? maxTokens ?? DEFAULT_CHAT_MAX_TOKENS;
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

  validateNoOpChatFields(body);
  validateOpenAIChatToolChoice(body);
  const stream = optionalBoolean(body, "stream") ?? false;
  const model = stringField(body, "model");
  const parsedTools = body.tool_choice === "none" ? undefined : parseOpenAIChatTools(body);
  const templateOptions = chatTemplateOptions(body);
  const temperature = optionalNumber(
    body,
    "temperature",
    (value) => value >= 0 && value <= 2,
    "a number between 0 and 2",
  );
  const topP = optionalNumber(body, "top_p", (value) => value > 0 && value <= 1, "0 < value <= 1");
  const topK = optionalInteger(body, "top_k", (value) => value > 0, "a positive integer");
  const ignoreEos = optionalBoolean(body, "ignore_eos");
  const seed = optionalInteger(body, "seed", (value) => value >= 0, "a non-negative integer");
  const stop = parseOpenAIStopSequences(body, "chat completions");
  const user = optionalString(body, "user");
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
        messages: parseOpenAIChatMessages(body),
        ...(parsedTools === undefined ? {} : { tools: parsedTools }),
        ...(Object.keys(templateOptions).length === 0 ? {} : { chatTemplate: templateOptions }),
      },
      sampling: {
        maxTokens: chatMaxTokens(body),
        ...(temperature === undefined ? {} : { temperature }),
        ...(topP === undefined ? {} : { topP }),
        ...(topK === undefined ? {} : { topK }),
        ...(ignoreEos === undefined ? {} : { ignoreEos }),
        ...(seed === undefined ? {} : { seed }),
        ...(stop === undefined ? {} : { stop }),
      },
      stream,
      protocol: "openai.chat_completions",
      ...(user === undefined ? {} : { metadata: { user } }),
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
