/**
 * OpenAI-compatible Responses protocol adapter.
 * @module
 */

import { isRecord, ServeError } from "../errors";
import type { GenerationInput, NormalizedGenerationRequest } from "../types";
import { parseOpenAIResponseInput } from "./openai-responses-input";
import { parseOpenAIStopSequences } from "./openai-stop";

export {
  formatOpenAIResponse,
  formatOpenAIResponsePending,
} from "./openai-responses-formatting";

export type OpenAIResponseUsage = {
  input_tokens: number;
  input_tokens_details: {
    cached_tokens: number;
  };
  output_tokens: number;
  output_tokens_details: {
    reasoning_tokens: number;
  };
  total_tokens: number;
};

export type OpenAIResponseOutputText = {
  type: "output_text";
  text: string;
  annotations: unknown[];
};

export type OpenAIResponseMessageItem = {
  id: string;
  type: "message";
  status: "completed";
  role: "assistant";
  content: OpenAIResponseOutputText[];
};

export type OpenAIResponseReasoningItem = {
  id: string;
  type: "reasoning";
  status: "completed";
  summary: unknown[];
  content: {
    type: "reasoning_text";
    text: string;
  }[];
};

export type OpenAIResponseOutputItem = OpenAIResponseReasoningItem | OpenAIResponseMessageItem;

export type OpenAIResponseObject = {
  id: string;
  object: "response";
  created_at: number;
  status: "completed" | "in_progress" | "incomplete";
  completed_at: number | null;
  error: null;
  incomplete_details: { reason: "max_output_tokens" } | null;
  instructions: string | null;
  max_output_tokens: number | null;
  model: string;
  output: OpenAIResponseOutputItem[];
  output_text: string;
  parallel_tool_calls: boolean;
  previous_response_id: null;
  reasoning: {
    effort: null;
    summary: null;
  };
  store: boolean;
  temperature: number | null;
  text: {
    format: {
      type: "text";
    };
  };
  tool_choice: "auto" | "none";
  tools: [];
  top_p: number | null;
  truncation: "disabled";
  usage: OpenAIResponseUsage | null;
  user: string | null;
  metadata: Record<string, string>;
};

export type NormalizedOpenAIResponse = {
  model: string;
  stream: boolean;
  streamOptions: {
    includeObfuscation: boolean;
  };
  instructions: string | null;
  maxOutputTokens: number | null;
  temperature: number | null;
  topP: number | null;
  toolChoice: "auto" | "none";
  parallelToolCalls: boolean;
  metadata: Record<string, string>;
  user: string | null;
  request: NormalizedGenerationRequest;
};

const DEFAULT_RESPONSE_MAX_OUTPUT_TOKENS = 16;

function stringField(record: Record<string, unknown>, key: string): string {
  const value = record[key];
  if (typeof value !== "string" || value.trim() === "") {
    throw new ServeError(`OpenAI responses: "${key}" must be a non-empty string.`, {
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
    throw new ServeError(`OpenAI responses: "${key}" must be a string.`, { param: key });
  }
  return value;
}

function optionalBoolean(record: Record<string, unknown>, key: string): boolean | undefined {
  const value = record[key];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "boolean") {
    throw new ServeError(`OpenAI responses: "${key}" must be a boolean.`, { param: key });
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
    throw new ServeError(`OpenAI responses: "${key}" must be ${description}.`, { param: key });
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

function rejectPresent(record: Record<string, unknown>, key: string, description: string): void {
  const value = record[key];
  if (value !== undefined && value !== null) {
    throw new ServeError(`OpenAI responses: "${key}" is not supported yet. ${description}`, {
      param: key,
    });
  }
}

function rejectTrueBoolean(
  record: Record<string, unknown>,
  key: string,
  description: string,
): void {
  const value = optionalBoolean(record, key);
  if (value === true) {
    throw new ServeError(`OpenAI responses: "${key}" is not supported yet. ${description}`, {
      param: key,
    });
  }
}

function parseInstructions(record: Record<string, unknown>): string | null {
  const instructions = optionalString(record, "instructions");
  return instructions === undefined ? null : instructions;
}

function parseMaxOutputTokens(record: Record<string, unknown>): {
  maxTokens: number;
  maxOutputTokens: number | null;
} {
  const maxOutputTokens = optionalInteger(
    record,
    "max_output_tokens",
    (value) => value >= 0,
    "a non-negative integer",
  );
  return {
    maxTokens: maxOutputTokens ?? DEFAULT_RESPONSE_MAX_OUTPUT_TOKENS,
    maxOutputTokens: maxOutputTokens ?? null,
  };
}

function parseTools(record: Record<string, unknown>): [] {
  const tools = record.tools;
  if (tools === undefined || tools === null) {
    return [];
  }
  if (!Array.isArray(tools)) {
    throw new ServeError('OpenAI responses: "tools" must be an array or null.', {
      param: "tools",
    });
  }
  if (tools.length > 0) {
    throw new ServeError("OpenAI responses: tools are not supported by this endpoint yet.", {
      param: "tools",
    });
  }
  return [];
}

function parseToolChoice(record: Record<string, unknown>): "auto" | "none" {
  const toolChoice = record.tool_choice;
  if (toolChoice === undefined || toolChoice === null || toolChoice === "auto") {
    return "auto";
  }
  if (toolChoice === "none") {
    return "none";
  }
  throw new ServeError(
    'OpenAI responses: "tool_choice" currently supports only "auto" or "none".',
    {
      param: "tool_choice",
    },
  );
}

function parseMetadata(record: Record<string, unknown>): Record<string, string> {
  const metadata = record.metadata;
  if (metadata === undefined || metadata === null) {
    return {};
  }
  if (!isRecord(metadata)) {
    throw new ServeError('OpenAI responses: "metadata" must be an object or null.', {
      param: "metadata",
    });
  }

  const parsed: Record<string, string> = {};
  for (const [key, value] of Object.entries(metadata)) {
    if (typeof value !== "string") {
      throw new ServeError('OpenAI responses: "metadata" values must be strings.', {
        param: "metadata",
      });
    }
    parsed[key] = value;
  }
  return parsed;
}

function parseStreamOptions(
  record: Record<string, unknown>,
  stream: boolean,
): NormalizedOpenAIResponse["streamOptions"] {
  const value = record.stream_options;
  if (value === undefined || value === null) {
    return { includeObfuscation: false };
  }
  if (!stream) {
    throw new ServeError('OpenAI responses: "stream_options" requires "stream": true.', {
      param: "stream_options",
    });
  }
  if (!isRecord(value)) {
    throw new ServeError('OpenAI responses: "stream_options" must be an object or null.', {
      param: "stream_options",
    });
  }
  const includeObfuscation = optionalBoolean(value, "include_obfuscation") ?? false;
  if (includeObfuscation) {
    throw new ServeError(
      'OpenAI responses: "stream_options.include_obfuscation" is not supported yet.',
      { param: "stream_options" },
    );
  }
  return { includeObfuscation };
}

function validateTextFormat(record: Record<string, unknown>): void {
  const text = record.text;
  if (text === undefined || text === null) {
    return;
  }
  if (!isRecord(text)) {
    throw new ServeError('OpenAI responses: "text" must be an object or null.', {
      param: "text",
    });
  }
  const format = text.format;
  if (format === undefined || format === null) {
    return;
  }
  if (!isRecord(format) || format.type !== "text") {
    throw new ServeError("OpenAI responses: only text output format is supported today.", {
      param: "text",
    });
  }
}

function validateModalities(record: Record<string, unknown>): void {
  const modalities = record.modalities;
  if (modalities === undefined || modalities === null) {
    return;
  }
  if (!Array.isArray(modalities) || modalities.length !== 1 || modalities[0] !== "text") {
    throw new ServeError('OpenAI responses: only ["text"] modalities are supported today.', {
      param: "modalities",
    });
  }
}

function validateStorage(record: Record<string, unknown>): void {
  rejectTrueBoolean(record, "store", "Local serving does not persist response state.");
  rejectTrueBoolean(record, "background", "Background responses require persisted jobs.");
}

function validateUnsupportedState(record: Record<string, unknown>): void {
  rejectPresent(
    record,
    "previous_response_id",
    "Stateful response continuation is not implemented.",
  );
  rejectPresent(record, "conversation", "Conversation state is not implemented.");
  rejectPresent(record, "prompt", "Prompt templates are not implemented.");
  rejectPresent(record, "prompt_cache_retention", "Prompt cache retention is not implemented.");
  rejectPresent(record, "include", "Response includes are not implemented.");
  rejectPresent(record, "max_tool_calls", "Tool execution is not implemented.");
}

function validateNoOpFields(record: Record<string, unknown>): void {
  parseTools(record);
  parseToolChoice(record);
  validateStorage(record);
  validateUnsupportedState(record);
  validateTextFormat(record);
  validateModalities(record);
  const truncation = optionalString(record, "truncation");
  if (truncation !== undefined && truncation !== "disabled") {
    throw new ServeError('OpenAI responses: only "truncation": "disabled" is supported today.', {
      param: "truncation",
    });
  }
  const reasoning = record.reasoning;
  if (reasoning !== undefined && reasoning !== null) {
    throw new ServeError("OpenAI responses: reasoning controls are not supported yet.", {
      param: "reasoning",
    });
  }
}

function responseTemplateFlag(
  record: Record<string, unknown>,
  key: "enable_thinking" | "preserve_thinking",
): boolean | undefined {
  return optionalBoolean(record, key);
}

function chatTemplateOptions(record: Record<string, unknown>) {
  const kwargs = record.chat_template_kwargs;
  if (kwargs !== undefined && kwargs !== null && !isRecord(kwargs)) {
    throw new ServeError('OpenAI responses: "chat_template_kwargs" must be an object or null.', {
      param: "chat_template_kwargs",
    });
  }
  const templateRecord = isRecord(kwargs) ? kwargs : {};
  const enableThinking =
    responseTemplateFlag(templateRecord, "enable_thinking") ??
    responseTemplateFlag(record, "enable_thinking");
  const preserveThinking =
    responseTemplateFlag(templateRecord, "preserve_thinking") ??
    responseTemplateFlag(record, "preserve_thinking");

  return {
    ...(enableThinking === undefined ? {} : { enableThinking }),
    ...(preserveThinking === undefined ? {} : { preserveThinking }),
  };
}

function responseInputWithTemplate(
  input: GenerationInput,
  templateOptions: { enableThinking?: boolean; preserveThinking?: boolean },
): GenerationInput {
  if (Object.keys(templateOptions).length === 0) {
    return input;
  }
  if (input.kind === "messages") {
    return { ...input, chatTemplate: templateOptions };
  }
  if (input.kind === "content") {
    return { ...input, chatTemplate: templateOptions };
  }
  return input;
}

/** Normalize an OpenAI Responses JSON body into one generation request. */
export function normalizeOpenAIResponseRequest(
  body: unknown,
  options: { id: string },
): NormalizedOpenAIResponse {
  if (!isRecord(body)) {
    throw new ServeError("OpenAI responses: request body must be a JSON object.");
  }

  validateNoOpFields(body);
  const model = stringField(body, "model");
  const stream = optionalBoolean(body, "stream") ?? false;
  const streamOptions = parseStreamOptions(body, stream);
  const instructions = parseInstructions(body);
  const input = parseOpenAIResponseInput(body, instructions);
  const maxTokens = parseMaxOutputTokens(body);
  const temperature = optionalNumber(
    body,
    "temperature",
    (value) => value >= 0 && value <= 2,
    "a number between 0 and 2",
  );
  const topP = optionalNumber(body, "top_p", (value) => value > 0 && value <= 1, "0 < value <= 1");
  const topK = optionalInteger(body, "top_k", (value) => value > 0, "a positive integer");
  const seed = optionalInteger(body, "seed", (value) => value >= 0, "a non-negative integer");
  const stop = parseOpenAIStopSequences(body, "responses");
  const metadata = parseMetadata(body);
  const user = optionalString(body, "user") ?? optionalString(body, "safety_identifier") ?? null;
  const promptCacheKey = optionalString(body, "prompt_cache_key");
  const toolChoice = parseToolChoice(body);
  const parallelToolCalls = optionalBoolean(body, "parallel_tool_calls") ?? true;
  const templateOptions = chatTemplateOptions(body);
  return {
    model,
    stream,
    streamOptions,
    instructions,
    maxOutputTokens: maxTokens.maxOutputTokens,
    temperature: temperature ?? null,
    topP: topP ?? null,
    toolChoice,
    parallelToolCalls,
    metadata,
    user,
    request: {
      id: options.id,
      model,
      input: responseInputWithTemplate(input, templateOptions),
      sampling: {
        maxTokens: maxTokens.maxTokens,
        ...(temperature === undefined ? {} : { temperature }),
        ...(topP === undefined ? {} : { topP }),
        ...(topK === undefined ? {} : { topK }),
        ...(seed === undefined ? {} : { seed }),
        ...(stop === undefined ? {} : { stop }),
      },
      stream,
      protocol: "openai.responses",
      metadata: {
        ...metadata,
        ...(user === null ? {} : { user }),
        ...(promptCacheKey === undefined ? {} : { promptCacheKey }),
      },
    },
  };
}
