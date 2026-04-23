/**
 * OpenAI-compatible completions protocol adapter.
 * @module
 */

import { isRecord, ServeError } from "../errors";
import type {
  GenerationInput,
  GenerationUsage,
  NormalizedFinishReason,
  NormalizedGenerationRequest,
  NormalizedGenerationResult,
} from "../types";

export type OpenAICompletionUsage = {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
};

export type OpenAICompletionChoice = {
  text: string;
  index: number;
  logprobs: null;
  finish_reason: "stop" | "length" | "content_filter" | null;
};

export type OpenAICompletionResponse = {
  id: string;
  object: "text_completion";
  created: number;
  model: string;
  choices: OpenAICompletionChoice[];
  usage?: OpenAICompletionUsage | null;
};

export type NormalizedCompletionBatch = {
  model: string;
  stream: boolean;
  streamOptions: {
    includeUsage: boolean;
  };
  requests: NormalizedGenerationRequest[];
};

const DEFAULT_COMPLETION_MAX_TOKENS = 16;

function stringField(record: Record<string, unknown>, key: string): string {
  const value = record[key];
  if (typeof value !== "string" || value.trim() === "") {
    throw new ServeError(`OpenAI completions: "${key}" must be a non-empty string.`, {
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
    throw new ServeError(`OpenAI completions: "${key}" must be a boolean.`, { param: key });
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
    throw new ServeError(`OpenAI completions: "${key}" must be ${description}.`, { param: key });
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

function tokenId(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= 0;
}

function tokenPrompt(value: unknown): readonly number[] | null {
  if (!Array.isArray(value) || !value.every(tokenId)) {
    return null;
  }
  if (value.length === 0) {
    throw new ServeError("OpenAI completions: token prompts must contain at least one token.", {
      param: "prompt",
    });
  }
  return value;
}

function promptInputs(record: Record<string, unknown>): GenerationInput[] {
  const value = record.prompt;
  if (typeof value === "string") {
    return [{ kind: "text", text: value }];
  }
  if (Array.isArray(value) && value.length === 0) {
    throw new ServeError('OpenAI completions: "prompt" must contain at least one prompt.', {
      param: "prompt",
    });
  }
  if (Array.isArray(value) && value.every((entry) => typeof entry === "string")) {
    return value.map((text) => ({ kind: "text", text }));
  }

  const tokens = tokenPrompt(value);
  if (tokens !== null) {
    return [{ kind: "tokens", tokenIds: [...tokens] }];
  }

  if (Array.isArray(value)) {
    const tokenPrompts = value.map(tokenPrompt);
    if (tokenPrompts.every((entry) => entry !== null)) {
      return tokenPrompts.map((entry) => ({ kind: "tokens", tokenIds: [...entry] }));
    }
  }

  throw new ServeError(
    'OpenAI completions: "prompt" must be a string, string array, token id array, or array of token id arrays.',
    { param: "prompt" },
  );
}

function stopSequences(record: Record<string, unknown>): readonly string[] | undefined {
  const value = record.stop;
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value === "string") {
    return [value];
  }
  if (Array.isArray(value) && value.every((entry) => typeof entry === "string")) {
    if (value.length > 4) {
      throw new ServeError('OpenAI completions: "stop" can contain at most 4 sequences.', {
        param: "stop",
      });
    }
    return value;
  }
  throw new ServeError('OpenAI completions: "stop" must be a string, string array, or null.', {
    param: "stop",
  });
}

function optionalString(record: Record<string, unknown>, key: string): string | undefined {
  const value = record[key];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "string") {
    throw new ServeError(`OpenAI completions: "${key}" must be a string.`, { param: key });
  }
  return value;
}

function optionalNoOpInteger(record: Record<string, unknown>, key: string, noOp: number): void {
  const value = optionalInteger(record, key, (entry) => entry > 0, "a positive integer");
  if (value !== undefined && value !== noOp) {
    throw new ServeError(`OpenAI completions: "${key}" currently supports only ${noOp}.`, {
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
    throw new ServeError(`OpenAI completions: "${key}" currently supports only ${supported}.`, {
      param: key,
    });
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
    throw new ServeError(`OpenAI completions: non-zero "${key}" is not supported yet.`, {
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
    throw new ServeError('OpenAI completions: "logit_bias" must be an object or null.', {
      param: "logit_bias",
    });
  }
  if (Object.keys(value).length > 0) {
    throw new ServeError('OpenAI completions: non-empty "logit_bias" is not supported yet.', {
      param: "logit_bias",
    });
  }
}

function rejectLogprobs(record: Record<string, unknown>): void {
  const value = record.logprobs;
  if (value === undefined || value === null) {
    return;
  }
  throw new ServeError('OpenAI completions: "logprobs" is not supported yet.', {
    param: "logprobs",
  });
}

function rejectSuffix(record: Record<string, unknown>): void {
  const value = record.suffix;
  if (value === undefined || value === null || value === "") {
    return;
  }
  if (typeof value !== "string") {
    throw new ServeError('OpenAI completions: "suffix" must be a string or null.', {
      param: "suffix",
    });
  }
  throw new ServeError('OpenAI completions: non-empty "suffix" is not supported yet.', {
    param: "suffix",
  });
}

function validateNoOpCompletionFields(record: Record<string, unknown>): void {
  optionalNoOpInteger(record, "n", 1);
  optionalNoOpInteger(record, "best_of", 1);
  rejectNonDefaultBoolean(record, "echo", false);
  rejectNonZeroPenalty(record, "presence_penalty");
  rejectNonZeroPenalty(record, "frequency_penalty");
  rejectLogitBias(record);
  rejectLogprobs(record);
  rejectSuffix(record);
}

function finishReason(reason: NormalizedFinishReason): OpenAICompletionChoice["finish_reason"] {
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

function combineUsage(
  results: readonly NormalizedGenerationResult[],
): Required<GenerationUsage> | undefined {
  let sawUsage = false;
  const usage: Required<GenerationUsage> = {
    promptTokens: 0,
    completionTokens: 0,
    totalTokens: 0,
  };

  for (const result of results) {
    const resultUsage = result.usage;
    if (resultUsage === undefined) {
      continue;
    }
    sawUsage = true;
    usage.promptTokens += resultUsage.promptTokens ?? 0;
    usage.completionTokens += resultUsage.completionTokens ?? 0;
    usage.totalTokens += resultUsage.totalTokens ?? 0;
  }

  return sawUsage ? usage : undefined;
}

function formatUsage(usage: Required<GenerationUsage>): OpenAICompletionUsage {
  return {
    prompt_tokens: usage.promptTokens,
    completion_tokens: usage.completionTokens,
    total_tokens: usage.totalTokens,
  };
}

function formatPartialUsage(usage: GenerationUsage): OpenAICompletionUsage {
  return {
    ...(usage.promptTokens === undefined ? {} : { prompt_tokens: usage.promptTokens }),
    ...(usage.completionTokens === undefined ? {} : { completion_tokens: usage.completionTokens }),
    ...(usage.totalTokens === undefined ? {} : { total_tokens: usage.totalTokens }),
  };
}

function streamOptions(
  record: Record<string, unknown>,
  stream: boolean,
): NormalizedCompletionBatch["streamOptions"] {
  const value = record.stream_options;
  if (value === undefined || value === null) {
    return { includeUsage: false };
  }
  if (!stream) {
    throw new ServeError('OpenAI completions: "stream_options" requires "stream": true.', {
      param: "stream_options",
    });
  }
  if (!isRecord(value)) {
    throw new ServeError('OpenAI completions: "stream_options" must be an object or null.', {
      param: "stream_options",
    });
  }

  const includeUsage = optionalBoolean(value, "include_usage") ?? false;
  const includeObfuscation = optionalBoolean(value, "include_obfuscation") ?? false;
  if (includeObfuscation) {
    throw new ServeError(
      'OpenAI completions: "stream_options.include_obfuscation" is not supported yet.',
      { param: "stream_options" },
    );
  }
  return { includeUsage };
}

/** Normalize an OpenAI completions JSON body into protocol-neutral generation requests. */
export function normalizeOpenAICompletionRequest(
  body: unknown,
  options: { id: string },
): NormalizedCompletionBatch {
  if (!isRecord(body)) {
    throw new ServeError("OpenAI completions: request body must be a JSON object.");
  }

  validateNoOpCompletionFields(body);
  const model = stringField(body, "model");
  const inputs = promptInputs(body);
  const maxTokens =
    optionalInteger(body, "max_tokens", (value) => value >= 0, "a non-negative integer") ??
    DEFAULT_COMPLETION_MAX_TOKENS;
  const temperature = optionalNumber(
    body,
    "temperature",
    (value) => value >= 0 && value <= 2,
    "a number between 0 and 2",
  );
  const topP = optionalNumber(body, "top_p", (value) => value > 0 && value <= 1, "0 < value <= 1");
  const topK = optionalInteger(body, "top_k", (value) => value > 0, "a positive integer");
  const stream = optionalBoolean(body, "stream") ?? false;
  const seed = optionalInteger(body, "seed", (value) => value >= 0, "a non-negative integer");
  const stop = stopSequences(body);
  const user = optionalString(body, "user");
  const parsedStreamOptions = streamOptions(body, stream);

  return {
    model,
    stream,
    streamOptions: parsedStreamOptions,
    requests: inputs.map((input, index) => ({
      id: inputs.length === 1 ? options.id : `${options.id}-${index}`,
      model,
      input,
      sampling: {
        maxTokens,
        ...(temperature === undefined ? {} : { temperature }),
        ...(topP === undefined ? {} : { topP }),
        ...(topK === undefined ? {} : { topK }),
        ...(seed === undefined ? {} : { seed }),
        ...(stop === undefined ? {} : { stop }),
      },
      stream,
      protocol: "openai.completions",
      metadata: {
        promptIndex: index,
        ...(user === undefined ? {} : { user }),
      },
    })),
  };
}

/** Format generation results as an OpenAI completions response. */
export function formatOpenAICompletionResponse(
  batch: NormalizedCompletionBatch,
  results: readonly NormalizedGenerationResult[],
  options: { id: string; created: number },
): OpenAICompletionResponse {
  const usage = combineUsage(results);
  return {
    id: options.id,
    object: "text_completion",
    created: options.created,
    model: batch.model,
    choices: results.map((result, index) => ({
      text: result.text,
      index,
      logprobs: null,
      finish_reason: finishReason(result.finishReason),
    })),
    ...(usage === undefined ? {} : { usage: formatUsage(usage) }),
  };
}

export function formatOpenAICompletionStreamChunk(
  request: NormalizedGenerationRequest,
  text: string,
  options: {
    id: string;
    created: number;
    finishReason?: NormalizedFinishReason | null;
    includeUsage?: boolean;
  },
): OpenAICompletionResponse {
  return {
    id: options.id,
    object: "text_completion",
    created: options.created,
    model: request.model,
    choices: [
      {
        text,
        index: 0,
        logprobs: null,
        finish_reason:
          options.finishReason === undefined || options.finishReason === null
            ? null
            : finishReason(options.finishReason),
      },
    ],
    ...(options.includeUsage ? { usage: null } : {}),
  };
}

export function formatOpenAICompletionUsageStreamChunk(
  batch: NormalizedCompletionBatch,
  usage: GenerationUsage | undefined,
  options: { id: string; created: number },
): OpenAICompletionResponse {
  return {
    id: options.id,
    object: "text_completion",
    created: options.created,
    model: batch.model,
    choices: [],
    usage: usage === undefined ? null : formatPartialUsage(usage),
  };
}
