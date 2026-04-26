/**
 * Server introspection response helpers.
 * @module
 */

import { jsonResponse, ServeError } from "./errors";
import {
  formatOpenAIModelResponse,
  formatOpenAIModelsResponse,
  parseOpenAIModelIdPath,
  type ServedModelInfo,
} from "./protocols/openai-models";
import type { GenerationEngine } from "./types";

export type ServeRuntimeLimits = {
  maxGeneratedTokens?: number;
  maxPromptTokens?: number;
  maxTotalTokens?: number;
  maxBatchSize?: number;
  batchWindowMs?: number;
  streamDecodeInterval?: number;
  maxConcurrentRequests?: number;
  gpuMemoryUtilization?: number;
};

export type ServeInfoModel = {
  id: string;
  context_window: number | null;
  max_prompt_tokens: number | null;
  max_total_tokens: number | null;
  effective_total_tokens: number | null;
};

export type ServeInfoResponse = {
  status: "ok";
  router: "@mlxts/serve";
  version: null;
  model_id: string | null;
  model_ids: string[];
  model_count: number;
  models: ServeInfoModel[];
  endpoints: string[];
  limits: {
    max_generated_tokens: number | null;
    max_prompt_tokens: number | null;
    max_total_tokens: number | null;
    max_client_batch_size: number | null;
    batch_window_ms: number | null;
    stream_decode_interval: number | null;
    max_concurrent_requests: number | null;
    gpu_memory_utilization: number | null;
  };
  capabilities: {
    completions: true;
    chat_completions: true;
    responses: "text_only";
    sse_streaming: boolean;
    batch_generation: boolean;
    reasoning_content: true;
    tool_calls: true;
  };
};

export type ServeInfoOptions = {
  engine: GenerationEngine;
  models?: readonly ServedModelInfo[];
  limits?: ServeRuntimeLimits;
};

const SERVE_ENDPOINTS = [
  "/health",
  "/info",
  "/v1/models",
  "/v1/completions",
  "/v1/chat/completions",
  "/v1/responses",
];

function numberOrNull(value: number | undefined): number | null {
  return value ?? null;
}

function formatServeInfoModel(model: ServedModelInfo, limits: ServeRuntimeLimits): ServeInfoModel {
  return {
    id: model.id,
    context_window: numberOrNull(model.admission?.contextWindow),
    max_prompt_tokens: numberOrNull(model.admission?.maxPromptTokens ?? limits.maxPromptTokens),
    max_total_tokens: numberOrNull(model.admission?.maxTotalTokens ?? limits.maxTotalTokens),
    effective_total_tokens: numberOrNull(model.admission?.effectiveTotalTokens),
  };
}

function servedModelById(models: readonly ServedModelInfo[], id: string): ServedModelInfo {
  const model = models.find((entry) => entry.id === id);
  if (model === undefined) {
    throw new ServeError(`Model "${id}" is not served by this endpoint.`, {
      code: "model_not_found",
      param: "model",
      status: 404,
    });
  }
  return model;
}

/** Format the lightweight `/info` operator response. */
export function formatServeInfoResponse(options: ServeInfoOptions): ServeInfoResponse {
  const models = options.models ?? [];
  const modelIds = models.map((model) => model.id);
  const limits = options.limits ?? {};
  return {
    status: "ok",
    router: "@mlxts/serve",
    version: null,
    model_id: modelIds[0] ?? null,
    model_ids: modelIds,
    model_count: modelIds.length,
    models: models.map((model) => formatServeInfoModel(model, limits)),
    endpoints: [...SERVE_ENDPOINTS],
    limits: {
      max_generated_tokens: numberOrNull(limits.maxGeneratedTokens),
      max_prompt_tokens: numberOrNull(limits.maxPromptTokens),
      max_total_tokens: numberOrNull(limits.maxTotalTokens),
      max_client_batch_size: numberOrNull(limits.maxBatchSize),
      batch_window_ms: numberOrNull(limits.batchWindowMs),
      stream_decode_interval: numberOrNull(limits.streamDecodeInterval),
      max_concurrent_requests: numberOrNull(limits.maxConcurrentRequests),
      gpu_memory_utilization: numberOrNull(limits.gpuMemoryUtilization),
    },
    capabilities: {
      completions: true,
      chat_completions: true,
      responses: "text_only",
      sse_streaming: options.engine.stream !== undefined,
      batch_generation: options.engine.generateBatch !== undefined,
      reasoning_content: true,
      tool_calls: true,
    },
  };
}

/** Return the `/info` response. */
export function serveInfoResponse(options: ServeInfoOptions): Response {
  return jsonResponse(formatServeInfoResponse(options));
}

export function openAIModelRouteResponse(
  pathname: string,
  options: { models?: readonly ServedModelInfo[]; now?: () => Date },
): Response | null {
  if (pathname === "/v1/models") {
    const created = Math.floor((options.now?.() ?? new Date()).getTime() / 1000);
    return jsonResponse(formatOpenAIModelsResponse(options.models ?? [], { created }));
  }

  if (pathname.startsWith("/v1/models/")) {
    const modelId = parseOpenAIModelIdPath(pathname);
    const created = Math.floor((options.now?.() ?? new Date()).getTime() / 1000);
    return jsonResponse(
      formatOpenAIModelResponse(servedModelById(options.models ?? [], modelId), { created }),
    );
  }

  return null;
}
