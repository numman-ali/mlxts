/**
 * Shared OpenAI-compatible usage formatting helpers.
 * @module
 */

import type { GenerationUsage } from "../types";

export type OpenAIPromptTokensDetails = {
  cached_tokens?: number;
  cache_write_tokens?: number;
};

export type OpenAICompletionLikeUsage = {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  prompt_tokens_details?: OpenAIPromptTokensDetails;
};

export function formatOpenAICompletionLikeUsage(usage: GenerationUsage): OpenAICompletionLikeUsage {
  return {
    ...(usage.promptTokens === undefined ? {} : { prompt_tokens: usage.promptTokens }),
    ...(usage.completionTokens === undefined ? {} : { completion_tokens: usage.completionTokens }),
    ...(usage.totalTokens === undefined ? {} : { total_tokens: usage.totalTokens }),
    ...(usage.promptTokens === undefined
      ? {}
      : { prompt_tokens_details: promptTokensDetails(usage) }),
  };
}

function promptTokensDetails(usage: GenerationUsage): OpenAIPromptTokensDetails {
  const cacheReadTokens = usage.cacheReadTokens ?? 0;
  const cacheWriteTokens = usage.cacheWriteTokens ?? 0;
  return {
    cached_tokens: cacheReadTokens,
    cache_write_tokens: cacheWriteTokens,
  };
}
