/**
 * Bun-native serving adapters for mlxts generation engines.
 * @module
 */

export {
  createMicroBatchingGenerationEngine,
  type MicroBatchingGenerationEngineOptions,
} from "./batching-engine";
export {
  type ConcurrencyLimitGenerationEngineOptions,
  createConcurrencyLimitGenerationEngine,
} from "./concurrency-engine";
export { ServeError } from "./errors";
export {
  effectiveTotalTokenLimit,
  estimateGenerationMemory,
  type GenerationMemoryEstimate,
  type ModelAdmissionMetadata,
  modelAdmissionMetadata,
  modelContextWindow,
} from "./model-context";
export {
  createModelRouterGenerationEngine,
  type ModelRouterGenerationEngineOptions,
} from "./model-router";
export {
  DEFAULT_MODEL_SERVER_BATCH_WINDOW_MS,
  DEFAULT_MODEL_SERVER_GPU_MEMORY_UTILIZATION,
  DEFAULT_MODEL_SERVER_HOSTNAME,
  DEFAULT_MODEL_SERVER_MAX_BATCH_SIZE,
  DEFAULT_MODEL_SERVER_MAX_CONCURRENT_REQUESTS,
  DEFAULT_MODEL_SERVER_MAX_GENERATED_TOKENS,
  DEFAULT_MODEL_SERVER_MAX_PROMPT_TOKENS,
  DEFAULT_MODEL_SERVER_MAX_TOTAL_TOKENS,
  DEFAULT_MODEL_SERVER_PORT,
  type LoadedModelServerEntry,
  type ModelServerRuntimeOptions,
  type RunningModelServer,
  type ServeLoadedModelOptions,
  type ServeLoadedModelsOptions,
  type ServeModelOptions,
  serveLoadedModel,
  serveLoadedModels,
  serveModel,
} from "./model-server";
export {
  type ServeModelSourceEntry,
  type ServeModelsOptions,
  type ServeModelsProgressContext,
  type ServeModelsRuntime,
  serveModels,
  serveModelsWithRuntime,
} from "./model-sources";
export {
  type AnthropicContentBlock,
  type AnthropicMessageResponse,
  type AnthropicStopReason,
  type AnthropicTextBlock,
  type AnthropicThinkingBlock,
  type AnthropicUsage,
  formatAnthropicMessageResponse,
  type NormalizedAnthropicMessage,
  normalizeAnthropicMessageRequest,
} from "./protocols/anthropic-messages";
export {
  createOpenAIChatCompletionReasoningStream,
  formatOpenAIChatCompletionResponse,
  formatOpenAIChatCompletionStreamChunk,
  formatOpenAIChatCompletionUsageStreamChunk,
  type NormalizedChatCompletion,
  normalizeOpenAIChatCompletionRequest,
  type OpenAIChatCompletionChoice,
  type OpenAIChatCompletionChunk,
  type OpenAIChatCompletionChunkChoice,
  type OpenAIChatCompletionMessage,
  type OpenAIChatCompletionResponse,
  type OpenAIChatCompletionStreamDelta,
  type OpenAIChatCompletionUsage,
} from "./protocols/openai-chat-completions";
export {
  formatOpenAICompletionResponse,
  formatOpenAICompletionStreamChunk,
  formatOpenAICompletionUsageStreamChunk,
  type NormalizedCompletionBatch,
  normalizeOpenAICompletionRequest,
  type OpenAICompletionChoice,
  type OpenAICompletionResponse,
  type OpenAICompletionUsage,
} from "./protocols/openai-completions";
export {
  formatOpenAIModelResponse,
  formatOpenAIModelsResponse,
  type OpenAIModelInfo,
  type OpenAIModelsResponse,
  type ServedModelInfo,
} from "./protocols/openai-models";
export {
  formatOpenAIResponse,
  type NormalizedOpenAIResponse,
  normalizeOpenAIResponseRequest,
  type OpenAIResponseMessageItem,
  type OpenAIResponseObject,
  type OpenAIResponseOutputItem,
  type OpenAIResponseOutputText,
  type OpenAIResponseReasoningItem,
  type OpenAIResponseUsage,
} from "./protocols/openai-responses";
export {
  createRequestLimitGenerationEngine,
  type RequestLimitGenerationEngineOptions,
} from "./request-limits";
export {
  createFetchHandler,
  type ServeAppOptions,
  type ServeServerOptions,
  startServeServer,
} from "./server";
export {
  formatServeInfoResponse,
  type ServeInfoResponse,
  type ServeRuntimeLimits,
} from "./server-info";
export {
  createTransformersGenerationEngine,
  type TransformersGenerationEngineOptions,
} from "./transformers-engine";
export {
  createQwen3_5ImageContentAdapter,
  type LoadedContentPrompt,
  type TransformersContentAdapter,
  type TransformersContentAdapterLoadContext,
  type TransformersContentAdapterModelContext,
} from "./transformers-engine-content";
export type {
  GenerationContentMessage,
  GenerationContentPart,
  GenerationEngine,
  GenerationInput,
  GenerationMediaSource,
  GenerationMemoryUsage,
  GenerationProtocol,
  GenerationSamplingOptions,
  GenerationStreamEvent,
  GenerationUsage,
  NormalizedFinishReason,
  NormalizedGenerationRequest,
  NormalizedGenerationResult,
  ServeEvent,
} from "./types";
