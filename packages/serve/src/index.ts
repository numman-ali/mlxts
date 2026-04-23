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
  createModelRouterGenerationEngine,
  type ModelRouterGenerationEngineOptions,
} from "./model-router";
export {
  DEFAULT_MODEL_SERVER_BATCH_WINDOW_MS,
  DEFAULT_MODEL_SERVER_HOSTNAME,
  DEFAULT_MODEL_SERVER_MAX_BATCH_SIZE,
  DEFAULT_MODEL_SERVER_MAX_CONCURRENT_REQUESTS,
  DEFAULT_MODEL_SERVER_MAX_GENERATED_TOKENS,
  DEFAULT_MODEL_SERVER_MAX_TOTAL_TOKENS,
  DEFAULT_MODEL_SERVER_PORT,
  type LoadedModelServerEntry,
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
  formatOpenAIModelsResponse,
  type OpenAIModelInfo,
  type OpenAIModelsResponse,
  type ServedModelInfo,
} from "./protocols/openai-models";
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
  createTransformersGenerationEngine,
  type TransformersGenerationEngineOptions,
} from "./transformers-engine";
export type {
  GenerationEngine,
  GenerationInput,
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
