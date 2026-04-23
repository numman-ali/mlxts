/**
 * Bun-native serving adapters for mlxts generation engines.
 * @module
 */

export {
  createMicroBatchingGenerationEngine,
  type MicroBatchingGenerationEngineOptions,
} from "./batching-engine";
export { ServeError } from "./errors";
export {
  createModelRouterGenerationEngine,
  type ModelRouterGenerationEngineOptions,
} from "./model-router";
export {
  DEFAULT_MODEL_SERVER_HOSTNAME,
  DEFAULT_MODEL_SERVER_MAX_GENERATED_TOKENS,
  DEFAULT_MODEL_SERVER_MAX_TOTAL_TOKENS,
  DEFAULT_MODEL_SERVER_PORT,
  type RunningModelServer,
  type ServeLoadedModelOptions,
  type ServeModelOptions,
  serveLoadedModel,
  serveModel,
} from "./model-server";
export {
  formatOpenAIChatCompletionResponse,
  type NormalizedChatCompletion,
  normalizeOpenAIChatCompletionRequest,
  type OpenAIChatCompletionChoice,
  type OpenAIChatCompletionMessage,
  type OpenAIChatCompletionResponse,
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
