/**
 * Pretrained decoder loading and explicit generation for mlxts.
 * @module
 */

export { AutoModel, AutoTokenizer } from "./auto";
export {
  type ChatMessage,
  type ChatTemplate,
  type ChatTool,
  type ChatToolCall,
  loadChatTemplate,
} from "./chat-template";
export {
  expandQwen3_5ImageTokens,
  prepareQwen3_5ImagePrompt,
} from "./families/qwen3_5/conditional";
export {
  type DecodedQwen3_5Image,
  loadQwen3_5VisionPreprocessor,
  type PreparedQwen3_5ImageBatch,
  parseQwen3_5VisionPreprocessorConfig,
  prepareQwen3_5ImageBatch,
  type Qwen3_5VisionPreprocessorConfig,
  smartResizeQwen3_5Image,
} from "./families/qwen3_5/preprocessing";
export {
  type ContinuousBatchEvent,
  type ContinuousBatchTokenRequest,
  type ContinuousBatchTokenSchedulerOptions,
  createContinuousBatchTokenScheduler,
  GenerationAbortError,
  generateBatchTokens,
  generatePreparedTokenEvents,
  generatePreparedTokens,
  generateStep,
  generateText,
  generateTextStream,
  generateTokenEvents,
  generateTokens,
  makePromptCache,
} from "./generation";
export {
  BatchKVCache,
  KVCache,
  LayerPatternKVCache,
  SlidingWindowKVCache,
} from "./infrastructure/cache";
export {
  createInteractionProfile,
  type InteractionProfile,
  loadInteractionProfile,
  type PromptCompilation,
} from "./interaction-profile";
export { loadCausalLM, loadPretrainedTokenizer, resolvePretrainedSource } from "./load";
export {
  type CausalLMAdapterFormat,
  type LoadCausalLMAdaptersOptions,
  loadCausalLMAdapters,
  type SaveCausalLMAdaptersOptions,
  saveCausalLMAdapters,
} from "./lora-adapters";
export {
  type LoRATargetPreset,
  type ResolvedLoRATargets,
  type ResolveLoRATargetsOptions,
  resolveLoRATargets,
} from "./lora-targets";
export type { QuantizedSnapshotResult, QuantizePretrainedSnapshotOptions } from "./quantize";
export { quantizePretrainedSnapshot } from "./quantize";
export { FAMILY_REGISTRY, resolveFamily } from "./registry";
export type {
  BaseModelConfig,
  BatchGenerationOptions,
  BatchTokenGenerationEvent,
  CausalLM,
  DecoderCache,
  FamilyRegistration,
  ForwardOptions,
  GenerationDefaults,
  GenerationOptions,
  GenerationResult,
  LoadCausalLMOptions,
  LoadPretrainedTokenizerOptions,
  LoadSourceOptions,
  PrefillProgressEvent,
  PretrainedLoadProgressEvent,
  SamplerOptions,
  SupportedModelFamily,
  TextGenerationResult,
  TokenGenerationEvent,
  TransformerBatchCache,
  TransformerCache,
} from "./types";
export {
  ConfigParseError,
  MissingWeightsError,
  UnsupportedModelError,
  WeightMismatchError,
} from "./types";
