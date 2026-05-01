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
export { parseCLIPTextConfig } from "./families/clip/config";
export {
  type LoadCLIPTextModelOptions,
  loadCLIPTextModel,
  loadCLIPTextModelWithProjection,
  resolveCLIPTextModelSource,
} from "./families/clip/load";
export {
  CLIPTextEmbeddings,
  CLIPTextModel,
  CLIPTextModelWithProjection,
  disposeCLIPTextModelOutput,
  disposeCLIPTextProjectionOutput,
} from "./families/clip/model";
export type {
  CLIPHiddenActivation,
  CLIPTextConfig,
  CLIPTextModelOptions,
  CLIPTextModelOutput,
  CLIPTextProjectionOutput,
} from "./families/clip/types";
export {
  type CLIPTextWeightLoadOptions,
  type CLIPTextWeightLoadResult,
  type CLIPTextWeightTarget,
  clipTextWeightPath,
  loadCLIPTextWeights,
} from "./families/clip/weights";
export { parseQwen2Config } from "./families/qwen2/config";
export {
  loadQwen2_5VLTextEncoder,
  loadQwen2CausalLM,
  type Qwen2TextCausalLM,
} from "./families/qwen2/load";
export { parseQwen3Config } from "./families/qwen3/config";
export { loadQwen3CausalLM, type Qwen3TextCausalLM } from "./families/qwen3/load";
export {
  loadQwen3_5ForConditionalGeneration,
  shouldLoadQwen3_5ForConditionalGeneration,
} from "./families/qwen3_5/load";
export {
  expandQwen3_5ImageTokens,
  expandQwen3_5ImageTokensFromGridThw,
  prepareQwen3_5ImagePrompt,
  prepareQwen3_5ImagePromptTokenPlan,
  QWEN3_5_IMAGE_MARKER,
  type Qwen3_5ImagePromptTokenPlan,
} from "./families/qwen3_5/multimodal/conditional";
export type { Qwen3_5ImageGridThw } from "./families/qwen3_5/multimodal/conditional-support";
export {
  type DecodedQwen3_5Image,
  loadQwen3_5VisionPreprocessor,
  type PreparedQwen3_5ImageBatch,
  parseQwen3_5VisionPreprocessorConfig,
  prepareQwen3_5ImageBatch,
  type Qwen3_5VisionPreprocessorConfig,
  qwen3_5ImageGridThwValues,
  smartResizeQwen3_5Image,
} from "./families/qwen3_5/multimodal/preprocessing";
export { parseT5EncoderConfig } from "./families/t5/config";
export {
  type LoadT5EncoderModelOptions,
  loadT5EncoderModel,
  resolveT5EncoderModelSource,
} from "./families/t5/load";
export { disposeT5EncoderModelOutput, T5EncoderModel } from "./families/t5/model";
export type {
  T5DenseActivation,
  T5EncoderConfig,
  T5EncoderModelOptions,
  T5EncoderModelOutput,
  T5FeedForwardProjection,
} from "./families/t5/types";
export {
  loadT5EncoderWeights,
  type T5EncoderWeightLoadOptions,
  type T5EncoderWeightLoadResult,
  t5EncoderWeightPath,
} from "./families/t5/weights";
export { parseWhisperConfig, parseWhisperFeatureExtractorConfig } from "./families/whisper/config";
export {
  createWhisperMelFilterBank,
  prepareWhisperAudioFeatures,
} from "./families/whisper/preprocessing";
export type {
  WhisperActivation,
  WhisperAudioFeatures,
  WhisperConfig,
  WhisperFeatureExtractorConfig,
} from "./families/whisper/types";
export {
  type ContinuousBatchAdmissionBudgetSnapshot,
  type ContinuousBatchAdmissionController,
  type ContinuousBatchAdmissionDecision,
  type ContinuousBatchAdmissionRequest,
  type ContinuousBatchAdmissionReservation,
  type ContinuousBatchEvent,
  type ContinuousBatchSchedulerEvent,
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
  type CacheLayerKind,
  KVCache,
  LayerPatternBatchKVCache,
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
} from "./lora/adapters";
export { expectTrainableModule } from "./lora/module-traversal";
export {
  type LoRATargetPreset,
  type ResolvedLoRATargets,
  type ResolveLoRATargetsOptions,
  resolveLoRATargets,
} from "./lora/targets";
export { disposePreparedPrompt, slicePreparedPrompt } from "./prepared-prompt";
export { createProgressReporter } from "./pretrained/progress";
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
  PreparedPrompt,
  PretrainedLoadProgressEvent,
  PromptCacheSnapshotEvent,
  SamplerOptions,
  SupportedModelFamily,
  TextGenerationResult,
  TokenGenerationEvent,
  TransformerBatchCache,
  TransformerCache,
  TransformerCacheForkOptions,
  TransformerCacheSnapshot,
} from "./types";
export {
  ConfigParseError,
  MissingWeightsError,
  UnsupportedModelError,
  WeightMismatchError,
} from "./types";
