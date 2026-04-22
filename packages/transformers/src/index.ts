/**
 * Pretrained decoder loading and explicit generation for mlxts.
 * @module
 */

export { AutoModel, AutoTokenizer } from "./auto";
export { type ChatMessage, type ChatTemplate, loadChatTemplate } from "./chat-template";
export {
  generateStep,
  generateText,
  generateTextStream,
  generateTokens,
  makePromptCache,
} from "./generation";
export { KVCache, LayerPatternKVCache, SlidingWindowKVCache } from "./infrastructure/cache";
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
  CausalLM,
  FamilyRegistration,
  ForwardOptions,
  GenerationDefaults,
  GenerationOptions,
  GenerationResult,
  LoadCausalLMOptions,
  LoadPretrainedTokenizerOptions,
  LoadSourceOptions,
  PretrainedLoadProgressEvent,
  SamplerOptions,
  SupportedModelFamily,
  TextGenerationResult,
  TransformerCache,
} from "./types";
export {
  ConfigParseError,
  MissingWeightsError,
  UnsupportedModelError,
  WeightMismatchError,
} from "./types";
