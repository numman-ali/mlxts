/**
 * Pretrained decoder loading and explicit generation for mlxts.
 * @module
 */

export { AutoModel, AutoTokenizer } from "./auto";
export { type ChatMessage, type ChatTemplate, loadChatTemplate } from "./chat-template";
export { generateStep, generateText, generateTokens, makePromptCache } from "./generation";
export { KVCache, LayerPatternKVCache, SlidingWindowKVCache } from "./infrastructure/cache";
export { loadCausalLM, loadPretrainedTokenizer, resolvePretrainedSource } from "./load";
export { FAMILY_REGISTRY, resolveFamily } from "./registry";
export type {
  BaseModelConfig,
  CausalLM,
  FamilyRegistration,
  ForwardOptions,
  GenerationOptions,
  GenerationResult,
  LoadCausalLMOptions,
  LoadPretrainedTokenizerOptions,
  LoadSourceOptions,
  PretrainedLoadProgressEvent,
  SamplerOptions,
  SupportedModelFamily,
  TransformerCache,
} from "./types";
export {
  ConfigParseError,
  MissingWeightsError,
  UnsupportedModelError,
  WeightMismatchError,
} from "./types";
