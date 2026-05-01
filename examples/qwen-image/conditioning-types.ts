import type { MxArray } from "@mlxts/core";
import type { QwenImageConditioning } from "@mlxts/diffusion";
import type { Tokenizer } from "@mlxts/tokenizers";
import type { LoadCausalLMOptions, LoadPretrainedTokenizerOptions } from "@mlxts/transformers";

export type QwenImagePromptConditioningOptions = {
  prompt: string;
  negativePrompt?: string;
  trueCfgScale?: number;
  maxSequenceLength?: number;
};

export type QwenImagePromptConditioning = Disposable & {
  readonly batchSize: number;
  readonly conditioning: QwenImageConditioning;
  readonly promptTruncated: boolean;
  readonly negativePromptTruncated: boolean;
};

export type QwenImagePromptConditioner = Disposable & {
  readonly pipelineKind: "qwen-image";
  encodePrompt(options: QwenImagePromptConditioningOptions): QwenImagePromptConditioning;
};

export type QwenImageTextModelOutput = {
  lastHiddenState: MxArray;
  hiddenStates?: MxArray[];
};

export type QwenImageQwen2_5VLTextModel = {
  runWithHiddenStates(
    inputIds: MxArray,
    options?: { outputHiddenStates?: boolean },
  ): QwenImageTextModelOutput;
};

export type QwenImageQwen2_5VLTextEncoder = Disposable & {
  readonly model: QwenImageQwen2_5VLTextModel;
};

export type QwenImagePromptConditionerComponents = {
  tokenizer: Tokenizer;
  textEncoder: QwenImageQwen2_5VLTextEncoder;
};

export type LoadQwenImagePromptConditionerOptions = {
  tokenizer?: LoadPretrainedTokenizerOptions;
  textEncoder?: LoadCausalLMOptions;
};
